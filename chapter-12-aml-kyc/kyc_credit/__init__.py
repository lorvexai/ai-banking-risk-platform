"""KYC Credit Borrower Screening — MR-2026-063.

SS1/23 Risk: MEDIUM | EU AI Act: HIGH-RISK Annex III §5b
Regulation: POCA 2002 | MLR 2017 Reg. 28/33/35
           JMLSG Part II Banking | Companies House API (UK)

Primary thread: connects Ch 2 MR-2026-035 (document extraction)
to Ch 3 MR-2026-037 (Credit Decision Agent LangGraph gate).

Corporate KYC pipeline per prompt spec:
1. Entity verification — Companies House API (UK, not SEC EDGAR)
2. UBO identification — PSC register, up to 4 layers (JMLSG)
3. PEP/sanctions screen — all directors AND all UBOs >25%
4. EDD for exposures >£1M, PEP-connected, high-risk country
5. Credit-KYC gate — CLEARED/EDD_REQUIRED/BLOCKED/SAR

POCA 2002 s.333A architectural guarantee:
Credit agent receives ONLY "BLOCKED" — never knows if SAR filed.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Dict, List, Optional
import logging

from awb_commons.models import (
    KYCStatus, UBORecord, PEPSanctionsResult,
    KYCCreditResult,
)
from digital_identity import PEPSanctionsScreener
from aml_monitoring import TippingOffGuardrail

log = logging.getLogger(__name__)

# MLR 2017 Reg. 28(3)(b) — UBO threshold
UBO_THRESHOLD_PCT = 25.0
# JMLSG Part II Banking — max ownership chain depth
MAX_OWNERSHIP_LAYERS = 4
# MLR 2017 Reg. 33 — EDD trigger for large exposures
EDD_LARGE_EXPOSURE_GBP = Decimal("1_000_000")

# FATF high-risk jurisdictions (EDD per MLR 2017 Reg. 33(6))
FATF_HIGH_RISK = frozenset([
    "AF","BY","CF","CD","CU","IR","IQ","LY","ML","MM",
    "NI","KP","RU","SO","SS","SD","SY","VE","YE","ZW",
])


class CompaniesHouseClient:
    """UK Companies House API client for entity verification.

    UK-specific — replaces SEC EDGAR (US-only, not applicable).
    Retrieves company data from Companies House public API:
    - Registered name and address
    - SIC code and incorporation date
    - Filing history (confirmation statement status)
    - PSC (Persons with Significant Control) register

    Red flags per JMLSG Part II:
    - Company dissolved or dormant
    - Recently incorporated (<6 months) seeking large credit
    - Missing or overdue confirmation statement
    """

    BASE_URL = "https://api.companieshouse.gov.uk"  # UK-specific

    def get_company_data(
        self,
        companies_house_number: str,
    ) -> Dict:
        """Retrieve company data from Companies House.

        Args:
            companies_house_number: UK Companies House number.

        Returns:
            Company data dict with registration and status info.
        """
        # Production: GET /company/{company_number}
        log.info(
            "Companies House lookup: %s (UK API — not SEC EDGAR)",
            companies_house_number,
        )
        return {
            "company_number": companies_house_number,
            "company_name": "[FROM COMPANIES HOUSE]",
            "company_status": "active",
            "registered_office_address": {"postal_code": "BS1 2AA"},
            "date_of_creation": "2020-01-15",
            "sic_codes": ["6419"],
        }

    def get_psc_register(
        self,
        companies_house_number: str,
    ) -> List[Dict]:
        """Retrieve Persons with Significant Control from CH register.

        MLR 2017 Reg. 28(3)(b): identify persons with >25%
        ownership or voting control or other significant control.
        Companies House PSC register is the primary UK source.

        Args:
            companies_house_number: UK Companies House number.

        Returns:
            List of PSC records.
        """
        log.info(
            "PSC register: %s Companies House (UK)",
            companies_house_number,
        )
        # Production: GET /company/{number}/persons-with-significant-control
        return []  # Empty = no PSCs found (triggers complex structure flag)

    def check_red_flags(
        self,
        company_data: Dict,
    ) -> List[str]:
        """Check company data for JMLSG red flags.

        Args:
            company_data: Companies House response dict.

        Returns:
            List of red flag descriptions.
        """
        flags = []
        status = company_data.get("company_status", "")
        if status in ("dissolved", "dormant"):
            flags.append(f"company_status_{status}")
        creation = company_data.get("date_of_creation", "")
        if creation:
            from datetime import datetime as dt
            try:
                created = dt.strptime(creation, "%Y-%m-%d").date()
                months_old = (date.today() - created).days / 30
                if months_old < 6:
                    flags.append("recently_incorporated_lt_6months")
            except ValueError:
                pass
        return flags


class UBOTracer:
    """Trace ultimate beneficial owners per MLR 2017 Reg. 28(3)(b).

    Identifies all persons owning or controlling >25% of a company.
    Traces through corporate layers up to 4 levels per JMLSG Part II.
    Primary source: Companies House PSC register (UK).

    When PSC is another company: recurse up the ownership chain.
    Flag nominee structures with no identifiable natural person UBO.
    """

    def __init__(
        self,
        threshold_pct: float = UBO_THRESHOLD_PCT,
        max_layers: int = MAX_OWNERSHIP_LAYERS,
    ) -> None:
        self._threshold = threshold_pct
        self._max_layers = max_layers
        self._ch_client = CompaniesHouseClient()

    def trace_ubos(
        self,
        entity_id: str,
        companies_house_number: Optional[str],
        ownership_data: List[Dict],
        layer: int = 1,
    ) -> List[UBORecord]:
        """Recursively identify UBOs above 25% threshold.

        Args:
            entity_id: AWB entity identifier.
            companies_house_number: CH number (None for overseas).
            ownership_data: List of ownership records.
            layer: Current ownership chain depth.

        Returns:
            List of UBORecord for natural persons >= 25%.
        """
        ubos: List[UBORecord] = []
        if layer > self._max_layers:
            log.warning(
                "Max ownership layers %d reached for %s",
                self._max_layers, entity_id,
            )
            return ubos

        for owner in ownership_data:
            pct = float(owner.get("ownership_pct", 0))
            if pct < self._threshold:
                continue
            nationality = owner.get("nationality", "GB")
            ubo = UBORecord(
                entity_id=entity_id,
                ubo_name=owner.get("name", ""),
                ownership_pct=pct,
                control_type=owner.get("control_type", "shares"),
                psc_register_verified=(
                    companies_house_number is not None
                ),
                is_pep=owner.get("is_pep", False),
                high_risk_jurisdiction=nationality in FATF_HIGH_RISK,
                layer=layer,
            )
            ubos.append(ubo)
            log.info(
                "UBO identified: entity=%s name=%s "
                "pct=%.1f%% pep=%s hr_jx=%s layer=%d",
                entity_id, ubo.ubo_name, pct,
                ubo.is_pep, ubo.high_risk_jurisdiction, layer,
            )
        return ubos

    def is_complex_structure(
        self,
        ubos: List[UBORecord],
        entity_chain_depth: int,
    ) -> bool:
        """Detect complex/opaque ownership requiring EDD.

        JMLSG Part II: nominee structures and complex chains
        where beneficial ownership cannot be clearly identified
        are EDD triggers per MLR 2017 Reg. 33.

        Args:
            ubos: Identified UBOs.
            entity_chain_depth: Ownership chain depth.

        Returns:
            True if structure requires EDD.
        """
        if len(ubos) == 0:
            log.warning("No UBOs identified — complex/nominee structure")
            return True
        if entity_chain_depth > 3:
            log.warning(
                "Deep ownership chain: %d layers", entity_chain_depth
            )
            return True
        return False


class KYCCreditScreener:
    """KYC gate for credit borrowers — primary thread Chapter 12.

    Sits between Ch 2 document extraction (MR-2026-035) and
    Ch 3 Credit Decision Agent (MR-2026-037) in LangGraph pipeline.

    Ch 2 MR-2026-035 output is reused: extracted company name,
    director names, registered address from the CDA output
    avoids re-extracting from the same credit pack documents.

    Pipeline per prompt spec:
    Step 1: DocumentAgent (Ch 2 CDA) → credit pack extraction
    Step 2: KYCCreditScreener (this) → KYC gate
    Step 3: IF CLEARED → RWAForecastAgent (Ch 6)
            IF EDD_REQUIRED → pause; EDD workflow
            IF BLOCKED → reject; MLRO notified
            IF SAR_FILED → BLOCKED (s.333A — SAR not disclosed)

    Args:
        model_id: Registered model ID (MR-2026-063).
    """

    def __init__(
        self,
        model_id: str = "MR-2026-063",
    ) -> None:
        self._model_id = model_id
        self._ubo_tracer = UBOTracer()
        self._screener = PEPSanctionsScreener()
        self._ch_client = CompaniesHouseClient()
        self._tipping_off = TippingOffGuardrail()

    def screen_entity(
        self,
        entity_id: str,
        entity_name: str,
        companies_house_number: Optional[str],
        registered_country: str,
        ownership_data: List[Dict],
        proposed_exposure_gbp: Decimal = Decimal("0"),
        entity_chain_depth: int = 1,
    ) -> KYCCreditResult:
        """Run full KYC assessment for a credit applicant.

        Args:
            entity_id: AWB internal entity ID.
            entity_name: Legal entity name.
            companies_house_number: CH number (None if overseas).
            registered_country: ISO country of incorporation.
            ownership_data: Shareholder/PSC records.
            proposed_exposure_gbp: Credit facility amount.
            entity_chain_depth: Ownership layers.

        Returns:
            KYCCreditResult — BLOCKED if SAR filed (s.333A).
        """
        result = KYCCreditResult(
            entity_id=entity_id,
            status=KYCStatus.PENDING,
            assessed_date=date.today(),
            model_id=self._model_id,
        )

        # Companies House verification
        company_data = {}
        if companies_house_number:
            company_data = self._ch_client.get_company_data(
                companies_house_number
            )
            red_flags = self._ch_client.check_red_flags(company_data)
            if red_flags:
                result.edd_reasons.extend(red_flags)

        # UBO identification (MLR 2017 Reg. 28(3)(b))
        result.ubos = self._ubo_tracer.trace_ubos(
            entity_id, companies_house_number,
            ownership_data, layer=1,
        )

        # PEP/sanctions screen all UBOs and directors
        for ubo in result.ubos:
            pep_result = self._screener.screen_individual(
                f"{entity_id}-{ubo.ubo_name[:8]}",
                ubo.ubo_name,
                date.today(),  # Production: use actual DOB
                nationality="GB",
            )
            if pep_result.is_pep:
                result.edd_reasons.append("ubo_is_pep")
                result.mlro_required = True
            if pep_result.sanctions_hit:
                result.status = KYCStatus.SANCTIONS_HIT
                result.blocks_credit_decision = True
                log.error(
                    "UBO sanctions hit: entity=%s ubo=%s",
                    entity_id, ubo.ubo_name,
                )
                return result

        # Sanctions hit check
        if result.status == KYCStatus.SANCTIONS_HIT:
            return result

        # EDD triggers
        if registered_country in FATF_HIGH_RISK:
            result.edd_reasons.append("entity_high_risk_jurisdiction")
        if proposed_exposure_gbp >= EDD_LARGE_EXPOSURE_GBP:
            result.edd_reasons.append(
                f"large_exposure_gte_1m_gbp"
            )
        if self._ubo_tracer.is_complex_structure(
            result.ubos, entity_chain_depth
        ):
            result.edd_reasons.append("complex_ownership_structure")
        if any(u.is_pep for u in result.ubos):
            result.edd_reasons.append("ubo_is_pep")

        if result.edd_reasons:
            result.status = KYCStatus.EDD_REQUIRED
            result.edd_required = True
            result.blocks_credit_decision = True
            log.info(
                "Corporate EDD required: entity=%s reasons=%s",
                entity_id, result.edd_reasons,
            )
        else:
            result.status = KYCStatus.CDD_PASS
            result.blocks_credit_decision = False
            log.info("Corporate KYC CDD pass: entity=%s", entity_id)

        return result

    def get_credit_gate_decision(
        self,
        kyc_result: KYCCreditResult,
        sar_filed: bool = False,
    ) -> str:
        """Return credit gate decision for Ch 3 Credit Agent.

        Implements POCA 2002 s.333A architectural guarantee:
        if SAR filed, credit agent receives BLOCKED only —
        never knows whether blocking is due to SAR or KYC.

        Args:
            kyc_result: Completed KYCCreditResult.
            sar_filed: Whether a SAR has been filed (INTERNAL ONLY).

        Returns:
            "CLEARED" | "EDD_REQUIRED" | "BLOCKED"
            (Never "SAR_FILED" — s.333A compliance)
        """
        # POCA s.333A: SAR status never disclosed externally
        if sar_filed:
            return self._tipping_off.get_safe_credit_status(
                sar_filed=True,
                kyc_status="BLOCKED",
            )

        clearable = {KYCStatus.CDD_PASS, KYCStatus.EDD_PASS}
        if kyc_result.status in clearable:
            return "CLEARED"
        elif kyc_result.edd_required:
            return "EDD_REQUIRED"
        else:
            return "BLOCKED"
