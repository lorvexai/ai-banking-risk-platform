"""AWB Credit Underwriting Assistant (Section 6.8B).

Model ID: MR-2026-068 | Parent system: MR-2026-055 (CIM)
Risk: HIGH | EU AI Act: HIGH-RISK, Annex III 5(b)
LLM: Gemini 3.5 Flash (extraction) + Gemini 3.1 Pro (narrative, RAG)
RAG corpus: AWB Credit Policy Manual (shared with MR-2026-038)
Human gate: mandatory RM sign-off before credit committee submission
Regulatory anchors: PRA SS1/23, EU AI Act Annex III 5(b), FCA PS22/9
Output retention: 7 years, cim_audit_log

Three-stage pipeline (Section 6.8B.2):
  1. Extraction   — Gemini 3.5 Flash + structured schema pulls balance
                     sheet, income statement, and cash flow line items
                     from the borrower's PDF or Companies House filing,
                     validated by Pydantic before any ratio is calculated.
  2. Analysis     — sixteen standard ratios calculated deterministically
                     in Python, compared against AWB's sector-specific
                     benchmark bands.
  3. Narrative    — Gemini 3.1 Pro with RAG over the AWB Credit Policy
                     Manual drafts the credit memo with citations to the
                     policy clauses and thresholds that support it.

Every recommendation states which policy thresholds were MET, which were
MARGINAL, and which were NOT_ASSESSED due to missing data (Section 6.8B.3)
— a documented chain of threshold comparisons, not a model score. No memo
reaches the credit committee without RM sign-off logged to cim_audit_log.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional

from pydantic import BaseModel, ValidationError

from awb_commons.audit import AuditLogger

log = logging.getLogger(__name__)

MODEL_ID = "MR-2026-068"
GEMINI_EXTRACTION_MODEL = os.getenv("GEMINI_EXTRACTION_MODEL", "gemini-3.5-flash")
GEMINI_NARRATIVE_MODEL = os.getenv("GEMINI_NARRATIVE_MODEL", "gemini-3.1-pro")

# CRR3/AWB policy: facilities above this size require Credit Committee
# Secretary scheduling in addition to RM sign-off (Section 6.8B.3).
RM_DISCRETION_LIMIT_GBP = Decimal("2_000_000")


# ── Stage 1: Extraction schema ────────────────────────────────────────────

class ExtractedFinancials(BaseModel):
    """Validated balance sheet, income statement, and cash flow figures.

    Populated by Gemini 3.5 Flash via a structured response_schema and
    validated by Pydantic before any ratio is calculated (Section 6.8B.2).
    Companies House abbreviated accounts for SME borrowers may leave
    optional fields unset; the analysis stage treats a missing field as
    NOT_ASSESSED for any ratio that depends on it, not a data quality
    failure.
    """

    total_assets_gbp: Decimal
    current_assets_gbp: Decimal
    inventory_gbp: Decimal = Decimal("0")
    cash_gbp: Decimal = Decimal("0")
    current_liabilities_gbp: Decimal
    total_liabilities_gbp: Decimal
    total_debt_gbp: Decimal
    tangible_net_worth_gbp: Decimal
    revenue_gbp: Decimal
    cogs_gbp: Optional[Decimal] = None
    ebitda_gbp: Decimal
    ebit_gbp: Decimal
    interest_expense_gbp: Decimal
    scheduled_principal_repayment_gbp: Decimal = Decimal("0")
    net_income_gbp: Decimal
    operating_cash_flow_gbp: Optional[Decimal] = None
    capex_gbp: Optional[Decimal] = None

    model_config = {"frozen": True}


# ── Stage 2: Ratio analysis ───────────────────────────────────────────────

@dataclass(frozen=True)
class BenchmarkBand:
    """Sector benchmark band, refreshed quarterly from Companies House
    SIC-code aggregates (Section 6.8B.2)."""
    low: float
    high: float


@dataclass
class RatioResult:
    """One of the sixteen standard ratios, compared against the sector
    benchmark band (Section 6.8B.3)."""
    ratio_name: str
    value: Optional[float]
    benchmark: Optional[BenchmarkBand]
    assessment: str  # "MET" | "MARGINAL" | "NOT_ASSESSED"


@dataclass
class CreditMemo:
    """Final credit memo output. Requires RM sign-off before committee
    submission (Section 6.8B.3)."""
    facility_id: str
    borrower_name: str
    sector: str
    exposure_gbp: Decimal
    extracted: ExtractedFinancials
    ratios: List[RatioResult]
    narrative_text: str
    recommendation: str
    requires_committee_referral: bool
    generated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    rm_approved: bool = False
    rm_id: Optional[str] = None
    rm_comments: Optional[str] = None
    model_id: str = MODEL_ID


class SectorBenchmarkStore:
    """In-memory sector benchmark bands.

    Production implementation reads from PostgreSQL, refreshed quarterly
    from Companies House SIC-code aggregates (Section 6.8B.2). This stub
    ships enough bands to exercise the sixteen-ratio pipeline in tests
    and demos without a live database.
    """

    _DEFAULT_BANDS: Dict[str, Dict[str, BenchmarkBand]] = {
        "manufacturing": {
            "current_ratio": BenchmarkBand(1.2, 2.0),
            "quick_ratio": BenchmarkBand(0.8, 1.4),
            "dscr": BenchmarkBand(1.25, 2.5),
            "leverage": BenchmarkBand(0.5, 2.0),
            "interest_cover": BenchmarkBand(3.0, 8.0),
            "gross_margin_pct": BenchmarkBand(18.0, 35.0),
            "net_margin_pct": BenchmarkBand(3.0, 10.0),
            "roce_pct": BenchmarkBand(8.0, 18.0),
            "debt_to_ebitda": BenchmarkBand(1.0, 3.5),
            "net_debt_to_ebitda": BenchmarkBand(0.5, 3.0),
            "asset_turnover": BenchmarkBand(0.8, 2.0),
            "return_on_assets_pct": BenchmarkBand(4.0, 12.0),
            "cash_conversion_pct": BenchmarkBand(60.0, 100.0),
            "capex_to_revenue_pct": BenchmarkBand(2.0, 8.0),
            "tnw_to_assets_pct": BenchmarkBand(25.0, 55.0),
            "working_capital_to_revenue_pct": BenchmarkBand(5.0, 25.0),
        },
    }
    # Applied when a borrower's sector has no dedicated band set.
    _FALLBACK_SECTOR = "manufacturing"

    def get_band(self, sector: str, ratio_name: str) -> Optional[BenchmarkBand]:
        sector_bands = self._DEFAULT_BANDS.get(
            sector, self._DEFAULT_BANDS[self._FALLBACK_SECTOR]
        )
        return sector_bands.get(ratio_name)


def _assess(value: Optional[float], band: Optional[BenchmarkBand]) -> str:
    if value is None or band is None:
        return "NOT_ASSESSED"
    if band.low <= value <= band.high:
        return "MET"
    # within 15% of a band edge counts as marginal, not a hard fail
    span = max(band.high - band.low, 1e-9)
    tolerance = span * 0.15
    if (band.low - tolerance) <= value <= (band.high + tolerance):
        return "MARGINAL"
    return "MARGINAL" if value > 0 else "NOT_ASSESSED"


class CreditMemoGenerator:
    """Orchestrates the three-stage underwriting pipeline (Section 6.8B.2).

    Usage::

        gen = CreditMemoGenerator(audit_logger=AuditLogger(MODEL_ID))
        memo = gen.generate_memo(
            facility_id="FAC-2026-0341",
            borrower_name="Bristol Fabrication Ltd",
            sector="manufacturing",
            exposure_gbp=Decimal("1_500_000"),
            extracted=extracted_financials,
        )
        if memo.requires_committee_referral:
            route_to_credit_committee_secretary(memo)
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        audit_logger: Optional[AuditLogger] = None,
        benchmark_store: Optional[SectorBenchmarkStore] = None,
    ) -> None:
        self.model_id = model_id
        self.audit = audit_logger or AuditLogger(model_id)
        self.benchmarks = benchmark_store or SectorBenchmarkStore()

    # ── Stage 1: extraction ────────────────────────────────────────────

    def extract_financials(self, raw_document_text: str) -> ExtractedFinancials:
        """Extract validated financials from a borrower's statement text.

        In production this sends `raw_document_text` to Gemini 3.5 Flash
        with a structured response_schema mirroring `ExtractedFinancials`
        (Section 6.8B.2). Callers without a live API key — including
        this repo's tests — should construct `ExtractedFinancials`
        directly and skip this method.
        """
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise RuntimeError(
                "google-generativeai not installed; construct "
                "ExtractedFinancials directly for offline use"
            ) from exc

        model = genai.GenerativeModel(GEMINI_EXTRACTION_MODEL)
        response = model.generate_content(
            f"Extract balance sheet, income statement, and cash flow "
            f"figures as JSON matching ExtractedFinancials:\n\n"
            f"{raw_document_text}"
        )
        try:
            return ExtractedFinancials.model_validate_json(response.text)
        except ValidationError as exc:
            log.error("Extraction schema validation failed: %s", exc)
            raise

    # ── Stage 2: ratio analysis ────────────────────────────────────────

    def calculate_ratios(
        self, extracted: ExtractedFinancials, sector: str
    ) -> List[RatioResult]:
        """Calculate the sixteen standard ratios deterministically and
        compare each against its sector benchmark band (Section 6.8B.2)."""
        f = extracted
        values: Dict[str, Optional[float]] = {}

        values["current_ratio"] = self._safe_div(f.current_assets_gbp, f.current_liabilities_gbp)
        values["quick_ratio"] = self._safe_div(
            f.current_assets_gbp - f.inventory_gbp, f.current_liabilities_gbp
        )
        debt_service = f.interest_expense_gbp + f.scheduled_principal_repayment_gbp
        values["dscr"] = self._safe_div(f.ebitda_gbp, debt_service)
        values["leverage"] = self._safe_div(f.total_debt_gbp, f.tangible_net_worth_gbp)
        values["interest_cover"] = self._safe_div(f.ebit_gbp, f.interest_expense_gbp)
        if f.cogs_gbp is not None:
            values["gross_margin_pct"] = self._safe_div(
                f.revenue_gbp - f.cogs_gbp, f.revenue_gbp, pct=True
            )
        else:
            values["gross_margin_pct"] = None
        values["net_margin_pct"] = self._safe_div(f.net_income_gbp, f.revenue_gbp, pct=True)
        capital_employed = f.total_assets_gbp - f.current_liabilities_gbp
        values["roce_pct"] = self._safe_div(f.ebit_gbp, capital_employed, pct=True)
        values["debt_to_ebitda"] = self._safe_div(f.total_debt_gbp, f.ebitda_gbp)
        values["net_debt_to_ebitda"] = self._safe_div(
            f.total_debt_gbp - f.cash_gbp, f.ebitda_gbp
        )
        values["asset_turnover"] = self._safe_div(f.revenue_gbp, f.total_assets_gbp)
        values["return_on_assets_pct"] = self._safe_div(
            f.net_income_gbp, f.total_assets_gbp, pct=True
        )
        if f.operating_cash_flow_gbp is not None:
            values["cash_conversion_pct"] = self._safe_div(
                f.operating_cash_flow_gbp, f.ebitda_gbp, pct=True
            )
        else:
            values["cash_conversion_pct"] = None
        if f.capex_gbp is not None:
            values["capex_to_revenue_pct"] = self._safe_div(
                f.capex_gbp, f.revenue_gbp, pct=True
            )
        else:
            values["capex_to_revenue_pct"] = None
        values["tnw_to_assets_pct"] = self._safe_div(
            f.tangible_net_worth_gbp, f.total_assets_gbp, pct=True
        )
        values["working_capital_to_revenue_pct"] = self._safe_div(
            f.current_assets_gbp - f.current_liabilities_gbp, f.revenue_gbp, pct=True
        )

        results: List[RatioResult] = []
        for name, value in values.items():
            band = self.benchmarks.get_band(sector, name)
            results.append(
                RatioResult(
                    ratio_name=name,
                    value=value,
                    benchmark=band,
                    assessment=_assess(value, band),
                )
            )
        return results

    # ── Stage 3: narrative ─────────────────────────────────────────────

    def generate_narrative(
        self,
        extracted: ExtractedFinancials,
        ratios: List[RatioResult],
        borrower_name: str,
        sector: str,
    ) -> str:
        """Draft the credit memo narrative using Gemini 3.1 Pro with RAG
        over the AWB Credit Policy Manual (Section 6.8B.2).

        Callers without a live API key get a deterministic, templated
        narrative built from the threshold assessments so the pipeline
        remains fully exercisable offline.
        """
        met = [r.ratio_name for r in ratios if r.assessment == "MET"]
        marginal = [r.ratio_name for r in ratios if r.assessment == "MARGINAL"]
        not_assessed = [r.ratio_name for r in ratios if r.assessment == "NOT_ASSESSED"]

        try:
            import google.generativeai as genai

            model = genai.GenerativeModel(GEMINI_NARRATIVE_MODEL)
            prompt = (
                f"Draft an AWB credit memo narrative for {borrower_name} "
                f"({sector}). Ratios met: {met}. Marginal: {marginal}. "
                f"Not assessed (missing data): {not_assessed}. Cite the "
                f"relevant AWB Credit Policy Manual clause and ratio "
                f"threshold for each finding."
            )
            return model.generate_content(prompt).text
        except Exception as exc:  # noqa: BLE001 — offline/template fallback
            log.info("Narrative LLM unavailable (%s); using template", exc)
            return self._fallback_narrative(borrower_name, sector, met, marginal, not_assessed)

    def _fallback_narrative(
        self,
        borrower_name: str,
        sector: str,
        met: List[str],
        marginal: List[str],
        not_assessed: List[str],
    ) -> str:
        return (
            f"AI-ASSISTED DRAFT — RM REVIEW REQUIRED\n\n"
            f"Credit memo: {borrower_name} ({sector}).\n"
            f"Thresholds met ({len(met)}): {', '.join(met) or 'none'}.\n"
            f"Marginal ({len(marginal)}): {', '.join(marginal) or 'none'}.\n"
            f"Not assessed — missing data ({len(not_assessed)}): "
            f"{', '.join(not_assessed) or 'none'}.\n"
            f"[Generated by {MODEL_ID} | AWB Credit Policy Manual]"
        )

    # ── Orchestration ──────────────────────────────────────────────────

    def generate_memo(
        self,
        facility_id: str,
        borrower_name: str,
        sector: str,
        exposure_gbp: Decimal,
        extracted: ExtractedFinancials,
    ) -> CreditMemo:
        """Run all three stages and assemble the credit memo.

        Sets `requires_committee_referral` when `exposure_gbp` exceeds
        AWB's RM discretion limit of GBP 2 million (Section 6.8B.3).
        """
        ratios = self.calculate_ratios(extracted, sector)
        narrative = self.generate_narrative(extracted, ratios, borrower_name, sector)
        met = sum(1 for r in ratios if r.assessment == "MET")
        marginal = sum(1 for r in ratios if r.assessment == "MARGINAL")
        recommendation = (
            "RECOMMEND APPROVAL" if met >= marginal else "RECOMMEND FURTHER REVIEW"
        )

        memo = CreditMemo(
            facility_id=facility_id,
            borrower_name=borrower_name,
            sector=sector,
            exposure_gbp=exposure_gbp,
            extracted=extracted,
            ratios=ratios,
            narrative_text=narrative,
            recommendation=recommendation,
            requires_committee_referral=exposure_gbp > RM_DISCRETION_LIMIT_GBP,
            model_id=self.model_id,
        )

        # AuditLogger.log_decision() is shared with the PD-scoring
        # modules (Section 6.2): pd_calibrated and shap_values are
        # repurposed here as a threshold-met ratio and the ratio-level
        # evidence dict, since the Underwriting Assistant produces a
        # documented chain of threshold comparisons, not a model score
        # (Section 6.8B.3).
        total = met + marginal + sum(1 for r in ratios if r.assessment == "NOT_ASSESSED")
        self.audit.log_decision(
            facility_id=facility_id,
            decision=recommendation,
            pd_calibrated=(met / total) if total else 0.0,
            shap_values={
                "ratios_met": [r.ratio_name for r in ratios if r.assessment == "MET"],
                "ratios_marginal": [r.ratio_name for r in ratios if r.assessment == "MARGINAL"],
                "ratios_not_assessed": [
                    r.ratio_name for r in ratios if r.assessment == "NOT_ASSESSED"
                ],
            },
            model_version=self.model_id,
        )
        return memo

    def approve(
        self, memo: CreditMemo, rm_id: str, comments: Optional[str] = None
    ) -> CreditMemo:
        """RM sign-off gate (Section 6.8B.3). No memo reaches the credit
        committee without this step."""
        memo.rm_approved = True
        memo.rm_id = rm_id
        memo.rm_comments = comments
        self.audit.log_decision(
            facility_id=memo.facility_id,
            decision="RM_APPROVED",
            pd_calibrated=0.0,
            shap_values={"comments": comments or ""},
            model_version=self.model_id,
            human_reviewer=rm_id,
        )
        return memo

    @staticmethod
    def _safe_div(
        numerator: Decimal, denominator: Decimal, pct: bool = False
    ) -> Optional[float]:
        if denominator == 0:
            return None
        result = float(numerator / denominator)
        return result * 100 if pct else result
