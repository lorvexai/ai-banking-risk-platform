"""Shared dataclass schemas for Chapter 12 AML, KYC, Financial Crime.

Model Registry:
    MR-2026-062: Digital Identity Verification Platform
                 SS1/23 Risk: HIGH | EU AI Act: HIGH-RISK Annex III §6
                 (biometric identification of natural persons)
    MR-2026-061: AML Transaction Monitoring System
                 SS1/23 Risk: HIGH | EU AI Act: LIMITED scope
    MR-2026-063: KYC Credit Borrower Screening
                 SS1/23 Risk: MEDIUM | EU AI Act: HIGH-RISK Annex III §5b

Primary regulations:
    POCA 2002 — UK primary AML statute (s.327–333A)
    MLR 2017  — Money Laundering Regulations 2017
    FCA SYSC 6.3 — systems and controls
    JMLSG Guidance — Part I and Part II Banking
    FATF 40 Recommendations — international standard
    NEVER: BSA / FinCEN / PATRIOT Act — US-only, not applicable
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import List, Optional
import logging

log = logging.getLogger(__name__)


class KYCStatus(Enum):
    """Customer KYC decision status."""
    PENDING = "PENDING"
    CDD_PASS = "CDD_PASS"           # Standard CDD cleared
    EDD_REQUIRED = "EDD_REQUIRED"   # Enhanced Due Diligence needed
    EDD_PASS = "EDD_PASS"           # EDD completed and cleared
    DECLINED = "DECLINED"           # Identity verification failed
    PEP_FLAGGED = "PEP_FLAGGED"     # Politically Exposed Person
    SANCTIONS_HIT = "SANCTIONS_HIT"  # OFSI/UN list match


class AlertPriority(Enum):
    """AML alert priority tier."""
    HIGH = "HIGH"       # score >= 0.70; MLRO escalation at 0.90
    MEDIUM = "MEDIUM"   # score 0.35–0.70
    LOW = "LOW"         # score < 0.35; auto-cleared


class SARStatus(Enum):
    """SAR lifecycle per POCA 2002 s.330."""
    DRAFT = "DRAFT"
    MLRO_REVIEW = "MLRO_REVIEW"
    LEGAL_CLEARED = "LEGAL_CLEARED"
    SUBMITTED = "SUBMITTED"             # Filed to NCA SubmitSAR
    NCA_ACKNOWLEDGED = "NCA_ACKNOWLEDGED"
    CONSENT_AWAITED = "CONSENT_AWAITED"  # s.335 moratorium (7 days)


@dataclass
class KYCDocumentExtract:
    """Structured output from Gemini 3.5 Flash document verification.

    Reuses MR-2026-035 (Chapter 2 Credit Document Analyser)
    extraction pipeline with a KYC-specific output schema.
    POCA 2002 / MLR 2017 Reg. 28: identify the customer using
    reliable, independent source documents.
    """
    document_type: str          # "passport", "driving_licence", "utility_bill"
    full_name: str
    date_of_birth: date
    document_number: str
    expiry_date: date
    issuing_country: str        # ISO 3166-1 alpha-2
    address: Optional[str] = None
    mrz_valid: bool = False     # Machine Readable Zone checksum
    confidence: float = 0.0    # 0.0–1.0 overall extraction confidence
    verification_status: str = "PENDING"
    model_id: str = "MR-2026-062"


@dataclass
class PEPSanctionsResult:
    """PEP and sanctions screening result.

    UK sources: OFSI (HM Treasury), UN SC Consolidated List.
    NOT OFAC (US Treasury) — UK firms screen against OFSI.
    MLR 2017 Reg. 35: enhanced obligations for PEPs.
    """
    customer_id: str
    name_screened: str
    is_pep: bool = False
    pep_category: Optional[str] = None  # "domestic", "foreign", "io"
    pep_look_back_months: int = 12      # MLR 2017 Reg. 35 — 12 months
    sanctions_hit: bool = False
    sanctions_lists_matched: List[str] = field(default_factory=list)
    match_score: float = 0.0    # Jaro-Winkler fuzzy match (0.0–1.0)
    # Thresholds per prompt spec:
    # >= 0.95: auto-block | 0.85–0.95: compliance review | < 0.85: clear
    requires_edd: bool = False
    screened_at: Optional[datetime] = None
    screening_source_version: Optional[str] = None


@dataclass
class KYCDecision:
    """Complete KYC onboarding decision record.

    Audit trail retained 7 years per AWB policy
    (SYSC 6.3.3R minimum 5 years; AWB exceeds this).
    JMLSG Part I para 7.28: retain CDD records.
    """
    customer_id: str
    decision_date: date
    status: KYCStatus
    document_extract: Optional[KYCDocumentExtract] = None
    pep_sanctions: Optional[PEPSanctionsResult] = None
    liveness_score: float = 0.0
    liveness_passed: bool = False
    edd_trigger: Optional[str] = None
    narrative: Optional[str] = None   # LLM-generated KYC narrative
    decided_by: str = "MR-2026-062"
    review_required: bool = False
    # UK GDPR / DPA 2018: biometric data not stored post-verification
    biometric_template_deleted: bool = True


@dataclass
class AlertResult:
    """AML alert produced by the transaction monitoring ML system.

    JMLSG Part I Chapter 6: alerts must be investigated;
    FCA SYSC 6.3.2: systems to identify suspicious transactions.
    Retention: 5 years from alert date (SYSC 6.3.3R);
    AWB retains 7 years.
    """
    alert_id: str
    transaction_id: str
    account_id: str
    score: float                # XGBoost probability 0.0–1.0
    priority: AlertPriority
    features: List[str] = field(default_factory=list)
    shap_values: dict = field(default_factory=dict)
    typology_matches: List[str] = field(default_factory=list)  # RAG
    network_risk_flag: bool = False
    created_at: Optional[datetime] = None
    model_id: str = "MR-2026-061"
    # Thresholds per prompt spec:
    # 0.35 = alert threshold | 0.70 = high-priority | 0.90 = auto-MLRO


@dataclass
class NetworkRiskSummary:
    """Output from NetworkX Louvain community detection.

    Detects coordinated structuring patterns across customer
    accounts. Key JMLSG typology: structuring / smurfing.
    """
    community_id: str
    member_account_ids: List[str]
    total_amount_gbp: Decimal
    transaction_count: int
    is_structuring_ring: bool = False
    centrality_accounts: List[str] = field(default_factory=list)
    detection_method: str = "louvain_community_detection"
    model_id: str = "MR-2026-061"


@dataclass
class SARDraft:
    """AI-assisted Suspicious Activity Report draft.

    POCA 2002 s.330: nominated officer (MLRO) must disclose
    knowledge or suspicion of money laundering to NCA.
    s.333A: tipping off is a criminal offence — customer
    must never be informed that a SAR has been filed.
    NCA SubmitSAR API — UK National Crime Agency (NOT FinCEN).
    Retention: 5 years from submission (SYSC 6.3.3R);
    AWB retains 7 years.
    """
    sar_id: str
    customer_id: str
    account_id: str
    alert_ids: List[str]
    total_suspicious_amount_gbp: Decimal
    # NCA SAR structure sections (b) and (c) — LLM-generated
    nature_of_suspicion: str
    typology_citation: str    # JMLSG typology reference
    financial_details: str
    status: SARStatus = SARStatus.DRAFT
    poca_section: str = "s.330"
    sar_type: str = "disclosure"  # "disclosure" or "consent" (s.335)
    mlro_id: Optional[str] = None
    nca_reference: Optional[str] = None
    submitted_at: Optional[datetime] = None
    # POCA 2002 s.333A compliance
    requires_mlro_approval: bool = True    # ALWAYS True
    tipping_off_guardrail_active: bool = True  # ALWAYS True
    model_id: str = "MR-2026-063"

    def __post_init__(self):
        # Architectural enforcement: these MUST always be True
        if not self.requires_mlro_approval:
            raise ValueError(
                "POCA 2002 s.331: requires_mlro_approval cannot "
                "be False. MLRO must approve all SARs."
            )
        if not self.tipping_off_guardrail_active:
            raise ValueError(
                "POCA 2002 s.333A: tipping_off_guardrail_active "
                "cannot be False. Criminal offence to disable."
            )


@dataclass
class UBORecord:
    """Ultimate Beneficial Owner record for corporate KYC.

    MLR 2017 Reg. 28(3)(b): identify persons owning >25%.
    Companies House PSC register: primary UK source.
    Up to 4 ownership layers per JMLSG Part II Banking guidance.
    """
    entity_id: str
    ubo_name: str
    ownership_pct: float
    control_type: str       # "shares", "voting_rights", "other_means"
    psc_register_verified: bool = False  # Companies House (UK)
    is_pep: bool = False
    high_risk_jurisdiction: bool = False
    layer: int = 1          # Ownership chain depth (max 4 per JMLSG)
    UBO_THRESHOLD_PCT: float = 25.0  # MLR 2017 Reg. 28(3)(b)

    @property
    def requires_edd(self) -> bool:
        """EDD required if PEP or high-risk jurisdiction (MLR 2017 Reg. 33)."""
        return self.is_pep or self.high_risk_jurisdiction


@dataclass
class KYCCreditResult:
    """KYC gate result for the Chapter 3 Credit Decision Agent.

    Integrates into MR-2026-037 LangGraph state machine.
    POCA 2002 s.333A: if SAR filed, status = BLOCKED only.
    Credit agent NEVER knows whether a SAR has been filed.
    """
    entity_id: str
    status: KYCStatus
    edd_required: bool = False
    edd_reasons: List[str] = field(default_factory=list)
    mlro_required: bool = False
    ubos: List[UBORecord] = field(default_factory=list)
    pep_sanctions: Optional[PEPSanctionsResult] = None
    assessed_date: Optional[date] = None
    # Credit gate integration
    blocks_credit_decision: bool = False
    # POCA s.333A: SAR status NEVER disclosed to credit agent
    # This field is deliberately absent — credit agent only sees status
    model_id: str = "MR-2026-063"
