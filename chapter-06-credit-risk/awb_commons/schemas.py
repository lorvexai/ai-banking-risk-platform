"""AWB Commons — shared Pydantic schemas for the credit platform.

Version: 1.0.0 | AWB-AI-2025 programme | June 2026
Namespace: awb_commons

This module defines the data contracts between:
  - Chapter 2: Credit Document Analyser (MR-2026-035)
  - Chapter 3: Credit Decision Agent (MR-2026-037)
  - Chapter 6: PD Model (MR-2026-043), Consumer Scoring (MR-2026-041)
  - Chapter 11: Basel Credit Risk Reporting, COREP C 02.00/C 08.00 (MR-2026-072)

Any change to schemas here requires approval from
the relevant model risk officers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from datetime import datetime


class CreditDecision(str, Enum):
    """Consumer origination decision — MR-2026-041."""
    APPROVE  = "APPROVE"
    REVIEW   = "REVIEW"      # Borderline — human queue
    DECLINE  = "DECLINE"


class ModelRiskRating(str, Enum):
    """PRA SS1/23 risk ratings."""
    HIGH   = "HIGH"
    MEDIUM = "MEDIUM"
    LOW    = "LOW"


@dataclass
class CreditFeatures:
    """14 features consumed by AWB Corporate PD Model.

    Data contract between Chapter 2 (CDA, MR-2026-035)
    and Chapter 6 (PD Model, MR-2026-043).
    Version: 1.0.0 — June 2026
    """
    # Financial ratios (6 features)
    debt_ebitda: float
    interest_cover: float
    current_ratio: float
    net_debt_equity: float
    ebitda_margin: float
    revenue_growth_yoy: float
    # Behavioural signals (4 features)
    days_past_due_12m_max: int
    utilisation_rate_12m_avg: float
    account_conduct_score: float
    payment_pattern_flag: int     # 0=clean, 1=irregular
    # Macroeconomic overlays (2 features)
    uk_gdp_growth_qoq: float
    sector_pmi_index: float
    # Facility characteristics (2 features)
    ltv_secured: float            # 0.0 for unsecured
    tenor_years_remaining: float


@dataclass
class PDModelResult:
    """Output of AWB Corporate PD Model (MR-2026-043).

    Consumed by Chapter 3 RWAForecastAgent via PDModelTool,
    Chapter 11 COREP C 07.00 pipeline, Chapter 16 dashboard.
    """
    pd_calibrated: float       # Platt-scaled PD 0.0–1.0
    shap_values: dict          # {feature_name: shap_value}
    base_value: float          # Mean PD across training set
    model_version: str         # SHA-256 of model artefact
    facility_id: str
    scored_at: datetime = field(
        default_factory=datetime.utcnow
    )

    def exceeds_pd_floor(self) -> bool:
        """CRR3 Art. 160 — PD floor 0.03%."""
        return self.pd_calibrated >= 0.0003

    def top_shap_factors(
        self, n: int = 5
    ) -> list[tuple[str, float]]:
        """Top-n SHAP contributors by absolute value."""
        return sorted(
            self.shap_values.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:n]


@dataclass
class ScoringResult:
    """Output of LightGBM Consumer Scorer (MR-2026-041)."""
    pd_calibrated: float
    decision: CreditDecision
    shap_top3_risk: list[dict]
    shap_top3_protect: list[dict]
    model_version: str
    application_id: str
    scored_at: datetime = field(
        default_factory=datetime.utcnow
    )


@dataclass
class RWAResult:
    """Output of CRR3RWACalculator.

    Used by Chapter 3 RWAForecastAgent and consumed by Chapter 11's
    Basel Credit Risk Reporting module (MR-2026-072) for COREP C 02.00/
    C 08.00 filing.
    """
    facility_id: str
    pd: float
    lgd: float
    ead: float
    maturity: float
    rwa_irb: float           # Pure IRB RWA (£)
    rwa_sa: float            # Standardised RWA (£)
    rwa_floor: float         # floor_rate × rwa_sa (£)
    rwa_effective: float     # max(irb, floor) (£)
    floor_binds: bool
    capital_req: float       # rwa_effective × Tier 1 ratio
    floor_rate: float        # CRR3 phase-in rate applied
    calculated_at: datetime = field(
        default_factory=datetime.utcnow
    )


@dataclass
class ValidationReport:
    """Full CRR3 Art. 176 validation results for MR-2026-043."""
    model_id: str
    validation_date: datetime
    auc_roc: float
    auc_pr: float
    brier_score: float
    hosmer_lemeshow_p: float
    gini_coefficient: float
    ks_statistic: float
    psi: float
    long_run_pd_deviation_bps: float
    pass_all: bool
    recalibrate_required: bool
    details: dict = field(default_factory=dict)

    # AWB thresholds per SS1/23 validation standard
    AUC_ROC_MIN           = 0.75
    AUC_PR_MIN            = 0.50
    BRIER_MAX             = 0.12
    HL_P_MIN              = 0.05
    GINI_MIN              = 0.50
    KS_MIN                = 0.40
    PSI_ALERT             = 0.10
    PSI_RECAL_TRIGGER     = 0.20
    PD_DEVIATION_MAX_BPS  = 15.0
