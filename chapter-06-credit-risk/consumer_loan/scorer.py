"""AWB Consumer Loan Origination — LightGBM Credit Scorer.

Model ID:    MR-2026-041
Risk rating: MEDIUM (PRA SS1/23)
EU AI Act:   HIGH-RISK Annex III §5b
ICT Asset:   CLO-2026-001
Chapter:     6 — Credit Risk Intelligence with AI

Architecture:
  LightGBM (leaf-wise, 14 features) →
  Platt scaling →
  SHAP TreeExplainer →
  Decision threshold →
  ScoringResult

All decisions logged to PostgreSQL audit log (7-year
retention, FCA COBS 9).
"""
from __future__ import annotations

import hashlib
import logging
import os
import pickle
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# Decision thresholds — reviewed quarterly by Model Risk
THRESHOLD_APPROVE  = float(os.getenv("CLO_APPROVE_THRESHOLD",  "0.08"))
THRESHOLD_DECLINE  = float(os.getenv("CLO_DECLINE_THRESHOLD",  "0.20"))
MODEL_VERSION_FILE = os.getenv("CLO_MODEL_PATH", "models/lgbm_consumer_v1.pkl")

# 14 feature names (must match CreditFeatures field order)
FEATURE_NAMES = [
    "debt_to_income",
    "housing_cost_ratio",
    "ob_income_volatility",
    "ob_discretionary_spend_ratio",
    "ob_overdraft_days_90",
    "ob_gambling_flag",
    "ob_bill_payment_timeliness",
    "ob_current_account_tenure_yrs",
    "ob_net_cashflow_trend",
    "credit_bureau_score",
    "existing_credit_utilisation",
    "adverse_history_flag",
    "employment_tenure_months",
    "requested_amount_to_income",
]


@dataclass
class ScoringResult:
    """Output of MR-2026-041 consumer credit scorer."""
    application_id: str
    pd_calibrated: float
    decision: str               # APPROVE | REVIEW | DECLINE
    shap_top3_risk: list[dict]  # Top-3 risk-increasing factors
    shap_top3_protect: list[dict]
    model_version: str
    raw_score: float = 0.0


class LightGBMCreditScorer:
    """MR-2026-041 — Retail consumer loan PD scorer.

    LightGBM (leaf-wise growth) + Platt scaling calibration
    + SHAP TreeExplainer for FCA PS22/9 decline explanations.

    EU AI Act Annex III §5b: HIGH-RISK — creditworthiness
    assessment of natural persons.

    Usage::

        scorer = LightGBMCreditScorer.load()
        result = scorer.predict("APP-001", features_dict)
        # result.decision ∈ {APPROVE, REVIEW, DECLINE}
        # result.shap_top3_risk  → used in decline letter
    """

    def __init__(
        self,
        model,          # LightGBM Booster
        calibrator,     # sklearn LogisticRegression (Platt)
        explainer,      # shap.TreeExplainer
        version: str,
    ) -> None:
        self._model      = model
        self._calibrator = calibrator
        self._explainer  = explainer
        self.version     = version

    # ── Public API ────────────────────────────────────────────────

    def predict(
        self,
        application_id: str,
        features: dict,
    ) -> ScoringResult:
        """Score a loan application.

        Args:
            application_id: AWB application reference.
            features: dict mapping FEATURE_NAMES to values.
                      Missing Open Banking features are
                      imputed before calling this method
                      (see FeatureEngineer.impute_missing).

        Returns:
            ScoringResult with decision and SHAP explanation.

        Raises:
            ValueError: if features dict is missing required keys.
        """
        x = self._to_array(features)
        raw  = float(self._model.predict(x)[0])
        pd_cal = self._platt_scale(raw)
        shap_vals = self._explainer.shap_values(x)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]  # binary: positive class
        sv = shap_vals[0]

        top3_risk, top3_prot = self._top3(features, sv)
        decision = self._decide(pd_cal)

        log.info(
            "MR-2026-041 app=%s pd=%.4f dec=%s ver=%s",
            application_id, pd_cal, decision, self.version,
        )
        return ScoringResult(
            application_id  = application_id,
            pd_calibrated   = round(pd_cal, 6),
            decision        = decision,
            shap_top3_risk  = top3_risk,
            shap_top3_protect = top3_prot,
            model_version   = self.version,
            raw_score       = raw,
        )

    # ── Internal helpers ──────────────────────────────────────────

    def _to_array(self, features: dict) -> np.ndarray:
        """Convert feature dict to ordered numpy array."""
        missing = [k for k in FEATURE_NAMES if k not in features]
        if missing:
            raise ValueError(
                f"Missing features: {missing}"
            )
        return np.array(
            [[features[k] for k in FEATURE_NAMES]],
            dtype=np.float64,
        )

    def _platt_scale(self, raw: float) -> float:
        """Apply Platt scaling calibration."""
        return float(
            self._calibrator.predict_proba([[raw]])[0, 1]
        )

    def _decide(self, pd_cal: float) -> str:
        """Apply decision thresholds."""
        if pd_cal < THRESHOLD_APPROVE:
            return "APPROVE"
        if pd_cal > THRESHOLD_DECLINE:
            return "DECLINE"
        return "REVIEW"

    def _top3(
        self,
        features: dict,
        shap_vals: np.ndarray,
    ) -> tuple[list[dict], list[dict]]:
        """Extract top-3 risk-increasing and risk-decreasing
        SHAP factors for the FCA decline letter.
        """
        pairs = list(zip(FEATURE_NAMES, shap_vals))
        pairs.sort(key=lambda x: x[1], reverse=True)

        def _fmt(name: str, sv: float) -> dict:
            return {
                "feature":     name,
                "shap_value":  round(float(sv), 4),
                "feature_val": features.get(name),
            }

        risk  = [_fmt(n, s) for n, s in pairs[:3]  if s > 0]
        prot  = [_fmt(n, s) for n, s in pairs[-3:] if s < 0]
        return risk[:3], prot[:3]

    @classmethod
    def load(cls, path: str = MODEL_VERSION_FILE) -> "LightGBMCreditScorer":
        """Load scorer from pickled artefact bundle.

        Artefact bundle (dict):
          model:      LightGBM Booster
          calibrator: sklearn LogisticRegression
          explainer:  shap.TreeExplainer
          sha256:     hex digest of this bundle
        """
        with open(path, "rb") as f:
            bundle = pickle.load(f)

        sha = hashlib.sha256(
            open(path, "rb").read()
        ).hexdigest()[:16]

        return cls(
            model      = bundle["model"],
            calibrator = bundle["calibrator"],
            explainer  = bundle["explainer"],
            version    = bundle.get("version", sha),
        )

    @classmethod
    def build_stub(cls) -> "LightGBMCreditScorer":
        """Build a deterministic stub for unit testing.

        Returns a scorer that maps debt_to_income linearly
        to PD — no LightGBM installation required.
        """
        return _StubScorer()


class _StubScorer(LightGBMCreditScorer):
    """Deterministic test stub — no model file needed."""

    def __init__(self) -> None:
        self.version = "stub-v0"

    def predict(
        self,
        application_id: str,
        features: dict,
    ) -> ScoringResult:
        dti = features.get("debt_to_income", 0.3)
        pd_cal = min(max(dti * 0.4, 0.01), 0.99)
        decision = (
            "APPROVE" if pd_cal < THRESHOLD_APPROVE
            else "DECLINE" if pd_cal > THRESHOLD_DECLINE
            else "REVIEW"
        )
        sv_stub = [0.05, -0.02, 0.03] + [0.0] * 11
        top3r = [{"feature": "debt_to_income",
                  "shap_value": 0.05, "feature_val": dti}]
        top3p = [{"feature": "credit_bureau_score",
                  "shap_value": -0.02, "feature_val": 720}]
        return ScoringResult(
            application_id   = application_id,
            pd_calibrated    = round(pd_cal, 6),
            decision         = decision,
            shap_top3_risk   = top3r,
            shap_top3_protect= top3p,
            model_version    = self.version,
            raw_score        = pd_cal,
        )
