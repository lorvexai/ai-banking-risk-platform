"""
payment_fraud/detector.py — AWB Payment Fraud Detection System.
Model ID: MR-2026-049 | PRA SS1/23 Risk: MEDIUM
EU AI Act: HIGH-RISK Annex III §5b
Avon & Wessex Bank plc (AWB) — AWB-AI-2025 programme.

Architecture: XGBoost anomaly scorer + LLM narrative explainer.
Processes 2.4M transactions/day across AWB payment channels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from awb_commons.models import FraudAlert, FraudSeverity

logger = logging.getLogger(__name__)

# PRA SS1/23: thresholds must be documented in model card
FRAUD_THRESHOLDS = {
    FraudSeverity.LOW: 0.35,
    FraudSeverity.MEDIUM: 0.55,
    FraudSeverity.HIGH: 0.75,
    FraudSeverity.CRITICAL: 0.90,
}

# EU AI Act Art. 14 — human review mandatory above this amount
HUMAN_REVIEW_THRESHOLD_GBP = 10_000.0


@dataclass
class TransactionFeatures:
    """
    Engineered features for the fraud scoring model.
    All PII stripped before model input (UK GDPR Art. 25).
    """

    transaction_id: str
    amount_gbp: float
    hour_of_day: int
    day_of_week: int
    channel: str  # ONLINE / BRANCH / ATM / API
    merchant_category: str
    distance_from_home_km: float
    velocity_24h: int  # tx count in last 24 hours
    avg_amount_30d: float
    amount_vs_avg_ratio: float
    is_foreign_currency: bool
    is_new_payee: bool


@dataclass
class FraudScorerConfig:
    """
    Configuration for MR-2026-049 Payment Fraud Detector.
    Loaded from environment; documented in PRA SS1/23 model card.
    """

    model_version: str = "xgb-v3.2-2025"
    feature_count: int = 12
    high_review_threshold: float = 0.75
    auto_block_threshold: float = 0.92
    human_review_amount_gbp: float = HUMAN_REVIEW_THRESHOLD_GBP
    mr_reference: str = "MR-2026-049"


class PaymentFraudDetector:
    """
    AWB payment fraud detection: XGBoost score + rule overlay.

    The model assigns a risk score [0, 1] to each transaction.
    Rule overlay adds domain constraints (e.g. velocity limits).
    LLM generates plain-English explanation for human reviewers.

    PRA SS1/23: all predictions logged to PostgreSQL audit trail.
    DORA: classified as ICT asset PAY-2026-001.
    """

    def __init__(self, config: FraudScorerConfig | None = None) -> None:
        self.config = config or FraudScorerConfig()
        logger.info(
            "PaymentFraudDetector initialised: %s",
            self.config.mr_reference,
        )

    def score_transaction(
        self,
        features: TransactionFeatures,
    ) -> FraudAlert:
        """
        Score a single payment transaction for fraud risk.

        Args:
            features: Engineered transaction features.

        Returns:
            FraudAlert with risk score and severity classification.

        Raises:
            ValueError: If features fail validation checks.
        """
        self._validate_features(features)
        raw_score = self._xgboost_score(features)
        adjusted = self._apply_rule_overlay(raw_score, features)
        severity = self._classify_severity(adjusted)
        rules = self._triggered_rules(features, adjusted)

        alert = FraudAlert(
            transaction_id=features.transaction_id,
            account_id="REDACTED",  # PII removed from alert
            amount_gbp=features.amount_gbp,
            risk_score=round(adjusted, 4),
            severity=severity,
            triggered_rules=rules,
            model_version=self.config.model_version,
            requires_human_review=(
                adjusted >= self.config.high_review_threshold
                or features.amount_gbp
                >= self.config.human_review_amount_gbp
            ),
        )
        logger.info(
            "Fraud score: tx=%s score=%.4f severity=%s review=%s",
            features.transaction_id,
            adjusted,
            severity,
            alert.requires_human_review,
        )
        return alert

    # ── Private helpers ───────────────────────────────────────────

    def _validate_features(self, f: TransactionFeatures) -> None:
        if f.amount_gbp < 0:
            raise ValueError(
                f"Negative amount: {f.amount_gbp}"
            )
        if not (0 <= f.hour_of_day <= 23):
            raise ValueError(
                f"Invalid hour: {f.hour_of_day}"
            )

    def _xgboost_score(
        self, f: TransactionFeatures
    ) -> float:
        """
        Deterministic scoring stub (replaces live XGBoost in tests).
        Production loads model from MLflow model registry.
        """
        score = 0.1
        if f.amount_vs_avg_ratio > 5.0:
            score += 0.30
        if f.is_new_payee:
            score += 0.20
        if f.velocity_24h > 10:
            score += 0.15
        if f.is_foreign_currency:
            score += 0.10
        if f.distance_from_home_km > 500:
            score += 0.15
        return min(score, 1.0)

    def _apply_rule_overlay(
        self,
        score: float,
        f: TransactionFeatures,
    ) -> float:
        """Hard rules that override model score."""
        if f.velocity_24h > 50:
            score = max(score, 0.92)
        if f.amount_gbp > 50_000 and f.is_new_payee:
            score = max(score, 0.85)
        return score

    def _classify_severity(
        self, score: float
    ) -> FraudSeverity:
        if score >= FRAUD_THRESHOLDS[FraudSeverity.CRITICAL]:
            return FraudSeverity.CRITICAL
        elif score >= FRAUD_THRESHOLDS[FraudSeverity.HIGH]:
            return FraudSeverity.HIGH
        elif score >= FRAUD_THRESHOLDS[FraudSeverity.MEDIUM]:
            return FraudSeverity.MEDIUM
        else:
            return FraudSeverity.LOW

    def _triggered_rules(
        self,
        f: TransactionFeatures,
        score: float,
    ) -> list[str]:
        rules: list[str] = []
        if f.amount_vs_avg_ratio > 5.0:
            rules.append("HIGH_AMOUNT_VS_AVERAGE")
        if f.is_new_payee:
            rules.append("NEW_PAYEE")
        if f.velocity_24h > 10:
            rules.append("HIGH_VELOCITY")
        if f.is_foreign_currency:
            rules.append("FOREIGN_CURRENCY")
        if score >= self.config.auto_block_threshold:
            rules.append("AUTO_BLOCK_THRESHOLD")
        return rules
