"""
exercises/fraud_model.py — Exercise 8.1 starter code.
Chapter 8: Operational Risk Detection and Management.
Avon & Wessex Bank plc (AWB) — AWB-AI-2025 programme.

EXERCISE 8.1: Train the AWB Payment Fraud XGBoost Model
Difficulty: ★★★☆☆ | Estimated time: 45 minutes

Task:
    Train an XGBoost payment fraud classifier using the 12
    features from Section 8.2.2. Your model must achieve:
        recall > 0.90 on the held-out test set
        FPR    < 0.05 on the held-out test set

    Then configure EU AI Act Art. 14 thresholds and verify
    the threshold configuration test passes.

GitHub: lorvenio/ai-banking-risk-platform/chapter_08/
Solution: chapter_08/solutions/fraud_model_solution.py
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# ── Synthetic data generation ─────────────────────────────────────


def generate_synthetic_transactions(
    n_samples: int = 10_000,
    fraud_rate: float = 0.012,
    seed: int = 42,
) -> tuple[list[dict], list[int]]:
    """
    Generate synthetic AWB transaction dataset.

    Mirrors the 12-feature set from Section 8.2.2.
    Class imbalance matches AWB production (1:84 ratio).

    Args:
        n_samples:  Total transactions to generate.
        fraud_rate: Fraction of fraudulent transactions.
        seed:       Random seed for reproducibility.

    Returns:
        Tuple of (feature_dicts, labels) where label=1 is fraud.
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    features: list[dict] = []
    labels: list[int] = []

    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud

    # Legitimate transactions
    for _ in range(n_legit):
        features.append({
            "amount_vs_avg_ratio": rng.gauss(1.1, 0.4),
            "cross_account_velocity_index": rng.gauss(1.2, 0.5),
            "is_new_payee": int(rng.random() < 0.08),
            "velocity_24h": max(1, int(rng.gauss(4, 2))),
            "is_foreign_currency": int(rng.random() < 0.05),
            "distance_from_home_km": abs(rng.gauss(8, 12)),
            "ip_subnet_velocity": rng.gauss(1.1, 0.3),
            "hour_of_day": rng.randint(7, 22),
            "merchant_category_risk": rng.randint(1, 3),
            "channel_risk_score": rng.choice(
                [0.0, 0.2, 0.3, 0.4]
            ),
            "account_age_days": rng.gauss(900, 400),
            "device_fingerprint_age_days": rng.gauss(
                180, 90
            ),
        })
        labels.append(0)

    # Fraudulent transactions (elevated risk features)
    for _ in range(n_fraud):
        features.append({
            "amount_vs_avg_ratio": abs(rng.gauss(8.5, 3.0)),
            "cross_account_velocity_index": abs(
                rng.gauss(12.0, 5.0)
            ),
            "is_new_payee": int(rng.random() < 0.82),
            "velocity_24h": max(3, int(rng.gauss(22, 8))),
            "is_foreign_currency": int(rng.random() < 0.45),
            "distance_from_home_km": abs(
                rng.gauss(320, 200)
            ),
            "ip_subnet_velocity": abs(rng.gauss(8.0, 3.0)),
            "hour_of_day": rng.choice(
                [2, 3, 4, 23, 0, 1]
            ),
            "merchant_category_risk": rng.randint(3, 5),
            "channel_risk_score": rng.choice([0.3, 0.4]),
            "account_age_days": abs(rng.gauss(120, 80)),
            "device_fingerprint_age_days": abs(
                rng.gauss(12, 10)
            ),
        })
        labels.append(1)

    # Shuffle
    combined = list(zip(features, labels))
    rng.shuffle(combined)
    features, labels = zip(*combined)
    return list(features), list(labels)


# ── EU AI Act Art. 14 threshold configuration ─────────────────────


@dataclass
class EUAIActThresholds:
    """
    EU AI Act Article 14 human oversight tier configuration.
    Thresholds are governance parameters (MRC approval required).

    Tiers:
        APPROVE: score < approve_max  — no human review
        REVIEW:  approve_max ≤ score < block_min  — 15-min SLA
        BLOCK:   score ≥ block_min  — auto-hold, 4-hr confirm
    """
    approve_max: float = 0.45
    block_min: float = 0.85
    mr_reference: str = "MR-2026-049"

    def classify(self, score: float) -> str:
        """Classify a score into APPROVE / REVIEW / BLOCK."""
        if score >= self.block_min:
            return "BLOCK"
        elif score >= self.approve_max:
            return "REVIEW"
        else:
            return "APPROVE"


# ── TODO: Implement your solution below ───────────────────────────


def train_fraud_model(
    X_train: list[dict],
    y_train: list[int],
) -> object:
    """
    TODO: Train an XGBoost model on the AWB synthetic dataset.

    Requirements:
        - Use XGBoost XGBClassifier
        - Handle class imbalance (fraud_rate ≈ 1.2%)
        - Apply 5-fold stratified cross-validation during tuning
        - Return fitted model object

    Hint: Use scale_pos_weight = n_negative / n_positive
    """
    raise NotImplementedError(
        "Implement train_fraud_model() — see Section 8.2.3"
    )


def evaluate_model(
    model: object,
    X_test: list[dict],
    y_test: list[int],
    thresholds: EUAIActThresholds,
) -> dict:
    """
    TODO: Evaluate the trained model.

    Must return a dict containing:
        recall:      float  — target > 0.90
        fpr:         float  — target < 0.05
        auc_roc:     float
        tier_counts: dict   — {APPROVE: n, REVIEW: n, BLOCK: n}

    The test test_eu_ai_act_threshold_config() checks that:
        - APPROVE rate > 0.80 (most transactions auto-approved)
        - REVIEW rate  < 0.15
        - BLOCK rate   < 0.05
    """
    raise NotImplementedError(
        "Implement evaluate_model() — see Section 8.2.4"
    )


# ── Test suite for the exercise ───────────────────────────────────


def test_eu_ai_act_threshold_config() -> None:
    """
    Verify EU AI Act Art. 14 threshold configuration.
    This is the success criterion for Exercise 8.1.
    """
    thresholds = EUAIActThresholds()

    # Verify tier boundaries
    assert thresholds.classify(0.10) == "APPROVE"
    assert thresholds.classify(0.45) == "REVIEW"
    assert thresholds.classify(0.85) == "BLOCK"
    assert thresholds.classify(0.44) == "APPROVE"
    assert thresholds.classify(0.84) == "REVIEW"
    assert thresholds.classify(1.00) == "BLOCK"

    # MR reference correct
    assert thresholds.mr_reference == "MR-2026-049"

    print("✅ test_eu_ai_act_threshold_config PASSED")


def test_synthetic_data_generation() -> None:
    """Verify synthetic dataset has correct properties."""
    features, labels = generate_synthetic_transactions(
        n_samples=1000, fraud_rate=0.012
    )
    assert len(features) == 1000
    assert len(labels) == 1000

    fraud_count = sum(labels)
    fraud_pct = fraud_count / len(labels)
    assert 0.008 <= fraud_pct <= 0.018, (
        f"Fraud rate {fraud_pct:.3f} outside expected range"
    )

    # All 12 features present
    expected_features = {
        "amount_vs_avg_ratio",
        "cross_account_velocity_index",
        "is_new_payee",
        "velocity_24h",
        "is_foreign_currency",
        "distance_from_home_km",
        "ip_subnet_velocity",
        "hour_of_day",
        "merchant_category_risk",
        "channel_risk_score",
        "account_age_days",
        "device_fingerprint_age_days",
    }
    assert set(features[0].keys()) == expected_features

    print("✅ test_synthetic_data_generation PASSED")


if __name__ == "__main__":
    # Run the tests that don't require a trained model
    test_synthetic_data_generation()
    test_eu_ai_act_threshold_config()

    print("\nGenerating dataset for training...")
    feats, labs = generate_synthetic_transactions(n_samples=14_400)
    fraud_n = sum(labs)
    print(f"Dataset: {len(feats)} transactions, {fraud_n} fraud")
    print("\nNext step: implement train_fraud_model() above.")
    print(
        "Target: recall > 0.90, FPR < 0.05 on 20% held-out set."
    )
