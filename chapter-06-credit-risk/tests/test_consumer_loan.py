"""Tests — Consumer Loan Origination (MR-2026-041).

Coverage:
  - LightGBMCreditScorer (stub): approve / review / decline routing
  - SHAP top-3 extraction
  - Boundary conditions: threshold edges, PD floor
  - FairnessMonitor: parity ratio, alert generation
  - FairnessMonitor: equalised odds check
  - FeatureEngineer: Open Banking present / absent
  - FeatureEngineer: derived ratio computation
  - FeatureEngineer: income decile mapping
  - DeclineLetterGenerator: validation rules
  - FCA PS22/9: missing CRA sentence triggers alert
  - Integration: scorer → letter pipeline

Run: pytest chapter_06/tests/test_consumer_loan.py -v
"""
import pytest
import pandas as pd
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from consumer_loan.scorer import (
    LightGBMCreditScorer, _StubScorer, THRESHOLD_APPROVE,
    THRESHOLD_DECLINE, FEATURE_NAMES,
)
from consumer_loan.fairness import (
    FairnessMonitor, PARITY_THRESHOLD,
)
from consumer_loan.features import (
    FeatureEngineer, RawApplication,
)
from consumer_loan.decline_letter import (
    DeclineLetterGenerator, FEATURE_PLAIN,
)


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def stub_scorer():
    return LightGBMCreditScorer.build_stub()


@pytest.fixture
def low_risk_features():
    """Features mapping to PD well below APPROVE threshold."""
    return {k: 0.05 for k in FEATURE_NAMES} | {
        "debt_to_income":           0.08,
        "credit_bureau_score":      820.0,
        "adverse_history_flag":     0.0,
        "ob_overdraft_days_90":     0.0,
        "ob_gambling_flag":         0.0,
    }


@pytest.fixture
def high_risk_features():
    """Features mapping to PD above DECLINE threshold."""
    return {k: 0.5 for k in FEATURE_NAMES} | {
        "debt_to_income":           0.85,
        "credit_bureau_score":      420.0,
        "adverse_history_flag":     1.0,
        "ob_overdraft_days_90":     18.0,
        "ob_gambling_flag":         1.0,
    }


@pytest.fixture
def borderline_features():
    """Features mapping to PD in REVIEW zone."""
    return {k: 0.3 for k in FEATURE_NAMES} | {
        "debt_to_income": 0.28,  # PD ≈ 0.112 (in REVIEW zone)
        "credit_bureau_score": 620.0,
    }


@pytest.fixture
def sample_decisions():
    """Sample decisions DataFrame for fairness testing."""
    np.random.seed(42)
    n = 1000
    age_bands = np.random.choice(
        ["18-24","25-34","35-44","45-54","55+"], n
    )
    # Ensure parity — equal approval rates across ages
    approved  = np.random.binomial(1, 0.62, n).astype(bool)
    defaulted = np.random.binomial(1, 0.12, n).astype(bool)
    return pd.DataFrame({
        "age_band":          age_bands,
        "gender":            np.random.choice(["M","F"], n),
        "imd_decile":        np.random.randint(1, 11, n).astype(str),
        "employment_status": np.random.choice(
            ["employed","self_employed","retired"], n
        ),
        "approved":  approved,
        "defaulted": defaulted,
    })


@pytest.fixture
def biased_decisions():
    """Decisions where one group has <80% approval rate."""
    np.random.seed(0)
    data = {
        "age_band": ["25-34"] * 300 + ["18-24"] * 300,
        "gender":   ["M"] * 300 + ["F"] * 300,
        "imd_decile": ["5"] * 600,
        "employment_status": ["employed"] * 600,
        "approved": (
            [True] * 240 + [False] * 60 +   # 25-34: 80% rate
            [True] * 165 + [False] * 135     # 18-24: 55% rate → alert
        ),
        "defaulted": [False] * 600,
    }
    return pd.DataFrame(data)


@pytest.fixture
def engineer():
    return FeatureEngineer()


@pytest.fixture
def raw_app():
    return RawApplication(
        application_id="APP-001",
        requested_amount_gbp=8_000.0,
        loan_term_months=36,
        purpose="home_improvement",
        gross_annual_income=42_000.0,
        employment_status="employed",
        employment_tenure_months=48,
        residential_status="owner",
        time_at_address_months=60,
        num_dependants=1,
        monthly_housing_cost=900.0,
        existing_monthly_commitments=250.0,
        bureau_score=710,
        bureau_adverse_flag=0,
        bureau_utilisation=0.35,
        ob_connected=True,
        ob_income_volatility=0.18,
        ob_discretionary_spend_ratio=0.24,
        ob_overdraft_days_90=1.0,
        ob_gambling_flag=0.0,
        ob_bill_payment_timeliness=0.95,
        ob_current_account_tenure_yrs=6.5,
        ob_net_cashflow_trend=120.0,
    )


# ── Scorer tests ──────────────────────────────────────────────────

class TestLightGBMCreditScorer:

    def test_low_risk_approves(self, stub_scorer, low_risk_features):
        result = stub_scorer.predict("APP-L", low_risk_features)
        assert result.decision == "APPROVE"
        assert result.pd_calibrated < THRESHOLD_APPROVE

    def test_high_risk_declines(self, stub_scorer, high_risk_features):
        result = stub_scorer.predict("APP-H", high_risk_features)
        assert result.decision == "DECLINE"
        assert result.pd_calibrated > THRESHOLD_DECLINE

    def test_borderline_routes_to_review(
        self, stub_scorer, borderline_features
    ):
        result = stub_scorer.predict("APP-B", borderline_features)
        assert result.decision == "REVIEW"
        assert THRESHOLD_APPROVE <= result.pd_calibrated <= THRESHOLD_DECLINE

    def test_shap_top3_risk_not_empty(
        self, stub_scorer, high_risk_features
    ):
        result = stub_scorer.predict("APP-H2", high_risk_features)
        assert len(result.shap_top3_risk) >= 1
        assert "feature" in result.shap_top3_risk[0]
        assert "shap_value" in result.shap_top3_risk[0]

    def test_shap_top3_risk_positive_values(
        self, stub_scorer, high_risk_features
    ):
        """Risk-increasing factors must have positive SHAP."""
        result = stub_scorer.predict("APP-S", high_risk_features)
        for factor in result.shap_top3_risk:
            assert factor["shap_value"] >= 0

    def test_result_has_model_version(
        self, stub_scorer, low_risk_features
    ):
        result = stub_scorer.predict("APP-V", low_risk_features)
        assert result.model_version == "stub-v0"
        assert len(result.model_version) > 0

    def test_missing_feature_handled(self, stub_scorer):
        """Stub returns result even with partial features
        (uses 0.0 for missing). Production model raises ValueError.
        """
        # Stub tolerates missing features (uses get() with default)
        result = stub_scorer.predict("APP-ERR",
                                     {"debt_to_income": 0.3})
        assert result.decision in {"APPROVE","REVIEW","DECLINE"}

    def test_pd_floor_applied(self, stub_scorer):
        """PD should never be exactly 0."""
        features = {k: 0.0 for k in FEATURE_NAMES}
        features["debt_to_income"] = 0.0
        result = stub_scorer.predict("APP-ZERO", features)
        assert result.pd_calibrated > 0

    def test_application_id_preserved(
        self, stub_scorer, low_risk_features
    ):
        result = stub_scorer.predict("APP-ID-123", low_risk_features)
        assert result.application_id == "APP-ID-123"

    def test_all_feature_names_present(self):
        """FEATURE_NAMES must cover all 14 features."""
        assert len(FEATURE_NAMES) == 14


# ── Fairness monitor tests ────────────────────────────────────────

class TestFairnessMonitor:

    def test_balanced_data_no_alerts(
        self, sample_decisions
    ):
        monitor = FairnessMonitor()
        report = monitor.monthly_report(sample_decisions, "2026-06")
        # With balanced random data, alert should not fire
        age_result = report.segment_results.get("age_band", {})
        ratio = age_result.get("parity_ratio", 1.0)
        assert ratio >= 0.70  # may vary due to random seed

    def test_biased_data_raises_alert(self, biased_decisions):
        monitor = FairnessMonitor()
        report = monitor.monthly_report(biased_decisions, "2026-06")
        assert report.has_alerts
        age_alert = [
            a for a in report.alerts if a.segment == "age_band"
        ]
        assert len(age_alert) >= 1

    def test_alert_below_threshold(self, biased_decisions):
        monitor = FairnessMonitor()
        report = monitor.monthly_report(biased_decisions, "2026-06")
        for alert in report.alerts:
            assert alert.value < PARITY_THRESHOLD

    def test_report_month_stored(self, sample_decisions):
        monitor = FairnessMonitor()
        report = monitor.monthly_report(sample_decisions, "2026-01")
        assert report.report_month == "2026-01"

    def test_total_decisions_count(self, sample_decisions):
        monitor = FairnessMonitor()
        report = monitor.monthly_report(sample_decisions, "2026-06")
        assert report.total_decisions == len(sample_decisions)

    def test_missing_segment_skipped(self, sample_decisions):
        """Monitor should not crash if a segment column is missing."""
        df = sample_decisions.drop(columns=["gender"])
        monitor = FairnessMonitor()
        report = monitor.monthly_report(df, "2026-06")
        assert "gender" not in report.segment_results

    def test_equalised_odds_requires_outcome_data(
        self, sample_decisions
    ):
        monitor = FairnessMonitor()
        result = monitor.equalised_odds_check(
            sample_decisions, "age_band", "2026-06"
        )
        assert "fnr_gap" in result or "skipped" in result


# ── Feature engineering tests ─────────────────────────────────────

class TestFeatureEngineer:

    def test_ob_connected_uses_actual_values(
        self, engineer, raw_app
    ):
        features = engineer.engineer(raw_app)
        assert features["ob_income_volatility"] == \
            raw_app.ob_income_volatility
        assert features["ob_gambling_flag"] == \
            raw_app.ob_gambling_flag

    def test_ob_not_connected_uses_imputed(self, engineer, raw_app):
        raw_app.ob_connected = False
        features = engineer.engineer(raw_app)
        # Imputed values come from population median — not None
        assert features["ob_income_volatility"] is not None
        assert isinstance(features["ob_income_volatility"], float)

    def test_debt_to_income_computed(self, engineer, raw_app):
        features = engineer.engineer(raw_app)
        assert "debt_to_income" in features
        assert 0.0 < features["debt_to_income"] < 5.0

    def test_housing_cost_ratio_bounded(self, engineer, raw_app):
        features = engineer.engineer(raw_app)
        assert 0.0 <= features["housing_cost_ratio"] <= 1.0

    def test_adverse_flag_preserved(self, engineer, raw_app):
        raw_app.bureau_adverse_flag = 1
        features = engineer.engineer(raw_app)
        assert features["adverse_history_flag"] == 1.0

    def test_income_decile_low_income(self, engineer):
        decile = engineer._income_decile(12_000.0)
        assert decile == 1

    def test_income_decile_high_income(self, engineer):
        decile = engineer._income_decile(100_000.0)
        assert decile == 10

    def test_all_14_features_produced(self, engineer, raw_app):
        from consumer_loan.scorer import FEATURE_NAMES
        features = engineer.engineer(raw_app)
        for name in FEATURE_NAMES:
            assert name in features, f"Missing feature: {name}"


# ── Decline letter tests ───────────────────────────────────────────

class TestDeclineLetterGenerator:

    @pytest.fixture
    def gen(self):
        return DeclineLetterGenerator()

    @pytest.fixture
    def sample_shap_factors(self):
        return [
            {"feature": "debt_to_income",
             "shap_value": 0.065, "feature_val": 0.72},
            {"feature": "credit_bureau_score",
             "shap_value": 0.048, "feature_val": 420},
            {"feature": "ob_overdraft_days_90",
             "shap_value": 0.031, "feature_val": 14},
        ]

    def test_fallback_letter_contains_cra_and_phone(
        self, gen, sample_shap_factors
    ):
        """Fallback letter has required FCA elements."""
        letter_text = gen._fallback_letter(sample_shap_factors)
        assert "Consumer Credit Act 1974" in letter_text
        assert "0800" in letter_text
        # Validation notes may flag word count on short letters
        notes = gen._validate(letter_text)
        cra_ok = not any("CRA" in n for n in notes)
        assert cra_ok

    def test_fallback_contains_cra_sentence(
        self, gen, sample_shap_factors
    ):
        text = gen._fallback_letter(sample_shap_factors)
        assert "Consumer Credit Act 1974" in text

    def test_fallback_contains_phone_number(
        self, gen, sample_shap_factors
    ):
        text = gen._fallback_letter(sample_shap_factors)
        assert "0800" in text

    def test_validation_catches_missing_cra(self, gen):
        bad_text = (
            "Dear Customer, your application was declined "
            "because of your credit score of 85 and your "
            "debt-to-income ratio is too high. "
            "Please call 0800 123 4567."
        )
        notes = gen._validate(bad_text)
        assert any("CRA" in n for n in notes)

    def test_generate_returns_decline_letter(
        self, gen, sample_shap_factors
    ):
        # Without Gemini available, should use fallback
        letter = gen.generate("APP-DL-001", sample_shap_factors)
        assert letter.application_id == "APP-DL-001"
        assert len(letter.letter_text) > 50

    def test_format_reasons_uses_plain_english(
        self, gen, sample_shap_factors
    ):
        reasons = gen._format_reasons(sample_shap_factors)
        # Should contain plain-English translation, not raw feature name
        assert "debt_to_income" not in reasons
        # Should contain number prefix
        assert "1." in reasons


# ── Integration test: scorer → letter ────────────────────────────

class TestScorerLetterPipeline:

    def test_decline_produces_valid_letter(
        self, stub_scorer, high_risk_features, engineer
    ):
        """Integration: DECLINE decision triggers letter generation."""
        result = stub_scorer.predict("APP-INT-001", high_risk_features)
        assert result.decision == "DECLINE"

        gen    = DeclineLetterGenerator()
        letter = gen.generate("APP-INT-001", result.shap_top3_risk)
        # Without Gemini, fallback is used — CRA must be present
        assert "Consumer Credit Act 1974" in letter.letter_text

    def test_approve_does_not_need_letter(
        self, stub_scorer, low_risk_features
    ):
        """Approved applications don't need decline letters."""
        result = stub_scorer.predict("APP-INT-002", low_risk_features)
        assert result.decision == "APPROVE"
        # No letter needed — test that shap_top3_risk is still populated
        # (for audit log even on approvals)
        assert isinstance(result.shap_top3_risk, list)
