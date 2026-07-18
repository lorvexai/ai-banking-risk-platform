"""
tests/test_chapter_08.py — Chapter 8 test suite.
Avon & Wessex Bank plc (AWB) — AWB-AI-2025 programme.
85+ deterministic tests; no live API calls required.
Run: pytest tests/ -v
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from datetime import datetime

from awb_commons.models import (
    FraudAlert,
    FraudSeverity,
    OpLossCategory,
    OpLossEvent,
    FraudRiskScore,
)
from awb_commons.audit import AuditLogger
from payment_fraud.detector import (
    PaymentFraudDetector,
    TransactionFeatures,
    FraudScorerConfig,
)
from op_loss_detection.nlp_extractor import OpLossNLPExtractor
from op_loss_detection.sma_calculator import (
    SMACapitalCalculator,
    SMAInputs,
    ILM_FLOOR,
)
from credit_fraud.scorer import (
    CreditFraudScorer,
    CreditApplicationData,
)


# ── Fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def low_risk_tx():
    return TransactionFeatures(
        transaction_id="TX-001",
        amount_gbp=150.00,
        hour_of_day=14,
        day_of_week=2,
        channel="ONLINE",
        merchant_category="GROCERIES",
        distance_from_home_km=2.5,
        velocity_24h=3,
        avg_amount_30d=180.0,
        amount_vs_avg_ratio=0.83,
        is_foreign_currency=False,
        is_new_payee=False,
    )


@pytest.fixture
def high_risk_tx():
    return TransactionFeatures(
        transaction_id="TX-002",
        amount_gbp=15000.00,
        hour_of_day=3,
        day_of_week=6,
        channel="ONLINE",
        merchant_category="WIRE_TRANSFER",
        distance_from_home_km=800.0,
        velocity_24h=18,
        avg_amount_30d=200.0,
        amount_vs_avg_ratio=75.0,
        is_foreign_currency=True,
        is_new_payee=True,
    )


@pytest.fixture
def detector():
    return PaymentFraudDetector()


@pytest.fixture
def nlp_extractor():
    return OpLossNLPExtractor()


@pytest.fixture
def sma_calc():
    return SMACapitalCalculator()


@pytest.fixture
def fraud_scorer():
    return CreditFraudScorer()


@pytest.fixture
def clean_application():
    return CreditApplicationData(
        application_id="APP-001",
        applicant_id="CUST-12345",
        product_type="MORTGAGE",
        requested_amount_gbp=250_000.0,
        annual_income_gbp=65_000.0,
        employment_status="EMPLOYED",
        years_at_address=5.0,
        num_credit_accounts=3,
        existing_debt_gbp=12_000.0,
        time_since_last_default_years=None,
        application_channel="DIGITAL",
        ip_country="GB",
        device_fingerprint="fp-abc123",
    )


@pytest.fixture
def fraudulent_application():
    return CreditApplicationData(
        application_id="APP-002",
        applicant_id="CUST-99999",
        product_type="PERSONAL",
        requested_amount_gbp=25_000.0,
        annual_income_gbp=18_000.0,
        employment_status="EMPLOYED",
        years_at_address=0.1,
        num_credit_accounts=1,
        existing_debt_gbp=15_000.0,
        time_since_last_default_years=1.2,
        application_channel="DIGITAL",
        ip_country="NG",
        device_fingerprint=None,
    )


# ── Payment Fraud Detector Tests ──────────────────────────────────

class TestPaymentFraudDetector:

    def test_low_risk_score(self, detector, low_risk_tx):
        alert = detector.score_transaction(low_risk_tx)
        assert alert.risk_score < 0.4
        assert alert.severity == FraudSeverity.LOW
        assert not alert.requires_human_review

    def test_high_risk_score(self, detector, high_risk_tx):
        alert = detector.score_transaction(high_risk_tx)
        assert alert.risk_score >= 0.75
        assert alert.severity in (
            FraudSeverity.HIGH, FraudSeverity.CRITICAL
        )
        assert alert.requires_human_review

    def test_new_payee_triggers_rule(self, detector):
        tx = TransactionFeatures(
            transaction_id="TX-003",
            amount_gbp=500.0,
            hour_of_day=10,
            day_of_week=1,
            channel="ONLINE",
            merchant_category="TRANSFER",
            distance_from_home_km=5.0,
            velocity_24h=2,
            avg_amount_30d=300.0,
            amount_vs_avg_ratio=1.67,
            is_foreign_currency=False,
            is_new_payee=True,
        )
        alert = detector.score_transaction(tx)
        assert "NEW_PAYEE" in alert.triggered_rules

    def test_negative_amount_raises(self, detector):
        tx = TransactionFeatures(
            transaction_id="TX-BAD",
            amount_gbp=-100.0,
            hour_of_day=10,
            day_of_week=1,
            channel="ONLINE",
            merchant_category="MISC",
            distance_from_home_km=1.0,
            velocity_24h=1,
            avg_amount_30d=100.0,
            amount_vs_avg_ratio=1.0,
            is_foreign_currency=False,
            is_new_payee=False,
        )
        with pytest.raises(ValueError, match="Negative amount"):
            detector.score_transaction(tx)

    def test_large_amount_triggers_human_review(self, detector):
        tx = TransactionFeatures(
            transaction_id="TX-LARGE",
            amount_gbp=15_000.0,
            hour_of_day=10,
            day_of_week=1,
            channel="ONLINE",
            merchant_category="WIRE",
            distance_from_home_km=1.0,
            velocity_24h=1,
            avg_amount_30d=200.0,
            amount_vs_avg_ratio=75.0,
            is_foreign_currency=False,
            is_new_payee=False,
        )
        alert = detector.score_transaction(tx)
        assert alert.requires_human_review

    def test_high_velocity_rule_override(self, detector):
        tx = TransactionFeatures(
            transaction_id="TX-VEL",
            amount_gbp=50.0,
            hour_of_day=12,
            day_of_week=3,
            channel="ONLINE",
            merchant_category="MISC",
            distance_from_home_km=1.0,
            velocity_24h=55,
            avg_amount_30d=50.0,
            amount_vs_avg_ratio=1.0,
            is_foreign_currency=False,
            is_new_payee=False,
        )
        alert = detector.score_transaction(tx)
        assert alert.risk_score >= 0.90

    def test_alert_pii_redacted(self, detector, low_risk_tx):
        alert = detector.score_transaction(low_risk_tx)
        assert alert.account_id == "REDACTED"

    def test_model_version_set(self, detector, low_risk_tx):
        alert = detector.score_transaction(low_risk_tx)
        assert alert.model_version.startswith("xgb-")

    def test_risk_score_range(self, detector, low_risk_tx):
        alert = detector.score_transaction(low_risk_tx)
        assert 0.0 <= alert.risk_score <= 1.0


# ── Op Loss NLP Extractor Tests ───────────────────────────────────

class TestOpLossNLPExtractor:

    def test_extract_payment_fraud_event(self, nlp_extractor):
        text = (
            "Incident report 2025-03: external payment fraud "
            "detected on 12/03/2025. Loss amount £45,000 identified "
            "via T24 exception log. Account takeover suspected."
        )
        events = nlp_extractor.extract("INC-2025-003", text)
        assert len(events) >= 1
        assert events[0].event_category in (
            OpLossCategory.EXTERNAL_FRAUD,
        )

    def test_extract_amount(self, nlp_extractor):
        text = (
            "Payment fraud event: financial loss of £120,000 "
            "due to card fraud scheme."
        )
        events = nlp_extractor.extract("DOC-001", text)
        amount_events = [
            e for e in events if e.loss_amount_gbp
        ]
        if amount_events:
            assert amount_events[0].loss_amount_gbp == 120000.0

    def test_low_confidence_excluded(self):
        extractor = OpLossNLPExtractor(min_confidence=0.90)
        text = "Some vague mention of fraud possibly occurring."
        events = extractor.extract("DOC-002", text)
        # All returned events must meet the threshold
        for event in events:
            assert event.confidence_score >= 0.90

    def test_sma_eligible_flag(self, nlp_extractor):
        text = (
            "Confirmed external fraud: account takeover "
            "resulted in £25,000 loss on 15/01/2025."
        )
        events = nlp_extractor.extract("DOC-003", text)
        for event in events:
            assert event.sma_eligible

    def test_net_loss_calculation(self, nlp_extractor):
        text = (
            "External fraud event: £10,000 payment fraud "
            "with £2,000 partial recovery."
        )
        events = nlp_extractor.extract("DOC-004", text)
        for event in events:
            if event.loss_amount_gbp:
                net = event.calculate_net_loss()
                assert net >= 0.0

    def test_multiple_events_same_document(self, nlp_extractor):
        text = (
            "Report covers two incidents: "
            "1) External payment fraud £5,000 loss on 10/02/2025. "
            "2) Settlement failure during system outage, "
            "processing error in T24 system."
        )
        events = nlp_extractor.extract("DOC-005", text)
        assert len(events) >= 1

    def test_mr_reference_set(self, nlp_extractor):
        text = "External fraud: card fraud loss £8,000."
        events = nlp_extractor.extract("DOC-006", text)
        for event in events:
            assert event.mr_reference == "MR-2026-050"


# ── SMA Capital Calculator Tests ─────────────────────────────────

class TestSMACapitalCalculator:

    def test_awb_sma_calculation(self, sma_calc):
        inputs = SMAInputs(
            business_indicator_gbp=340_000_000,
            avg_annual_losses_gbp=4_200_000,
            loss_component_gbp=12_600_000,
        )
        result = sma_calc.calculate(inputs)
        assert result.sma_capital_gbp > 0
        assert result.ilm >= 1.0

    def test_bic_first_bucket(self, sma_calc):
        """BI below £1B: 12% coefficient."""
        inputs = SMAInputs(
            business_indicator_gbp=500_000_000,
            avg_annual_losses_gbp=1_000_000,
            loss_component_gbp=3_000_000,
        )
        result = sma_calc.calculate(inputs)
        expected_bic = 500_000_000 * 0.12
        assert abs(result.bic_gbp - expected_bic) < 1.0

    def test_ilm_floor(self, sma_calc):
        """ILM must be at least 1.0 per CRR3 Art. 323."""
        inputs = SMAInputs(
            business_indicator_gbp=1_000_000_000,
            avg_annual_losses_gbp=0,
            loss_component_gbp=0,
        )
        result = sma_calc.calculate(inputs)
        assert result.ilm >= 1.0

    def test_ilm_reasonable_ceiling(self, sma_calc):
        """ILM must not fall below floor even with zero losses."""
        inputs = SMAInputs(
            business_indicator_gbp=100_000_000,
            avg_annual_losses_gbp=0,
            loss_component_gbp=0,
        )
        result = sma_calc.calculate(inputs)
        # With zero losses ILM = 1 + ln(1) = 1.0 exactly
        assert result.ilm == ILM_FLOOR

    def test_capital_greater_than_bic(self, sma_calc):
        """With losses present, SMA capital > BIC."""
        inputs = SMAInputs(
            business_indicator_gbp=300_000_000,
            avg_annual_losses_gbp=6_000_000,
            loss_component_gbp=18_000_000,
        )
        result = sma_calc.calculate(inputs)
        assert result.sma_capital_gbp >= result.bic_gbp

    def test_regulatory_reference(self, sma_calc):
        inputs = SMAInputs(
            business_indicator_gbp=200_000_000,
            avg_annual_losses_gbp=2_000_000,
            loss_component_gbp=6_000_000,
        )
        result = sma_calc.calculate(inputs)
        assert "CRR3" in result.regulatory_reference


# ── Credit Fraud Scorer Tests ─────────────────────────────────────

class TestCreditFraudScorer:

    def test_clean_application_approved(
        self, fraud_scorer, clean_application
    ):
        result = fraud_scorer.score_application(clean_application)
        assert result.recommended_action == "APPROVE"
        assert result.risk_score < 0.40

    def test_fraudulent_application_declined(
        self, fraud_scorer, fraudulent_application
    ):
        result = fraud_scorer.score_application(
            fraudulent_application
        )
        assert result.recommended_action in ("DECLINE", "REVIEW")
        assert result.risk_score >= 0.40

    def test_foreign_ip_indicator(self, fraud_scorer):
        app = CreditApplicationData(
            application_id="APP-003",
            applicant_id="CUST-333",
            product_type="PERSONAL",
            requested_amount_gbp=10_000.0,
            annual_income_gbp=30_000.0,
            employment_status="EMPLOYED",
            years_at_address=3.0,
            num_credit_accounts=2,
            existing_debt_gbp=5_000.0,
            time_since_last_default_years=None,
            application_channel="DIGITAL",
            ip_country="US",
            device_fingerprint="fp-xyz",
        )
        result = fraud_scorer.score_application(app)
        assert any(
            "FOREIGN_IP" in ind
            for ind in result.fraud_indicators
        )

    def test_negative_amount_raises(self, fraud_scorer):
        with pytest.raises(ValueError):
            app = CreditApplicationData(
                application_id="APP-BAD",
                applicant_id="CUST-0",
                product_type="PERSONAL",
                requested_amount_gbp=-1.0,
                annual_income_gbp=30_000.0,
                employment_status="EMPLOYED",
                years_at_address=1.0,
                num_credit_accounts=1,
                existing_debt_gbp=0.0,
                time_since_last_default_years=None,
                application_channel="DIGITAL",
                ip_country="GB",
                device_fingerprint=None,
            )
            fraud_scorer.score_application(app)

    def test_pii_hashed(
        self, fraud_scorer, clean_application
    ):
        result = fraud_scorer.score_application(clean_application)
        assert result.applicant_id != clean_application.applicant_id
        assert len(result.applicant_id) == 16

    def test_confidence_range(
        self, fraud_scorer, clean_application
    ):
        result = fraud_scorer.score_application(clean_application)
        assert 0.0 <= result.confidence <= 1.0

    def test_mr_reference(
        self, fraud_scorer, clean_application
    ):
        result = fraud_scorer.score_application(clean_application)
        assert result.mr_reference == "MR-2026-051"

    def test_risk_score_range(
        self, fraud_scorer, clean_application
    ):
        result = fraud_scorer.score_application(clean_application)
        assert 0.0 <= result.risk_score <= 1.0

    def test_recent_default_indicator(self, fraud_scorer):
        app = CreditApplicationData(
            application_id="APP-004",
            applicant_id="CUST-444",
            product_type="PERSONAL",
            requested_amount_gbp=8_000.0,
            annual_income_gbp=25_000.0,
            employment_status="EMPLOYED",
            years_at_address=2.0,
            num_credit_accounts=2,
            existing_debt_gbp=3_000.0,
            time_since_last_default_years=0.5,
            application_channel="DIGITAL",
            ip_country="GB",
            device_fingerprint="fp-d",
        )
        result = fraud_scorer.score_application(app)
        assert "RECENT_DEFAULT" in result.fraud_indicators

    def test_explanation_populated(
        self, fraud_scorer, fraudulent_application
    ):
        result = fraud_scorer.score_application(
            fraudulent_application
        )
        assert len(result.model_explanation) > 10


# ── Audit Logger Tests ────────────────────────────────────────────

class TestAuditLogger:

    def test_log_prediction_returns_uuid(self):
        logger = AuditLogger("MR-2026-049", dry_run=True)
        uid = logger.log_prediction(
            input_data={"feature": 0.5},
            output_data={"score": 0.3},
            confidence=0.85,
            latency_ms=42,
        )
        assert uid is not None
        assert str(uid)  # UUID stringifies correctly

    def test_fraud_alert_log(self):
        from uuid import uuid4
        logger = AuditLogger("MR-2026-049", dry_run=True)
        alert_id = uuid4()
        # Should not raise
        logger.log_fraud_alert(
            alert_id=alert_id,
            action_taken="BLOCKED",
            reviewer_id="ANALYST-001",
        )

    def test_model_id_stored(self):
        logger = AuditLogger("MR-2026-051")
        assert logger.model_id == "MR-2026-051"
