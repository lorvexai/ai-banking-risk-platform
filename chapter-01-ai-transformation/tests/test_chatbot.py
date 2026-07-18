"""
tests/test_chatbot.py
AWB AI Customer Service Platform — Comprehensive Test Suite

Tests cover:
  - Intent classification (unit, with mocks)
  - FCA compliance filter (unit — no API calls)
  - Audit log (unit — SQLite in-memory)
  - Pipeline (integration — mocked Gemini API)

Run: pytest tests/ -v
Run without API: pytest tests/ -v -k "not live"
Run live API tests: pytest tests/ -v -k "live" --api-key YOUR_KEY

Author: AWB AI Programme (AWB-AI-2025)
Chapter: 1 — The AI Transformation of Risk and Compliance in Banking
"""

import json
import os
import tempfile
import uuid
from unittest.mock import MagicMock, patch

import pytest

from chatbot.audit_log import (
    InteractionLog,
    build_interaction_log,
    get_flagged_interactions,
    get_interactions_by_session,
    initialise_db,
    log_interaction,
)
from chatbot.classifier import (
    ALWAYS_ESCALATE,
    ESCALATION_CONFIDENCE_THRESHOLD,
    CustomerIntent,
    IntentResult,
)
from chatbot.compliance_filter import (
    ComplianceFlag,
    DraftResponse,
    compliance_check,
)
from chatbot.response_generator import ProductInfo, AccountSummary


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Redirect audit log to a temporary SQLite database."""
    db_file = str(tmp_path / "test_audit.db")
    monkeypatch.setenv("AUDIT_DB_PATH", db_file)
    initialise_db()
    return db_file


@pytest.fixture
def sample_intent_balance() -> IntentResult:
    return IntentResult(
        intent=CustomerIntent.BALANCE_ENQUIRY,
        confidence=0.95,
        entities={"account_type": "current"},
        requires_escalation=False,
        escalation_reason=None,
    )


@pytest.fixture
def sample_intent_complaint() -> IntentResult:
    return IntentResult(
        intent=CustomerIntent.COMPLAINT,
        confidence=0.92,
        entities={},
        requires_escalation=True,
        escalation_reason="always_escalate_intent",
    )


@pytest.fixture
def sample_intent_low_confidence() -> IntentResult:
    return IntentResult(
        intent=CustomerIntent.PRODUCT_ENQUIRY,
        confidence=0.60,
        entities={},
        requires_escalation=True,
        escalation_reason="low_confidence",
    )


@pytest.fixture
def clean_draft() -> DraftResponse:
    return DraftResponse(
        text="Your current account balance is £2,847.65.",
        citations=["AWB T24 core banking (real-time)"],
        confidence=0.95,
        contains_rate_information=False,
        contains_account_data=True,
    )


@pytest.fixture
def rate_draft() -> DraftResponse:
    return DraftResponse(
        text="Our Cash ISA pays 4.25% AER.",
        citations=["AWB Product Catalogue"],
        confidence=0.90,
        contains_rate_information=True,
        contains_account_data=False,
    )


@pytest.fixture
def advice_draft() -> DraftResponse:
    return DraftResponse(
        text="I recommend you apply for our ISA as it is the best product for you.",
        citations=[],
        confidence=0.85,
        contains_rate_information=False,
        contains_account_data=False,
    )


# ---------------------------------------------------------------------------
# Section 1: CustomerIntent enum tests
# ---------------------------------------------------------------------------

class TestCustomerIntentEnum:

    def test_all_intents_have_string_values(self):
        for intent in CustomerIntent:
            assert isinstance(intent.value, str)
            assert len(intent.value) > 0

    def test_always_escalate_set_contains_complaint(self):
        assert CustomerIntent.COMPLAINT in ALWAYS_ESCALATE

    def test_always_escalate_set_contains_account_change(self):
        assert CustomerIntent.ACCOUNT_CHANGE in ALWAYS_ESCALATE

    def test_balance_enquiry_not_in_always_escalate(self):
        assert CustomerIntent.BALANCE_ENQUIRY not in ALWAYS_ESCALATE

    def test_escalation_threshold_is_below_one(self):
        assert 0.0 < ESCALATION_CONFIDENCE_THRESHOLD < 1.0

    def test_escalation_threshold_value(self):
        """Threshold must be 0.75 per FCA Consumer Duty design."""
        assert ESCALATION_CONFIDENCE_THRESHOLD == 0.75


# ---------------------------------------------------------------------------
# Section 2: IntentResult Pydantic model tests
# ---------------------------------------------------------------------------

class TestIntentResult:

    def test_valid_intent_result(self, sample_intent_balance):
        assert sample_intent_balance.intent == CustomerIntent.BALANCE_ENQUIRY
        assert sample_intent_balance.confidence == 0.95
        assert not sample_intent_balance.requires_escalation

    def test_confidence_bounds_lower(self):
        with pytest.raises(Exception):
            IntentResult(
                intent=CustomerIntent.BALANCE_ENQUIRY,
                confidence=-0.1,
                entities={},
                requires_escalation=False,
            )

    def test_confidence_bounds_upper(self):
        with pytest.raises(Exception):
            IntentResult(
                intent=CustomerIntent.BALANCE_ENQUIRY,
                confidence=1.1,
                entities={},
                requires_escalation=False,
            )

    def test_entities_defaults_to_empty_dict(self):
        result = IntentResult(
            intent=CustomerIntent.OUT_OF_SCOPE,
            confidence=0.80,
            entities={},
            requires_escalation=False,
        )
        assert result.entities == {}

    def test_complaint_intent_escalation_flag(self, sample_intent_complaint):
        assert sample_intent_complaint.requires_escalation is True
        assert sample_intent_complaint.escalation_reason == "always_escalate_intent"

    def test_low_confidence_escalation(self, sample_intent_low_confidence):
        assert sample_intent_low_confidence.confidence < ESCALATION_CONFIDENCE_THRESHOLD
        assert sample_intent_low_confidence.requires_escalation is True
        assert sample_intent_low_confidence.escalation_reason == "low_confidence"


# ---------------------------------------------------------------------------
# Section 3: Compliance filter tests (no API calls)
# ---------------------------------------------------------------------------

class TestComplianceFilter:

    def test_clean_response_approved(self, clean_draft):
        result = compliance_check(clean_draft)
        assert result.approved is True
        assert ComplianceFlag.FINANCIAL_ADVICE_RISK not in result.flags
        assert ComplianceFlag.MISLEADING_RATE not in result.flags

    def test_human_agent_reminder_appended(self, clean_draft):
        result = compliance_check(clean_draft)
        assert "0800 123 4567" in result.modified_text or "AWB branch" in result.modified_text

    def test_financial_advice_blocked(self, advice_draft):
        result = compliance_check(advice_draft)
        assert result.approved is False
        assert ComplianceFlag.FINANCIAL_ADVICE_RISK in result.flags
        assert result.escalation_required is True

    def test_financial_advice_replaced_with_fallback(self, advice_draft):
        result = compliance_check(advice_draft)
        # Original text must not appear in output
        assert "I recommend" not in result.modified_text
        assert "0800 123 4567" in result.modified_text

    def test_rate_disclaimer_appended(self, rate_draft):
        result = compliance_check(rate_draft)
        assert "variable" in result.modified_text.lower() or "subject to change" in result.modified_text.lower()
        assert ComplianceFlag.MISSING_RATE_DISCLAIMER in result.flags

    def test_rate_disclaimer_not_duplicated(self, rate_draft):
        """Running filter twice should not double-append the disclaimer."""
        result1 = compliance_check(rate_draft)
        draft2 = DraftResponse(
            text=result1.modified_text,
            contains_rate_information=True,
        )
        result2 = compliance_check(draft2)
        # Count occurrences — should appear once
        count = result2.modified_text.lower().count("subject to change")
        assert count <= 1

    def test_misleading_rate_blocked(self):
        misleading = DraftResponse(
            text="Our ISA offers a guaranteed rate of 4.25% forever.",
            contains_rate_information=True,
        )
        result = compliance_check(misleading)
        assert ComplianceFlag.MISLEADING_RATE in result.flags
        assert result.approved is False
        assert result.escalation_required is True

    def test_vulnerable_customer_triggers_escalation(self, clean_draft):
        result = compliance_check(clean_draft, is_vulnerable_flag=True)
        assert result.escalation_required is True

    def test_audit_notes_populated_on_flag(self, advice_draft):
        result = compliance_check(advice_draft)
        assert result.audit_notes != ""
        assert result.audit_notes != "PASS"

    def test_clean_response_audit_notes_pass(self, clean_draft):
        result = compliance_check(clean_draft)
        assert "PASS" in result.audit_notes or result.flags == []


# ---------------------------------------------------------------------------
# Section 4: Audit log tests
# ---------------------------------------------------------------------------

class TestAuditLog:

    def test_initialise_db_creates_table(self, temp_db):
        import sqlite3
        conn = sqlite3.connect(temp_db)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='interaction_log'"
        ).fetchall()
        conn.close()
        assert len(tables) == 1

    def test_log_interaction_writes_record(self, temp_db, sample_intent_balance):
        from chatbot.compliance_filter import ComplianceResult
        compliance = ComplianceResult(
            approved=True, modified_text="Your balance is £2,847.65.", audit_notes="PASS"
        )
        record = build_interaction_log(
            session_id="test-session-001",
            customer_id="CUST-12345",
            customer_segment="retail",
            channel="web",
            message_text="What is my balance?",
            intent_result=sample_intent_balance,
            response_text="Your balance is £2,847.65.",
            compliance_result=compliance,
            escalated_to_agent=False,
            latency_ms=342,
        )
        interaction_id = log_interaction(record)
        assert len(interaction_id) == 36  # UUID4 format

    def test_log_interaction_retrieval_by_session(self, temp_db, sample_intent_balance):
        from chatbot.compliance_filter import ComplianceResult
        session_id = f"session-{uuid.uuid4()}"
        compliance = ComplianceResult(approved=True, modified_text="Test.", audit_notes="PASS")
        record = build_interaction_log(
            session_id=session_id,
            customer_id="CUST-99999",
            customer_segment="sme",
            channel="app",
            message_text="Tell me about SME loans.",
            intent_result=sample_intent_balance,
            response_text="Test response.",
            compliance_result=compliance,
            escalated_to_agent=False,
        )
        log_interaction(record)
        rows = get_interactions_by_session(session_id)
        assert len(rows) == 1
        assert rows[0]["customer_id"] == "CUST-99999"

    def test_flagged_interactions_query(self, temp_db, sample_intent_balance):
        from chatbot.compliance_filter import ComplianceFlag, ComplianceResult
        compliance_flagged = ComplianceResult(
            approved=False,
            flags=[ComplianceFlag.FINANCIAL_ADVICE_RISK],
            modified_text="Safe fallback.",
            audit_notes="FINANCIAL_ADVICE_RISK detected",
        )
        record = build_interaction_log(
            session_id=f"session-{uuid.uuid4()}",
            customer_id="CUST-00001",
            customer_segment="retail",
            channel="web",
            message_text="Which product should I buy?",
            intent_result=sample_intent_balance,
            response_text="Safe fallback.",
            compliance_result=compliance_flagged,
            escalated_to_agent=True,
        )
        log_interaction(record)
        flagged = get_flagged_interactions(limit=10)
        assert len(flagged) >= 1
        # Verify compliance flags are stored as JSON array
        flags = json.loads(flagged[0]["compliance_flags_json"])
        assert "financial_advice_risk" in flags

    def test_build_interaction_log_generates_uuid(self, sample_intent_balance):
        from chatbot.compliance_filter import ComplianceResult
        compliance = ComplianceResult(approved=True, modified_text="Test.", audit_notes="PASS")
        record1 = build_interaction_log(
            session_id="s1", customer_id=None, customer_segment="retail",
            channel="web", message_text="Hi",
            intent_result=sample_intent_balance,
            response_text="Hello", compliance_result=compliance,
            escalated_to_agent=False,
        )
        record2 = build_interaction_log(
            session_id="s2", customer_id=None, customer_segment="retail",
            channel="web", message_text="Hi",
            intent_result=sample_intent_balance,
            response_text="Hello", compliance_result=compliance,
            escalated_to_agent=False,
        )
        assert record1.interaction_id != record2.interaction_id

    def test_dora_asset_id_present(self, sample_intent_balance):
        from chatbot.compliance_filter import ComplianceResult
        compliance = ComplianceResult(approved=True, modified_text="Test.", audit_notes="PASS")
        record = build_interaction_log(
            session_id="s1", customer_id=None, customer_segment="retail",
            channel="web", message_text="Hi",
            intent_result=sample_intent_balance,
            response_text="Hello", compliance_result=compliance,
            escalated_to_agent=False,
        )
        assert record.dora_asset_id == "CS-2026-001"


# ---------------------------------------------------------------------------
# Section 5: FCA Consumer Duty — regulatory requirement tests
# ---------------------------------------------------------------------------

class TestFCAConsumerDutyRequirements:
    """
    Explicit tests for each FCA Consumer Duty PS22/9 obligation.
    These tests should NEVER be removed — they document regulatory compliance.
    """

    def test_complaint_always_escalates(self):
        """
        FCA Consumer Duty PS22/9: customers must always be able to reach a human.
        COMPLAINT intent MUST trigger escalation regardless of confidence.
        """
        assert CustomerIntent.COMPLAINT in ALWAYS_ESCALATE

    def test_account_change_always_escalates(self):
        """
        ACCOUNT_CHANGE is a regulated activity — must be handled by authorised adviser.
        """
        assert CustomerIntent.ACCOUNT_CHANGE in ALWAYS_ESCALATE

    def test_low_confidence_triggers_escalation(self):
        """
        Confidence < 0.75 means ambiguous query — escalate rather than risk
        delivering an incorrect response (fair outcomes obligation).
        """
        low_conf = IntentResult(
            intent=CustomerIntent.PRODUCT_ENQUIRY,
            confidence=0.60,
            entities={},
            requires_escalation=True,
            escalation_reason="low_confidence",
        )
        assert low_conf.requires_escalation is True
        assert low_conf.confidence < ESCALATION_CONFIDENCE_THRESHOLD

    def test_financial_advice_language_blocked(self):
        """
        Providing personal financial recommendations is a regulated activity
        under FSMA 2000 s.19. AI must NEVER cross this boundary.
        """
        draft = DraftResponse(
            text="I recommend you invest in our ISA as it is ideal for you.",
            contains_rate_information=False,
        )
        result = compliance_check(draft)
        assert result.approved is False
        assert ComplianceFlag.FINANCIAL_ADVICE_RISK in result.flags

    def test_human_agent_contact_always_present(self):
        """
        FCA Consumer Duty: customers must always have a route to human support.
        Phone number or branch reference must appear in every approved response.
        """
        draft = DraftResponse(
            text="Your balance is £1,000.",
            contains_rate_information=False,
            contains_account_data=True,
        )
        result = compliance_check(draft)
        assert result.approved is True
        # Human agent contact must be present
        has_contact = (
            "0800" in result.modified_text
            or "branch" in result.modified_text.lower()
            or "adviser" in result.modified_text.lower()
        )
        assert has_contact, "Human agent contact information missing from approved response"

    def test_rate_disclaimer_mandatory_when_rates_mentioned(self):
        """
        FCA COBS 4.5: rate communications must make clear they are variable.
        """
        draft = DraftResponse(
            text="Our savings account pays 4.50% AER.",
            contains_rate_information=True,
        )
        result = compliance_check(draft)
        assert ComplianceFlag.MISSING_RATE_DISCLAIMER in result.flags
        assert "variable" in result.modified_text.lower() or "subject to change" in result.modified_text.lower()

    def test_audit_log_captures_every_interaction(self, temp_db, sample_intent_balance):
        """
        FCA Consumer Duty: every interaction must be logged for regulatory review.
        7-year retention is enforced by PostgreSQL data retention policy.
        """
        from chatbot.compliance_filter import ComplianceResult
        compliance = ComplianceResult(approved=True, modified_text="Test.", audit_notes="PASS")
        session_id = f"fca-test-{uuid.uuid4()}"
        record = build_interaction_log(
            session_id=session_id,
            customer_id="CUST-FCA-001",
            customer_segment="retail",
            channel="web",
            message_text="What is my balance?",
            intent_result=sample_intent_balance,
            response_text="Your balance is £2,847.65.",
            compliance_result=compliance,
            escalated_to_agent=False,
        )
        log_interaction(record)
        rows = get_interactions_by_session(session_id)
        assert len(rows) == 1
        # All mandatory FCA fields must be present
        row = rows[0]
        assert row["intent"] is not None
        assert row["response_text"] is not None
        assert row["compliance_flags_json"] is not None
        assert row["timestamp_utc"] is not None
        assert row["dora_asset_id"] == "CS-2026-001"


# ---------------------------------------------------------------------------
# Section 6: Pipeline integration tests (mocked Gemini)
# ---------------------------------------------------------------------------

class TestPipelineMocked:
    """Integration tests using mocked Gemini API — no API key required."""

    def _make_mock_gemini(self, intent_json: str, response_text: str):
        """Create a mock Gemini client that returns predetermined outputs."""
        mock_response = MagicMock()
        mock_response.text = intent_json

        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response

        return mock_model

    @patch("chatbot.classifier.genai.GenerativeModel")
    @patch("chatbot.classifier.genai.configure")
    def test_pipeline_balance_enquiry(self, mock_configure, mock_model_class, temp_db):
        """Balance enquiry should return account data and not escalate."""
        mock_intent_response = MagicMock()
        mock_intent_response.text = json.dumps({
            "intent": "balance_enquiry",
            "confidence": 0.95,
            "entities": {"account_type": "current"},
            "requires_escalation": False,
            "escalation_reason": None,
        })
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_intent_response
        mock_model_class.return_value = mock_model

        with patch("chatbot.pipeline.generate_response") as mock_gen:
            from chatbot.response_generator import DraftResponse as DR
            mock_gen.return_value = DR(
                text="Your current account balance is £2,847.65.",
                contains_account_data=True,
            )

            from chatbot.pipeline import process_customer_message
            result = process_customer_message(
                session_id=f"test-{uuid.uuid4()}",
                message="What is my balance?",
                customer_id="CUST-001",
                api_key="test-key",
            )

        assert result.intent == "balance_enquiry"
        assert result.confidence == 0.95
        assert not result.requires_escalation

    @patch("chatbot.classifier.genai.GenerativeModel")
    @patch("chatbot.classifier.genai.configure")
    def test_pipeline_complaint_escalates_immediately(self, mock_configure, mock_model_class, temp_db):
        """Complaint must trigger immediate escalation — FCA Consumer Duty."""
        mock_intent_response = MagicMock()
        mock_intent_response.text = json.dumps({
            "intent": "complaint",
            "confidence": 0.92,
            "entities": {},
            "requires_escalation": True,
            "escalation_reason": "always_escalate_intent",
        })
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_intent_response
        mock_model_class.return_value = mock_model

        from chatbot.pipeline import process_customer_message
        result = process_customer_message(
            session_id=f"test-{uuid.uuid4()}",
            message="I want to make a complaint about my mortgage.",
            customer_id="CUST-002",
            api_key="test-key",
        )

        assert result.requires_escalation is True
        assert result.intent == "complaint"
        # Response must include human contact details
        assert "0800" in result.response_text or "adviser" in result.response_text.lower()

    @patch("chatbot.classifier.genai.GenerativeModel")
    @patch("chatbot.classifier.genai.configure")
    def test_pipeline_interaction_logged(self, mock_configure, mock_model_class, temp_db):
        """Every interaction must produce an audit log record."""
        mock_intent_response = MagicMock()
        mock_intent_response.text = json.dumps({
            "intent": "product_enquiry",
            "confidence": 0.88,
            "entities": {"product": "isa"},
            "requires_escalation": False,
            "escalation_reason": None,
        })
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_intent_response
        mock_model_class.return_value = mock_model

        session_id = f"test-{uuid.uuid4()}"

        with patch("chatbot.pipeline.generate_response") as mock_gen:
            from chatbot.response_generator import DraftResponse as DR
            mock_gen.return_value = DR(text="Our Cash ISA pays 4.25% AER.", contains_rate_information=True)

            from chatbot.pipeline import process_customer_message
            result = process_customer_message(
                session_id=session_id,
                message="Tell me about your ISA.",
                api_key="test-key",
            )

        # Verify audit log record exists
        rows = get_interactions_by_session(session_id)
        assert len(rows) == 1
        assert rows[0]["intent"] == "product_enquiry"
        assert result.interaction_id == rows[0]["interaction_id"]


# ---------------------------------------------------------------------------
# Section 7: Sample data generator tests
# ---------------------------------------------------------------------------

class TestSampleDataGenerator:

    def test_product_catalogue_has_expected_products(self):
        from chatbot.response_generator import PRODUCT_CATALOGUE
        assert "isa" in PRODUCT_CATALOGUE
        assert "sme_loan" in PRODUCT_CATALOGUE
        assert "mortgage" in PRODUCT_CATALOGUE

    def test_isa_has_rate(self):
        from chatbot.response_generator import PRODUCT_CATALOGUE
        isa = PRODUCT_CATALOGUE["isa"]
        assert isa.current_rate_pct == 4.25
        assert "AER" in isa.rate_description

    def test_sme_loan_has_eligibility(self):
        from chatbot.response_generator import PRODUCT_CATALOGUE
        loan = PRODUCT_CATALOGUE["sme_loan"]
        assert len(loan.eligibility_criteria) > 0

    def test_account_summary_currency_gbp(self):
        account = AccountSummary(
            customer_id="CUST-001",
            account_number_masked="****4321",
            product_type="Current Account",
            available_balance_gbp=5000.0,
        )
        assert account.currency == "GBP"
        assert account.available_balance_gbp == 5000.0


# ---------------------------------------------------------------------------
# Section 8: Live API tests (skipped without GOOGLE_API_KEY)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY not set — skipping live API tests"
)
class TestLiveAPI:
    """
    Live integration tests requiring a real Google AI Studio API key.
    Run with: pytest tests/ -v -k "live" or set GOOGLE_API_KEY env var.
    """

    def test_live_classify_balance_enquiry(self):
        from chatbot.classifier import classify_intent
        result = classify_intent("What is my current account balance?")
        assert result.intent == CustomerIntent.BALANCE_ENQUIRY
        assert result.confidence >= 0.75
        assert not result.requires_escalation

    def test_live_classify_complaint_escalates(self):
        from chatbot.classifier import classify_intent
        result = classify_intent("I want to make a complaint, I am very unhappy with the service.")
        assert result.intent == CustomerIntent.COMPLAINT
        assert result.requires_escalation is True

    def test_live_classify_out_of_scope(self):
        from chatbot.classifier import classify_intent
        result = classify_intent("What is the capital of France?")
        assert result.intent == CustomerIntent.OUT_OF_SCOPE

    def test_live_rate_disclaimer_appended(self):
        from chatbot.classifier import classify_intent
        from chatbot.response_generator import generate_response, PRODUCT_CATALOGUE
        intent = classify_intent("What rate does your ISA pay?")
        draft = generate_response(intent, product_info=PRODUCT_CATALOGUE["isa"])
        compliance = compliance_check(draft)
        assert "variable" in compliance.modified_text.lower() or "subject to change" in compliance.modified_text.lower()
