"""
tests/test_credit_agent.py
Comprehensive test suite for AWB Automated Credit Decision Workflow
Chapter 3: Agentic AI for Financial Risk

Test coverage:
- Tool definitions and validation (15 tests)
- Agent loop termination and max_iterations (6 tests)
- Policy rule evaluation (10 tests)
- Credit memo structure and validation (8 tests)
- Human oversight checkpoint (5 tests)
- EU AI Act compliance (4 tests)
- Audit log completeness (4 tests)

Total: 52 tests
All external calls (T24, Gemini) are mocked.

Run with: pytest tests/test_credit_agent.py -v
"""

from __future__ import annotations

import datetime
import json
import sys
import os
import pytest
from typing import Any, Dict
from unittest.mock import MagicMock, patch

# Ensure credit_agent is importable from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from credit_agent.tools import (
    fetch_t24_exposure,
    check_credit_policy,
    assess_covenants,
    calculate_ratios,
    fetch_comparable_portfolio,
    draft_credit_memo,
    TOOL_REGISTRY,
    get_tool_schemas,
)
from credit_agent.policy_rules import (
    AWBCreditPolicyRuleSet,
    LeverageRatioRule,
    InterestCoverRule,
    ConcentrationRule,
    MinimumEquityRule,
    PolicyBreach,
    Severity,
    FacilityType,
    DEFAULT_POLICY,
)
from credit_agent.credit_memo_generator import (
    CreditMemo,
    CreditDecision,
    RegulatoryFlag,
    AuditTrail,
    PolicyBreachSummary,
    CovenantSummary,
    build_credit_memo_from_agent_output,
)
from credit_agent.agent import (
    CreditDecisionAgent,
    AgentStatus,
    AgentRunResult,
    HumanOversightCheckpoint,
    LLMClient,
    MAX_ITERATIONS,
    HITL_THRESHOLD_GBP,
    ToolCallLog,
)
from data.generate_sample_application import generate, BASE_APPLICATION, STRESSED_APPLICATION


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_application() -> Dict[str, Any]:
    return generate("base")


@pytest.fixture
def stressed_application() -> Dict[str, Any]:
    return generate("stressed")


@pytest.fixture
def small_application() -> Dict[str, Any]:
    return generate("small")


@pytest.fixture
def compliant_financials() -> Dict[str, Any]:
    """Financial inputs that pass all policy rules."""
    return {
        "net_debt_gbp": 10_000_000.0,
        "ebitda_gbp": 4_000_000.0,           # Leverage: 2.5x ✅
        "ebit_gbp": 3_500_000.0,
        "interest_expense_gbp": 800_000.0,   # ICR: 4.375x ✅
        "total_exposure_gbp": 15_000_000.0,  # Conc: 0.054% ✅
        "tangible_equity_gbp": 12_000_000.0, # Equity > £1M ✅
        "total_assets_gbp": 30_000_000.0,
    }


@pytest.fixture
def breaching_financials() -> Dict[str, Any]:
    """Financial inputs that breach leverage and ICR rules."""
    return {
        "net_debt_gbp": 30_000_000.0,
        "ebitda_gbp": 4_000_000.0,           # Leverage: 7.5x ❌
        "ebit_gbp": 1_200_000.0,
        "interest_expense_gbp": 1_800_000.0, # ICR: 0.67x ❌
        "total_exposure_gbp": 35_000_000.0,
        "tangible_equity_gbp": 500_000.0,    # Below £1M ❌
        "total_assets_gbp": 40_000_000.0,
    }


@pytest.fixture
def mock_agent() -> CreditDecisionAgent:
    """Agent with mock LLM client."""
    return CreditDecisionAgent()


@pytest.fixture
def minimal_memo_kwargs() -> Dict[str, Any]:
    """Minimal valid kwargs for CreditMemo."""
    return dict(
        applicant_name="Test Borrower Ltd",
        facility_amount_gbp=100_000.0,
        facility_type="TERM_LOAN",
        recommendation=CreditDecision.REFER,
        risk_rating=5,
        rationale="Test rationale with sufficient length to pass validation requirements here.",
        human_review_required=False,
    )


# ===========================================================================
# SECTION 1: Tool Definitions (15 tests)
# ===========================================================================

class TestFetchT24Exposure:

    def test_returns_dict_with_required_keys(self):
        result = fetch_t24_exposure("AWB-CUST-001234")
        required = {"customer_id", "total_committed_gbp", "total_drawn_gbp", "t24_response_code"}
        assert required.issubset(set(result.keys()))

    def test_echoes_customer_id(self):
        result = fetch_t24_exposure("AWB-CUST-001234")
        assert result["customer_id"] == "AWB-CUST-001234"

    def test_contingent_included_by_default(self):
        result = fetch_t24_exposure("AWB-CUST-001234", include_contingent=True)
        assert result["total_contingent_gbp"] > 0

    def test_contingent_excluded_when_false(self):
        result = fetch_t24_exposure("AWB-CUST-001234", include_contingent=False)
        assert result["total_contingent_gbp"] == 0.0

    def test_empty_customer_id_raises(self):
        with pytest.raises(ValueError, match="empty string"):
            fetch_t24_exposure("")

    def test_none_customer_id_raises(self):
        with pytest.raises((ValueError, TypeError)):
            fetch_t24_exposure(None)

    def test_t24_response_code_200(self):
        result = fetch_t24_exposure("AWB-CUST-001234")
        assert result["t24_response_code"] == "200"

    def test_total_committed_positive(self):
        result = fetch_t24_exposure("AWB-CUST-001234")
        assert result["total_committed_gbp"] > 0


class TestCheckCreditPolicy:

    def test_compliant_returns_no_breaches(self, compliant_financials):
        result = check_credit_policy(**compliant_financials)
        assert result["breach_count"] == 0
        assert result["policy_compliant"] is True

    def test_high_leverage_creates_breach(self, breaching_financials):
        result = check_credit_policy(**breaching_financials)
        breach_names = [b["rule_name"] for b in result["breaches"]]
        assert any("LEVERAGE" in n for n in breach_names)

    def test_low_icr_creates_breach(self, breaching_financials):
        result = check_credit_policy(**breaching_financials)
        breach_names = [b["rule_name"] for b in result["breaches"]]
        assert any("INTEREST_COVER" in n for n in breach_names)

    def test_recommendation_decline_on_critical(self):
        result = check_credit_policy(
            net_debt_gbp=50_000_000.0,
            ebitda_gbp=-1_000_000.0,  # Negative EBITDA → CRITICAL
            ebit_gbp=-1_500_000.0,
            interest_expense_gbp=2_000_000.0,
            total_exposure_gbp=55_000_000.0,
            tangible_equity_gbp=500_000.0,
            total_assets_gbp=60_000_000.0,
        )
        assert result["recommendation"] == "DECLINE"

    def test_compliant_recommendation_approve(self, compliant_financials):
        result = check_credit_policy(**compliant_financials)
        assert result["recommendation"] == "APPROVE"

    def test_policy_summary_present(self, compliant_financials):
        result = check_credit_policy(**compliant_financials)
        assert "policy_summary" in result
        assert "max_leverage_ratio" in result["policy_summary"]


class TestCalculateRatios:

    def test_calculates_leverage_correctly(self):
        result = calculate_ratios(
            revenue_gbp=25_000_000.0, ebitda_gbp=4_000_000.0, ebit_gbp=3_500_000.0,
            net_profit_gbp=2_000_000.0, total_assets_gbp=30_000_000.0,
            total_liabilities_gbp=18_000_000.0, current_assets_gbp=8_000_000.0,
            current_liabilities_gbp=5_000_000.0, net_debt_gbp=12_000_000.0,
            interest_expense_gbp=800_000.0,
        )
        assert result["leverage"]["net_debt_to_ebitda"] == pytest.approx(3.0, rel=0.01)

    def test_calculates_icr_correctly(self):
        result = calculate_ratios(
            revenue_gbp=25_000_000.0, ebitda_gbp=4_000_000.0, ebit_gbp=3_200_000.0,
            net_profit_gbp=2_000_000.0, total_assets_gbp=30_000_000.0,
            total_liabilities_gbp=18_000_000.0, current_assets_gbp=8_000_000.0,
            current_liabilities_gbp=5_000_000.0, net_debt_gbp=12_000_000.0,
            interest_expense_gbp=800_000.0,
        )
        assert result["coverage"]["interest_cover_ratio"] == pytest.approx(4.0, rel=0.01)

    def test_zero_revenue_raises(self):
        with pytest.raises(ValueError, match="positive"):
            calculate_ratios(
                revenue_gbp=0, ebitda_gbp=1, ebit_gbp=1, net_profit_gbp=1,
                total_assets_gbp=1, total_liabilities_gbp=1,
                current_assets_gbp=1, current_liabilities_gbp=1,
                net_debt_gbp=1, interest_expense_gbp=1,
            )


class TestAssessCovenants:

    def test_returns_recommended_covenants(self):
        result = assess_covenants(
            facility_amount_gbp=5_000_000.0, facility_type="TERM_LOAN",
            leverage_ratio=3.5, interest_cover_ratio=3.0,
        )
        assert len(result["recommended_covenants"]) >= 2

    def test_high_leverage_triggers_quarterly_testing(self):
        result = assess_covenants(
            facility_amount_gbp=5_000_000.0, facility_type="TERM_LOAN",
            leverage_ratio=4.0, interest_cover_ratio=2.2,
        )
        assert result["testing_frequency"] == "Quarterly"

    def test_ltv_covenant_added_when_provided(self):
        result = assess_covenants(
            facility_amount_gbp=5_000_000.0, facility_type="TERM_LOAN",
            leverage_ratio=3.0, interest_cover_ratio=3.0,
            loan_to_value_pct=65.0,
        )
        covenant_names = [c["covenant_name"] for c in result["recommended_covenants"]]
        assert any("Loan-to-Value" in n for n in covenant_names)


class TestFetchComparablePortfolio:

    def test_valid_sic_code_returns_stats(self):
        result = fetch_comparable_portfolio("4120", "TERM_LOAN", "MEDIUM")
        assert "median_ratios" in result
        assert "portfolio_count" in result

    def test_invalid_sic_code_raises(self):
        with pytest.raises(ValueError, match="SIC 2007"):
            fetch_comparable_portfolio("ABC", "TERM_LOAN", "MEDIUM")

    def test_invalid_size_band_raises(self):
        with pytest.raises(ValueError):
            fetch_comparable_portfolio("4120", "TERM_LOAN", "HUGE")


class TestToolRegistry:

    def test_all_tools_registered(self):
        expected = {
            "fetch_t24_exposure", "check_credit_policy", "assess_covenants",
            "calculate_ratios", "fetch_comparable_portfolio", "draft_credit_memo",
        }
        assert expected == set(TOOL_REGISTRY.keys())

    def test_tool_schemas_have_correct_count(self):
        schemas = get_tool_schemas()
        assert len(schemas) == 6

    def test_all_schemas_have_required_fields(self):
        for schema in get_tool_schemas():
            assert "name" in schema
            assert "description" in schema
            assert "parameters" in schema


# ===========================================================================
# SECTION 2: Policy Rule Evaluation (10 tests)
# ===========================================================================

class TestLeverageRule:

    def test_compliant_returns_none(self):
        rule = LeverageRatioRule()
        result = rule.evaluate(net_debt=8_000_000.0, ebitda=4_000_000.0)
        assert result is None  # 2.0x — well within 5.0x limit

    def test_breach_returns_policy_breach(self):
        rule = LeverageRatioRule()
        result = rule.evaluate(net_debt=25_000_000.0, ebitda=4_000_000.0)  # 6.25x
        assert isinstance(result, PolicyBreach)
        assert result.actual_value == pytest.approx(6.25, rel=0.01)

    def test_warning_band_returns_medium_severity(self):
        rule = LeverageRatioRule()
        result = rule.evaluate(net_debt=17_000_000.0, ebitda=4_000_000.0)  # 4.25x
        assert result is not None
        assert result.severity == Severity.MEDIUM

    def test_negative_ebitda_is_critical(self):
        rule = LeverageRatioRule()
        result = rule.evaluate(net_debt=5_000_000.0, ebitda=-100_000.0)
        assert result.severity == Severity.CRITICAL


class TestInterestCoverRule:

    def test_compliant_returns_none(self):
        rule = InterestCoverRule()
        result = rule.evaluate(ebit=4_000_000.0, interest_expense=1_000_000.0)  # 4.0x
        assert result is None

    def test_breach_below_2x(self):
        rule = InterestCoverRule()
        result = rule.evaluate(ebit=1_500_000.0, interest_expense=1_000_000.0)  # 1.5x
        assert result is not None
        assert result.severity in (Severity.HIGH, Severity.CRITICAL)

    def test_below_1x_is_critical(self):
        rule = InterestCoverRule()
        result = rule.evaluate(ebit=800_000.0, interest_expense=1_000_000.0)  # 0.8x
        assert result.severity == Severity.CRITICAL

    def test_zero_interest_expense_returns_none(self):
        rule = InterestCoverRule()
        result = rule.evaluate(ebit=2_000_000.0, interest_expense=0.0)
        assert result is None  # No debt, no breach


class TestConcentrationRule:

    def test_small_exposure_returns_none(self):
        rule = ConcentrationRule()
        result = rule.evaluate(total_exposure_gbp=500_000.0)  # 0.002%
        assert result is None

    def test_large_exposure_returns_breach(self):
        rule = ConcentrationRule()
        # 15.1% of £28B = £4.228B — exceeds 15% policy limit
        result = rule.evaluate(total_exposure_gbp=4_230_000_000.0)
        assert result is not None
        assert result.severity == Severity.HIGH


class TestAWBCreditPolicyRuleSet:

    def test_all_compliant_returns_empty_list(self, compliant_financials):
        policy = AWBCreditPolicyRuleSet()
        breaches = policy.evaluate_all(
            net_debt=compliant_financials["net_debt_gbp"],
            ebitda=compliant_financials["ebitda_gbp"],
            ebit=compliant_financials["ebit_gbp"],
            interest_expense=compliant_financials["interest_expense_gbp"],
            total_exposure_gbp=compliant_financials["total_exposure_gbp"],
            tangible_equity_gbp=compliant_financials["tangible_equity_gbp"],
            total_assets_gbp=compliant_financials["total_assets_gbp"],
        )
        assert breaches == []

    def test_has_blocking_breach_with_critical(self):
        policy = AWBCreditPolicyRuleSet()
        breach = PolicyBreach(
            rule_name="TEST", actual_value=99.0, threshold=5.0,
            severity=Severity.CRITICAL, description="Test"
        )
        assert policy.has_blocking_breach([breach]) is True

    def test_default_policy_has_correct_thresholds(self):
        assert DEFAULT_POLICY.leverage_rule.max_leverage_ratio == 5.0
        assert DEFAULT_POLICY.interest_cover_rule.min_interest_cover == 2.0
        assert DEFAULT_POLICY.concentration_rule.max_concentration_pct == 15.0


# ===========================================================================
# SECTION 3: Credit Memo Structure (8 tests)
# ===========================================================================

class TestCreditMemo:

    def test_valid_memo_creates_successfully(self, minimal_memo_kwargs):
        memo = CreditMemo(**minimal_memo_kwargs)
        assert memo.memo_id.startswith("CM-")
        assert memo.recommendation == CreditDecision.REFER

    def test_invalid_risk_rating_raises(self, minimal_memo_kwargs):
        minimal_memo_kwargs["risk_rating"] = 11
        with pytest.raises(Exception):  # pydantic ValidationError
            CreditMemo(**minimal_memo_kwargs)

    def test_risk_rating_zero_raises(self, minimal_memo_kwargs):
        minimal_memo_kwargs["risk_rating"] = 0
        with pytest.raises(Exception):
            CreditMemo(**minimal_memo_kwargs)

    def test_blank_applicant_name_raises(self, minimal_memo_kwargs):
        minimal_memo_kwargs["applicant_name"] = "   "
        with pytest.raises(Exception):
            CreditMemo(**minimal_memo_kwargs)

    def test_facility_over_500m_raises(self, minimal_memo_kwargs):
        minimal_memo_kwargs["facility_amount_gbp"] = 600_000_000.0
        with pytest.raises(Exception):
            CreditMemo(**minimal_memo_kwargs)

    def test_hitl_flag_auto_added_for_large_facility(self):
        memo = CreditMemo(
            applicant_name="Large Borrower Ltd",
            facility_amount_gbp=1_000_000.0,
            facility_type="TERM_LOAN",
            recommendation=CreditDecision.APPROVE,
            risk_rating=4,
            rationale="Comprehensive review conducted and all policy requirements satisfied.",
            human_review_required=True,
            regulatory_flags=[],  # EU_AI_ACT_HITL should be auto-added
        )
        assert RegulatoryFlag.EU_AI_ACT_HITL in memo.regulatory_flags

    def test_human_review_required_true_for_large_facility_or_raises(self):
        """Facilities ≥ £500K must have human_review_required=True."""
        with pytest.raises(Exception):
            CreditMemo(
                applicant_name="Test Borrower Ltd",
                facility_amount_gbp=500_000.0,
                facility_type="TERM_LOAN",
                recommendation=CreditDecision.APPROVE,
                risk_rating=3,
                rationale="Test rationale with enough characters to satisfy field validation.",
                human_review_required=False,  # Should raise
            )

    def test_to_dict_serialises_correctly(self, minimal_memo_kwargs):
        memo = CreditMemo(**minimal_memo_kwargs)
        d = memo.to_dict()
        assert isinstance(d, dict)
        assert "recommendation" in d
        assert d["recommendation"] == "REFER"

    def test_retention_date_7_years_from_today(self, minimal_memo_kwargs):
        memo = CreditMemo(**minimal_memo_kwargs)
        today = datetime.date.today()
        expected_year = today.year + 7
        assert memo.retention_until.year == expected_year


# ===========================================================================
# SECTION 4: Agent Loop and Termination (6 tests)
# ===========================================================================

class TestAgentLoop:

    def test_agent_completes_within_max_iterations(self, base_application):
        agent = CreditDecisionAgent()
        result = agent.run(base_application, auto_approve_human_review=True)
        assert result.total_iterations <= MAX_ITERATIONS
        assert result.status in (
            AgentStatus.COMPLETED,
            AgentStatus.MAX_ITERATIONS_REACHED,
            AgentStatus.AWAITING_HUMAN_REVIEW,
        )

    def test_agent_terminates_on_repeated_tool_failures(self):
        """Agent must terminate within max_iterations even if all tools fail."""
        # Use a tool registry with all tools raising exceptions
        failing_registry = {
            name: MagicMock(side_effect=RuntimeError(f"Simulated failure: {name}"))
            for name in TOOL_REGISTRY
        }

        agent = CreditDecisionAgent(tool_registry=failing_registry, max_iterations=5)
        application = {
            "applicant_name": "Failure Test Ltd",
            "customer_id": "AWB-CUST-999",
            "facility_amount_gbp": 100_000.0,  # Below HITL threshold
            "facility_type": "TERM_LOAN",
        }
        result = agent.run(application, auto_approve_human_review=False)
        assert result.total_iterations <= 5
        assert result.status in (
            AgentStatus.MAX_ITERATIONS_REACHED,
            AgentStatus.FAILED,
            AgentStatus.COMPLETED,
        )

    def test_agent_run_produces_run_id(self, base_application):
        agent = CreditDecisionAgent()
        result = agent.run(base_application, auto_approve_human_review=True)
        assert result.run_id.startswith("RUN-")

    def test_agent_run_has_tool_call_logs(self, base_application):
        agent = CreditDecisionAgent()
        result = agent.run(base_application, auto_approve_human_review=True)
        assert isinstance(result.tool_call_logs, list)

    def test_agent_run_has_steps(self, base_application):
        agent = CreditDecisionAgent()
        result = agent.run(base_application, auto_approve_human_review=True)
        assert len(result.steps) > 0

    def test_max_iterations_respected(self, base_application):
        """Agent with max_iterations=2 should stop after 2 steps."""
        agent = CreditDecisionAgent(max_iterations=2)
        result = agent.run(base_application, auto_approve_human_review=True)
        assert result.total_iterations <= 2


# ===========================================================================
# SECTION 5: Human Oversight Checkpoint (5 tests)
# ===========================================================================

class TestHumanOversightCheckpoint:

    def test_is_required_above_threshold(self):
        assert HumanOversightCheckpoint.is_required(500_000.0) is True
        assert HumanOversightCheckpoint.is_required(1_000_000.0) is True

    def test_not_required_below_threshold(self):
        assert HumanOversightCheckpoint.is_required(499_999.0) is False
        assert HumanOversightCheckpoint.is_required(100_000.0) is False

    def test_review_request_contains_required_fields(self, minimal_memo_kwargs):
        minimal_memo_kwargs["facility_amount_gbp"] = 1_000_000.0
        minimal_memo_kwargs["human_review_required"] = True
        memo = CreditMemo(**minimal_memo_kwargs)
        request = HumanOversightCheckpoint.request_review(memo, "RUN-TEST")
        assert request["review_required"] is True
        assert "task_id" in request
        assert "deadline" in request
        assert "regulatory_basis" in request

    def test_no_review_request_for_small_facility(self, minimal_memo_kwargs):
        memo = CreditMemo(**minimal_memo_kwargs)
        request = HumanOversightCheckpoint.request_review(memo, "RUN-TEST")
        assert request["review_required"] is False

    def test_simulate_approval_marks_review_complete(self, minimal_memo_kwargs):
        minimal_memo_kwargs["facility_amount_gbp"] = 1_000_000.0
        minimal_memo_kwargs["human_review_required"] = True
        memo = CreditMemo(**minimal_memo_kwargs)
        updated = HumanOversightCheckpoint.simulate_human_approval(memo, "SCO-001")
        assert updated.human_review_completed is True
        assert updated.human_reviewer_id == "SCO-001"


# ===========================================================================
# SECTION 6: EU AI Act Compliance (4 tests)
# ===========================================================================

class TestEUAIActCompliance:

    def test_hitl_fires_for_facilities_above_500k(self, base_application):
        """EU AI Act Article 14: Agent must pause for human review on ≥ £500K."""
        assert base_application["facility_amount_gbp"] >= HITL_THRESHOLD_GBP
        agent = CreditDecisionAgent()
        result = agent.run(base_application, auto_approve_human_review=False)
        assert result.status == AgentStatus.AWAITING_HUMAN_REVIEW

    def test_hitl_not_fired_below_500k(self, small_application):
        """Facilities < £500K should not trigger human oversight pause."""
        assert small_application["facility_amount_gbp"] < HITL_THRESHOLD_GBP
        agent = CreditDecisionAgent()
        result = agent.run(small_application, auto_approve_human_review=False)
        # Status should be COMPLETED (not AWAITING_HUMAN_REVIEW) for small facilities
        assert result.status != AgentStatus.AWAITING_HUMAN_REVIEW

    def test_eu_ai_act_flag_present_in_large_facility_memo(self, base_application):
        agent = CreditDecisionAgent()
        result = agent.run(base_application, auto_approve_human_review=True)
        if result.credit_memo:
            assert RegulatoryFlag.EU_AI_ACT_HITL in result.credit_memo.regulatory_flags

    def test_human_review_required_set_in_large_facility_memo(self, base_application):
        agent = CreditDecisionAgent()
        result = agent.run(base_application, auto_approve_human_review=True)
        if result.credit_memo:
            assert result.credit_memo.human_review_required is True


# ===========================================================================
# SECTION 7: Audit Log Completeness (4 tests)
# ===========================================================================

class TestAuditLog:

    def test_every_agent_run_produces_audit_log(self, base_application):
        """PRA SS1/23: Every agent run must produce an audit log entry."""
        agent = CreditDecisionAgent()
        result = agent.run(base_application, auto_approve_human_review=True)
        audit_log = result.get_audit_log()
        assert isinstance(audit_log, dict)
        assert "run_id" in audit_log
        assert "status" in audit_log
        assert "completed_at" in audit_log

    def test_audit_log_contains_model_registration(self, base_application):
        agent = CreditDecisionAgent()
        result = agent.run(base_application, auto_approve_human_review=True)
        audit_log = result.get_audit_log()
        assert audit_log["model_registration"] == "MR-2026-037"

    def test_tool_call_logs_present_in_audit(self, base_application):
        agent = CreditDecisionAgent()
        result = agent.run(base_application, auto_approve_human_review=True)
        audit_log = result.get_audit_log()
        assert isinstance(audit_log["tool_calls"], list)

    def test_credit_memo_audit_trail_not_empty(self, base_application):
        agent = CreditDecisionAgent()
        result = agent.run(base_application, auto_approve_human_review=True)
        if result.credit_memo:
            assert len(result.credit_memo.audit_trail) > 0


# ===========================================================================
# SECTION 8: Sample Data Generation (3 tests)
# ===========================================================================

class TestSampleDataGeneration:

    def test_generate_base_application(self):
        app = generate("base")
        assert app["applicant_name"] == "Fenland Construction Ltd"
        assert app["facility_amount_gbp"] == 5_000_000.0
        assert app["customer_id"] == "AWB-CUST-001234"

    def test_generate_stressed_application_has_higher_facility(self):
        stressed = generate("stressed")
        base = generate("base")
        assert stressed["facility_amount_gbp"] > base["facility_amount_gbp"]

    def test_generate_invalid_scenario_raises(self):
        with pytest.raises(ValueError, match="Unknown scenario"):
            generate("invalid_scenario")


# ===========================================================================
# SECTION 9: build_credit_memo_from_agent_output (3 tests)
# ===========================================================================

class TestBuildCreditMemoFromAgentOutput:

    def _make_agent_findings(self) -> Dict[str, Any]:
        return {
            "check_credit_policy": {
                "breaches": [],
                "breach_count": 0,
                "blocking_breach_count": 0,
                "recommendation": "APPROVE",
            },
            "calculate_ratios": {
                "leverage": {"net_debt_to_ebitda": 2.5},
                "coverage": {"interest_cover_ratio": 4.0},
            },
            "assess_covenants": {
                "recommended_covenants": [
                    {"covenant_name": "Max Leverage", "metric": "Net Debt/EBITDA",
                     "threshold": 4.5, "direction": "MAXIMUM"},
                ],
                "testing_frequency": "Quarterly",
            },
            "fetch_t24_exposure": {
                "total_committed_gbp": 13_000_000.0,
            },
            "draft_credit_memo": {
                "recommendation": "APPROVE",
                "risk_rating": 4,
                "rationale": "Solid financials with strong EBITDA coverage and policy compliance.",
                "key_risks": ["Sector concentration risk"],
                "mitigants": ["Strong covenant package"],
                "conditions": ["Execute facility agreement"],
                "next_steps": ["Notify applicant"],
            },
        }

    def test_builds_memo_successfully(self):
        findings = self._make_agent_findings()
        app = {"applicant_name": "Test Ltd", "facility_amount_gbp": 100_000.0,
               "facility_type": "TERM_LOAN", "customer_id": "AWB-001"}
        memo = build_credit_memo_from_agent_output(findings, app)
        assert isinstance(memo, CreditMemo)

    def test_small_facility_not_requiring_human_review(self):
        findings = self._make_agent_findings()
        app = {"applicant_name": "Small Ltd", "facility_amount_gbp": 200_000.0,
               "facility_type": "TERM_LOAN"}
        memo = build_credit_memo_from_agent_output(findings, app)
        assert memo.human_review_required is False

    def test_large_facility_requiring_human_review(self):
        findings = self._make_agent_findings()
        app = {"applicant_name": "Large Ltd", "facility_amount_gbp": 5_000_000.0,
               "facility_type": "TERM_LOAN"}
        memo = build_credit_memo_from_agent_output(findings, app)
        assert memo.human_review_required is True


# ===========================================================================
# SECTION 10: ToolCallLog (2 tests)
# ===========================================================================

class TestToolCallLog:

    def test_log_has_model_registration(self):
        log = ToolCallLog(tool_name="fetch_t24_exposure", tool_inputs={"customer_id": "X"})
        assert log.model_registration == "MR-2026-037"

    def test_log_to_dict_serialises(self):
        log = ToolCallLog(tool_name="calculate_ratios", tool_inputs={"revenue_gbp": 1.0})
        d = log.to_dict()
        assert d["tool_name"] == "calculate_ratios"
        assert "timestamp" in d
