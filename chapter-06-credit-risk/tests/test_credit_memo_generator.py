"""Tests for the Credit Underwriting Assistant (Section 6.8B, MR-2026-068).

Exercises all three pipeline stages offline (no Gemini API key required):
extraction is bypassed by constructing ExtractedFinancials directly,
narrative generation falls back to the deterministic template.
"""
import sys
import os
from decimal import Decimal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from underwriting.credit_memo_generator import (
    CreditMemoGenerator,
    ExtractedFinancials,
    RM_DISCRETION_LIMIT_GBP,
    MODEL_ID,
)
from awb_commons.audit import AuditLogger


# AWB illustrative SME manufacturing borrower
HEALTHY_FINANCIALS = ExtractedFinancials(
    total_assets_gbp=Decimal("5_000_000"),
    current_assets_gbp=Decimal("2_000_000"),
    inventory_gbp=Decimal("400_000"),
    cash_gbp=Decimal("300_000"),
    current_liabilities_gbp=Decimal("1_200_000"),
    total_liabilities_gbp=Decimal("2_500_000"),
    total_debt_gbp=Decimal("1_800_000"),
    tangible_net_worth_gbp=Decimal("2_500_000"),
    revenue_gbp=Decimal("6_000_000"),
    cogs_gbp=Decimal("4_200_000"),
    ebitda_gbp=Decimal("900_000"),
    ebit_gbp=Decimal("700_000"),
    interest_expense_gbp=Decimal("120_000"),
    scheduled_principal_repayment_gbp=Decimal("180_000"),
    net_income_gbp=Decimal("400_000"),
    operating_cash_flow_gbp=Decimal("650_000"),
    capex_gbp=Decimal("250_000"),
)

# Companies House abbreviated accounts — cogs/cash-flow/capex omitted
SME_ABBREVIATED_FINANCIALS = ExtractedFinancials(
    total_assets_gbp=Decimal("800_000"),
    current_assets_gbp=Decimal("300_000"),
    current_liabilities_gbp=Decimal("250_000"),
    total_liabilities_gbp=Decimal("500_000"),
    total_debt_gbp=Decimal("350_000"),
    tangible_net_worth_gbp=Decimal("300_000"),
    revenue_gbp=Decimal("1_100_000"),
    ebitda_gbp=Decimal("140_000"),
    ebit_gbp=Decimal("110_000"),
    interest_expense_gbp=Decimal("28_000"),
    net_income_gbp=Decimal("60_000"),
)


@pytest.fixture
def generator() -> CreditMemoGenerator:
    logger = AuditLogger(MODEL_ID)
    logger.clear_for_test()
    return CreditMemoGenerator(audit_logger=logger)


class TestRatioCalculation:
    def test_sixteen_ratios_calculated(self, generator):
        ratios = generator.calculate_ratios(HEALTHY_FINANCIALS, "manufacturing")
        assert len(ratios) == 16
        assert len({r.ratio_name for r in ratios}) == 16

    def test_current_ratio_value(self, generator):
        ratios = generator.calculate_ratios(HEALTHY_FINANCIALS, "manufacturing")
        current = next(r for r in ratios if r.ratio_name == "current_ratio")
        assert current.value == pytest.approx(2_000_000 / 1_200_000)

    def test_missing_fields_are_not_assessed(self, generator):
        ratios = generator.calculate_ratios(SME_ABBREVIATED_FINANCIALS, "manufacturing")
        gross_margin = next(r for r in ratios if r.ratio_name == "gross_margin_pct")
        cash_conversion = next(r for r in ratios if r.ratio_name == "cash_conversion_pct")
        assert gross_margin.assessment == "NOT_ASSESSED"
        assert cash_conversion.assessment == "NOT_ASSESSED"
        assert gross_margin.value is None

    def test_unknown_sector_falls_back_to_default_bands(self, generator):
        ratios = generator.calculate_ratios(HEALTHY_FINANCIALS, "unknown_sector")
        assert len(ratios) == 16
        assert any(r.benchmark is not None for r in ratios)

    def test_zero_denominator_is_not_assessed(self, generator):
        # Zero interest expense AND zero scheduled principal repayment
        # zeroes DSCR's denominator (debt service), not its numerator.
        no_debt_service = HEALTHY_FINANCIALS.model_copy(
            update={
                "interest_expense_gbp": Decimal("0"),
                "scheduled_principal_repayment_gbp": Decimal("0"),
            }
        )
        ratios = generator.calculate_ratios(no_debt_service, "manufacturing")
        dscr = next(r for r in ratios if r.ratio_name == "dscr")
        assert dscr.value is None
        assert dscr.assessment == "NOT_ASSESSED"


class TestNarrativeGeneration:
    def test_fallback_narrative_offline(self, generator):
        ratios = generator.calculate_ratios(HEALTHY_FINANCIALS, "manufacturing")
        narrative = generator.generate_narrative(
            HEALTHY_FINANCIALS, ratios, "Bristol Fabrication Ltd", "manufacturing"
        )
        assert "Bristol Fabrication Ltd" in narrative
        assert MODEL_ID in narrative


class TestMemoOrchestration:
    def test_generate_memo_below_discretion_limit(self, generator):
        memo = generator.generate_memo(
            facility_id="FAC-2026-0341",
            borrower_name="Bristol Fabrication Ltd",
            sector="manufacturing",
            exposure_gbp=Decimal("1_500_000"),
            extracted=HEALTHY_FINANCIALS,
        )
        assert memo.requires_committee_referral is False
        assert memo.model_id == MODEL_ID
        assert len(memo.ratios) == 16
        assert memo.rm_approved is False

    def test_generate_memo_above_discretion_limit_requires_referral(self, generator):
        memo = generator.generate_memo(
            facility_id="FAC-2026-0912",
            borrower_name="Wessex Engineering plc",
            sector="manufacturing",
            exposure_gbp=RM_DISCRETION_LIMIT_GBP + Decimal("1"),
            extracted=HEALTHY_FINANCIALS,
        )
        assert memo.requires_committee_referral is True

    def test_no_memo_reaches_committee_without_rm_signoff(self, generator):
        memo = generator.generate_memo(
            facility_id="FAC-2026-0341",
            borrower_name="Bristol Fabrication Ltd",
            sector="manufacturing",
            exposure_gbp=Decimal("1_500_000"),
            extracted=HEALTHY_FINANCIALS,
        )
        assert memo.rm_approved is False
        approved = generator.approve(memo, rm_id="RM-014", comments="Looks fine")
        assert approved.rm_approved is True
        assert approved.rm_id == "RM-014"

    def test_audit_trail_written_on_generate_and_approve(self, generator):
        memo = generator.generate_memo(
            facility_id="FAC-2026-0341",
            borrower_name="Bristol Fabrication Ltd",
            sector="manufacturing",
            exposure_gbp=Decimal("1_500_000"),
            extracted=HEALTHY_FINANCIALS,
        )
        generator.approve(memo, rm_id="RM-014")
        records = generator.audit.get_records(facility_id="FAC-2026-0341")
        event_types = [r["event_type"] for r in records]
        assert event_types.count("DECISION") == 2  # generate + approve
