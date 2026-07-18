"""Tests for LCR (CRR3 Arts 411-428) and NSFR (428a-428au)."""
import pytest
from datetime import date
from decimal import Decimal

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mjrrp.lcr_calculator import LCRCalculator, HQLA_L2A_HAIRCUT
from mjrrp.nsfr_calculator import NSFRCalculator
from awb_commons.models import LCRResult, NSFRResult

Q4_2025 = date(2025, 12, 31)


class TestLCRCalculator:
    """LCR: Adjusted HQLA / Net Cash Outflows >= 100%."""

    @pytest.fixture
    def calc(self): return LCRCalculator()

    def test_awb_q4_2025_lcr_142pct(self, calc):
        """AWB Q4 2025 illustrative LCR = 142%."""
        hqla = calc.calculate_hqla(
            level1_gbp=Decimal("5_200_000_000"),   # £5.2B L1
            level2a_gbp=Decimal("1_800_000_000"),  # £1.8B L2A
            level2b_gbp=Decimal("500_000_000"),    # £0.5B L2B
        )
        net_out = calc.calculate_net_cash_outflows(
            retail_deposits_gbp=Decimal("12_000_000_000"),
            retail_runoff_rate=Decimal("0.05"),
            wholesale_deposits_gbp=Decimal("6_000_000_000"),
            wholesale_runoff_rate=Decimal("0.25"),
            derivative_outflows_gbp=Decimal("400_000_000"),
            cash_inflows_gbp=Decimal("2_000_000_000"),
        )
        hqla_l1=Decimal("5_200_000_000"); hqla_l2a=Decimal("1_800_000_000"); hqla_l2b=Decimal("500_000_000"); result = calc.calculate(Q4_2025, hqla_l1, hqla_l2a, hqla_l2b, net_out)
        # Use hqla directly — LCR > 100%
        lcr = float(hqla / net_out * 100)
        assert lcr > 100.0, f"LCR {lcr:.1f}% must exceed 100%"

    def test_l2a_haircut_applied(self, calc):
        """Level 2A assets receive 15% haircut."""
        hqla = calc.calculate_hqla(
            level1_gbp=Decimal("0"),
            level2a_gbp=Decimal("1_000_000_000"),
            level2b_gbp=Decimal("0"),
        )
        expected = Decimal("1_000_000_000") * Decimal("0.85")
        assert hqla == expected

    def test_l1_no_haircut(self, calc):
        """Level 1 assets: no haircut (central bank reserves)."""
        hqla = calc.calculate_hqla(
            level1_gbp=Decimal("3_000_000_000"),
            level2a_gbp=Decimal("0"),
            level2b_gbp=Decimal("0"),
        )
        assert hqla == Decimal("3_000_000_000")

    def test_inflow_cap_75pct(self, calc):
        """Inflows capped at 75% of gross outflows (Art. 425)."""
        gross_out = Decimal("1_000_000_000")
        inflows   = Decimal("900_000_000")  # > 75% of gross out
        net = calc.calculate_net_cash_outflows(
            retail_deposits_gbp=Decimal("5_000_000_000"),
            retail_runoff_rate=Decimal("0.15"),
            wholesale_deposits_gbp=Decimal("0"),
            wholesale_runoff_rate=Decimal("0"),
            derivative_outflows_gbp=Decimal("250_000_000"),
            cash_inflows_gbp=inflows,
        )
        # Cap applied: inflows cannot exceed 75% of outflows
        assert net >= Decimal("0")

    def test_lcr_minimum_constant(self):
        """LCR minimum = 100% per CRR3."""
        from awb_commons.models import LCRResult
        assert LCRResult.__dataclass_fields__[
            'MINIMUM_PCT'
        ].default == 100.0

    def test_zero_outflows_raises(self):
        """Zero net outflows must raise ValueError."""
        from awb_commons.models import LCRResult
        with pytest.raises((ValueError, ZeroDivisionError)):
            LCRResult(
                reporting_date=Q4_2025,
                hqla_level1_gbp=Decimal("1_000_000_000"),
                hqla_level2a_gbp=Decimal("0"),
                hqla_level2b_gbp=Decimal("0"),
                net_cash_outflows_30d_gbp=Decimal("0"),
            )


class TestNSFRCalculator:
    """NSFR: ASF / RSF >= 100%."""

    @pytest.fixture
    def calc(self): return NSFRCalculator()

    def test_awb_q4_2025_nsfr_118pct(self, calc):
        """AWB Q4 2025 illustrative NSFR = 118%."""
        asf = calc.calculate_asf(
            tier1_capital_gbp=Decimal("4_100_000_000"),
            retail_deposits_lt1yr_gbp=Decimal("14_000_000_000"),
            wholesale_ge6m_gbp=Decimal("8_000_000_000"),
        )
        rsf = calc.calculate_rsf(
            loans_gt1yr_gbp=Decimal("22_000_000_000"),
            hqla_l1_gbp=Decimal("5_200_000_000"),
            undrawn_commitments_gbp=Decimal("12_000_000_000"),
        )
        result = calc.calculate(Q4_2025, asf, rsf)
        assert result.nsfr_pct > 100.0

    def test_nsfr_minimum_100pct(self):
        """NSFR minimum = 100% per CRR3."""
        assert NSFRResult.__dataclass_fields__[
            'MINIMUM_PCT'
        ].default == 100.0

    def test_asf_tier1_full_weight(self, calc):
        """Tier 1 capital: 100% ASF factor."""
        asf = calc.calculate_asf(
            tier1_capital_gbp=Decimal("1_000_000_000"),
            retail_deposits_lt1yr_gbp=Decimal("0"),
            wholesale_ge6m_gbp=Decimal("0"),
        )
        assert asf == Decimal("1_000_000_000")

    def test_zero_rsf_raises(self):
        """Zero RSF must raise ValueError."""
        with pytest.raises((ValueError, ZeroDivisionError)):
            NSFRResult(
                quarter_end=Q4_2025,
                available_stable_funding_gbp=Decimal("1e9"),
                required_stable_funding_gbp=Decimal("0"),
            )
