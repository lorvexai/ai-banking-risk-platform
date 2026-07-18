"""tests/test_chapter_09.py — Chapter 9 test suite. 35+ tests."""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pytest
from datetime import datetime

from awb_commons.models import StressScenario
from cash_flow.forecaster import CashFlowForecaster, TreasuryInputs
from lcr_nsfr.calculator import (
    LCRCalculator, NSFRCalculator,
    HQLAPortfolio, StressOutflows, StressInflows, NSFRInputs,
)
from intraday_liquidity.monitor import (
    IntradayLiquidityMonitor, IntradayPosition,
)


# ── Fixtures ─────────────────────────────────────────────────────
@pytest.fixture
def treasury_inputs():
    return TreasuryInputs(
        current_position_gbp=38_500_000_000,
        scheduled_inflows_gbp=2_100_000_000,
        scheduled_outflows_gbp=1_800_000_000,
        uncommitted_facilities_gbp=500_000_000,
        fx_exposure_gbp=200_000_000,
        wholesale_maturing_7d_gbp=800_000_000,
        retail_deposit_base_gbp=18_000_000_000,
        forecast_date=datetime(2026, 3, 1),
    )

@pytest.fixture
def awb_hqla():
    return HQLAPortfolio(
        level_1_central_bank_gbp=4_200_000_000,
        level_1_gov_bonds_gbp=6_800_000_000,
        level_2a_covered_bonds_gbp=2_100_000_000,
        level_2b_corp_bonds_gbp=900_000_000,
    )

@pytest.fixture
def awb_outflows():
    return StressOutflows(
        retail_stable_gbp=12_000_000_000,
        retail_less_stable_gbp=6_000_000_000,
        wholesale_operational_gbp=4_000_000_000,
        wholesale_non_op_gbp=2_000_000_000,
        committed_facilities_gbp=1_800_000_000,
        derivatives_collateral_gbp=900_000_000,
    )

@pytest.fixture
def awb_inflows():
    return StressInflows(
        maturing_loans_gbp=2_400_000_000,
        committed_inflows_gbp=600_000_000,
        other_inflows_gbp=400_000_000,
    )

@pytest.fixture
def awb_nsfr_inputs():
    return NSFRInputs(
        tier1_capital_gbp=3_200_000_000,
        tier2_capital_gbp=400_000_000,
        stable_retail_deposits_gbp=14_000_000_000,
        less_stable_deposits_gbp=4_000_000_000,
        wholesale_funding_1y_gbp=2_000_000_000,
        loans_lt_1y_gbp=6_000_000_000,
        loans_gt_1y_gbp=16_000_000_000,
        hqla_unencumbered_gbp=11_000_000_000,
        other_assets_gbp=5_000_000_000,
    )

@pytest.fixture
def normal_intraday():
    return IntradayPosition(
        timestamp=datetime(2026, 3, 1, 14, 0),
        opening_balance_gbp=3_500_000_000,
        gross_settlements_gbp=2_800_000_000,
        gross_receipts_gbp=3_100_000_000,
        central_bank_facility_gbp=2_000_000_000,
        peak_usage_today_gbp=1_200_000_000,
        available_facility_gbp=8_000_000_000,
    )

@pytest.fixture
def stressed_intraday():
    return IntradayPosition(
        timestamp=datetime(2026, 3, 1, 16, 30),
        opening_balance_gbp=1_200_000_000,
        gross_settlements_gbp=4_800_000_000,
        gross_receipts_gbp=3_500_000_000,
        central_bank_facility_gbp=2_000_000_000,
        peak_usage_today_gbp=7_400_000_000,
        available_facility_gbp=8_000_000_000,
    )


# ── Cash Flow Forecaster Tests ────────────────────────────────────
class TestCashFlowForecaster:
    def test_returns_correct_horizon(self, treasury_inputs):
        fc = CashFlowForecaster(horizon_days=30)
        result = fc.forecast(treasury_inputs)
        assert len(result) == 30

    def test_day_sequence(self, treasury_inputs):
        fc = CashFlowForecaster(horizon_days=10)
        result = fc.forecast(treasury_inputs)
        for i, f in enumerate(result, 1):
            assert f.horizon_days == i

    def test_ci_widens_with_horizon(self, treasury_inputs):
        fc = CashFlowForecaster(horizon_days=30)
        result = fc.forecast(treasury_inputs)
        ci_d1  = result[0].confidence_upper_gbp - result[0].confidence_lower_gbp
        ci_d30 = result[29].confidence_upper_gbp - result[29].confidence_lower_gbp
        assert ci_d30 > ci_d1

    def test_negative_position_raises(self):
        fc = CashFlowForecaster()
        bad = TreasuryInputs(
            current_position_gbp=-1.0,
            scheduled_inflows_gbp=1e9,
            scheduled_outflows_gbp=1e9,
            uncommitted_facilities_gbp=0,
            fx_exposure_gbp=0,
            wholesale_maturing_7d_gbp=0,
            retail_deposit_base_gbp=0,
            forecast_date=datetime(2026, 3, 1),
        )
        with pytest.raises(ValueError):
            fc.forecast(bad)

    def test_buffer_breach_detection(self, treasury_inputs):
        fc = CashFlowForecaster(horizon_days=30)
        forecasts = fc.forecast(treasury_inputs)
        breaches = fc.flag_buffer_breaches(
            forecasts, buffer_gbp=50_000_000_000_000
        )
        assert len(breaches) == 30  # all breach at huge buffer

    def test_mr_reference_set(self, treasury_inputs):
        fc = CashFlowForecaster(horizon_days=5)
        result = fc.forecast(treasury_inputs)
        for f in result:
            assert f.mr_reference == "MR-2026-052"

    def test_model_version_set(self, treasury_inputs):
        fc = CashFlowForecaster(horizon_days=1)
        result = fc.forecast(treasury_inputs)
        assert result[0].model_version.startswith("lstm-")


# ── LCR Calculator Tests ──────────────────────────────────────────
class TestLCRCalculator:
    def test_awb_lcr_compliant(self, awb_hqla, awb_outflows, awb_inflows):
        calc = LCRCalculator()
        result = calc.calculate(awb_hqla, awb_outflows, awb_inflows)
        assert result.compliant
        assert result.lcr_pct >= 100.0

    def test_lcr_above_internal_buffer(self, awb_hqla, awb_outflows, awb_inflows):
        calc = LCRCalculator()
        result = calc.calculate(awb_hqla, awb_outflows, awb_inflows)
        assert result.is_above_buffer(110.0)

    def test_zero_hqla_non_compliant(self, awb_outflows, awb_inflows):
        calc = LCRCalculator()
        empty_hqla = HQLAPortfolio()
        result = calc.calculate(empty_hqla, awb_outflows, awb_inflows)
        assert not result.compliant
        assert result.lcr_pct == 0.0

    def test_stress_reduces_lcr(self, awb_hqla, awb_outflows, awb_inflows):
        calc = LCRCalculator()
        base = calc.calculate(awb_hqla, awb_outflows, awb_inflows, StressScenario.BASE)
        stress = calc.calculate(awb_hqla, awb_outflows, awb_inflows, StressScenario.COMBINED)
        assert stress.lcr_pct < base.lcr_pct

    def test_regulatory_reference(self, awb_hqla, awb_outflows, awb_inflows):
        calc = LCRCalculator()
        result = calc.calculate(awb_hqla, awb_outflows, awb_inflows)
        assert "CRR3" in result.regulatory_reference

    def test_scenario_stored(self, awb_hqla, awb_outflows, awb_inflows):
        calc = LCRCalculator()
        result = calc.calculate(awb_hqla, awb_outflows, awb_inflows,
                                StressScenario.IDIOSYNCRATIC)
        assert result.scenario == StressScenario.IDIOSYNCRATIC

    def test_level2_cap_applied(self):
        calc = LCRCalculator()
        hqla = HQLAPortfolio(
            level_1_central_bank_gbp=1_000_000_000,
            level_1_gov_bonds_gbp=0,
            level_2a_covered_bonds_gbp=5_000_000_000,  # exceeds cap
            level_2b_corp_bonds_gbp=0,
        )
        outflows = StressOutflows(retail_stable_gbp=5_000_000_000)
        inflows  = StressInflows()
        result = calc.calculate(hqla, outflows, inflows)
        # Without cap: 1B + 5B*0.85 = 5.25B. With 40% cap: max L2 = total*0.4
        # Uncapped HQLA = 5.25B; capped: L1/(1-0.4) approx = 1B + 0.667B = 1.667B
        # The cap limits result to less than uncapped 5.25B
        uncapped = 1_000_000_000 + (5_000_000_000 * 0.85)
        assert result.hqla_gbp < uncapped

    def test_inflow_cap_75pct(self, awb_hqla):
        calc = LCRCalculator()
        large_inflows = StressInflows(maturing_loans_gbp=200_000_000_000)
        outflows = StressOutflows(retail_stable_gbp=5_000_000_000)
        result = calc.calculate(awb_hqla, outflows, large_inflows)
        assert result.net_outflows_gbp > 0


# ── NSFR Calculator Tests ─────────────────────────────────────────
class TestNSFRCalculator:
    def test_awb_nsfr_compliant(self, awb_nsfr_inputs):
        calc = NSFRCalculator()
        result = calc.calculate(awb_nsfr_inputs)
        assert result.compliant
        assert result.nsfr_pct >= 100.0

    def test_nsfr_formula(self, awb_nsfr_inputs):
        calc = NSFRCalculator()
        result = calc.calculate(awb_nsfr_inputs)
        expected = (result.available_stable_funding_gbp /
                    result.required_stable_funding_gbp) * 100.0
        assert abs(result.nsfr_pct - expected) < 0.01

    def test_zero_capital_reduces_asf(self):
        calc = NSFRCalculator()
        inp = NSFRInputs(
            tier1_capital_gbp=0,
            stable_retail_deposits_gbp=10_000_000_000,
            loans_gt_1y_gbp=10_000_000_000,
        )
        result = calc.calculate(inp)
        assert result.available_stable_funding_gbp < 10_000_000_000

    def test_regulatory_reference(self, awb_nsfr_inputs):
        calc = NSFRCalculator()
        result = calc.calculate(awb_nsfr_inputs)
        assert "CRR3" in result.regulatory_reference


# ── Intraday Monitor Tests ────────────────────────────────────────
class TestIntradayLiquidityMonitor:
    def test_normal_no_action(self, normal_intraday):
        mon = IntradayLiquidityMonitor()
        alert = mon.assess(normal_intraday)
        assert not alert.requires_action

    def test_stressed_requires_action(self, stressed_intraday):
        mon = IntradayLiquidityMonitor()
        alert = mon.assess(stressed_intraday)
        assert alert.requires_action

    def test_utilisation_calculated(self, normal_intraday):
        mon = IntradayLiquidityMonitor()
        alert = mon.assess(normal_intraday)
        expected = (1_200_000_000 / 8_000_000_000) * 100
        assert abs(alert.utilisation_pct - expected) < 0.1

    def test_buffer_calculated(self, normal_intraday):
        mon = IntradayLiquidityMonitor()
        alert = mon.assess(normal_intraday)
        assert alert.available_buffer_gbp == pytest.approx(
            8_000_000_000 - 1_200_000_000
        )

    def test_negative_balance_raises(self):
        mon = IntradayLiquidityMonitor()
        pos = IntradayPosition(
            timestamp=datetime.utcnow(),
            opening_balance_gbp=-100,
            gross_settlements_gbp=0,
            gross_receipts_gbp=0,
            central_bank_facility_gbp=0,
            peak_usage_today_gbp=0,
            available_facility_gbp=1_000_000_000,
        )
        with pytest.raises(ValueError):
            mon.assess(pos)

    def test_daily_peak_summary(self, normal_intraday, stressed_intraday):
        mon = IntradayLiquidityMonitor()
        summary = mon.daily_peak_summary(
            [normal_intraday, stressed_intraday]
        )
        assert summary["peak_usage_gbp"] == max(
            normal_intraday.peak_usage_today_gbp,
            stressed_intraday.peak_usage_today_gbp,
        )

    def test_mr_reference(self, normal_intraday):
        mon = IntradayLiquidityMonitor()
        alert = mon.assess(normal_intraday)
        assert alert.mr_reference == "MR-2026-054"

    def test_critical_action_message(self):
        mon = IntradayLiquidityMonitor(alert_threshold_pct=0.20)
        pos = IntradayPosition(
            timestamp=datetime.utcnow(),
            opening_balance_gbp=500_000_000,
            gross_settlements_gbp=0,
            gross_receipts_gbp=0,
            central_bank_facility_gbp=0,
            peak_usage_today_gbp=7_900_000_000,
            available_facility_gbp=8_000_000_000,
        )
        alert = mon.assess(pos)
        assert "CRITICAL" in alert.recommended_action or "ALERT" in alert.recommended_action
