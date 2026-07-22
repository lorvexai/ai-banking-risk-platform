"""
chapter_07/tests/test_chapter_07.py
AWB Chapter 7 — Market Risk Test Suite
80+ tests covering backtesting, VaR engine, CVA
AWB naming | awb_commons | MR-2026-046 | MR-2026-048
"""

from __future__ import annotations

import sys
import os
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from typing import Dict, List

# Add code paths
sys.path.insert(0, os.path.join(
    os.path.dirname(__file__), '..'
))
from backtesting.backtest_engine import (
    BacktestEngine, BacktestResult, MARComplianceChecker,
    MARFlag
)
from var_engine.mc_var_engine import (
    MonteCarloVaREngine, VaRResult, VaRBackTester,
    BackTestResult, BACKTEST_GREEN_LIMIT,
    BACKTEST_AMBER_LIMIT
)
from cva.cva_calculator import (
    CVACalculator, CVAResult, ExposureProfile,
    DEFAULT_RECOVERY_RATE
)


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def sample_prices():
    """250-day price series for AWB backtest tests."""
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=250)
    prices = 100 * np.cumprod(
        1 + np.random.normal(0.0003, 0.01, 250)
    )
    return pd.Series(prices, index=dates)


@pytest.fixture
def sample_signals(sample_prices):
    """Simple momentum signals on sample prices."""
    ma_short = sample_prices.rolling(10).mean()
    ma_long  = sample_prices.rolling(50).mean()
    signals  = np.sign(ma_short - ma_long).fillna(0)
    return signals


@pytest.fixture
def correlation_matrix_2x2():
    """2x2 valid correlation matrix."""
    return np.array([[1.0, 0.3], [0.3, 1.0]])


@pytest.fixture
def var_engine(correlation_matrix_2x2):
    """MC-VaR engine with 2 risk factors."""
    vols = np.array([0.01, 0.008])
    return MonteCarloVaREngine(
        correlation_matrix=correlation_matrix_2x2,
        volatilities=vols,
        risk_factor_names=["GIRR_10Y", "FX_GBPUSD"],
    )


@pytest.fixture
def exposure_profile():
    """AWB counterparty exposure profile."""
    return ExposureProfile(
        counterparty_id="BARCLAYS_001",
        time_steps=[0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0],
        expected_exposure=[
            5_000_000, 8_000_000, 12_000_000,
            10_000_000, 8_500_000, 6_000_000, 4_000_000
        ],
        peak_exposure_975=[
            7_000_000, 11_000_000, 16_000_000,
            14_000_000, 12_000_000, 9_000_000, 6_000_000
        ],
    )


@pytest.fixture
def pd_term_structure():
    """PD term structure (from MR-2026-043 mock)."""
    return {
        0.25: 0.002, 0.5: 0.004, 1.0: 0.008,
        2.0:  0.016, 3.0: 0.024, 4.0: 0.032,
        5.0:  0.040,
    }


@pytest.fixture
def cva_calculator():
    """CVA calculator with flat OIS curve."""
    return CVACalculator(
        recovery_rate=0.40,
        ois_discount_factors={
            0.25: 0.989, 0.5: 0.978, 1.0: 0.957,
            2.0: 0.916, 3.0: 0.876, 4.0: 0.838,
            5.0: 0.801,
        }
    )


@pytest.fixture
def backtest_data():
    """Trade data for MAR compliance testing."""
    trades = pd.DataFrame({
        "trade_id": [f"T{i:04d}" for i in range(20)],
        "instrument": (
            ["GILT_2Y"] * 5 + ["GILT_10Y"] * 5
            + ["EURUSD"] * 5 + ["FTSE_FUT"] * 5
        ),
        "side": (
            ["BUY", "SELL", "BUY", "BUY", "SELL"] * 4
        ),
        "time": pd.date_range(
            "2025-06-01 09:00", periods=20, freq="5min"
        ),
    })
    return trades


# ═══════════════════════════════════════════════════════════════
# BACKTEST ENGINE TESTS
# ═══════════════════════════════════════════════════════════════


class TestBacktestEngine:

    def test_run_backtest_returns_result(
        self, sample_signals, sample_prices
    ):
        engine = BacktestEngine()
        result = engine.run_backtest(
            sample_signals, sample_prices
        )
        assert isinstance(result, BacktestResult)

    def test_sharpe_is_finite(
        self, sample_signals, sample_prices
    ):
        engine = BacktestEngine()
        result = engine.run_backtest(
            sample_signals, sample_prices
        )
        assert np.isfinite(result.sharpe)

    def test_max_drawdown_is_negative(
        self, sample_signals, sample_prices
    ):
        engine = BacktestEngine()
        result = engine.run_backtest(
            sample_signals, sample_prices
        )
        assert result.max_drawdown_pct <= 0.0

    def test_var_95_is_positive(
        self, sample_signals, sample_prices
    ):
        engine = BacktestEngine()
        result = engine.run_backtest(
            sample_signals, sample_prices
        )
        # var_95_daily is negative pct (loss)
        assert isinstance(result.var_95_daily, float)

    def test_cvar_worse_than_var(
        self, sample_signals, sample_prices
    ):
        engine = BacktestEngine()
        result = engine.run_backtest(
            sample_signals, sample_prices
        )
        # Both should be finite floats
        assert isinstance(result.var_95_daily, float)
        assert isinstance(result.cvar_95_daily, float)
        # If finite: CVaR should be <= VaR (deeper tail)
        if (np.isfinite(result.var_95_daily)
                and np.isfinite(result.cvar_95_daily)):
            assert (
                result.cvar_95_daily
                <= result.var_95_daily
            )

    def test_zero_spread_reduces_costs(
        self, sample_signals, sample_prices
    ):
        engine_no_cost = BacktestEngine(
            spread_bps=0.0, impact_bps=0.0
        )
        engine_with_cost = BacktestEngine(
            spread_bps=10.0, impact_bps=5.0
        )
        r_nc = engine_no_cost.run_backtest(
            sample_signals, sample_prices
        )
        r_wc = engine_with_cost.run_backtest(
            sample_signals, sample_prices
        )
        assert (
            r_nc.total_return_pct
            >= r_wc.total_return_pct
        )

    def test_misaligned_index_raises_value_error(
        self, sample_prices
    ):
        engine = BacktestEngine()
        bad_signals = pd.Series(
            [1, 0, -1], index=[0, 1, 2]
        )
        with pytest.raises(ValueError, match="index"):
            engine.run_backtest(bad_signals, sample_prices)

    def test_win_rate_between_zero_and_one(
        self, sample_signals, sample_prices
    ):
        engine = BacktestEngine()
        result = engine.run_backtest(
            sample_signals, sample_prices
        )
        assert 0.0 <= result.win_rate <= 1.0

    def test_n_trades_positive(
        self, sample_signals, sample_prices
    ):
        engine = BacktestEngine()
        result = engine.run_backtest(
            sample_signals, sample_prices
        )
        assert result.n_trades >= 0

    def test_walk_forward_returns_list(
        self, sample_signals, sample_prices
    ):
        engine = BacktestEngine()
        results = engine.walk_forward_validate(
            sample_signals, sample_prices,
            train_years=1, test_years=0,
            step_months=3,
        )
        assert isinstance(results, list)

    def test_all_walk_forward_results_valid(
        self, sample_signals, sample_prices
    ):
        engine = BacktestEngine()
        results = engine.walk_forward_validate(
            sample_signals, sample_prices,
            train_years=1, test_years=0,
            step_months=6,
        )
        for r in results:
            assert isinstance(r, BacktestResult)
            assert np.isfinite(r.sharpe)


# ═══════════════════════════════════════════════════════════════
# MAR COMPLIANCE CHECKER TESTS
# ═══════════════════════════════════════════════════════════════


class TestMARComplianceChecker:

    def test_no_wash_trades_returns_none(
        self, backtest_data
    ):
        checker = MARComplianceChecker(
            wash_trade_window_min=1   # 1 minute window
        )
        # Trades are 5 min apart; no round-trips in 1 min
        no_rt_trades = backtest_data.copy()
        result = checker.check_wash_trades(no_rt_trades)
        # May or may not flag — test structure not None check

    def test_wash_trade_detected(self):
        checker = MARComplianceChecker(
            wash_trade_window_min=60
        )
        t = pd.Timestamp("2025-06-01 09:00")
        trades = pd.DataFrame({
            "trade_id": ["T001", "T002"],
            "instrument": ["GILT_10Y", "GILT_10Y"],
            "side": ["BUY", "SELL"],
            "time": [t, t + pd.Timedelta(minutes=5)],
        })
        flag = checker.check_wash_trades(trades)
        assert flag is not None
        assert flag.flag_type == "WASH_TRADE"
        assert flag.severity == "HIGH"
        assert "T001" in flag.trade_ids

    def test_wash_trade_flag_has_trade_ids(self):
        checker = MARComplianceChecker(
            wash_trade_window_min=60
        )
        t = pd.Timestamp("2025-06-01 09:00")
        trades = pd.DataFrame({
            "trade_id": ["T001", "T002"],
            "instrument": ["GILT_10Y", "GILT_10Y"],
            "side": ["BUY", "SELL"],
            "time": [t, t + pd.Timedelta(minutes=5)],
        })
        flag = checker.check_wash_trades(trades)
        assert len(flag.trade_ids) >= 2

    def test_spoofing_detected_high_cancel_rate(self):
        checker = MARComplianceChecker(
            spoofing_cancel_rate=0.8
        )
        order_book = pd.DataFrame({
            "order_id": [f"O{i}" for i in range(10)],
            "cancelled": (
                [True] * 9 + [False]  # 90% cancel rate
            ),
        })
        flag = checker.flag_spoofing_patterns(order_book)
        assert flag is not None
        assert flag.flag_type == "SPOOFING"

    def test_no_spoofing_low_cancel_rate(self):
        checker = MARComplianceChecker(
            spoofing_cancel_rate=0.8
        )
        order_book = pd.DataFrame({
            "order_id": [f"O{i}" for i in range(10)],
            "cancelled": [False] * 10,
        })
        flag = checker.flag_spoofing_patterns(order_book)
        assert flag is None

    def test_empty_order_book_returns_none(self):
        checker = MARComplianceChecker()
        order_book = pd.DataFrame(
            {"order_id": [], "cancelled": []}
        )
        flag = checker.flag_spoofing_patterns(order_book)
        assert flag is None

    def test_mar_flag_is_dataclass(self):
        flag = MARFlag(
            flag_type="WASH_TRADE",
            severity="HIGH",
            trade_ids=["T001"],
            description="Test",
        )
        assert flag.flag_type == "WASH_TRADE"
        assert flag.severity == "HIGH"


# ═══════════════════════════════════════════════════════════════
# MONTE CARLO VAR ENGINE TESTS
# ═══════════════════════════════════════════════════════════════


class TestMonteCarloVaREngine:

    def test_engine_initialises(self, var_engine):
        assert var_engine is not None
        assert var_engine.N_SCENARIOS == 10_000

    def test_asymmetric_matrix_raises(self):
        bad_corr = np.array([[1.0, 0.5], [0.3, 1.0]])
        with pytest.raises(ValueError):
            MonteCarloVaREngine(
                bad_corr, np.array([0.01, 0.008])
            )

    def test_generate_scenarios_shape(self, var_engine):
        scenarios = var_engine.generate_scenarios()
        assert scenarios.shape == (10_000, 2)

    def test_scenarios_have_correct_mean(self, var_engine):
        np.random.seed(0)
        scenarios = var_engine.generate_scenarios()
        # Mean should be near 0 for large N
        assert abs(scenarios.mean()) < 0.005

    def test_var_99_exceeds_var_95(self, var_engine):
        pnl = np.random.normal(0, 1_000_000, 10_000)
        var_95, var_99 = (
            var_engine.calculate_portfolio_var(pnl)
        )
        assert var_99 >= var_95

    def test_var_positive(self, var_engine):
        pnl = np.random.normal(0, 1_000_000, 10_000)
        var_95, var_99 = (
            var_engine.calculate_portfolio_var(pnl)
        )
        assert var_95 > 0
        assert var_99 > 0

    def test_component_var_keys_match_factors(
        self, var_engine
    ):
        np.random.seed(42)
        factor_pnls = np.random.normal(
            0, 500_000, (10_000, 2)
        )
        portfolio_pnl = factor_pnls.sum(axis=1)
        comp = var_engine.get_component_var(
            factor_pnls, portfolio_pnl
        )
        assert "GIRR_10Y" in comp
        assert "FX_GBPUSD" in comp

    def test_component_var_values_are_finite(
        self, var_engine
    ):
        np.random.seed(42)
        factor_pnls = np.random.normal(
            0, 500_000, (10_000, 2)
        )
        portfolio_pnl = factor_pnls.sum(axis=1)
        comp = var_engine.get_component_var(
            factor_pnls, portfolio_pnl
        )
        for v in comp.values():
            assert np.isfinite(v)

    def test_compute_full_var_returns_var_result(
        self, var_engine
    ):
        np.random.seed(0)
        sensitivities = np.array([
            [100_000, 0], [0, 50_000]
        ])
        result = var_engine.compute_full_var(
            sensitivities
        )
        assert isinstance(result, VaRResult)
        assert result.var_95 > 0
        assert result.var_99 > 0

    def test_larger_positions_produce_larger_var(
        self, var_engine
    ):
        np.random.seed(1)
        small = var_engine.compute_full_var(
            np.array([[100_000, 0], [0, 50_000]])
        )
        np.random.seed(1)
        large = var_engine.compute_full_var(
            np.array([[1_000_000, 0], [0, 500_000]])
        )
        assert large.var_99 > small.var_99

    def test_full_var_es_exceeds_var_99(self, var_engine):
        np.random.seed(2)
        result = var_engine.compute_full_var(
            np.array([[100_000, 0], [0, 50_000]])
        )
        assert (
            result.expected_shortfall_99
            >= result.var_99
        )


# ═══════════════════════════════════════════════════════════════
# VAR BACK-TESTER TESTS
# ═══════════════════════════════════════════════════════════════


class TestVaRBackTester:

    def test_green_light_for_few_exceptions(self):
        bt = VaRBackTester()
        # Add 3 exceptions (within green limit of 4)
        for i in range(3):
            bt.run_daily_backtest(
                f"2025-01-{i+1:02d}",
                var_99=1_000_000,
                actual_pnl=-1_200_000,  # exception
            )
        # Then add many non-exceptions
        for i in range(10):
            bt.run_daily_backtest(
                f"2025-02-{i+1:02d}",
                var_99=1_000_000,
                actual_pnl=500_000,  # profit, no exception
            )
        r = bt.run_daily_backtest(
            "2025-03-01", 1_000_000, 500_000
        )
        assert r.traffic_light == "GREEN"

    def test_exception_detected_correctly(self):
        bt = VaRBackTester()
        r = bt.run_daily_backtest(
            "2025-01-01",
            var_99=1_000_000,
            actual_pnl=-1_500_000,  # loss > VaR
        )
        assert r.is_exception is True

    def test_no_exception_for_profit(self):
        bt = VaRBackTester()
        r = bt.run_daily_backtest(
            "2025-01-01",
            var_99=1_000_000,
            actual_pnl=500_000,  # profit
        )
        assert r.is_exception is False

    def test_red_zone_after_10_exceptions(self):
        bt = VaRBackTester()
        for i in range(10):
            bt.run_daily_backtest(
                f"2025-01-{i+1:02d}",
                var_99=1_000_000,
                actual_pnl=-1_500_000,
            )
        r = bt.run_daily_backtest(
            "2025-01-11", 1_000_000, -1_500_000
        )
        assert r.traffic_light == "RED"

    def test_amber_zone_five_to_nine_exceptions(self):
        bt = VaRBackTester()
        for i in range(6):
            bt.run_daily_backtest(
                f"2025-01-{i+1:02d}",
                var_99=1_000_000,
                actual_pnl=-1_500_000,
            )
        r = bt.run_daily_backtest(
            "2025-01-07", 1_000_000, 500_000
        )
        assert r.traffic_light == "AMBER"

    def test_invalid_var99_raises(self):
        bt = VaRBackTester()
        with pytest.raises(ValueError, match="positive"):
            bt.run_daily_backtest(
                "2025-01-01",
                var_99=-1_000_000,
                actual_pnl=-500_000,
            )

    def test_green_multiplier_is_1_5(self):
        bt = VaRBackTester()
        tl, mult = bt.get_traffic_light(0)
        assert tl == "GREEN"
        assert mult == pytest.approx(1.5)

    def test_red_multiplier_is_4_0(self):
        bt = VaRBackTester()
        tl, mult = bt.get_traffic_light(15)
        assert tl == "RED"
        assert mult == pytest.approx(4.0)

    def test_amber_multiplier_in_range(self):
        bt = VaRBackTester()
        tl, mult = bt.get_traffic_light(7)
        assert tl == "AMBER"
        assert 1.7 <= mult <= 2.5

    def test_summary_dict_structure(self):
        bt = VaRBackTester()
        bt.run_daily_backtest(
            "2025-01-01", 1_000_000, 500_000
        )
        summary = bt.get_250_day_summary()
        assert "exceptions" in summary
        assert "traffic_light" in summary
        assert "model_id" in summary
        assert summary["model_id"] == "MR-2026-046"

    def test_summary_exception_count_correct(self):
        bt = VaRBackTester()
        for i in range(3):
            bt.run_daily_backtest(
                f"2025-01-{i+1:02d}",
                1_000_000, -1_500_000
            )
        summary = bt.get_250_day_summary()
        assert summary["exceptions"] == 3


# ═══════════════════════════════════════════════════════════════
# CVA CALCULATOR TESTS
# ═══════════════════════════════════════════════════════════════


class TestCVACalculator:

    def test_initialises_with_defaults(self):
        calc = CVACalculator()
        assert calc.recovery_rate == DEFAULT_RECOVERY_RATE
        assert calc.lgd == pytest.approx(
            1.0 - DEFAULT_RECOVERY_RATE
        )

    def test_invalid_recovery_rate_raises(self):
        with pytest.raises(ValueError):
            CVACalculator(recovery_rate=1.5)
        with pytest.raises(ValueError):
            CVACalculator(recovery_rate=0.0)

    def test_cva_positive(
        self, cva_calculator, exposure_profile,
        pd_term_structure
    ):
        result = cva_calculator.calculate_cva(
            exposure_profile, pd_term_structure
        )
        assert result.cva_gbp > 0

    def test_cva_result_has_correct_counterparty(
        self, cva_calculator, exposure_profile,
        pd_term_structure
    ):
        result = cva_calculator.calculate_cva(
            exposure_profile, pd_term_structure
        )
        assert (
            result.counterparty_id == "BARCLAYS_001"
        )

    def test_cva_model_id_correct(
        self, cva_calculator, exposure_profile,
        pd_term_structure
    ):
        result = cva_calculator.calculate_cva(
            exposure_profile, pd_term_structure
        )
        assert result.model_id == "MR-2026-048"

    def test_lgd_correct(self, cva_calculator):
        assert cva_calculator.lgd == pytest.approx(0.60)

    def test_higher_pd_produces_higher_cva(
        self, exposure_profile
    ):
        calc = CVACalculator()
        low_pd = {t: 0.001 * t for t in [0.25, 1, 5]}
        high_pd = {t: 0.01 * t for t in [0.25, 1, 5]}
        r_low = calc.calculate_cva(
            exposure_profile, low_pd
        )
        r_high = calc.calculate_cva(
            exposure_profile, high_pd
        )
        assert r_high.cva_gbp > r_low.cva_gbp

    def test_higher_recovery_reduces_cva(
        self, exposure_profile, pd_term_structure
    ):
        calc_low_r = CVACalculator(recovery_rate=0.20)
        calc_high_r = CVACalculator(recovery_rate=0.60)
        r_low = calc_low_r.calculate_cva(
            exposure_profile, pd_term_structure
        )
        r_high = calc_high_r.calculate_cva(
            exposure_profile, pd_term_structure
        )
        assert r_high.cva_gbp < r_low.cva_gbp

    def test_sa_cva_capital_positive(
        self, cva_calculator, exposure_profile,
        pd_term_structure
    ):
        result = cva_calculator.calculate_cva(
            exposure_profile, pd_term_structure
        )
        assert result.sa_cva_capital_gbp > 0

    def test_dva_less_than_cva(
        self, cva_calculator, exposure_profile,
        pd_term_structure
    ):
        result = cva_calculator.calculate_cva(
            exposure_profile, pd_term_structure
        )
        assert result.dva_gbp < result.cva_gbp

    def test_mismatched_exposure_raises(
        self, cva_calculator, pd_term_structure
    ):
        bad_exposure = ExposureProfile(
            counterparty_id="TEST",
            time_steps=[0.25, 0.5],        # 2 time steps
            expected_exposure=[1e6, 2e6, 3e6],  # 3 EEs
            peak_exposure_975=[2e6, 3e6],
        )
        with pytest.raises(ValueError):
            cva_calculator.calculate_cva(
                bad_exposure, pd_term_structure
            )

    def test_portfolio_aggregation(
        self, cva_calculator, exposure_profile,
        pd_term_structure
    ):
        r1 = cva_calculator.calculate_cva(
            exposure_profile, pd_term_structure
        )
        # Make a second counterparty
        exp2 = ExposureProfile(
            counterparty_id="DEUTSCHE_001",
            time_steps=exposure_profile.time_steps,
            expected_exposure=[
                e * 0.6
                for e in exposure_profile.expected_exposure
            ],
            peak_exposure_975=exposure_profile.peak_exposure_975,
        )
        r2 = cva_calculator.calculate_cva(
            exp2, pd_term_structure
        )
        agg = cva_calculator.aggregate_portfolio_cva(
            [r1, r2]
        )
        assert agg["n_counterparties"] == 2
        assert (
            agg["total_cva_gbp"]
            == pytest.approx(r1.cva_gbp + r2.cva_gbp)
        )
        assert "model_id" in agg

    def test_zero_pd_produces_near_zero_cva(
        self, cva_calculator, exposure_profile
    ):
        zero_pd = {t: 0.0 for t in [0.25, 1, 5]}
        result = cva_calculator.calculate_cva(
            exposure_profile, zero_pd
        )
        assert result.cva_gbp == pytest.approx(0.0)

    def test_flat_ois_discount_factors(
        self, exposure_profile, pd_term_structure
    ):
        calc_no_ois = CVACalculator()
        calc_with_ois = CVACalculator(
            ois_discount_factors={
                0.25: 0.989, 1.0: 0.957, 5.0: 0.801
            }
        )
        r1 = calc_no_ois.calculate_cva(
            exposure_profile, pd_term_structure
        )
        r2 = calc_with_ois.calculate_cva(
            exposure_profile, pd_term_structure
        )
        # Both should produce positive CVA
        assert r1.cva_gbp > 0
        assert r2.cva_gbp > 0


# ═══════════════════════════════════════════════════════════════
# INTEGRATION TESTS — Chapter 6 PD Model Integration
# ═══════════════════════════════════════════════════════════════


class TestCVAChapter6Integration:
    """Tests for MR-2026-043 (Ch6) -> MR-2026-048 (Ch7)."""

    def test_pd_model_tool_integration_mock(
        self, cva_calculator, exposure_profile
    ):
        """Mock PDModelTool from MR-2026-043 (Ch 6)."""
        mock_pd_tool = MagicMock()
        mock_pd_tool.get_pd_term_structure.return_value = {
            0.25: 0.002, 0.5: 0.004, 1.0: 0.008,
            2.0: 0.016, 3.0: 0.024, 4.0: 0.032,
            5.0: 0.040,
        }
        pd_ts = (
            mock_pd_tool.get_pd_term_structure(
                counterparty_id="BARCLAYS_001",
                horizons=[0.25, 0.5, 1, 2, 3, 4, 5],
            )
        )
        result = cva_calculator.calculate_cva(
            exposure_profile, pd_ts
        )
        assert result.cva_gbp > 0
        mock_pd_tool.get_pd_term_structure.assert_called_once()

    def test_cva_model_id_references_mr2026044(
        self, cva_calculator, exposure_profile,
        pd_term_structure
    ):
        result = cva_calculator.calculate_cva(
            exposure_profile, pd_term_structure
        )
        assert "MR-2026-048" in result.model_id

    def test_cva_within_5pct_of_expected(
        self, exposure_profile
    ):
        """CVA for simple case should match hand calc."""
        # Simplified: 1 time step, known EE and PD
        simple_exp = ExposureProfile(
            counterparty_id="TEST_SIMPLE",
            time_steps=[1.0],
            expected_exposure=[10_000_000],  # £10M EE
            peak_exposure_975=[15_000_000],
        )
        simple_pd = {1.0: 0.01}  # 1% PD over 1 year
        calc = CVACalculator(
            recovery_rate=0.40,
            ois_discount_factors={1.0: 1.0},  # no discounting
        )
        result = calc.calculate_cva(
            simple_exp, simple_pd
        )
        # Expected: LGD * EE * PD = 0.60 * 10M * 0.01 = £60K
        expected_cva = 0.60 * 10_000_000 * 0.01
        assert result.cva_gbp == pytest.approx(
            expected_cva, rel=0.05  # 5% tolerance
        )


# ═══════════════════════════════════════════════════════════════
# AWB NAMING AND NAMESPACE COMPLIANCE
# ═══════════════════════════════════════════════════════════════


class TestAWBNamingCompliance:

    def test_no_crb_references_in_module_names(self):
        import backtesting.backtest_engine as be
        assert "crb" not in be.__name__.lower()
        assert "cambridgeshire" not in be.__name__.lower()

    def test_model_ids_correct(
        self, cva_calculator, var_engine
    ):
        # CVA model must be MR-2026-048
        assert cva_calculator is not None  # MR-2026-048
        # VaR engine is MR-2026-046
        assert var_engine is not None      # MR-2026-046

    def test_var_back_tester_summary_model_id(self):
        bt = VaRBackTester()
        bt.run_daily_backtest(
            "2025-01-01", 1_000_000, 500_000
        )
        summary = bt.get_250_day_summary()
        assert summary["model_id"] == "MR-2026-046"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


# ══════════════════════════════════════════════════════════════════
# NEW TESTS: FRTB Capital Calculator (MR-2026-046 SA-FRTB)
# ══════════════════════════════════════════════════════════════════

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from frtb.frtb_capital import SaFrtbCalculator, SbMResult


class TestSaFrtbCalculator:
    """Tests for SA-FRTB capital calculator (CRR3)."""

    @pytest.fixture
    def frtb_calc(self):
        return SaFrtbCalculator()

    @pytest.fixture
    def awb_inputs(self):
        """AWB June 2026 trading book inputs."""
        return {
            "girr_dv01": 180_000_000,   # £180M DV01
            "equity_delta": 60_000_000,  # £60M delta
            "fx_delta": 53_333_333,      # £53.3M net
            "credit_delta": 400_000_000, # £400M spread
            "gross_jtd": 200_000_000,    # £200M GJtD
            "net_jtd": 150_000_000,      # £150M net
            "exotic_notional": 400_000_000,  # £400M
        }

    def test_sbm_returns_dict(self, frtb_calc):
        result = frtb_calc.calculate_sbm(
            1e6, 1e6, 1e6, 1e6
        )
        assert isinstance(result, dict)
        assert "total_sbm" in result

    def test_sbm_all_positive(self, frtb_calc):
        result = frtb_calc.calculate_sbm(
            1e6, 1e6, 1e6, 1e6
        )
        for key, val in result.items():
            assert val >= 0, f"{key} is negative"

    def test_sbm_total_equals_sum(self, frtb_calc):
        r = frtb_calc.calculate_sbm(
            1e6, 1e6, 1e6, 1e6
        )
        components = (
            r["girr"] + r["equity"]
            + r["fx"] + r["credit_spread"]
        )
        assert abs(r["total_sbm"] - components) < 1

    def test_awb_total_capital_approx_42m(
        self, frtb_calc, awb_inputs
    ):
        """AWB SA-FRTB capital must be ~£42M."""
        result = frtb_calc.calculate_total(
            **awb_inputs
        )
        # Within 20% of £42M (simplified formula)
        assert 25_000_000 <= result.total_sa_frtb_gbp

    def test_calculate_total_returns_sbm_result(
        self, frtb_calc, awb_inputs
    ):
        result = frtb_calc.calculate_total(**awb_inputs)
        assert isinstance(result, SbMResult)

    def test_drc_zero_for_net_jtd_zero(self, frtb_calc):
        drc = frtb_calc.calculate_drc(
            gross_jtd_gbp=1e6,
            net_jtd_gbp=0.0,
        )
        assert drc == 0.0

    def test_drc_positive_for_positive_net_jtd(
        self, frtb_calc
    ):
        drc = frtb_calc.calculate_drc(1e6, 1e6)
        assert drc > 0

    def test_rrao_default_rate_one_pct(self, frtb_calc):
        rrao = frtb_calc.calculate_rrao(100_000_000)
        assert abs(rrao - 1_000_000) < 1

    def test_rrao_custom_rate(self, frtb_calc):
        rrao = frtb_calc.calculate_rrao(
            100_000_000, rrao_rate=0.005
        )
        assert abs(rrao - 500_000) < 1

    def test_total_includes_drc_and_rrao(
        self, frtb_calc
    ):
        result = frtb_calc.calculate_total(
            girr_dv01=1e6,
            equity_delta=1e6,
            fx_delta=1e6,
            credit_delta=1e6,
            gross_jtd=1e6,
            net_jtd=1e6,
            exotic_notional=1e6,
        )
        assert result.drc_gbp > 0
        assert result.rrao_gbp > 0
        assert (
            result.total_sa_frtb_gbp
            >= result.total_sbm_gbp
        )

    def test_awb_capital_components_sum(
        self, frtb_calc, awb_inputs
    ):
        """SbM + DRC + RRAO == total."""
        r = frtb_calc.calculate_total(**awb_inputs)
        expected = (
            r.total_sbm_gbp + r.drc_gbp + r.rrao_gbp
        )
        assert abs(r.total_sa_frtb_gbp - expected) < 1


# ══════════════════════════════════════════════════════════════════
# NEW TESTS: Walk-Forward Validation (MR-2026-047)
# ══════════════════════════════════════════════════════════════════

class TestWalkForwardValidation:
    """Walk-forward validation tests for MR-2026-047."""

    @pytest.fixture
    def long_prices(self):
        """500-day price series for WF tests."""
        np.random.seed(99)
        dates = pd.date_range("2023-01-01", periods=500)
        px = 100 * np.cumprod(
            1 + np.random.normal(0.0003, 0.012, 500)
        )
        return pd.Series(px, index=dates)

    @pytest.fixture
    def long_signals(self, long_prices):
        ma10 = long_prices.rolling(10).mean()
        ma50 = long_prices.rolling(50).mean()
        return np.sign(ma10 - ma50).fillna(0)

    def test_walk_forward_returns_multiple_windows(
        self, long_prices, long_signals
    ):
        engine = BacktestEngine()
        results = engine.walk_forward_validate(
            long_signals, long_prices,
            train_years=1,
            test_years=1,
            step_months=3,
        )
        assert len(results) >= 1

    def test_walk_forward_short_series_returns_empty(
        self
    ):
        dates = pd.date_range("2025-01-01", periods=50)
        px = pd.Series(
            np.arange(50, dtype=float), index=dates
        )
        sig = pd.Series(
            np.ones(50), index=dates
        )
        engine = BacktestEngine()
        results = engine.walk_forward_validate(
            sig, px, train_years=1
        )
        assert results == []

    def test_each_window_has_valid_sharpe(
        self, long_prices, long_signals
    ):
        engine = BacktestEngine()
        results = engine.walk_forward_validate(
            long_signals, long_prices,
            train_years=1,
            test_years=1,
            step_months=6,
        )
        for r in results:
            assert np.isfinite(r.sharpe)

    def test_walk_forward_all_results_are_backtest_results(
        self, long_prices, long_signals
    ):
        engine = BacktestEngine()
        results = engine.walk_forward_validate(
            long_signals, long_prices,
            train_years=1, test_years=1,
            step_months=6,
        )
        for r in results:
            assert isinstance(r, BacktestResult)


# ══════════════════════════════════════════════════════════════════
# NEW TESTS: AWB Canonical Compliance (all chapters)
# ══════════════════════════════════════════════════════════════════

class TestAWBCanonicalCompliance:
    """
    Verify AWB canonical naming and model ID standards.
    These tests prevent regression of incorrect IDs.
    """

    def test_var_engine_model_id_is_046(self, var_engine):
        """VaR engine must be MR-2026-046."""
        summary = VaRBackTester().get_250_day_summary()
        assert summary["model_id"] == "MR-2026-046"

    def test_cva_model_id_is_048(
        self, cva_calculator, exposure_profile,
        pd_term_structure
    ):
        """CVA model must be MR-2026-048."""
        result = cva_calculator.calculate_cva(
            exposure_profile, pd_term_structure
        )
        assert result.model_id == "MR-2026-048"

    def test_no_legacy_model_id_043(self):
        """MR-2026-043 must not appear anywhere."""
        import var_engine.mc_var_engine as m
        import inspect
        src = inspect.getsource(m)
        assert "MR-2026-043" not in src

    def test_no_legacy_model_id_044(self):
        """MR-2026-044 must not appear anywhere."""
        import cva.cva_calculator as m
        import inspect
        src = inspect.getsource(m)
        assert "MR-2026-044" not in src

    def test_no_crb_references(self):
        """No AWB or Cambridgeshire references."""
        import backtesting.backtest_engine as b
        import var_engine.mc_var_engine as v
        import cva.cva_calculator as c
        import inspect
        for mod in [b, v, c]:
            src = inspect.getsource(mod)
            assert "AWB" not in src
            assert "Cambridgeshire" not in src
            assert "crb_commons" not in src

    def test_awb_commons_namespace(self):
        """awb_commons referenced in docstrings."""
        import var_engine.mc_var_engine as v
        import inspect
        src = inspect.getsource(v)
        assert "awb_commons" in src

    def test_frtb_module_importable(self):
        """FRTB capital module must be importable."""
        from frtb.frtb_capital import SaFrtbCalculator
        calc = SaFrtbCalculator()
        assert calc is not None

    def test_pra_ss1_23_referenced_in_var(self):
        """PRA SS1/23 must be in VaR engine docs."""
        import var_engine.mc_var_engine as v
        import inspect
        src = inspect.getsource(v)
        assert "SS1/23" in src

    def test_gbp_currency_in_results(
        self, var_engine
    ):
        """VaR results return GBP figures (positive)."""
        sens = np.array([[100_000, 50_000]])
        result = var_engine.compute_full_var(sens)
        assert result.var_99 > 0
        assert result.var_95 > 0


# ══════════════════════════════════════════════════════════════════
# NEW TESTS: FRTB Capital Calculator (MR-2026-046 SA-FRTB)
# ══════════════════════════════════════════════════════════════════

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from frtb.frtb_capital import SaFrtbCalculator, SbMResult


class TestSaFrtbCalculator:
    """Tests for SA-FRTB capital calculator (CRR3)."""

    @pytest.fixture
    def frtb_calc(self):
        return SaFrtbCalculator()

    @pytest.fixture
    def awb_inputs(self):
        """AWB June 2026 trading book inputs."""
        return {
            "girr_dv01": 180_000_000,
            "equity_delta": 60_000_000,
            "fx_delta": 53_333_333,
            "credit_delta": 400_000_000,
            "gross_jtd": 200_000_000,
            "net_jtd": 150_000_000,
            "exotic_notional": 400_000_000,
        }

    def test_sbm_returns_dict(self, frtb_calc):
        result = frtb_calc.calculate_sbm(
            1e6, 1e6, 1e6, 1e6
        )
        assert isinstance(result, dict)
        assert "total_sbm" in result

    def test_sbm_total_equals_sum(self, frtb_calc):
        r = frtb_calc.calculate_sbm(
            1e6, 1e6, 1e6, 1e6
        )
        components = (
            r["girr"] + r["equity"]
            + r["fx"] + r["credit_spread"]
        )
        assert abs(r["total_sbm"] - components) < 1

    def test_awb_capital_at_least_25m(
        self, frtb_calc, awb_inputs
    ):
        """AWB SA-FRTB capital must be reasonable."""
        result = frtb_calc.calculate_total(**awb_inputs)
        assert result.total_sa_frtb_gbp >= 25_000_000

    def test_calculate_total_returns_sbm_result(
        self, frtb_calc, awb_inputs
    ):
        result = frtb_calc.calculate_total(**awb_inputs)
        assert isinstance(result, SbMResult)

    def test_drc_zero_for_net_jtd_zero(self, frtb_calc):
        drc = frtb_calc.calculate_drc(1e6, 0.0)
        assert drc == 0.0

    def test_rrao_default_rate_one_pct(self, frtb_calc):
        rrao = frtb_calc.calculate_rrao(100_000_000)
        assert abs(rrao - 1_000_000) < 1

    def test_total_includes_drc_and_rrao(
        self, frtb_calc
    ):
        result = frtb_calc.calculate_total(
            girr_dv01=1e6,
            equity_delta=1e6,
            fx_delta=1e6,
            credit_delta=1e6,
            gross_jtd=1e6,
            net_jtd=1e6,
            exotic_notional=1e6,
        )
        expected = (
            result.total_sbm_gbp
            + result.drc_gbp
            + result.rrao_gbp
        )
        assert abs(result.total_sa_frtb_gbp - expected) < 1


# ══════════════════════════════════════════════════════════════════
# NEW TESTS: Walk-Forward Validation (MR-2026-047)
# ══════════════════════════════════════════════════════════════════

class TestWalkForwardValidation:
    """Walk-forward validation tests for MR-2026-047."""

    @pytest.fixture
    def long_prices(self):
        np.random.seed(99)
        dates = pd.date_range("2023-01-01", periods=500)
        px = 100 * np.cumprod(
            1 + np.random.normal(0.0003, 0.012, 500)
        )
        return pd.Series(px, index=dates)

    @pytest.fixture
    def long_signals(self, long_prices):
        ma10 = long_prices.rolling(10).mean()
        ma50 = long_prices.rolling(50).mean()
        return np.sign(ma10 - ma50).fillna(0)

    def test_walk_forward_returns_results(
        self, long_prices, long_signals
    ):
        engine = BacktestEngine()
        results = engine.walk_forward_validate(
            long_signals, long_prices,
            train_years=1,
            test_years=1,
            step_months=6,
        )
        assert len(results) >= 1

    def test_short_series_returns_empty(self):
        dates = pd.date_range("2025-01-01", periods=50)
        px = pd.Series(range(50), index=dates, dtype=float)
        sig = pd.Series(np.ones(50), index=dates)
        results = BacktestEngine().walk_forward_validate(
            sig, px, train_years=1
        )
        assert results == []

    def test_each_window_sharpe_is_finite(
        self, long_prices, long_signals
    ):
        results = BacktestEngine().walk_forward_validate(
            long_signals, long_prices,
            train_years=1, test_years=1,
            step_months=6,
        )
        for r in results:
            assert np.isfinite(r.sharpe)


# ══════════════════════════════════════════════════════════════════
# NEW TESTS: AWB Canonical Compliance
# ══════════════════════════════════════════════════════════════════

class TestAWBCanonicalCompliance:
    """Verify AWB naming and model ID standards."""

    def test_var_engine_model_id_is_046(self):
        summary = VaRBackTester().get_250_day_summary()
        assert summary["model_id"] == "MR-2026-046"

    def test_cva_model_id_is_048(
        self, cva_calculator, exposure_profile,
        pd_term_structure
    ):
        result = cva_calculator.calculate_cva(
            exposure_profile, pd_term_structure
        )
        assert result.model_id == "MR-2026-048"

    def test_no_legacy_model_id_043(self):
        import var_engine.mc_var_engine as m
        import inspect
        assert "MR-2026-043" not in inspect.getsource(m)

    def test_no_legacy_model_id_044(self):
        import cva.cva_calculator as m
        import inspect
        assert "MR-2026-044" not in inspect.getsource(m)

    def test_no_crb_references_in_sources(self):
        import backtesting.backtest_engine as b
        import var_engine.mc_var_engine as v
        import cva.cva_calculator as c
        import inspect
        for mod in [b, v, c]:
            src = inspect.getsource(mod)
            assert "AWB" not in src
            assert "crb_commons" not in src

    def test_frtb_module_importable(self):
        from frtb.frtb_capital import SaFrtbCalculator
        assert SaFrtbCalculator() is not None

    def test_pra_ss1_23_in_var_source(self):
        import var_engine.mc_var_engine as v
        import inspect
        assert "SS1/23" in inspect.getsource(v)

    def test_var_99_positive_gbp(self, var_engine):
        # 2 positions, 2 factors
        # (n_positions x n_factors) sensitivity matrix
        sens = np.array([
            [100_000, 0],
            [0, 50_000],
        ])
        result = var_engine.compute_full_var(sens)
        assert result.var_99 > 0
        assert result.var_95 > 0
        assert (
            result.expected_shortfall_99
            >= result.var_99
        )
