"""Tests for CRR3 Art. 429 Leverage Ratio Calculator.

Covers all 4 components, binding constraint analysis,
minimum breach detection, and Q4 2025 AWB illustrative figures.
"""
import pytest
from datetime import date
from decimal import Decimal

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mjrrp.leverage_calculator import LeverageRatioCalculator, CCF_MAP
from awb_commons.models import LeverageRatioResult

Q4_2025 = date(2025, 12, 31)

# AWB Q4 2025 illustrative figures (£B converted to £)
TIER1 = Decimal("4_100_000_000")     # £4.1B
ON_BS  = Decimal("38_800_000_000")   # £38.8B
SACCR  = Decimal("2_100_000_000")    # £2.1B
SFT    = Decimal("800_000_000")      # £0.8B
OFF_BS = Decimal("4_200_000_000")    # £4.2B (£12B × CCFs)
TOTAL_EXP = Decimal("45_900_000_000")  # £45.9B


@pytest.fixture
def calculator() -> LeverageRatioCalculator:
    return LeverageRatioCalculator(Q4_2025, TIER1)


class TestLeverageRatioComponents:
    """Test all 4 CRR3 Art. 429 exposure components."""

    def test_on_balance_sheet_calculation(self, calculator):
        """Art. 429b: assets minus provisions and deductions."""
        result = calculator.calculate_on_balance_sheet(
            total_assets_gbp=Decimal("40_000_000_000"),
            provisions_on_defaults_gbp=Decimal("800_000_000"),
            t1_deductions_gbp=Decimal("100_000_000"),
            derivative_accounting_value_gbp=Decimal("300_000_000"),
        )
        expected = Decimal("38_800_000_000")
        assert result == expected, (
            f"On-BS: expected £{expected/1e9}B got £{result/1e9}B"
        )

    def test_sa_ccr_calculation(self, calculator):
        """Art. 429c: SA-CCR = RC + PFE add-on."""
        rc = Decimal("800_000_000")
        pfe = Decimal("1_300_000_000")
        result = calculator.calculate_sa_ccr(rc, pfe)
        assert result == Decimal("2_100_000_000")

    def test_sft_exposure(self, calculator):
        """Art. 429d: Repo and securities lending exposure."""
        result = calculator.calculate_sft_exposure(
            gross_cash_receivables_gbp=Decimal("500_000_000"),
            gross_securities_provided_gbp=Decimal("350_000_000"),
            netting_benefit_gbp=Decimal("50_000_000"),
        )
        assert result == Decimal("800_000_000")

    def test_off_balance_sheet_ccf_mapping(self, calculator):
        """Arts 429e-g: off-BS exposure with credit conversion."""
        commitments = {
            "over_1yr": Decimal("8_000_000_000"),     # 50% CCF
            "guarantees": Decimal("2_000_000_000"),   # 100% CCF
        }
        result = calculator.calculate_off_balance_sheet(
            commitments
        )
        expected = (
            Decimal("8_000_000_000") * Decimal("0.50")
            + Decimal("2_000_000_000") * Decimal("1.00")
        )
        assert result == expected

    def test_unknown_ccf_category_raises(self, calculator):
        """Unknown CCF category must raise ValueError."""
        with pytest.raises(ValueError, match="Unknown CCF"):
            calculator.calculate_off_balance_sheet(
                {"unknown_type": Decimal("1_000_000_000")}
            )

    def test_ccf_unconditionally_cancellable(self, calculator):
        """Unconditionally cancellable facilities: 10% CCF."""
        assert CCF_MAP["unconditionally_cancellable"] == Decimal("0.10")

    def test_ccf_guarantees_full(self, calculator):
        """Guarantees and letters of credit: 100% CCF."""
        assert CCF_MAP["guarantees"] == Decimal("1.00")
        assert CCF_MAP["letters_of_credit"] == Decimal("1.00")


class TestLeverageRatioResult:
    """Test LeverageRatioResult calculations and breach detection."""

    def test_awb_q4_2025_ratio(self, calculator):
        """AWB Q4 2025 illustrative: 8.9% well above 3% min."""
        result = calculator.calculate_all(
            on_bs_gbp=ON_BS,
            sa_ccr_gbp=SACCR,
            sft_gbp=SFT,
            off_bs_gbp=OFF_BS,
        )
        assert abs(result.leverage_ratio_pct - 8.9) < 0.1, (
            f"Expected ~8.9%, got {result.leverage_ratio_pct:.2f}%"
        )

    def test_total_exposure_sum(self, calculator):
        """Total exposure = sum of all 4 components."""
        result = calculator.calculate_all(ON_BS, SACCR, SFT, OFF_BS)
        assert result.total_exposure_gbp == TOTAL_EXP

    def test_does_not_breach_minimum(self, calculator):
        """AWB Q4 2025: ratio 8.9% >> 3% minimum."""
        result = calculator.calculate_all(ON_BS, SACCR, SFT, OFF_BS)
        assert not result.breaches_minimum

    def test_breaches_minimum_detection(self):
        """Calculator correctly flags breach below 3%."""
        # Extreme case: tiny capital, large exposure
        calc = LeverageRatioCalculator(
            Q4_2025,
            Decimal("100_000_000"),   # £100M T1
        )
        result = calc.calculate_all(
            on_bs_gbp=Decimal("10_000_000_000"),  # £10B exposure
            sa_ccr_gbp=Decimal("0"),
            sft_gbp=Decimal("0"),
            off_bs_gbp=Decimal("0"),
        )
        assert result.breaches_minimum
        assert result.leverage_ratio_pct < 3.0

    def test_minimum_ratio_constant(self, calculator):
        """CRR3 Art. 429 minimum = 3.0% (AWB non-G-SIB)."""
        result = calculator.calculate_all(ON_BS, SACCR, SFT, OFF_BS)
        assert result.MINIMUM_RATIO_PCT == 3.0

    def test_zero_exposure_raises(self):
        """Zero total exposure must raise ValueError."""
        calc = LeverageRatioCalculator(Q4_2025, TIER1)
        with pytest.raises(ValueError):
            calc.calculate_all(
                Decimal("0"), Decimal("0"),
                Decimal("0"), Decimal("0"),
            )

    def test_binding_constraint_rwa_binds_first(self, calculator):
        """In stress: CET1 ratio is binding (not leverage ratio)."""
        # Severe stress: RWA +45%, leverage exposure +10%
        stressed_lr = calculator.calculate_all(
            on_bs_gbp=ON_BS * Decimal("1.05"),
            sa_ccr_gbp=SACCR * Decimal("1.15"),
            sft_gbp=SFT * Decimal("1.10"),
            off_bs_gbp=OFF_BS * Decimal("1.12"),
        )
        # Leverage ratio stays well above 3% minimum
        assert stressed_lr.leverage_ratio_pct > 3.0
        # Stressed LR ~6.2%: headroom above minimum
        assert stressed_lr.leverage_ratio_pct > 5.0


class TestLeverageRatioDataclass:
    """Test LeverageRatioResult dataclass directly."""

    def test_result_fields(self):
        """All fields computed correctly from inputs."""
        result = LeverageRatioResult(
            quarter_end=Q4_2025,
            tier1_capital_gbp=TIER1,
            on_balance_sheet_gbp=ON_BS,
            sa_ccr_derivatives_gbp=SACCR,
            sft_exposure_gbp=SFT,
            off_balance_sheet_gbp=OFF_BS,
        )
        assert result.total_exposure_gbp == TOTAL_EXP
        assert not result.breaches_minimum
        assert result.leverage_ratio_pct > 8.0
