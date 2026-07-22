"""Leverage Ratio Calculator — CRR3 Article 429.

Model ID: MR-2026-071 | Risk: HIGH | COREP return: C 47.00
Quarterly deadline: 12 business days after period end.
Minimum: 3.0% general | 3.5% G-SIBs (AWB = 3.0% applies).
"""
from __future__ import annotations
from datetime import date
from decimal import Decimal
from typing import Dict
import logging

from awb_commons.models import LeverageRatioResult

log = logging.getLogger(__name__)

# CRR3 Arts 429e-429g credit conversion factors
CCF_MAP: Dict[str, Decimal] = {
    "unconditionally_cancellable": Decimal("0.10"),
    "up_to_1yr": Decimal("0.20"),
    "over_1yr": Decimal("0.50"),
    "guarantees": Decimal("1.00"),
    "letters_of_credit": Decimal("1.00"),
}


class LeverageRatioCalculator:
    """Calculate CRR3 Art. 429 leverage ratio — all 4 components.

    Formula: Tier 1 Capital / Total Leverage Exposure Measure >= 3%

    Component breakdown:
        (a) On-balance-sheet (Art. 429b)
        (b) SA-CCR derivatives (Art. 429c)
        (c) Securities Financing Transactions (Art. 429d)
        (d) Off-balance-sheet commitments (Arts 429e-429g)

    AWB Q4 2025 illustrative result: 8.9% (well above 3% min).

    Args:
        quarter_end: Reporting quarter end date.
        tier1_capital_gbp: Tier 1 capital after all deductions.

    Example:
        >>> calc = LeverageRatioCalculator(
        ...     date(2025, 12, 31),
        ...     Decimal("4_100_000_000"),
        ... )
        >>> result = calc.calculate_all(
        ...     Decimal("38_800_000_000"),
        ...     Decimal("2_100_000_000"),
        ...     Decimal("800_000_000"),
        ...     Decimal("4_200_000_000"),
        ... )
        >>> print(f"{result.leverage_ratio_pct:.1f}%")
        8.9%
    """

    def __init__(
        self,
        quarter_end: date,
        tier1_capital_gbp: Decimal,
    ) -> None:
        self._quarter_end = quarter_end
        self._tier1 = tier1_capital_gbp

    def calculate_on_balance_sheet(
        self,
        total_assets_gbp: Decimal,
        provisions_on_defaults_gbp: Decimal,
        t1_deductions_gbp: Decimal,
        derivative_accounting_value_gbp: Decimal,
    ) -> Decimal:
        """CRR3 Art. 429b on-balance-sheet exposure.

        Total IFRS assets minus: provisions on defaulted
        exposures, regulatory Tier 1 deductions, and accounting
        value of derivatives (replaced by SA-CCR component).

        Args:
            total_assets_gbp: IFRS balance sheet total.
            provisions_on_defaults_gbp: Art. 429b(1)(a).
            t1_deductions_gbp: Art. 429b(1)(b).
            derivative_accounting_value_gbp: Replaced by SA-CCR.

        Returns:
            Art. 429b on-balance-sheet exposure measure.
        """
        exposure = (
            total_assets_gbp
            - provisions_on_defaults_gbp
            - t1_deductions_gbp
            - derivative_accounting_value_gbp
        )
        log.info(
            "On-BS (Art.429b): £%.2fB from £%.2fB assets",
            float(exposure) / 1e9,
            float(total_assets_gbp) / 1e9,
        )
        return exposure

    def calculate_sa_ccr(
        self,
        replacement_cost_gbp: Decimal,
        potential_future_exposure_gbp: Decimal,
    ) -> Decimal:
        """CRR3 Art. 429c SA-CCR derivative exposure.

        SA-CCR exposure = RC + PFE aggregate add-on.
        Higher than accounting fair value — captures future risk.
        AWB Q4 2025: 340 OTC contracts → £2.1B SA-CCR exposure.

        Args:
            replacement_cost_gbp: Current net replacement cost.
            potential_future_exposure_gbp: PFE per Annex II.

        Returns:
            SA-CCR exposure for leverage denominator.
        """
        sa_ccr = (
            replacement_cost_gbp
            + potential_future_exposure_gbp
        )
        log.info(
            "SA-CCR (Art.429c): £%.3fB = RC + PFE",
            float(sa_ccr) / 1e9,
        )
        return sa_ccr

    def calculate_sft_exposure(
        self,
        gross_cash_receivables_gbp: Decimal,
        gross_securities_provided_gbp: Decimal,
        netting_benefit_gbp: Decimal = Decimal("0"),
    ) -> Decimal:
        """CRR3 Art. 429d SFT exposure measure.

        Covers repos, reverse repos, and securities lending.
        Gross positions minus eligible master agreement netting.
        AWB Q4 2025: £3.2B SFT book → £0.8B leverage exposure.

        Returns:
            SFT leverage exposure measure.
        """
        sft = (
            gross_cash_receivables_gbp
            + gross_securities_provided_gbp
            - netting_benefit_gbp
        )
        log.info(
            "SFT (Art.429d): £%.3fB exposure",
            float(sft) / 1e9,
        )
        return sft

    def calculate_off_balance_sheet(
        self,
        commitments: Dict[str, Decimal],
    ) -> Decimal:
        """CRR3 Arts 429e-429g off-balance-sheet exposure.

        Converts nominal commitments to exposure using CCFs.
        AWB Q4 2025: £12B committed facilities → £4.2B exposure.

        Args:
            commitments: Nominal amounts by CCF category.
                Valid keys: unconditionally_cancellable,
                up_to_1yr, over_1yr, guarantees, letters_of_credit

        Returns:
            Off-balance-sheet leverage exposure.

        Raises:
            ValueError: If unknown CCF category supplied.
        """
        total = Decimal("0")
        for category, nominal in commitments.items():
            if category not in CCF_MAP:
                raise ValueError(
                    f"Unknown CCF category: '{category}'. "
                    f"Valid: {list(CCF_MAP.keys())}"
                )
            exposure = nominal * CCF_MAP[category]
            total += exposure
            log.debug(
                "Off-BS: %s £%.2fB × CCF %s = £%.3fB",
                category,
                float(nominal) / 1e9,
                CCF_MAP[category],
                float(exposure) / 1e9,
            )
        log.info(
            "Off-BS total (Arts 429e-g): £%.3fB",
            float(total) / 1e9,
        )
        return total

    def calculate_all(
        self,
        on_bs_gbp: Decimal,
        sa_ccr_gbp: Decimal,
        sft_gbp: Decimal,
        off_bs_gbp: Decimal,
    ) -> LeverageRatioResult:
        """Assemble 4 components into LeverageRatioResult.

        Args:
            on_bs_gbp: Art. 429b on-balance-sheet exposure.
            sa_ccr_gbp: Art. 429c derivative SA-CCR exposure.
            sft_gbp: Art. 429d SFT exposure.
            off_bs_gbp: Arts 429e-g off-balance-sheet exposure.

        Returns:
            LeverageRatioResult with ratio, total, breach flag.
        """
        result = LeverageRatioResult(
            quarter_end=self._quarter_end,
            tier1_capital_gbp=self._tier1,
            on_balance_sheet_gbp=on_bs_gbp,
            sa_ccr_derivatives_gbp=sa_ccr_gbp,
            sft_exposure_gbp=sft_gbp,
            off_balance_sheet_gbp=off_bs_gbp,
        )
        log.info(
            "Leverage ratio %s: %.2f%% "
            "(T1=£%.2fB / Exp=£%.2fB) "
            "min=%.1f%% breach=%s",
            self._quarter_end,
            result.leverage_ratio_pct,
            float(self._tier1) / 1e9,
            float(result.total_exposure_gbp) / 1e9,
            result.MINIMUM_RATIO_PCT,
            result.breaches_minimum,
        )
        return result
