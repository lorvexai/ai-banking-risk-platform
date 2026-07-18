"""
exercises/exercise_2.py — Exercise 8.2 starter code.
Chapter 8: Operational Risk Detection and Management.
Avon & Wessex Bank plc (AWB) — AWB-AI-2025 programme.

EXERCISE 8.2: Build the Full SMA Capital Impact Model
Difficulty: ★★★★☆ | Estimated time: 45 minutes

Task:
    Using SMACapitalCalculator, project AWB's ILM trajectory
    from 2025 to 2030, assuming the AI fraud systems reduce
    average annual losses by 35% per year from their 2024
    baseline of £12.8M.

    Produce a year-by-year table showing:
        year | avg_annual_loss | ilm | sma_capital | saving_vs_pre_ai

    Then calculate the cumulative 6-year capital saving and
    compare it to the £330K build cost.

    Regulatory reference: CRR3 Articles 316-323

GitHub: lorvenio/ai-banking-risk-platform/chapter_08/
Solution: chapter_08/solutions/exercise_2_solution.py
"""

from __future__ import annotations

import sys
import os
import logging
from dataclasses import dataclass

# Add parent directory to path so imports work from exercises/
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from op_loss_detection.sma_calculator import (
    SMACapitalCalculator,
    SMAInputs,
    SMAResult,
)

logger = logging.getLogger(__name__)

# ── AWB canonical constants ───────────────────────────────────────

AWB_BIC_GBP = 52_000_000          # £52M Business Indicator Component
AWB_BI_GBP  = 643_000_000         # £643M Business Indicator
AWB_PRE_AI_AVG_LOSS_GBP = 12_800_000   # £12.8M 10-yr avg (2024)
AWB_BUILD_COST_GBP      = 330_000      # £330K total build cost
AWB_COST_OF_CAPITAL_PCT = 0.08         # 8% PRA hurdle rate
AI_LOSS_REDUCTION_RATE  = 0.35         # 35% per year reduction
PROJECTION_YEARS        = 6            # 2025-2030


@dataclass
class SMAProjectionRow:
    """One year of the SMA capital projection."""
    year: int
    avg_annual_loss_gbp: float
    loss_component_gbp: float
    ilm: float
    sma_capital_gbp: float
    saving_vs_pre_ai_gbp: float
    annual_capital_saving_gbp: float  # at 8% cost of capital


# ── TODO: Implement your solution below ───────────────────────────


def project_ilm_trajectory(
    calculator: SMACapitalCalculator,
    base_avg_loss: float = AWB_PRE_AI_AVG_LOSS_GBP,
    reduction_rate: float = AI_LOSS_REDUCTION_RATE,
    years: int = PROJECTION_YEARS,
    start_year: int = 2025,
) -> list[SMAProjectionRow]:
    """
    TODO: Project AWB's ILM and SMA capital over 6 years.

    The 10-year rolling average loss used in the ILM formula
    changes as new (lower) AI-era losses enter the window.

    Simplifying assumption for this exercise:
        avg_annual_loss in year N = base_avg_loss * (1 - rate)^N
        loss_component = 15 * avg_annual_loss  (CRR3 Art. 318)

    The AWB BIC remains constant at £52M (not affected by AI).

    Args:
        calculator:     SMACapitalCalculator instance.
        base_avg_loss:  2024 pre-AI 10-year average loss.
        reduction_rate: Annual loss reduction from AI systems.
        years:          Number of projection years.
        start_year:     First year of projection.

    Returns:
        List of SMAProjectionRow, one per year.
    """
    raise NotImplementedError(
        "Implement project_ilm_trajectory() "
        "— see Section 8.3.5 and 8.5.1"
    )


def calculate_cumulative_roi(
    rows: list[SMAProjectionRow],
    build_cost: float = AWB_BUILD_COST_GBP,
    cost_of_capital: float = AWB_COST_OF_CAPITAL_PCT,
) -> dict:
    """
    TODO: Calculate the cumulative ROI of the AI programme.

    Returns a dict containing:
        total_capital_saving_gbp:   float  (sum of annual savings)
        roi_multiple:               float  (saving / build_cost)
        payback_months:             float
        npv_gbp:                    float  (at 8% hurdle)

    Hint: NPV = sum(saving_year_n / (1 + r)^n) - build_cost
          where n = 1, 2, ..., 6
    """
    raise NotImplementedError(
        "Implement calculate_cumulative_roi()"
    )


def print_projection_table(
    rows: list[SMAProjectionRow],
) -> None:
    """
    TODO: Print a formatted year-by-year projection table.

    Expected format:
        Year | Avg Loss (£M) | ILM  | SMA Capital (£M) | Saving (£M)
        2025 |          8.32 | 2.81 |           146.1  |       6.8
        2026 |          5.41 | 2.61 |           135.7  |      17.2
        ...

    All amounts in £M rounded to 1 decimal place.
    ILM rounded to 2 decimal places.
    """
    raise NotImplementedError(
        "Implement print_projection_table()"
    )


# ── Tests ─────────────────────────────────────────────────────────


def test_projection_row_count() -> None:
    """Projection must cover exactly PROJECTION_YEARS years."""
    calc = SMACapitalCalculator()
    rows = project_ilm_trajectory(calc)
    assert len(rows) == PROJECTION_YEARS, (
        f"Expected {PROJECTION_YEARS} rows, got {len(rows)}"
    )
    print("✅ test_projection_row_count PASSED")


def test_ilm_decreasing() -> None:
    """ILM must decrease each year as losses fall."""
    calc = SMACapitalCalculator()
    rows = project_ilm_trajectory(calc)
    for i in range(1, len(rows)):
        assert rows[i].ilm <= rows[i - 1].ilm, (
            f"ILM increased in year {rows[i].year}: "
            f"{rows[i-1].ilm:.3f} → {rows[i].ilm:.3f}"
        )
    print("✅ test_ilm_decreasing PASSED")


def test_pre_ai_capital_baseline() -> None:
    """
    Pre-AI SMA capital should be approximately £152.9M.
    ILM at £12.8M avg loss and £52M BIC ≈ 2.94.
    """
    calc = SMACapitalCalculator()
    pre_ai_inputs = SMAInputs(
        business_indicator_gbp=AWB_BI_GBP,
        avg_annual_losses_gbp=AWB_PRE_AI_AVG_LOSS_GBP,
        loss_component_gbp=AWB_PRE_AI_AVG_LOSS_GBP * 15,
    )
    result = calc.calculate(pre_ai_inputs)
    # ILM should be approximately 2.94
    assert 2.5 <= result.ilm <= 3.5, (
        f"Pre-AI ILM {result.ilm:.3f} outside expected range"
    )
    print(
        f"✅ test_pre_ai_capital_baseline PASSED "
        f"(ILM={result.ilm:.3f}, capital=£{result.sma_capital_gbp/1e6:.1f}M)"
    )


def test_cumulative_roi_positive() -> None:
    """Total capital saving must exceed build cost."""
    calc = SMACapitalCalculator()
    rows = project_ilm_trajectory(calc)
    roi = calculate_cumulative_roi(rows)

    assert roi["total_capital_saving_gbp"] > AWB_BUILD_COST_GBP, (
        "Total capital saving must exceed £330K build cost"
    )
    assert roi["roi_multiple"] > 1.0, (
        f"ROI multiple {roi['roi_multiple']:.1f}× must exceed 1.0×"
    )
    print(
        f"✅ test_cumulative_roi_positive PASSED "
        f"(ROI={roi['roi_multiple']:.1f}×, "
        f"NPV=£{roi['npv_gbp']/1e6:.1f}M)"
    )


if __name__ == "__main__":
    print("Exercise 8.2: AWB SMA Capital Impact Model\n")

    # These tests work without the implementation
    test_pre_ai_capital_baseline()

    print(
        "\nPre-AI baseline verified. "
        "Now implement the three TODO functions above."
    )
    print(
        f"Pre-AI: £12.8M avg loss × 15 = £192M LC "
        f"→ ILM ~2.94 → SMA ~£152.9M"
    )
    print(
        f"Target: model AI-driven reduction to ILM ~2.72 "
        f"by 2026 (saving £11.5M capital)"
    )
    print(
        f"\nOnce implemented, run: "
        f"python exercise_2.py (tests auto-run)"
    )
