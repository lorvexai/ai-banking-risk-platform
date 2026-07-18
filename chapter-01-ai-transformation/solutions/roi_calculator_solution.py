"""
chapter_01/solutions/roi_calculator_solution.py
AWB AI Customer Service Platform — ROI Calculator (Solution)

Exercise 1.1 complete solution.
Reference: github.com/lorvenio/ai-banking-risk-platform/chapter_01/

Author: AWB AI Programme (AWB-AI-2025)
Chapter: 1 — The AI Transformation of Risk and Compliance
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AWB canonical cost figures (Section 1.3)
# ---------------------------------------------------------------------------

CALL_CENTRE_FTE = 240
AVG_SALARY_GBP = 28_000
OVERHEAD_RATE = 0.40
AI_SYSTEM_COST_GBP = 940_000
RESIDUAL_MANUAL_GBP = 1_990_000
BUILD_COST_GBP = 280_000
HURDLE_RATE = 0.08

# Sensitivity multipliers (Section 1.3)
CONSERVATIVE_ADJ = -0.30
OPTIMISTIC_ADJ = +0.25


@dataclass
class ROIResult:
    """Result container for a single ROI scenario.

    Args:
        scenario: Scenario name.
        automation_rate: Effective adjustment applied.
        gross_saving_gbp: Annual saving before AI costs.
        net_saving_gbp: Annual saving after all costs.
        payback_months: Months to recover build cost.
        npv_3yr_gbp: 3-year NPV at 8% hurdle rate.
    """
    scenario: str
    automation_rate: float
    gross_saving_gbp: float
    net_saving_gbp: float
    payback_months: float
    npv_3yr_gbp: float


def calculate_base_saving() -> float:
    """Calculate the base annual net saving.

    Returns:
        Base annual net saving in GBP (Section 1.3).
    """
    manual_cost = (
        CALL_CENTRE_FTE * AVG_SALARY_GBP * (1 + OVERHEAD_RATE)
    )
    return manual_cost - AI_SYSTEM_COST_GBP - RESIDUAL_MANUAL_GBP


def calculate_npv(
    annual_saving: float,
    build_cost: float,
    hurdle_rate: float,
    years: int = 3,
) -> float:
    """Calculate NPV for a stream of equal annual savings.

    Args:
        annual_saving: Net saving per year in GBP.
        build_cost:    One-off build cost in GBP.
        hurdle_rate:   Discount rate (e.g., 0.08 for 8%).
        years:         Number of years to discount.

    Returns:
        NPV in GBP.

    Formula: sum(saving / (1+r)^t, t=1..years) - build_cost
    """
    pv_total = sum(
        annual_saving / (1 + hurdle_rate) ** t
        for t in range(1, years + 1)
    )
    return pv_total - build_cost


def _build_result(
    scenario: str,
    adjustment: float,
    base_saving: float,
) -> ROIResult:
    """Build a single ROIResult for the given scenario.

    Args:
        scenario:    Human-readable scenario name.
        adjustment:  Multiplier delta (e.g., -0.30 for Conservative).
        base_saving: Base annual net saving in GBP.

    Returns:
        Populated ROIResult dataclass.
    """
    adjusted_saving = base_saving * (1 + adjustment)
    gross = (
        CALL_CENTRE_FTE * AVG_SALARY_GBP * (1 + OVERHEAD_RATE)
    )
    payback = (
        BUILD_COST_GBP / adjusted_saving * 12
        if adjusted_saving > 0
        else float("inf")
    )
    npv = calculate_npv(
        annual_saving=adjusted_saving,
        build_cost=BUILD_COST_GBP,
        hurdle_rate=HURDLE_RATE,
    )
    return ROIResult(
        scenario=scenario,
        automation_rate=adjustment,
        gross_saving_gbp=gross,
        net_saving_gbp=adjusted_saving,
        payback_months=payback,
        npv_3yr_gbp=npv,
    )


def run_sensitivity(
    base_saving: float,
) -> list[ROIResult]:
    """Run Conservative / Base / Optimistic scenarios.

    Args:
        base_saving: Base annual net saving in GBP.

    Returns:
        List of three ROIResult objects.
    """
    return [
        _build_result("Conservative", CONSERVATIVE_ADJ, base_saving),
        _build_result("Base", 0.0, base_saving),
        _build_result("Optimistic", OPTIMISTIC_ADJ, base_saving),
    ]


def print_sensitivity_table(results: list[ROIResult]) -> None:
    """Print a formatted sensitivity table to stdout.

    Args:
        results: List of ROIResult from run_sensitivity.
    """
    header = (
        f"{'Scenario':<14} | "
        f"{'Adjustment':>10} | "
        f"{'Net Saving/yr':>14} | "
        f"{'Payback':>9} | "
        f"{'3yr NPV':>12}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.scenario:<14} | "
            f"{r.automation_rate:>+9.0%} | "
            f"£{r.net_saving_gbp:>12,.0f} | "
            f"{r.payback_months:>7.1f}m | "
            f"£{r.npv_3yr_gbp:>10,.0f}"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    base = calculate_base_saving()
    logger.info("Base annual saving: £%,.0f", base)
    results = run_sensitivity(base)
    print_sensitivity_table(results)
