"""
chapter_01/exercises/roi_calculator.py
AWB AI Customer Service Platform — ROI Calculator (Starter)

Exercise 1.1: Build the AWB Customer Service ROI Calculator
Difficulty: ★★☆☆☆ | Estimated time: 20 minutes

Task: Extend the basic ROI model to include a 3-scenario
sensitivity analysis (Conservative -30%, Base, Optimistic +25%).
Your output should produce a formatted table matching the
style in Section 1.3.

Success criterion: all three scenarios produce different NPV
values at 8% discount rate.

Author: AWB AI Programme (AWB-AI-2025)
Chapter: 1 — The AI Transformation of Risk and Compliance
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AWB Customer Service Platform — canonical cost figures (Section 1.3)
# ---------------------------------------------------------------------------

CALL_CENTRE_FTE = 240
AVG_SALARY_GBP = 28_000
OVERHEAD_RATE = 0.40
AI_SYSTEM_COST_GBP = 940_000
RESIDUAL_MANUAL_GBP = 1_990_000
BUILD_COST_GBP = 280_000
HURDLE_RATE = 0.08  # 8% NPV discount rate


@dataclass
class ROIResult:
    """Result container for a single ROI scenario.

    Args:
        scenario: Scenario name (Conservative/Base/Optimistic).
        automation_rate: Proportion of calls automated (0–1).
        gross_saving_gbp: Annual saving before AI costs.
        net_saving_gbp: Annual saving after AI costs.
        payback_months: Months to recover build cost.
        npv_3yr_gbp: 3-year NPV at hurdle rate.
    """
    scenario: str
    automation_rate: float
    gross_saving_gbp: float
    net_saving_gbp: float
    payback_months: float
    npv_3yr_gbp: float


def calculate_base_saving() -> float:
    """Calculate the base annual saving.

    Uses canonical AWB figures from Section 1.3:
    call centre FTE cost minus AI system cost minus
    residual manual handling cost.

    Returns:
        Base annual net saving in GBP.
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
        NPV in GBP (negative = destroys value).
    """
    # TODO: Implement NPV calculation
    # Formula: sum(saving / (1 + r)^t for t in 1..years) - build_cost
    raise NotImplementedError(
        "Implement calculate_npv — see Section 1.6 ROI Framework"
    )


def run_sensitivity(
    base_saving: float,
) -> list[ROIResult]:
    """Run Conservative / Base / Optimistic scenarios.

    Args:
        base_saving: Base annual net saving in GBP.

    Returns:
        List of three ROIResult objects, one per scenario.

    Scenarios:
        Conservative: -30% automation rate
        Base:          0% (no adjustment)
        Optimistic:   +25% automation rate
    """
    # TODO: Implement all three scenarios
    # Each should adjust the base_saving by the scenario multiplier
    # and call calculate_npv to get the 3-year NPV
    raise NotImplementedError(
        "Implement run_sensitivity — use CONSERVATIVE, BASE, OPTIMISTIC"
    )


def print_sensitivity_table(results: list[ROIResult]) -> None:
    """Print a formatted sensitivity table to stdout.

    Args:
        results: List of ROIResult objects from run_sensitivity.
    """
    header = (
        f"{'Scenario':<14} | "
        f"{'Automation':>10} | "
        f"{'Net Saving/yr':>14} | "
        f"{'Payback':>9} | "
        f"{'3yr NPV':>12}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.scenario:<14} | "
            f"{r.automation_rate:>9.0%} | "
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
