"""
exercises/lcr_calculator.py — Exercise 9.1 starter.
Chapter 9: Liquidity Risk — AWB-AI-2025 Programme.

Task: Build LCR from 28 EBA data elements.
Target: LCR matches expected AWB June 2026 value of 138%.

Model: MR-2026-073 | PRA SS1/23 Risk: LOW
"""
from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from lcr_nsfr.calculator import (
    LCRCalculator,
    HQLAPortfolio,
    StressOutflows,
    StressInflows,
)
from awb_commons.models import StressScenario


# ── AWB June 2026 HQLA Portfolio ────────────────────────────────
# Level 1 (no haircut): BoE reserves + UK gilts
# Level 2A (15% haircut): covered bonds rated AA- or better
# Level 2B (50% haircut): corporate bonds / RMBS
awb_hqla = HQLAPortfolio(
    level_1_central_bank_gbp=4_200_000_000,  # BoE reserves
    level_1_gov_bonds_gbp=6_200_000_000,     # UK gilts
    level_2a_covered_bonds_gbp=4_235_000_000, # covered bonds
    level_2b_corp_bonds_gbp=0,               # no L2B at AWB
)

# ── AWB June 2026 Stress Outflows (30-day) ───────────────────────
# CRR3 Art. 422-424 run-off rates applied by LCRCalculator:
# retail_stable: 5% | retail_less_stable: 10%
# wholesale_non_op: 100% | committed_facilities: 10%
awb_outflows = StressOutflows(
    retail_stable_gbp=12_000_000_000,        # insured deposits
    retail_less_stable_gbp=0,               # none at AWB
    wholesale_operational_gbp=4_500_000_000, # op. deposits
    wholesale_non_op_gbp=850_000_000,        # non-op wholesale
    committed_facilities_gbp=3_200_000_000,  # credit lines
    derivatives_collateral_gbp=200_000_000,  # collateral calls
)

# ── AWB June 2026 Stress Inflows (30-day) ────────────────────────
# Cap: max 75% of gross outflows (CRR3 Art. 425)
awb_inflows = StressInflows(
    maturing_loans_gbp=1_200_000_000,
    committed_inflows_gbp=400_000_000,
    other_inflows_gbp=0,
)


def exercise_lcr() -> None:
    """
    TODO: Calculate AWB's LCR using the inputs above.

    Steps:
    1. Instantiate LCRCalculator
    2. Call calculate() with awb_hqla, awb_outflows, awb_inflows
    3. Print: HQLA, net outflows, LCR%, compliant flag
    4. Verify: result.lcr_pct should be between 130% and 145%
       (AWB June 2026 reported LCR = 138%)
    5. Bonus: run with StressScenario.PRA_CST_SEVERE and confirm
       LCR remains >= 100% (AWB CST result = 108%)
    """
    # YOUR CODE HERE
    calc = LCRCalculator()

    # Step 1-2: calculate base LCR
    # result = calc.calculate(...)

    # Step 3: print results
    # print(f"HQLA: £{result.hqla_gbp/1e9:.1f}B")
    # print(f"Net outflows: £{result.net_outflows_gbp/1e9:.1f}B")
    # print(f"LCR: {result.lcr_pct:.1f}%")
    # print(f"Compliant: {result.compliant}")

    # Step 4: assert within expected range
    # assert 130 <= result.lcr_pct <= 145, (
    #     f"Expected ~138%, got {result.lcr_pct:.1f}%"
    # )

    # Step 5 bonus: stress scenario
    # stress = calc.calculate(
    #     awb_hqla, awb_outflows, awb_inflows,
    #     scenario=StressScenario.PRA_CST_SEVERE
    # )
    # assert stress.lcr_pct >= 100, "Must survive PRA CST severe"

    print("Exercise 9.1: implement the steps above.")
    print("Solution: chapter_09/solutions/lcr_calculator_sol.py")


if __name__ == "__main__":
    exercise_lcr()
