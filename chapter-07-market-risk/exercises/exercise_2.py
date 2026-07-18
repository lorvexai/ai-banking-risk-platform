"""
chapter_07/exercises/exercise_2.py
Exercise 7.2: Connect CVA to the Chapter 6 PD Model
via awb_commons PDModelTool

Difficulty: ★★★★☆ | Estimated time: 45 minutes

Task: Wire CVACalculator (MR-2026-048) to the Chapter 6
Corporate PD Model (MR-2026-040) using the
awb_commons.PDModelTool interface.

Steps:
  1. Call PDModelTool with stub counterparty (Barclays,
     BBB rating, 3-year facility) to get PD term structure
  2. Feed PD term structure into CVACalculator
  3. Verify CVA is in range £180,000–£240,000
  4. Calculate SA-CVA capital (unhedged = CVA amount)

Extension:
  Change recovery rate 40% -> 25% (stressed scenario).
  Document the sensitivity as a model card commentary.

Target CVA range (AWB June 2026): £180K–£240K

Complete solution:
  github.com/lorvenio/ai-banking-risk-platform/
  chapter_07/solutions/

AWB | awb_commons | MR-2026-048 exercise
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cva.cva_calculator import (
    CVACalculator,
    ExposureProfile,
    CVAResult,
)
from typing import Dict


# ── AWB Barclays IRS Counterparty (stub data) ────────────────
# This stub replaces the live MR-2026-040 PDModelTool call.
# In production: from awb_commons import PDModelTool
# pd_tool = PDModelTool(model_id="MR-2026-040")
# pd_ts = pd_tool.get_term_structure(
#     counterparty_id="BARCLAYS_001",
#     rating="BBB",
#     facility_years=3,
# )

BARCLAYS_EXPOSURE = ExposureProfile(
    counterparty_id="BARCLAYS_001",
    time_steps=[0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0],
    expected_exposure=[
        3_500_000,   # £3.5M at 3 months
        5_200_000,   # £5.2M at 6 months
        8_100_000,   # £8.1M at 1 year  (peak region)
        10_500_000,  # £10.5M at 2 years
        11_800_000,  # £11.8M at 3 years (peak)
        9_200_000,   # £9.2M at 4 years
        6_400_000,   # £6.4M at 5 years
    ],
    peak_exposure_975=[
        5_000_000, 7_500_000, 11_500_000,
        14_800_000, 16_500_000, 13_000_000, 9_000_000
    ],
)


def get_pd_term_structure_from_mr_2026_040(
    rating: str = "BBB",
) -> Dict[float, float]:
    """
    Stub for MR-2026-040 PDModelTool output.

    In production this calls:
        awb_commons.PDModelTool.get_term_structure()

    Returns cumulative PD at each time horizon.
    BBB-rated counterparty (Barclays proxy).

    Args:
        rating: Internal rating grade
    Returns:
        Dict {time_years: cumulative_pd}
    """
    # AWB internal PD curve for BBB counterparty
    # Source: MR-2026-040 (Chapter 6 Corporate PD Model)
    # Calibrated to UK corporate PD data, June 2026
    pd_curves = {
        "BBB": {
            0.25: 0.0018, 0.5: 0.0036,
            1.0: 0.0072, 2.0: 0.0144,
            3.0: 0.0216, 4.0: 0.0288,
            5.0: 0.0360,
        },
        "BB": {
            0.25: 0.0045, 0.5: 0.0090,
            1.0: 0.0180, 2.0: 0.0360,
            3.0: 0.0540, 4.0: 0.0720,
            5.0: 0.0900,
        },
        "A": {
            0.25: 0.0007, 0.5: 0.0014,
            1.0: 0.0028, 2.0: 0.0056,
            3.0: 0.0084, 4.0: 0.0112,
            5.0: 0.0140,
        },
    }
    return pd_curves.get(
        rating, pd_curves["BBB"]
    )


def calculate_barclays_cva(
    recovery_rate: float = 0.40,
) -> CVAResult:
    """
    Calculate CVA for Barclays IRS using MR-2026-048.

    Step 1: Get PD term structure (from MR-2026-040 stub)
    Step 2: Calculate CVA using CVACalculator
    Step 3: Return CVAResult with SA-CVA capital

    Args:
        recovery_rate: Recovery assumption (default 40%)
    Returns:
        CVAResult with CVA, DVA, SA-CVA capital
    """
    # TODO: Step 1 — get PD term structure
    # pd_ts = get_pd_term_structure_from_mr_2026_040(
    #     rating="BBB"
    # )

    # TODO: Step 2 — instantiate CVACalculator
    # calc = CVACalculator(recovery_rate=recovery_rate)

    # TODO: Step 3 — call calculate_cva
    # result = calc.calculate_cva(
    #     exposure=BARCLAYS_EXPOSURE,
    #     pd_term_structure=pd_ts,
    # )

    # TODO: Step 4 — verify CVA in range £180K–£240K
    # assert 180_000 <= result.cva_gbp <= 240_000, (
    #     f"CVA {result.cva_gbp} outside expected range"
    # )

    # TODO: Step 5 — print SA-CVA capital
    # print(f"CVA: £{result.cva_gbp:,.0f}")
    # print(f"SA-CVA capital: "
    #       f"£{result.sa_cva_capital_gbp:,.0f}")

    raise NotImplementedError(
        "Complete the TODO steps above."
    )


def sensitivity_analysis() -> None:
    """
    Extension task: stressed recovery rate sensitivity.

    Compare CVA and SA-CVA capital at:
      - Base: recovery rate = 40% (CRR3 Art. 274)
      - Stress: recovery rate = 25%

    Write a one-paragraph model card commentary
    documenting the sensitivity for MR-2026-048.
    """
    # TODO: Call calculate_barclays_cva() twice:
    # base_result = calculate_barclays_cva(0.40)
    # stress_result = calculate_barclays_cva(0.25)
    #
    # Compute:
    # cva_increase = stress_result.cva_gbp
    #                - base_result.cva_gbp
    # capital_increase = (
    #     stress_result.sa_cva_capital_gbp
    #     - base_result.sa_cva_capital_gbp
    # )
    # print(f"CVA increase under stress: "
    #       f"£{cva_increase:,.0f}")
    # print(f"Capital increase under stress: "
    #       f"£{capital_increase:,.0f}")
    raise NotImplementedError(
        "Complete sensitivity_analysis()."
    )


if __name__ == "__main__":
    print("Exercise 7.2: CVA Integration")
    print("=" * 40)
    try:
        result = calculate_barclays_cva()
        print(f"CVA (MR-2026-048): £{result.cva_gbp:,.0f}")
        print(f"SA-CVA capital: "
              f"£{result.sa_cva_capital_gbp:,.0f}")
    except NotImplementedError:
        print("Complete the TODOs in exercise_2.py")
    print("\nSensitivity analysis:")
    try:
        sensitivity_analysis()
    except NotImplementedError:
        print("Complete sensitivity_analysis()")
