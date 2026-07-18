"""Exercise 11.2: Generate Leverage Ratio CoRep C 47.00.

Difficulty: 4/5 stars  |  Estimated time: 45 minutes

Task:
    Using the synthetic AWB Q4 2025 balance sheet data below,
    calculate all four CRR3 Article 429 exposure components and
    produce a CoRep C 47.00 XBRL output that passes EBA DPM
    leverage-specific validation rules.

    Your result should reproduce AWB's Q4 2025 leverage ratio
    of 4.2% within 0.05 percentage points.

Reference:
    - AWB leverage_calculator.py (mjrrp module)
    - AWB xbrl_filer.py (mjrrp module)
    - CRR3 Article 429, 429b, 429c, 429d, 429e-429g

GitHub:
    github.com/lorvenio/ai-banking-risk-platform/
    chapter_011/solutions/leverage_ratio.py
"""
from decimal import Decimal
from datetime import date

# ── Synthetic AWB Q4 2025 Balance Sheet Data ───────────────────
QUARTER_END = date(2025, 12, 31)
TIER1_CAPITAL_GBP = Decimal("1_680_000_000")  # CRR3 Art. 26-36

# Art. 429b on-balance-sheet inputs
TOTAL_ASSETS_GBP = Decimal("40_000_000_000")
PROVISIONS_ON_DEFAULTS_GBP = Decimal("820_000_000")
T1_DEDUCTIONS_GBP = Decimal("80_000_000")
DERIVATIVE_ACCOUNTING_VALUE_GBP = Decimal("900_000_000")

# Art. 429c SA-CCR derivative inputs
REPLACEMENT_COST_GBP = Decimal("450_000_000")
POTENTIAL_FUTURE_EXPOSURE_GBP = Decimal("750_000_000")

# Art. 429d SFT inputs
GROSS_CASH_RECEIVABLES_GBP = Decimal("280_000_000")
GROSS_SECURITIES_PROVIDED_GBP = Decimal("160_000_000")
NETTING_BENEFIT_GBP = Decimal("40_000_000")

# Arts 429e-429g off-balance-sheet commitments
COMMITMENTS = {
    "unconditionally_cancellable": Decimal("3_000_000_000"),
    "up_to_1yr": Decimal("2_500_000_000"),
    "over_1yr": Decimal("4_000_000_000"),
    "guarantees": Decimal("500_000_000"),
    "letters_of_credit": Decimal("200_000_000"),
}

TARGET_LEVERAGE_RATIO_PCT = 4.2  # +/- 0.05 tolerance


def calculate_leverage_ratio():
    """TODO: Implement the leverage ratio calculation.

    Steps:
        1. Import LeverageRatioCalculator from mjrrp.leverage_calculator
        2. Calculate on-balance-sheet exposure (Art. 429b)
        3. Calculate SA-CCR exposure (Art. 429c)
        4. Calculate SFT exposure (Art. 429d)
        5. Calculate off-balance-sheet exposure (Arts 429e-429g)
        6. Assemble into LeverageRatioResult
        7. Verify result.leverage_ratio_pct is ~4.2%

    Returns:
        LeverageRatioResult with all four components.
    """
    raise NotImplementedError(
        "Implement calculate_leverage_ratio() "
        "using LeverageRatioCalculator from mjrrp module."
    )


def generate_corep_c47(result) -> str:
    """TODO: Generate COREP C 47.00 XBRL output.

    Args:
        result: LeverageRatioResult from calculate_leverage_ratio()

    Returns:
        XBRL XML string for C 47.00 submission.

    Hint: Use xbrl_filer.EBAXBRLFiler with return_code='C 47.00'
    """
    raise NotImplementedError(
        "Implement generate_corep_c47() "
        "using EBAXBRLFiler from mjrrp.xbrl_filer."
    )


if __name__ == "__main__":
    result = calculate_leverage_ratio()
    print(f"Leverage ratio: {result.leverage_ratio_pct:.2f}%")
    print(f"Target: {TARGET_LEVERAGE_RATIO_PCT}% "
          f"(tolerance +/- 0.05%)")
    tolerance = abs(
        result.leverage_ratio_pct - TARGET_LEVERAGE_RATIO_PCT
    )
    print(f"Difference: {tolerance:.3f}% "
          f"{'PASS' if tolerance <= 0.05 else 'FAIL'}")
    xbrl = generate_corep_c47(result)
    print(f"\nXBRL output ({len(xbrl)} chars):")
    print(xbrl[:300] + "...")
