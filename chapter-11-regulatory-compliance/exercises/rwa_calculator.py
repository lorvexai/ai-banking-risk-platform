"""Exercise 11.1: Calculate RWA for a 5-loan portfolio.

Difficulty: 3/5 stars  |  Estimated time: 30 minutes

Task:
    Using the standardised approach (CRR3 Articles 112-141),
    calculate the Risk-Weighted Assets for the five AWB
    corporate loans below. Apply the correct risk weight
    for each exposure class and verify your total RWA
    against the expected answer.

    Your total RWA should be within +/- £1 million of
    the reference answer.

GitHub:
    github.com/lorvenio/ai-banking-risk-platform/
    chapter_011/solutions/rwa_calculator.py
"""
from decimal import Decimal

# ── AWB synthetic loan portfolio ────────────────────────────────
# Each entry: (exposure_gbp, crr3_article, exposure_class, rw_pct)
# Risk weights per CRR3 Articles 112-141
PORTFOLIO = [
    {
        "description": "Investment-grade UK corporate (no rating)",
        "exposure_gbp": Decimal("5_000_000"),
        "crr3_article": "Art. 122",
        "exposure_class": "corporate",
        "risk_weight_pct": Decimal("100"),  # unrated corporate
    },
    {
        "description": "Residential mortgage (LTV 65%)",
        "exposure_gbp": Decimal("2_500_000"),
        "crr3_article": "Art. 125",
        "exposure_class": "retail_residential",
        "risk_weight_pct": Decimal("35"),
    },
    {
        "description": "Commercial real estate (standard)",
        "exposure_gbp": Decimal("8_000_000"),
        "crr3_article": "Art. 126",
        "exposure_class": "commercial_re",
        "risk_weight_pct": Decimal("100"),
    },
    {
        "description": "HVCRE (high-volatility CRE)",
        "exposure_gbp": Decimal("3_000_000"),
        "crr3_article": "Art. 126(2)",
        "exposure_class": "hvcre",
        "risk_weight_pct": Decimal("150"),  # war story error case
    },
    {
        "description": "UK sovereign gilts",
        "exposure_gbp": Decimal("10_000_000"),
        "crr3_article": "Art. 114",
        "exposure_class": "sovereign",
        "risk_weight_pct": Decimal("0"),
    },
]

EXPECTED_TOTAL_RWA_GBP = Decimal("15_375_000")


def calculate_portfolio_rwa(portfolio):
    """TODO: Calculate RWA for each loan and total.

    Args:
        portfolio: List of loan dicts (see PORTFOLIO above).

    Returns:
        Tuple of (List[Decimal], Decimal):
            - individual RWAs in same order as portfolio
            - total RWA

    Formula: RWA = exposure_gbp * risk_weight_pct / 100
    """
    raise NotImplementedError(
        "Implement RWA calculation for each "
        "exposure class in the portfolio."
    )


if __name__ == "__main__":
    individual, total = calculate_portfolio_rwa(PORTFOLIO)
    print("AWB Test Portfolio RWA Calculation")
    print("-" * 50)
    for loan, rwa in zip(PORTFOLIO, individual):
        print(f"{loan['description'][:35]:35s} "
              f"RW={loan['risk_weight_pct']:>3}%  "
              f"RWA=£{rwa:>12,.0f}")
    print("-" * 50)
    print(f"{'Total RWA':35s}       "
          f"    £{total:>12,.0f}")
    diff = abs(total - EXPECTED_TOTAL_RWA_GBP)
    status = "PASS" if diff <= 1_000_000 else "FAIL"
    print(f"\nExpected: £{EXPECTED_TOTAL_RWA_GBP:,.0f}  "
          f"Diff: £{diff:,.0f}  [{status}]")
