"""
data/generate_sample_application.py
Generate realistic AWB credit application data for Chapter 3 testing.

Generates a sample credit application for Fenland Construction Ltd,
a fictional UK SME borrower consistent with AWB's corporate/SME lending book.

Usage:
    python data/generate_sample_application.py
    python data/generate_sample_application.py --output data/sample_application.json
    python data/generate_sample_application.py --scenario stressed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Sample applications
# ---------------------------------------------------------------------------

BASE_APPLICATION: Dict[str, Any] = {
    "application_id": "APP-2026-001847",
    "submitted_date": "2026-03-10",
    "applicant_name": "Fenland Construction Ltd",
    "customer_id": "AWB-CUST-001234",
    "company_number": "04821567",
    "registered_address": {
        "line1": "Unit 14, Enterprise Park",
        "line2": "Ely Road",
        "city": "Cambridge",
        "postcode": "CB25 0AY",
        "country": "United Kingdom",
    },
    "industry_code": "4120",
    "industry_description": "Construction of residential and non-residential buildings",
    "years_trading": 12,
    "number_of_employees": 87,

    # --- Facility request ---
    "facility_amount_gbp": 5_000_000.0,
    "facility_type": "TERM_LOAN",
    "facility_purpose": (
        "Acquisition of commercial property at Newmarket Road, Cambridge, "
        "to consolidate operations and reduce annual rental expenditure of £420,000."
    ),
    "facility_tenor_years": 7,
    "proposed_interest_rate_pct": 6.75,
    "repayment_structure": "Quarterly principal + interest",
    "proposed_security": "First legal charge over Newmarket Road property (est. value £7.2M)",

    # --- Financial summary (last 3 years) ---
    "financials": {
        "year_1": {
            "year_end": "2025-03-31",
            "revenue_gbp": 24_800_000.0,
            "ebitda_gbp": 4_200_000.0,
            "ebit_gbp": 3_500_000.0,
            "net_profit_gbp": 2_050_000.0,
            "total_assets_gbp": 31_500_000.0,
            "total_liabilities_gbp": 19_200_000.0,
            "current_assets_gbp": 8_100_000.0,
            "current_liabilities_gbp": 5_300_000.0,
            "net_debt_gbp": 14_200_000.0,
            "interest_expense_gbp": 890_000.0,
            "capital_expenditure_gbp": 580_000.0,
            "tangible_equity_gbp": 12_300_000.0,
        },
        "year_2": {
            "year_end": "2024-03-31",
            "revenue_gbp": 21_400_000.0,
            "ebitda_gbp": 3_750_000.0,
            "ebit_gbp": 3_100_000.0,
            "net_profit_gbp": 1_820_000.0,
            "total_assets_gbp": 28_900_000.0,
            "total_liabilities_gbp": 17_800_000.0,
            "net_debt_gbp": 12_800_000.0,
            "interest_expense_gbp": 810_000.0,
            "tangible_equity_gbp": 11_100_000.0,
        },
        "year_3": {
            "year_end": "2023-03-31",
            "revenue_gbp": 18_200_000.0,
            "ebitda_gbp": 3_100_000.0,
            "ebit_gbp": 2_600_000.0,
            "net_profit_gbp": 1_450_000.0,
            "total_assets_gbp": 25_400_000.0,
            "total_liabilities_gbp": 16_200_000.0,
            "net_debt_gbp": 10_500_000.0,
            "interest_expense_gbp": 720_000.0,
            "tangible_equity_gbp": 9_200_000.0,
        },
        "latest_management_accounts": {
            "period_end": "2025-12-31",
            "revenue_gbp": 13_200_000.0,
            "ebitda_gbp": 2_400_000.0,
            "comment": "9-month management accounts; seasonality expected in Q4",
        },
    },

    # --- Post-facility financials (proforma) ---
    "proforma_financials": {
        "net_debt_gbp": 19_200_000.0,   # Existing + new £5M facility
        "ebitda_gbp": 4_500_000.0,       # Forecast Year 1 post-acquisition
        "ebit_gbp": 3_800_000.0,
        "interest_expense_gbp": 950_000.0,
        "stressed_interest_expense_gbp": 1_250_000.0,  # +200bps PRA stress
        "total_exposure_gbp": 18_700_000.0,  # T24 existing (£13.7M) + new (£5M)
        "total_assets_gbp": 38_700_000.0,   # Including property acquisition
        "total_liabilities_gbp": 26_700_000.0,
        "current_assets_gbp": 8_100_000.0,
        "current_liabilities_gbp": 5_500_000.0,
        "tangible_equity_gbp": 12_000_000.0,
        "loan_to_value_pct": 69.4,  # £5M facility / £7.2M property value
    },

    # --- Directors and ownership ---
    "directors": [
        {
            "name": "James Whitfield",
            "role": "CEO",
            "ownership_pct": 55.0,
            "years_with_company": 12,
        },
        {
            "name": "Sarah Whitfield",
            "role": "Finance Director",
            "ownership_pct": 30.0,
            "years_with_company": 10,
        },
        {
            "name": "Rajesh Patel",
            "role": "Operations Director",
            "ownership_pct": 15.0,
            "years_with_company": 7,
        },
    ],
    "personal_guarantees_offered": True,
    "guarantee_amount_gbp": 2_500_000.0,

    # --- Key contracts and pipeline ---
    "key_contracts": [
        {
            "client": "Cambridgeshire County Council",
            "contract_value_gbp": 8_400_000.0,
            "contract_end": "2027-06-30",
            "type": "Housing development (48 units)",
        },
        {
            "client": "NHS Property Services",
            "contract_value_gbp": 3_200_000.0,
            "contract_end": "2026-09-30",
            "type": "Medical centre refurbishment",
        },
    ],
    "pipeline_value_gbp": 18_000_000.0,

    # --- Banking relationship ---
    "crb_relationship_start": "2014-01-15",
    "relationship_manager": "Claire Thompson, AWB Corporate Banking Cambridge",
    "existing_products": [
        "Revolving Credit Facility £5M (FAC-2021-0847)",
        "Term Loan £8M (FAC-2019-0312)",
        "Trade Finance Lines £500K",
        "Business Current Account",
        "FX Forward Contracts",
    ],
    "arrears_history": "None — zero missed payments in 12-year relationship",
}

# ---------------------------------------------------------------------------
# Stressed scenario
# ---------------------------------------------------------------------------

STRESSED_APPLICATION: Dict[str, Any] = {
    **BASE_APPLICATION,
    "application_id": "APP-2026-001848",
    "applicant_name": "Fenland Construction Ltd (Stressed Scenario)",
    "facility_amount_gbp": 12_000_000.0,  # Triggers CRITICAL concentration breach
    "proforma_financials": {
        **BASE_APPLICATION["proforma_financials"],
        "net_debt_gbp": 26_200_000.0,      # Leverage ~5.8x — breaches 5.0x policy
        "ebitda_gbp": 4_500_000.0,
        "ebit_gbp": 2_800_000.0,           # ICR 1.8x — breaches 2.0x minimum
        "interest_expense_gbp": 1_550_000.0,
        "total_exposure_gbp": 25_700_000.0,
        "tangible_equity_gbp": 12_000_000.0,
        "total_assets_gbp": 38_700_000.0,
        "total_liabilities_gbp": 26_700_000.0,
        "current_assets_gbp": 8_100_000.0,
        "current_liabilities_gbp": 5_500_000.0,
    },
    "_scenario": "STRESSED — Multiple policy breaches expected",
}

# ---------------------------------------------------------------------------
# Small facility (below HITL threshold)
# ---------------------------------------------------------------------------

SMALL_FACILITY_APPLICATION: Dict[str, Any] = {
    **BASE_APPLICATION,
    "application_id": "APP-2026-001849",
    "applicant_name": "Ely Joinery Services Ltd",
    "customer_id": "AWB-CUST-005678",
    "facility_amount_gbp": 250_000.0,    # Below £500K HITL threshold
    "facility_type": "REVOLVING_CREDIT",
    "facility_purpose": "Working capital facility for seasonal cash flow management.",
    "_scenario": "SMALL FACILITY — No human review required",
}


# ---------------------------------------------------------------------------
# Output function
# ---------------------------------------------------------------------------

def generate(scenario: str = "base") -> Dict[str, Any]:
    """
    Generate a credit application for the specified scenario.

    Args:
        scenario: "base", "stressed", or "small".

    Returns:
        Credit application dict.
    """
    mapping = {
        "base": BASE_APPLICATION,
        "stressed": STRESSED_APPLICATION,
        "small": SMALL_FACILITY_APPLICATION,
    }
    if scenario not in mapping:
        raise ValueError(f"Unknown scenario '{scenario}'. Choose from: {list(mapping.keys())}")
    return mapping[scenario]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a sample AWB credit application for Chapter 3 testing."
    )
    parser.add_argument(
        "--scenario",
        choices=["base", "stressed", "small"],
        default="base",
        help="Application scenario to generate (default: base)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: print to stdout)",
    )
    args = parser.parse_args()

    application = generate(args.scenario)
    output_json = json.dumps(application, indent=2)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_json, encoding="utf-8")
        print(f"✅ Credit application written to {output_path}", file=sys.stderr)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
