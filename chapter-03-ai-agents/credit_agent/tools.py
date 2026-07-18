"""
credit_agent/tools.py
AWB Automated Credit Decision Workflow — Tool Definitions
Chapter 3: Agentic AI for Financial Risk

Each tool is a pure Python function with:
- Type hints and docstrings
- Input validation
- Mock T24/external responses (replace with live integration in production)
- Structured output (dict) for agent consumption

Tool registry maps tool names → callables for ReAct loop dispatch.

Regulatory context:
- PRA SS1/23: Each tool call is logged as a model decision step (MR-2026-037).
- EU AI Act Annex III: Tools forming part of a credit scoring system are
  subject to conformity assessment and human oversight requirements.
- FCA Consumer Duty (PS22/9): Outputs must be explainable to the customer.
"""

from __future__ import annotations

import datetime
import json
import time
import uuid
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _require(value: Any, name: str, expected_type: type) -> None:
    """Raise ValueError if value is None or wrong type."""
    if value is None:
        raise ValueError(f"Required parameter '{name}' is missing.")
    if not isinstance(value, expected_type):
        raise TypeError(
            f"Parameter '{name}' must be {expected_type.__name__}, "
            f"got {type(value).__name__}."
        )


def _require_positive(value: float, name: str) -> None:
    """Raise ValueError if value is not strictly positive."""
    if value <= 0:
        raise ValueError(f"Parameter '{name}' must be positive; got {value}.")


# ---------------------------------------------------------------------------
# Tool 1: fetch_t24_exposure
# ---------------------------------------------------------------------------

def fetch_t24_exposure(
    customer_id: str,
    include_contingent: bool = True,
) -> Dict[str, Any]:
    """
    Fetch the applicant's existing credit exposure from Temenos T24.

    In production this calls the T24 OFS (Online Financial Services) REST API
    at endpoint /api/v1/customer/{customer_id}/exposure.

    Args:
        customer_id: AWB's internal T24 customer identifier (e.g. "AWB-CUST-001234").
        include_contingent: If True, include contingent liabilities (guarantees, LCs).

    Returns:
        dict with keys:
            customer_id (str): Echo of input.
            existing_facilities (list): Current live facilities.
            total_committed_gbp (float): Total committed exposure in £.
            total_drawn_gbp (float): Currently drawn amount in £.
            total_contingent_gbp (float): Contingent exposure in £.
            data_as_at (str): ISO-8601 timestamp of T24 data snapshot.
            t24_response_code (str): "200" on success.

    Raises:
        ValueError: If customer_id is empty or malformed.
    """
    _require(customer_id, "customer_id", str)
    if not customer_id.strip():
        raise ValueError("customer_id cannot be an empty string.")
    if len(customer_id) > 50:
        raise ValueError("customer_id exceeds maximum length of 50 characters.")

    # --- Mock T24 response (replace with live T24 OFS call in production) ---
    mock_facilities = [
        {
            "facility_id": "FAC-2021-0847",
            "type": "REVOLVING_CREDIT",
            "committed_gbp": 5_000_000.0,
            "drawn_gbp": 3_200_000.0,
            "maturity_date": "2026-03-31",
            "interest_rate_pct": 6.25,
            "security": "First charge over trading assets",
        },
        {
            "facility_id": "FAC-2019-0312",
            "type": "TERM_LOAN",
            "committed_gbp": 8_000_000.0,
            "drawn_gbp": 6_500_000.0,
            "maturity_date": "2029-12-31",
            "interest_rate_pct": 5.75,
            "security": "Fixed and floating charge",
        },
    ]

    contingent = []
    total_contingent = 0.0
    if include_contingent:
        contingent = [
            {
                "type": "PERFORMANCE_BOND",
                "amount_gbp": 500_000.0,
                "beneficiary": "Highways England",
                "expiry": "2025-06-30",
            }
        ]
        total_contingent = 500_000.0

    total_committed = sum(f["committed_gbp"] for f in mock_facilities)
    total_drawn = sum(f["drawn_gbp"] for f in mock_facilities)

    return {
        "customer_id": customer_id,
        "existing_facilities": mock_facilities,
        "contingent_liabilities": contingent,
        "total_committed_gbp": total_committed,
        "total_drawn_gbp": total_drawn,
        "total_contingent_gbp": total_contingent,
        "data_as_at": datetime.datetime.utcnow().isoformat() + "Z",
        "t24_response_code": "200",
        "data_source": "T24 OFS REST API v2 (mock)",
    }


# ---------------------------------------------------------------------------
# Tool 2: check_credit_policy
# ---------------------------------------------------------------------------

def check_credit_policy(
    net_debt_gbp: float,
    ebitda_gbp: float,
    ebit_gbp: float,
    interest_expense_gbp: float,
    total_exposure_gbp: float,
    tangible_equity_gbp: float,
    total_assets_gbp: float,
    stressed_interest_expense_gbp: Optional[float] = None,
    facility_type: str = "TERM_LOAN",
) -> Dict[str, Any]:
    """
    Evaluate applicant financials against AWB's credit policy rule set.

    Applies all rules defined in policy_rules.AWBCreditPolicyRuleSet and
    returns a structured assessment including individual breach details.

    Args:
        net_debt_gbp: Total net debt post-facility (£).
        ebitda_gbp: Earnings before interest, taxes, depreciation, amortisation (£).
        ebit_gbp: Earnings before interest and taxes (£).
        interest_expense_gbp: Annual interest expense (£).
        total_exposure_gbp: Total AWB exposure (existing + proposed facility) (£).
        tangible_equity_gbp: Borrower's tangible net worth (£).
        total_assets_gbp: Borrower's total assets (£).
        stressed_interest_expense_gbp: Interest at +200bps stress scenario (£, optional).
        facility_type: One of TERM_LOAN, REVOLVING_CREDIT, OVERDRAFT (default: TERM_LOAN).

    Returns:
        dict with keys:
            policy_compliant (bool): True if no CRITICAL or HIGH breaches.
            breaches (list): List of PolicyBreach dicts.
            breach_count (int): Total number of breaches.
            blocking_breach_count (int): CRITICAL + HIGH severity breaches.
            policy_summary (dict): Threshold values applied.
            recommendation (str): APPROVE, REFER, or DECLINE based on breaches.
    """
    from credit_agent.policy_rules import (
        AWBCreditPolicyRuleSet,
        FacilityType,
        Severity,
    )

    # Input validation
    for name, val in [
        ("net_debt_gbp", net_debt_gbp),
        ("ebitda_gbp", ebitda_gbp),
        ("ebit_gbp", ebit_gbp),
        ("interest_expense_gbp", interest_expense_gbp),
        ("total_exposure_gbp", total_exposure_gbp),
        ("tangible_equity_gbp", tangible_equity_gbp),
        ("total_assets_gbp", total_assets_gbp),
    ]:
        _require(val, name, (int, float))

    # Map facility type string to enum
    try:
        ft = FacilityType(facility_type.upper())
    except ValueError:
        ft = FacilityType.TERM_LOAN

    policy = AWBCreditPolicyRuleSet()
    breaches = policy.evaluate_all(
        net_debt=net_debt_gbp,
        ebitda=ebitda_gbp,
        ebit=ebit_gbp,
        interest_expense=interest_expense_gbp,
        total_exposure_gbp=total_exposure_gbp,
        tangible_equity_gbp=tangible_equity_gbp,
        total_assets_gbp=total_assets_gbp,
        stressed_interest_expense=stressed_interest_expense_gbp,
        facility_type=ft,
    )

    blocking = [b for b in breaches if b.severity in (Severity.CRITICAL, Severity.HIGH)]
    advisory = [b for b in breaches if b.severity in (Severity.MEDIUM, Severity.LOW)]

    if any(b.severity == Severity.CRITICAL for b in breaches):
        recommendation = "DECLINE"
    elif blocking:
        recommendation = "REFER"
    elif advisory:
        recommendation = "REFER"
    else:
        recommendation = "APPROVE"

    return {
        "policy_compliant": len(blocking) == 0,
        "breaches": [b.to_dict() for b in breaches],
        "breach_count": len(breaches),
        "blocking_breach_count": len(blocking),
        "advisory_breach_count": len(advisory),
        "recommendation": recommendation,
        "policy_summary": policy.summary,
        "evaluated_at": datetime.datetime.utcnow().isoformat() + "Z",
    }


# ---------------------------------------------------------------------------
# Tool 3: assess_covenants
# ---------------------------------------------------------------------------

def assess_covenants(
    facility_amount_gbp: float,
    facility_type: str,
    leverage_ratio: float,
    interest_cover_ratio: float,
    loan_to_value_pct: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Recommend financial covenants for the proposed facility.

    Covenants are set at a headroom above policy limits to provide an
    early warning before a policy breach. Headroom is calibrated to AWB's
    standard covenant framework (AWB-COV-2024-001).

    Args:
        facility_amount_gbp: Proposed facility amount (£).
        facility_type: TERM_LOAN, REVOLVING_CREDIT, etc.
        leverage_ratio: Current net debt / EBITDA.
        interest_cover_ratio: Current EBIT / interest expense.
        loan_to_value_pct: LTV % for asset-backed facilities (optional).

    Returns:
        dict with keys:
            recommended_covenants (list): Covenant definitions.
            testing_frequency (str): Quarterly / Semi-annual / Annual.
            reporting_requirements (list): Financial information obligations.
            cure_period_days (int): Days to remedy a covenant breach.
    """
    _require(facility_amount_gbp, "facility_amount_gbp", (int, float))
    _require_positive(facility_amount_gbp, "facility_amount_gbp")
    _require(leverage_ratio, "leverage_ratio", (int, float))
    _require(interest_cover_ratio, "interest_cover_ratio", (int, float))

    covenants = []

    # Leverage covenant — set at 10% headroom above current ratio
    leverage_covenant = min(round(leverage_ratio * 1.10, 1), 4.5)
    covenants.append({
        "covenant_name": "Maximum Leverage Ratio",
        "metric": "Net Debt / EBITDA",
        "threshold": leverage_covenant,
        "direction": "MAXIMUM",
        "headroom_pct": 10.0,
        "rationale": f"Set at {leverage_covenant:.1f}x vs current {leverage_ratio:.2f}x (10% headroom).",
    })

    # Interest cover covenant
    icr_covenant = max(round(interest_cover_ratio * 0.90, 1), 1.75)
    covenants.append({
        "covenant_name": "Minimum Interest Cover Ratio",
        "metric": "EBIT / Interest Expense",
        "threshold": icr_covenant,
        "direction": "MINIMUM",
        "headroom_pct": 10.0,
        "rationale": f"Set at {icr_covenant:.1f}x vs current {interest_cover_ratio:.2f}x.",
    })

    # LTV covenant for asset-backed facilities
    if loan_to_value_pct is not None:
        ltv_covenant = min(round(loan_to_value_pct + 5.0, 0), 75.0)
        covenants.append({
            "covenant_name": "Maximum Loan-to-Value",
            "metric": "Outstanding Loan / Appraised Asset Value",
            "threshold": ltv_covenant,
            "direction": "MAXIMUM",
            "headroom_pct": 5.0,
            "rationale": f"Set at {ltv_covenant:.0f}% vs current {loan_to_value_pct:.1f}% LTV.",
        })

    # Determine testing frequency based on risk profile
    if leverage_ratio > 3.5 or interest_cover_ratio < 2.5:
        testing_frequency = "Quarterly"
        cure_period_days = 30
    elif facility_amount_gbp > 10_000_000:
        testing_frequency = "Semi-annual"
        cure_period_days = 45
    else:
        testing_frequency = "Annual"
        cure_period_days = 60

    reporting_requirements = [
        "Audited annual accounts within 180 days of financial year end",
        "Management accounts within 60 days of quarter end",
        "Compliance certificate signed by CFO within 15 days of test date",
        "Borrowing base certificate monthly (if revolving facility)",
        "Notification of material adverse change within 5 business days",
    ]

    return {
        "recommended_covenants": covenants,
        "testing_frequency": testing_frequency,
        "reporting_requirements": reporting_requirements,
        "cure_period_days": cure_period_days,
        "covenant_framework": "AWB-COV-2024-001",
        "assessed_at": datetime.datetime.utcnow().isoformat() + "Z",
    }


# ---------------------------------------------------------------------------
# Tool 4: calculate_ratios
# ---------------------------------------------------------------------------

def calculate_ratios(
    revenue_gbp: float,
    ebitda_gbp: float,
    ebit_gbp: float,
    net_profit_gbp: float,
    total_assets_gbp: float,
    total_liabilities_gbp: float,
    current_assets_gbp: float,
    current_liabilities_gbp: float,
    net_debt_gbp: float,
    interest_expense_gbp: float,
    capital_expenditure_gbp: float = 0.0,
) -> Dict[str, Any]:
    """
    Calculate a comprehensive set of financial ratios for credit analysis.

    Ratios are calculated per standard UK accounting conventions (UK GAAP / IFRS).

    Args:
        revenue_gbp: Annual revenue (£).
        ebitda_gbp: EBITDA (£).
        ebit_gbp: EBIT (£).
        net_profit_gbp: Net profit after tax (£).
        total_assets_gbp: Total assets (£).
        total_liabilities_gbp: Total liabilities (£).
        current_assets_gbp: Current assets (£).
        current_liabilities_gbp: Current liabilities (£).
        net_debt_gbp: Net debt (£).
        interest_expense_gbp: Annual interest expense (£).
        capital_expenditure_gbp: Annual capex (£), default 0.

    Returns:
        dict with calculated ratios grouped by category.
    """
    for name, val in [
        ("revenue_gbp", revenue_gbp),
        ("ebitda_gbp", ebitda_gbp),
        ("total_assets_gbp", total_assets_gbp),
    ]:
        _require(val, name, (int, float))
        _require_positive(val, name)

    def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
        return round(numerator / denominator, 4) if denominator != 0 else default

    equity_gbp = total_assets_gbp - total_liabilities_gbp
    free_cash_flow_gbp = ebitda_gbp - capital_expenditure_gbp

    profitability = {
        "ebitda_margin_pct": round(safe_div(ebitda_gbp, revenue_gbp) * 100, 2),
        "ebit_margin_pct": round(safe_div(ebit_gbp, revenue_gbp) * 100, 2),
        "net_profit_margin_pct": round(safe_div(net_profit_gbp, revenue_gbp) * 100, 2),
        "return_on_assets_pct": round(safe_div(net_profit_gbp, total_assets_gbp) * 100, 2),
        "return_on_equity_pct": round(safe_div(net_profit_gbp, equity_gbp) * 100, 2) if equity_gbp > 0 else None,
    }

    leverage = {
        "net_debt_to_ebitda": round(safe_div(net_debt_gbp, ebitda_gbp), 2),
        "net_debt_to_equity": round(safe_div(net_debt_gbp, equity_gbp), 2) if equity_gbp > 0 else None,
        "total_debt_to_assets_pct": round(safe_div(total_liabilities_gbp, total_assets_gbp) * 100, 2),
        "equity_ratio_pct": round(safe_div(equity_gbp, total_assets_gbp) * 100, 2) if equity_gbp > 0 else None,
    }

    coverage = {
        "interest_cover_ratio": round(safe_div(ebit_gbp, interest_expense_gbp), 2),
        "debt_service_cover_ratio": round(safe_div(free_cash_flow_gbp, interest_expense_gbp), 2),
        "ebitda_to_interest": round(safe_div(ebitda_gbp, interest_expense_gbp), 2),
    }

    liquidity = {
        "current_ratio": round(safe_div(current_assets_gbp, current_liabilities_gbp), 2),
        "quick_ratio": round(
            safe_div(current_assets_gbp * 0.7, current_liabilities_gbp), 2
        ),  # Approximation; use actual liquid assets in production
    }

    return {
        "profitability": profitability,
        "leverage": leverage,
        "coverage": coverage,
        "liquidity": liquidity,
        "summary": {
            "revenue_gbp": revenue_gbp,
            "ebitda_gbp": ebitda_gbp,
            "net_debt_gbp": net_debt_gbp,
            "equity_gbp": equity_gbp,
            "free_cash_flow_gbp": free_cash_flow_gbp,
        },
        "calculated_at": datetime.datetime.utcnow().isoformat() + "Z",
    }


# ---------------------------------------------------------------------------
# Tool 5: fetch_comparable_portfolio
# ---------------------------------------------------------------------------

def fetch_comparable_portfolio(
    industry_code: str,
    facility_type: str,
    facility_size_band: str,
) -> Dict[str, Any]:
    """
    Fetch AWB's comparable portfolio statistics for peer benchmarking.

    Queries AWB's internal analytics warehouse for median financial ratios
    of similarly-sized borrowers in the same industry sector.

    Args:
        industry_code: UK SIC 2007 industry code (e.g. "4120" = Construction).
        facility_type: TERM_LOAN, REVOLVING_CREDIT, etc.
        facility_size_band: One of SMALL (<£2M), MEDIUM (£2M–£10M), LARGE (>£10M).

    Returns:
        dict with portfolio median / quartile ratios for benchmarking.

    Raises:
        ValueError: If industry_code is not a 4-digit string.
    """
    _require(industry_code, "industry_code", str)
    if not industry_code.strip().isdigit() or len(industry_code.strip()) != 4:
        raise ValueError(
            f"industry_code must be a 4-digit UK SIC 2007 code; got '{industry_code}'."
        )

    valid_bands = ("SMALL", "MEDIUM", "LARGE")
    if facility_size_band.upper() not in valid_bands:
        raise ValueError(f"facility_size_band must be one of {valid_bands}.")

    # Mock portfolio statistics (in production: query analytics DWH)
    mock_stats = {
        "industry_code": industry_code,
        "industry_description": "Construction of residential and non-residential buildings",
        "facility_type": facility_type,
        "facility_size_band": facility_size_band,
        "portfolio_count": 47,
        "median_ratios": {
            "net_debt_to_ebitda": 3.2,
            "interest_cover_ratio": 3.1,
            "ebitda_margin_pct": 12.4,
            "current_ratio": 1.45,
            "equity_ratio_pct": 32.0,
        },
        "quartile_25": {
            "net_debt_to_ebitda": 2.1,
            "interest_cover_ratio": 4.2,
            "ebitda_margin_pct": 9.8,
        },
        "quartile_75": {
            "net_debt_to_ebitda": 4.3,
            "interest_cover_ratio": 2.1,
            "ebitda_margin_pct": 15.6,
        },
        "default_rate_12m_pct": 1.8,
        "default_rate_36m_pct": 4.2,
        "data_as_at": "2025-12-31",
        "data_source": "AWB Analytics DWH v2 (mock)",
    }

    return mock_stats


# ---------------------------------------------------------------------------
# Tool 6: draft_credit_memo
# ---------------------------------------------------------------------------

def draft_credit_memo(
    applicant_name: str,
    facility_amount_gbp: float,
    facility_type: str,
    policy_assessment: Dict[str, Any],
    financial_ratios: Dict[str, Any],
    covenant_assessment: Dict[str, Any],
    existing_exposure: Dict[str, Any],
    portfolio_benchmarks: Dict[str, Any],
    agent_recommendation: str,
    risk_rating: int,
) -> Dict[str, Any]:
    """
    Draft a structured credit memorandum from agent findings.

    This tool synthesises all prior tool outputs into a structured memo
    suitable for Credit Committee review. The memo is stored as audit
    evidence per PRA SS1/23 Section 5.3.

    Args:
        applicant_name: Legal name of the borrower.
        facility_amount_gbp: Proposed facility amount (£).
        facility_type: Facility type string.
        policy_assessment: Output from check_credit_policy().
        financial_ratios: Output from calculate_ratios().
        covenant_assessment: Output from assess_covenants().
        existing_exposure: Output from fetch_t24_exposure().
        portfolio_benchmarks: Output from fetch_comparable_portfolio().
        agent_recommendation: APPROVE / DECLINE / REFER.
        risk_rating: Integer 1–10 (1 = lowest risk, 10 = highest risk).

    Returns:
        dict representing the structured credit memo.

    Raises:
        ValueError: If risk_rating is outside 1–10 range.
    """
    _require(applicant_name, "applicant_name", str)
    _require(facility_amount_gbp, "facility_amount_gbp", (int, float))
    _require_positive(facility_amount_gbp, "facility_amount_gbp")
    _require(risk_rating, "risk_rating", int)

    if not 1 <= risk_rating <= 10:
        raise ValueError(f"risk_rating must be between 1 and 10; got {risk_rating}.")

    valid_recommendations = ("APPROVE", "DECLINE", "REFER")
    if agent_recommendation not in valid_recommendations:
        raise ValueError(
            f"agent_recommendation must be one of {valid_recommendations}."
        )

    # Build key risks from policy breaches
    key_risks = []
    for breach in policy_assessment.get("breaches", []):
        key_risks.append(
            f"{breach['rule_name']}: {breach['description']} "
            f"(Severity: {breach['severity']})"
        )

    if not key_risks:
        key_risks = ["No material policy breaches identified."]

    # Build mitigants
    mitigants = []
    for covenant in covenant_assessment.get("recommended_covenants", []):
        mitigants.append(
            f"{covenant['covenant_name']} covenant at "
            f"{covenant['threshold']} ({covenant['direction']})"
        )

    # Build conditions
    conditions = [
        "Execution of facility agreement and security documentation.",
        "Receipt of satisfactory valuation report (if asset-backed).",
        "No material adverse change between credit approval and drawdown.",
        "Compliance certificate from CFO confirming covenant compliance at first drawdown.",
    ]

    if facility_amount_gbp >= 500_000:
        conditions.append(
            "Human-in-the-loop review completed by Senior Credit Officer "
            "(EU AI Act 2024 Article 14 — mandatory for facilities ≥ £500,000)."
        )

    # Next steps
    next_steps = [
        "Forward credit memo to Credit Committee for formal approval.",
        "Obtain legal review of facility documentation.",
        "Commission independent valuation (if applicable).",
        "Notify applicant of decision within 5 business days (FCA Consumer Duty).",
    ]

    memo_id = f"CM-{datetime.datetime.utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"

    return {
        "memo_id": memo_id,
        "applicant_name": applicant_name,
        "facility_amount_gbp": facility_amount_gbp,
        "facility_type": facility_type,
        "recommendation": agent_recommendation,
        "risk_rating": risk_rating,
        "risk_rating_scale": "1 (lowest) to 10 (highest)",
        "key_risks": key_risks,
        "mitigants": mitigants,
        "conditions": conditions,
        "next_steps": next_steps,
        "rationale": (
            f"Agent recommendation is {agent_recommendation} based on: "
            f"{len(policy_assessment.get('breaches', []))} policy breach(es), "
            f"leverage ratio of {financial_ratios.get('leverage', {}).get('net_debt_to_ebitda', 'N/A')}x, "
            f"interest cover of {financial_ratios.get('coverage', {}).get('interest_cover_ratio', 'N/A')}x. "
            f"Risk rating: {risk_rating}/10."
        ),
        "regulatory_references": [
            "PRA SS1/23 Section 5.3 — Credit memo as model audit evidence",
            "EU AI Act 2024 Annex III — High-risk AI system (credit scoring)",
            "EU AI Act 2024 Article 14 — Human oversight for material credit decisions",
            "FCA Consumer Duty PS22/9 — Customer outcome documentation",
        ],
        "model_registration": "MR-2026-037",
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "requires_human_review": facility_amount_gbp >= 500_000,
    }


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------

TOOL_REGISTRY: Dict[str, Any] = {
    "fetch_t24_exposure": fetch_t24_exposure,
    "check_credit_policy": check_credit_policy,
    "assess_covenants": assess_covenants,
    "calculate_ratios": calculate_ratios,
    "fetch_comparable_portfolio": fetch_comparable_portfolio,
    "draft_credit_memo": draft_credit_memo,
}


def get_tool_schemas() -> List[Dict[str, Any]]:
    """
    Return Gemini-compatible function declarations for all registered tools.

    These schemas are included in the LLM system prompt so the model can
    reason about which tool to call and with what parameters.
    """
    return [
        {
            "name": "fetch_t24_exposure",
            "description": (
                "Fetch the applicant's existing credit exposure from AWB's Temenos T24 "
                "core banking system. Call this first to understand existing indebtedness."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string", "description": "AWB T24 customer ID"},
                    "include_contingent": {"type": "boolean", "description": "Include contingent liabilities"},
                },
                "required": ["customer_id"],
            },
        },
        {
            "name": "check_credit_policy",
            "description": (
                "Evaluate applicant financials against AWB's credit policy rule set. "
                "Returns policy breaches and a APPROVE/REFER/DECLINE recommendation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "net_debt_gbp": {"type": "number"},
                    "ebitda_gbp": {"type": "number"},
                    "ebit_gbp": {"type": "number"},
                    "interest_expense_gbp": {"type": "number"},
                    "total_exposure_gbp": {"type": "number"},
                    "tangible_equity_gbp": {"type": "number"},
                    "total_assets_gbp": {"type": "number"},
                    "facility_type": {"type": "string"},
                },
                "required": [
                    "net_debt_gbp", "ebitda_gbp", "ebit_gbp",
                    "interest_expense_gbp", "total_exposure_gbp",
                    "tangible_equity_gbp", "total_assets_gbp",
                ],
            },
        },
        {
            "name": "assess_covenants",
            "description": "Recommend financial covenants for the proposed facility.",
            "parameters": {
                "type": "object",
                "properties": {
                    "facility_amount_gbp": {"type": "number"},
                    "facility_type": {"type": "string"},
                    "leverage_ratio": {"type": "number"},
                    "interest_cover_ratio": {"type": "number"},
                    "loan_to_value_pct": {"type": "number"},
                },
                "required": ["facility_amount_gbp", "facility_type", "leverage_ratio", "interest_cover_ratio"],
            },
        },
        {
            "name": "calculate_ratios",
            "description": "Calculate comprehensive financial ratios for credit analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "revenue_gbp": {"type": "number"},
                    "ebitda_gbp": {"type": "number"},
                    "ebit_gbp": {"type": "number"},
                    "net_profit_gbp": {"type": "number"},
                    "total_assets_gbp": {"type": "number"},
                    "total_liabilities_gbp": {"type": "number"},
                    "current_assets_gbp": {"type": "number"},
                    "current_liabilities_gbp": {"type": "number"},
                    "net_debt_gbp": {"type": "number"},
                    "interest_expense_gbp": {"type": "number"},
                    "capital_expenditure_gbp": {"type": "number"},
                },
                "required": [
                    "revenue_gbp", "ebitda_gbp", "ebit_gbp", "net_profit_gbp",
                    "total_assets_gbp", "total_liabilities_gbp",
                    "current_assets_gbp", "current_liabilities_gbp",
                    "net_debt_gbp", "interest_expense_gbp",
                ],
            },
        },
        {
            "name": "fetch_comparable_portfolio",
            "description": "Fetch AWB's comparable portfolio statistics for peer benchmarking.",
            "parameters": {
                "type": "object",
                "properties": {
                    "industry_code": {"type": "string", "description": "4-digit UK SIC 2007 code"},
                    "facility_type": {"type": "string"},
                    "facility_size_band": {"type": "string", "enum": ["SMALL", "MEDIUM", "LARGE"]},
                },
                "required": ["industry_code", "facility_type", "facility_size_band"],
            },
        },
        {
            "name": "draft_credit_memo",
            "description": (
                "Draft a structured credit memorandum from all agent findings. "
                "Call this last, after all analysis tools have been executed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "applicant_name": {"type": "string"},
                    "facility_amount_gbp": {"type": "number"},
                    "facility_type": {"type": "string"},
                    "policy_assessment": {"type": "object"},
                    "financial_ratios": {"type": "object"},
                    "covenant_assessment": {"type": "object"},
                    "existing_exposure": {"type": "object"},
                    "portfolio_benchmarks": {"type": "object"},
                    "agent_recommendation": {"type": "string", "enum": ["APPROVE", "DECLINE", "REFER"]},
                    "risk_rating": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": [
                    "applicant_name", "facility_amount_gbp", "facility_type",
                    "policy_assessment", "financial_ratios", "covenant_assessment",
                    "existing_exposure", "portfolio_benchmarks",
                    "agent_recommendation", "risk_rating",
                ],
            },
        },
    ]
