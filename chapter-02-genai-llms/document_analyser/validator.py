"""
document_analyser/validator.py
AWB Credit Document Analyser — Financial Data Validator

Cross-validates extracted financial data and detects red flags per AWB
credit policy. Applies EBA margin of conservatism to uncertain debt figures.

Red flag thresholds (AWB Credit Policy v3.2, January 2026):
  - Revenue decline > 20% YoY → P2 flag (material deterioration)
  - Leverage ratio > 5.0x → P1 flag (exceeds AWB policy maximum)
  - Interest cover < 2.0x → P1 flag (covenant breach risk)
  - Negative equity → P1 flag (insolvency risk)
  - Current ratio < 1.0x → P2 flag (short-term liquidity concern)

EBA Guidelines on IRB (margin of conservatism):
  Apply +10% to net debt where extraction confidence < 0.80 to ensure
  capital calculations use conservative estimates.

Regulatory:
  - PRA SS1/23 MR-2026-035: validation is part of model governance
  - EBA Guidelines on IRB credit risk models: Section 4.3 (margin of conservatism)
  - Basel III/CRR3: conservative bias in credit risk parameter estimation

Author: AWB AI Programme (AWB-AI-2025)
Chapter: 2 — LLM Foundations for Financial Document Analysis
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

from document_analyser.extractor import FieldExtraction, FinancialSummary, RangeConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Red flag taxonomy
# ---------------------------------------------------------------------------

class RedFlagSeverity(str, Enum):
    P1 = "P1"   # Immediate escalation to Credit Committee
    P2 = "P2"   # Senior analyst review required
    P3 = "P3"   # Note for file — monitor next reporting period


@dataclass
class RedFlag:
    """A single credit risk red flag identified during validation."""
    flag_code: str
    severity: RedFlagSeverity
    description: str
    metric_name: str
    actual_value: float | None
    threshold: float
    threshold_direction: str   # "above" | "below"
    awb_policy_reference: str
    remediation: str


@dataclass
class ValidationResult:
    """
    Complete validation output for one FinancialSummary.

    Passed to analyst for review before credit decision.
    PRA SS1/23: validation result stored alongside extraction audit record.
    """
    document_id: str
    validation_passed: bool            # False if any P1 flag raised
    red_flags: list[RedFlag] = field(default_factory=list)
    cross_validation_issues: list[str] = field(default_factory=list)
    conservatism_adjustments: list[str] = field(default_factory=list)
    adjusted_net_debt: float | None = None   # Post-conservatism-adjustment value
    adjusted_leverage: float | None = None   # Recalculated with adjusted net debt
    analyst_notes: str = ""

    @property
    def p1_flags(self) -> list[RedFlag]:
        return [f for f in self.red_flags if f.severity == RedFlagSeverity.P1]

    @property
    def p2_flags(self) -> list[RedFlag]:
        return [f for f in self.red_flags if f.severity == RedFlagSeverity.P2]


# ---------------------------------------------------------------------------
# Red flag detection
# ---------------------------------------------------------------------------

# AWB credit policy thresholds
MAX_LEVERAGE = 5.0          # AWB Credit Policy v3.2 §4.1
MIN_INTEREST_COVER = 2.0    # AWB Credit Policy v3.2 §4.2
MAX_REVENUE_DECLINE = 0.20  # 20% YoY
MIN_CURRENT_RATIO = 1.0     # Short-term liquidity floor


def detect_red_flags(
    current: FinancialSummary,
    prior_year_revenue: float | None = None,
) -> list[RedFlag]:
    """
    Detect credit risk red flags in an extracted FinancialSummary.

    Args:
        current:            Extracted financial summary for current period.
        prior_year_revenue: Prior year revenue (£000s) for YoY comparison.
                            If None, revenue trend check is skipped.

    Returns:
        List of RedFlag instances, ordered by severity (P1 first).
    """
    flags: list[RedFlag] = []

    # --- P1: Leverage ratio exceeds policy maximum ---
    if (current.leverage_ratio.value is not None
            and isinstance(current.leverage_ratio.value, (int, float))):
        lev = float(current.leverage_ratio.value)
        if lev > MAX_LEVERAGE:
            flags.append(RedFlag(
                flag_code="CR-LEV-001",
                severity=RedFlagSeverity.P1,
                description=f"Leverage ratio {lev:.1f}x exceeds AWB policy maximum of {MAX_LEVERAGE}x",
                metric_name="leverage_ratio",
                actual_value=lev,
                threshold=MAX_LEVERAGE,
                threshold_direction="above",
                awb_policy_reference="AWB Credit Policy v3.2 §4.1",
                remediation=(
                    "Escalate to Credit Committee. Obtain management plan for deleveraging. "
                    "Consider covenant package requiring leverage < 5x within 12 months."
                ),
            ))
            logger.warning("P1 RED FLAG: leverage_ratio", extra={"value": lev})

    # --- P1: Interest cover below minimum ---
    if (current.interest_cover.value is not None
            and isinstance(current.interest_cover.value, (int, float))):
        ic = float(current.interest_cover.value)
        if ic < MIN_INTEREST_COVER:
            flags.append(RedFlag(
                flag_code="CR-IC-001",
                severity=RedFlagSeverity.P1,
                description=f"Interest cover {ic:.1f}x below AWB policy minimum of {MIN_INTEREST_COVER}x",
                metric_name="interest_cover",
                actual_value=ic,
                threshold=MIN_INTEREST_COVER,
                threshold_direction="below",
                awb_policy_reference="AWB Credit Policy v3.2 §4.2",
                remediation=(
                    "Escalate to Credit Committee. Request 3-year cash flow projections. "
                    "Consider interest reserve or cash sweep covenant."
                ),
            ))
            logger.warning("P1 RED FLAG: interest_cover", extra={"value": ic})

    # --- P1: Negative equity (total assets < total liabilities proxy) ---
    if (current.net_debt.value is not None and current.total_assets.value is not None
            and isinstance(current.net_debt.value, (int, float))
            and isinstance(current.total_assets.value, (int, float))):
        if float(current.net_debt.value) > float(current.total_assets.value):
            flags.append(RedFlag(
                flag_code="CR-NEQ-001",
                severity=RedFlagSeverity.P1,
                description="Net debt exceeds total assets — potential negative equity",
                metric_name="net_debt_vs_total_assets",
                actual_value=float(current.net_debt.value),
                threshold=float(current.total_assets.value),
                threshold_direction="above",
                awb_policy_reference="AWB Credit Policy v3.2 §4.5",
                remediation=(
                    "Immediate escalation required. Obtain full balance sheet. "
                    "Legal review of security position."
                ),
            ))
            logger.warning("P1 RED FLAG: negative equity proxy")

    # --- P2: Revenue decline > 20% YoY ---
    if (prior_year_revenue is not None
            and current.revenue.value is not None
            and isinstance(current.revenue.value, (int, float))
            and prior_year_revenue > 0):
        decline = (prior_year_revenue - float(current.revenue.value)) / prior_year_revenue
        if decline > MAX_REVENUE_DECLINE:
            flags.append(RedFlag(
                flag_code="CR-REV-001",
                severity=RedFlagSeverity.P2,
                description=(
                    f"Revenue declined {decline:.1%} YoY "
                    f"(from £{prior_year_revenue:,.0f}k to £{current.revenue.value:,.0f}k)"
                ),
                metric_name="revenue_yoy_decline",
                actual_value=decline,
                threshold=MAX_REVENUE_DECLINE,
                threshold_direction="above",
                awb_policy_reference="AWB Credit Policy v3.2 §4.3",
                remediation=(
                    "Obtain management commentary on revenue decline. "
                    "Review forward order book. Consider covenant headroom stress test."
                ),
            ))
            logger.warning("P2 RED FLAG: revenue_decline", extra={"decline_pct": decline})

    # --- P2: Current ratio below 1.0x ---
    if (current.current_ratio.value is not None
            and isinstance(current.current_ratio.value, (int, float))):
        cr = float(current.current_ratio.value)
        if cr < MIN_CURRENT_RATIO:
            flags.append(RedFlag(
                flag_code="CR-LIQ-001",
                severity=RedFlagSeverity.P2,
                description=f"Current ratio {cr:.2f}x below minimum {MIN_CURRENT_RATIO}x — liquidity concern",
                metric_name="current_ratio",
                actual_value=cr,
                threshold=MIN_CURRENT_RATIO,
                threshold_direction="below",
                awb_policy_reference="AWB Credit Policy v3.2 §4.4",
                remediation=(
                    "Obtain detailed working capital analysis. "
                    "Consider revolving credit facility to support liquidity."
                ),
            ))
            logger.warning("P2 RED FLAG: current_ratio", extra={"value": cr})

    # Sort: P1 first, then P2, then P3
    severity_order = {RedFlagSeverity.P1: 0, RedFlagSeverity.P2: 1, RedFlagSeverity.P3: 2}
    flags.sort(key=lambda f: severity_order[f.severity])
    return flags


# ---------------------------------------------------------------------------
# Cross-validation (extracted vs. stated ratios)
# ---------------------------------------------------------------------------

def cross_validate_ratios(summary: FinancialSummary) -> list[str]:
    """
    Check calculated ratios against extracted ratios.
    Detects transcription errors or model extraction mistakes.

    Returns list of discrepancy descriptions (empty = no issues found).
    """
    issues: list[str] = []
    tolerance = 0.05   # 5% tolerance for rounding differences

    # Check leverage: extracted vs. calculated (net_debt / ebitda)
    if (summary.leverage_ratio.value is not None
            and summary.net_debt.value is not None
            and summary.ebitda.value is not None
            and isinstance(summary.ebitda.value, (int, float))
            and float(summary.ebitda.value) != 0):
        extracted_lev = float(summary.leverage_ratio.value)
        calculated_lev = float(summary.net_debt.value) / float(summary.ebitda.value)
        if abs(extracted_lev - calculated_lev) / max(abs(calculated_lev), 0.01) > tolerance:
            issues.append(
                f"LEVERAGE DISCREPANCY: extracted {extracted_lev:.2f}x vs. "
                f"calculated {calculated_lev:.2f}x (net_debt/EBITDA). "
                f"Verify source document."
            )

    # Check EBITDA margin: extracted vs. calculated (ebitda / revenue)
    if (summary.ebitda_margin_pct.value is not None
            and summary.ebitda.value is not None
            and summary.revenue.value is not None
            and isinstance(summary.revenue.value, (int, float))
            and float(summary.revenue.value) != 0):
        extracted_margin = float(summary.ebitda_margin_pct.value)
        calculated_margin = (float(summary.ebitda.value) / float(summary.revenue.value)) * 100
        if abs(extracted_margin - calculated_margin) > 2.0:   # 2pp tolerance
            issues.append(
                f"EBITDA MARGIN DISCREPANCY: extracted {extracted_margin:.1f}% vs. "
                f"calculated {calculated_margin:.1f}%. Verify source document."
            )

    # Check current ratio: extracted vs. calculated (current_assets / current_liabilities)
    if (summary.current_ratio.value is not None
            and summary.current_assets.value is not None
            and summary.current_liabilities.value is not None
            and isinstance(summary.current_liabilities.value, (int, float))
            and float(summary.current_liabilities.value) != 0):
        extracted_cr = float(summary.current_ratio.value)
        calculated_cr = (
            float(summary.current_assets.value) / float(summary.current_liabilities.value)
        )
        if abs(extracted_cr - calculated_cr) / max(abs(calculated_cr), 0.01) > tolerance:
            issues.append(
                f"CURRENT RATIO DISCREPANCY: extracted {extracted_cr:.2f}x vs. "
                f"calculated {calculated_cr:.2f}x. Verify source document."
            )

    if issues:
        logger.warning("Cross-validation issues found", extra={"issues": issues})
    return issues


# ---------------------------------------------------------------------------
# EBA margin of conservatism
# ---------------------------------------------------------------------------

CONSERVATISM_UPLIFT = 0.10   # 10% uplift on uncertain debt figures (EBA Guidelines §4.3)
CONSERVATISM_CONFIDENCE_THRESHOLD = 0.80


def apply_conservatism(summary: FinancialSummary) -> tuple[float | None, float | None, list[str]]:
    """
    Apply EBA margin of conservatism to net debt where extraction confidence
    is below threshold.

    Returns:
        (adjusted_net_debt, adjusted_leverage, list_of_adjustment_descriptions)

    EBA reference: EBA/GL/2017/16 §4.3 — margin of conservatism for uncertain inputs.
    """
    adjustments: list[str] = []
    adjusted_net_debt = None
    adjusted_leverage = None

    if (summary.net_debt.value is not None
            and isinstance(summary.net_debt.value, (int, float))):
        original_debt = float(summary.net_debt.value)

        if summary.net_debt.confidence < CONSERVATISM_CONFIDENCE_THRESHOLD:
            adjusted_net_debt = original_debt * (1 + CONSERVATISM_UPLIFT)
            adjustments.append(
                f"Net debt adjusted from £{original_debt:,.0f}k to "
                f"£{adjusted_net_debt:,.0f}k (+{CONSERVATISM_UPLIFT:.0%} EBA conservatism "
                f"— extraction confidence {summary.net_debt.confidence:.2f} < "
                f"{CONSERVATISM_CONFIDENCE_THRESHOLD})"
            )
            logger.info(
                "EBA conservatism applied to net debt",
                extra={
                    "original": original_debt,
                    "adjusted": adjusted_net_debt,
                    "confidence": summary.net_debt.confidence,
                },
            )

            # Recalculate leverage with adjusted net debt
            if (summary.ebitda.value is not None
                    and isinstance(summary.ebitda.value, (int, float))
                    and float(summary.ebitda.value) != 0):
                adjusted_leverage = adjusted_net_debt / float(summary.ebitda.value)
                adjustments.append(
                    f"Leverage recalculated: {adjusted_leverage:.2f}x "
                    f"(using adjusted net debt)"
                )

    return adjusted_net_debt, adjusted_leverage, adjustments


# ---------------------------------------------------------------------------
# Main validation function
# ---------------------------------------------------------------------------

def validate_extraction(
    summary: FinancialSummary,
    prior_year_revenue: float | None = None,
) -> ValidationResult:
    """
    Run full validation suite on an extracted FinancialSummary.

    Args:
        summary:            Extracted FinancialSummary from extractor.py.
        prior_year_revenue: Prior year revenue (£000s) for YoY trend check.

    Returns:
        ValidationResult with red flags, cross-validation issues, and adjustments.
    """
    red_flags = detect_red_flags(summary, prior_year_revenue)
    cross_val_issues = cross_validate_ratios(summary)
    adj_debt, adj_leverage, conservatism_notes = apply_conservatism(summary)

    # validation_passed = False if any P1 flag raised
    p1_count = sum(1 for f in red_flags if f.severity == RedFlagSeverity.P1)
    passed = p1_count == 0

    result = ValidationResult(
        document_id=summary.document_id,
        validation_passed=passed,
        red_flags=red_flags,
        cross_validation_issues=cross_val_issues,
        conservatism_adjustments=conservatism_notes,
        adjusted_net_debt=adj_debt,
        adjusted_leverage=adj_leverage,
        analyst_notes=(
            f"PRA SS1/23 MR-2026-035 validation complete. "
            f"{len(red_flags)} red flag(s) identified "
            f"({p1_count} P1). "
            f"Human oversight required before credit decision."
        ),
    )

    logger.info(
        "Validation complete",
        extra={
            "document_id": summary.document_id,
            "passed": passed,
            "p1_flags": p1_count,
            "p2_flags": len(result.p2_flags),
            "cross_val_issues": len(cross_val_issues),
        },
    )

    return result
