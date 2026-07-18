"""
credit_agent/policy_rules.py
AWB Credit Policy Rules — Avon & Wessex Bank plc
Agentic AI: Automated Credit Decision Workflow (Chapter 3)

Regulatory context:
- PRA SS1/23: Policy rules form part of the model's decision boundary;
  any change to thresholds constitutes a model change requiring validation.
- EU AI Act 2024 Annex III: Credit-scoring rules are high-risk AI components.
- CRR3 / Basel IV: Leverage ratio and concentration limits align with
  AWB's Internal Capital Adequacy Assessment Process (ICAAP).

Model registration: MR-2026-037 (AWB Automated Credit Decision Workflow)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
import datetime


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    """Policy breach severity levels — align with AWB Risk Appetite Statement."""
    LOW = "LOW"          # Advisory; does not block approval
    MEDIUM = "MEDIUM"    # Requires second-line sign-off
    HIGH = "HIGH"        # Automatic REFER to Credit Committee
    CRITICAL = "CRITICAL"  # Automatic DECLINE; cannot be overridden


class FacilityType(str, Enum):
    """Supported facility types for policy rule applicability."""
    TERM_LOAN = "TERM_LOAN"
    REVOLVING_CREDIT = "REVOLVING_CREDIT"
    OVERDRAFT = "OVERDRAFT"
    TRADE_FINANCE = "TRADE_FINANCE"
    MORTGAGE = "MORTGAGE"


# ---------------------------------------------------------------------------
# Policy Breach
# ---------------------------------------------------------------------------

@dataclass
class PolicyBreach:
    """
    Represents a single credit policy rule violation.

    Stored as part of the audit log (PRA SS1/23 Section 5.3) and included
    verbatim in the credit memo for regulatory traceability.
    """
    rule_name: str
    actual_value: float
    threshold: float
    severity: Severity
    description: str
    remediation_options: List[str] = field(default_factory=list)
    detected_at: datetime.datetime = field(default_factory=datetime.datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "rule_name": self.rule_name,
            "actual_value": round(self.actual_value, 4),
            "threshold": round(self.threshold, 4),
            "severity": self.severity.value,
            "description": self.description,
            "remediation_options": self.remediation_options,
            "detected_at": self.detected_at.isoformat(),
        }

    def __str__(self) -> str:
        return (
            f"[{self.severity.value}] {self.rule_name}: "
            f"actual={self.actual_value:.2f}, threshold={self.threshold:.2f} — "
            f"{self.description}"
        )


# ---------------------------------------------------------------------------
# Individual Policy Rules
# ---------------------------------------------------------------------------

@dataclass
class LeverageRatioRule:
    """
    Maximum net debt / EBITDA leverage ratio.

    AWB Credit Policy CP-2024-001, Section 4.2.
    Basel IV LR framework: CRR3 Article 429 (leverage exposure measure).
    """
    rule_name: str = "MAX_LEVERAGE_RATIO"
    max_leverage_ratio: float = 5.0          # Net debt / EBITDA
    warning_threshold: float = 4.0          # Trigger enhanced monitoring
    applicable_facility_types: List[FacilityType] = field(
        default_factory=lambda: [
            FacilityType.TERM_LOAN,
            FacilityType.REVOLVING_CREDIT,
        ]
    )

    def evaluate(
        self,
        net_debt: float,
        ebitda: float,
        facility_type: FacilityType = FacilityType.TERM_LOAN,
    ) -> Optional[PolicyBreach]:
        """
        Returns a PolicyBreach if the leverage ratio exceeds policy limits,
        or None if compliant.

        Args:
            net_debt: Total net debt in £ (existing + proposed facility).
            ebitda: Earnings before interest, taxes, depreciation, amortisation (£).
            facility_type: The type of credit facility being assessed.

        Returns:
            PolicyBreach or None.
        """
        if ebitda <= 0:
            return PolicyBreach(
                rule_name=self.rule_name,
                actual_value=float("inf"),
                threshold=self.max_leverage_ratio,
                severity=Severity.CRITICAL,
                description="Negative or zero EBITDA; leverage ratio undefined. DECLINE mandatory.",
                remediation_options=["Applicant must demonstrate positive EBITDA trajectory."],
            )

        ratio = net_debt / ebitda

        if ratio > self.max_leverage_ratio:
            severity = Severity.HIGH if ratio <= self.max_leverage_ratio * 1.25 else Severity.CRITICAL
            return PolicyBreach(
                rule_name=self.rule_name,
                actual_value=ratio,
                threshold=self.max_leverage_ratio,
                severity=severity,
                description=(
                    f"Net debt/EBITDA of {ratio:.2f}x exceeds AWB policy maximum of "
                    f"{self.max_leverage_ratio:.1f}x (CP-2024-001 Section 4.2)."
                ),
                remediation_options=[
                    "Reduce requested facility size.",
                    "Provide additional equity injection to reduce net debt.",
                    "Demonstrate EBITDA improvement plan with audited forecasts.",
                ],
            )

        if ratio > self.warning_threshold:
            return PolicyBreach(
                rule_name=self.rule_name + "_WARNING",
                actual_value=ratio,
                threshold=self.warning_threshold,
                severity=Severity.MEDIUM,
                description=(
                    f"Net debt/EBITDA of {ratio:.2f}x exceeds early-warning threshold of "
                    f"{self.warning_threshold:.1f}x. Enhanced monitoring covenants required."
                ),
                remediation_options=[
                    "Include leverage maintenance covenant at 4.5x.",
                    "Require quarterly financial reporting.",
                ],
            )

        return None


@dataclass
class InterestCoverRule:
    """
    Minimum interest cover ratio (ICR): EBIT / interest expense.

    AWB Credit Policy CP-2024-001, Section 4.3.
    Stressed ICR tested at +200bps rate increase per PRA stress testing guidance.
    """
    rule_name: str = "MIN_INTEREST_COVER"
    min_interest_cover: float = 2.0          # EBIT / interest expense
    stressed_min_interest_cover: float = 1.5  # Under +200bps stress

    def evaluate(
        self,
        ebit: float,
        interest_expense: float,
        stressed_interest_expense: Optional[float] = None,
    ) -> Optional[PolicyBreach]:
        """
        Evaluates base and stressed interest cover ratios.

        Args:
            ebit: Earnings before interest and taxes (£).
            interest_expense: Annual interest expense (£).
            stressed_interest_expense: Interest expense at +200bps (optional).

        Returns:
            PolicyBreach for the most severe breach, or None if compliant.
        """
        if interest_expense <= 0:
            # No debt; no interest cover breach possible
            return None

        icr = ebit / interest_expense

        if icr < self.min_interest_cover:
            severity = Severity.CRITICAL if icr < 1.0 else Severity.HIGH
            return PolicyBreach(
                rule_name=self.rule_name,
                actual_value=icr,
                threshold=self.min_interest_cover,
                severity=severity,
                description=(
                    f"Interest cover ratio of {icr:.2f}x is below AWB policy minimum of "
                    f"{self.min_interest_cover:.1f}x (CP-2024-001 Section 4.3)."
                ),
                remediation_options=[
                    "Reduce facility size to lower debt service requirements.",
                    "Provide interest rate hedging evidence (fixed-rate swap).",
                    "Demonstrate EBIT improvement plan.",
                ],
            )

        # Check stressed ICR if provided
        if stressed_interest_expense and stressed_interest_expense > 0:
            stressed_icr = ebit / stressed_interest_expense
            if stressed_icr < self.stressed_min_interest_cover:
                return PolicyBreach(
                    rule_name=self.rule_name + "_STRESSED",
                    actual_value=stressed_icr,
                    threshold=self.stressed_min_interest_cover,
                    severity=Severity.MEDIUM,
                    description=(
                        f"Stressed ICR (at +200bps) of {stressed_icr:.2f}x is below AWB "
                        f"stressed threshold of {self.stressed_min_interest_cover:.1f}x."
                    ),
                    remediation_options=[
                        "Consider fixed-rate facility or interest rate cap.",
                        "Include ICR maintenance covenant at 1.75x.",
                    ],
                )

        return None


@dataclass
class ConcentrationRule:
    """
    Maximum single-obligor concentration as % of AWB's total lending book.

    AWB Credit Policy CP-2024-002, Section 2.1.
    Aligned with CRR3 Article 395 (large exposure limit: 25% of Tier 1 capital).
    AWB's internal limit is more conservative at 15% of total lending book.
    """
    rule_name: str = "MAX_CONCENTRATION_PCT"
    max_concentration_pct: float = 15.0      # % of total lending book
    warning_pct: float = 10.0               # Early warning trigger
    total_lending_book_gbp: float = 28_000_000_000.0  # AWB lending book: £28B

    def evaluate(
        self,
        total_exposure_gbp: float,
    ) -> Optional[PolicyBreach]:
        """
        Evaluates single-obligor concentration risk.

        Args:
            total_exposure_gbp: Total proposed exposure in £ (existing + new facility).

        Returns:
            PolicyBreach or None.
        """
        concentration_pct = (total_exposure_gbp / self.total_lending_book_gbp) * 100.0

        if concentration_pct > self.max_concentration_pct:
            return PolicyBreach(
                rule_name=self.rule_name,
                actual_value=concentration_pct,
                threshold=self.max_concentration_pct,
                severity=Severity.HIGH,
                description=(
                    f"Single-obligor concentration of {concentration_pct:.2f}% exceeds AWB "
                    f"policy limit of {self.max_concentration_pct:.1f}% (CP-2024-002 Section 2.1)."
                ),
                remediation_options=[
                    "Reduce facility size.",
                    "Syndicate a portion of the facility to reduce AWB's net hold.",
                    "Obtain credit risk insurance for the excess exposure.",
                ],
            )

        if concentration_pct > self.warning_pct:
            return PolicyBreach(
                rule_name=self.rule_name + "_WARNING",
                actual_value=concentration_pct,
                threshold=self.warning_pct,
                severity=Severity.LOW,
                description=(
                    f"Single-obligor concentration of {concentration_pct:.2f}% exceeds "
                    f"early-warning threshold of {self.warning_pct:.1f}%."
                ),
                remediation_options=[
                    "Monitor concentration quarterly.",
                    "Consider syndication for future drawdowns.",
                ],
            )

        return None


@dataclass
class MinimumEquityRule:
    """
    Minimum tangible net worth / equity requirement.

    AWB Credit Policy CP-2024-001, Section 4.4.
    Ensures borrower has meaningful skin-in-the-game.
    """
    rule_name: str = "MIN_TANGIBLE_EQUITY"
    min_equity_gbp: float = 1_000_000.0      # £1M minimum tangible net worth
    min_equity_ratio: float = 0.20           # 20% of total assets

    def evaluate(
        self,
        tangible_equity_gbp: float,
        total_assets_gbp: float,
    ) -> Optional[PolicyBreach]:
        """Evaluate minimum equity requirements."""
        if tangible_equity_gbp < self.min_equity_gbp:
            return PolicyBreach(
                rule_name=self.rule_name + "_ABSOLUTE",
                actual_value=tangible_equity_gbp,
                threshold=self.min_equity_gbp,
                severity=Severity.HIGH,
                description=(
                    f"Tangible net worth of £{tangible_equity_gbp:,.0f} is below AWB "
                    f"minimum of £{self.min_equity_gbp:,.0f} (CP-2024-001 Section 4.4)."
                ),
                remediation_options=[
                    "Equity injection from shareholders.",
                    "Personal guarantee from directors.",
                ],
            )

        if total_assets_gbp > 0:
            equity_ratio = tangible_equity_gbp / total_assets_gbp
            if equity_ratio < self.min_equity_ratio:
                return PolicyBreach(
                    rule_name=self.rule_name + "_RATIO",
                    actual_value=equity_ratio * 100,
                    threshold=self.min_equity_ratio * 100,
                    severity=Severity.MEDIUM,
                    description=(
                        f"Equity ratio of {equity_ratio*100:.1f}% is below AWB minimum of "
                        f"{self.min_equity_ratio*100:.0f}% (CP-2024-001 Section 4.4)."
                    ),
                    remediation_options=[
                        "Additional equity injection.",
                        "Asset disposal to reduce balance sheet leverage.",
                    ],
                )

        return None


# ---------------------------------------------------------------------------
# Policy Rule Set — aggregates all rules
# ---------------------------------------------------------------------------

@dataclass
class AWBCreditPolicyRuleSet:
    """
    Complete AWB credit policy rule set for automated evaluation.

    This dataclass is the single source of truth for credit policy thresholds.
    Any change to thresholds must be approved via the Model Change Management
    process (PRA SS1/23 Section 7) and documented in the model changelog.

    Policy document: AWB-CREDIT-POLICY-2024 v3.1
    Effective date: 01 January 2024
    Next review: 01 January 2025
    Approved by: Credit Committee (CC-2023-142)
    """
    leverage_rule: LeverageRatioRule = field(default_factory=LeverageRatioRule)
    interest_cover_rule: InterestCoverRule = field(default_factory=InterestCoverRule)
    concentration_rule: ConcentrationRule = field(default_factory=ConcentrationRule)
    equity_rule: MinimumEquityRule = field(default_factory=MinimumEquityRule)

    # Summary thresholds (for documentation and reporting)
    @property
    def summary(self) -> dict:
        return {
            "max_leverage_ratio": self.leverage_rule.max_leverage_ratio,
            "min_interest_cover": self.interest_cover_rule.min_interest_cover,
            "max_concentration_pct": self.concentration_rule.max_concentration_pct,
            "min_equity_gbp": self.equity_rule.min_equity_gbp,
            "policy_document": "AWB-CREDIT-POLICY-2024 v3.1",
            "effective_date": "2024-01-01",
        }

    def evaluate_all(
        self,
        net_debt: float,
        ebitda: float,
        ebit: float,
        interest_expense: float,
        total_exposure_gbp: float,
        tangible_equity_gbp: float,
        total_assets_gbp: float,
        stressed_interest_expense: Optional[float] = None,
        facility_type: FacilityType = FacilityType.TERM_LOAN,
    ) -> List[PolicyBreach]:
        """
        Evaluate all credit policy rules and return a list of breaches.

        Args:
            net_debt: Total net debt including proposed facility (£).
            ebitda: EBITDA (£).
            ebit: EBIT (£).
            interest_expense: Annual interest expense (£).
            total_exposure_gbp: Total AWB exposure post-facility (£).
            tangible_equity_gbp: Borrower's tangible net worth (£).
            total_assets_gbp: Borrower's total assets (£).
            stressed_interest_expense: Interest at +200bps stress (£, optional).
            facility_type: Facility type for rule applicability.

        Returns:
            List of PolicyBreach objects (empty if all rules pass).
        """
        breaches: List[PolicyBreach] = []

        result = self.leverage_rule.evaluate(net_debt, ebitda, facility_type)
        if result:
            breaches.append(result)

        result = self.interest_cover_rule.evaluate(ebit, interest_expense, stressed_interest_expense)
        if result:
            breaches.append(result)

        result = self.concentration_rule.evaluate(total_exposure_gbp)
        if result:
            breaches.append(result)

        result = self.equity_rule.evaluate(tangible_equity_gbp, total_assets_gbp)
        if result:
            breaches.append(result)

        return breaches

    def has_blocking_breach(self, breaches: List[PolicyBreach]) -> bool:
        """Returns True if any breach has CRITICAL or HIGH severity."""
        return any(b.severity in (Severity.CRITICAL, Severity.HIGH) for b in breaches)


# ---------------------------------------------------------------------------
# Module-level default instance
# ---------------------------------------------------------------------------

DEFAULT_POLICY = AWBCreditPolicyRuleSet()
