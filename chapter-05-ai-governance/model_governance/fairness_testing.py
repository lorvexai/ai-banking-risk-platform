"""
model_governance/fairness_testing.py
AWB AI Governance Platform — Fairness Testing
Chapter 5: Model Risk Management (PRA SS1/23)

Implements algorithmic fairness testing aligned with:
- UK Equality Act 2010: nine protected characteristics
- EU AI Act 2024 Article 10(5): training data quality and bias avoidance
- EU AI Act 2024 Recital 44: fundamental rights impact assessment
- FCA Consumer Duty PS22/3: fair outcomes across all customer segments

Fairness metrics implemented:
1. Demographic Parity: approval rates equal across groups
2. Equal Opportunity: true positive rates equal across groups
3. Disparate Impact Ratio: ratio of group approval rate to reference group rate
   (4/5ths rule: ratio < 0.80 triggers review)

Protected characteristics (UK Equality Act 2010 Section 4):
- Age, Disability, Gender reassignment, Marriage/civil partnership,
  Pregnancy/maternity, Race, Religion or belief, Sex, Sexual orientation
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DISPARATE_IMPACT_THRESHOLD = 0.80   # 4/5ths rule; ratios below this trigger review
DEMOGRAPHIC_PARITY_TOLERANCE = 0.05  # ±5 percentage points tolerance


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class FairnessMetric(str, Enum):
    DEMOGRAPHIC_PARITY = "DEMOGRAPHIC_PARITY"
    EQUAL_OPPORTUNITY = "EQUAL_OPPORTUNITY"
    DISPARATE_IMPACT = "DISPARATE_IMPACT"
    CALIBRATION = "CALIBRATION"


class FairnessResult(str, Enum):
    PASS = "PASS"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"   # Below threshold but not prohibited
    FAIL = "FAIL"                         # Material disparity requiring remediation


# ---------------------------------------------------------------------------
# Group statistics
# ---------------------------------------------------------------------------

@dataclass
class GroupStatistics:
    """
    Model outcome statistics for a single demographic group.

    approval_rate: proportion of applicants approved
    true_positive_rate: proportion of actual positives correctly approved (equal opportunity)
    false_positive_rate: proportion of actual negatives incorrectly approved
    sample_size: number of observations (for statistical significance)
    """
    group_name: str
    characteristic: str          # e.g. "age_group", "gender", "ethnicity"
    approval_rate: float
    true_positive_rate: Optional[float] = None
    false_positive_rate: Optional[float] = None
    sample_size: int = 0

    def __post_init__(self) -> None:
        for attr, val in [
            ("approval_rate", self.approval_rate),
        ]:
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{attr} must be between 0 and 1; got {val}.")


# ---------------------------------------------------------------------------
# Fairness test result
# ---------------------------------------------------------------------------

@dataclass
class FairnessTestResult:
    """Result for a single fairness metric and characteristic."""
    metric: FairnessMetric
    characteristic: str
    reference_group: str
    result: FairnessResult
    worst_ratio: float          # Minimum disparate impact ratio (or 1 - max gap)
    failing_groups: List[str]   # Groups below the threshold
    group_values: Dict[str, float]  # Metric value per group
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric.value,
            "characteristic": self.characteristic,
            "reference_group": self.reference_group,
            "result": self.result.value,
            "worst_ratio": round(self.worst_ratio, 4),
            "failing_groups": self.failing_groups,
            "group_values": {k: round(v, 4) for k, v in self.group_values.items()},
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Fairness report
# ---------------------------------------------------------------------------

@dataclass
class FairnessReport:
    """
    Comprehensive fairness assessment report for an AI model.

    Produced as part of the model validation exercise (PRA SS1/23 Section 3.2)
    and stored as audit evidence for EU AI Act fundamental rights impact
    assessment obligations.

    Remediation is required if:
    - Any disparate impact ratio < 0.80 (4/5ths rule)
    - Any demographic parity gap > DEMOGRAPHIC_PARITY_TOLERANCE
    """
    model_id: str
    model_name: str
    assessment_date: datetime.date = field(default_factory=datetime.date.today)
    assessor: str = ""
    test_results: List[FairnessTestResult] = field(default_factory=list)
    overall_result: FairnessResult = FairnessResult.PASS
    remediation_required: bool = False
    remediation_actions: List[str] = field(default_factory=list)
    regulatory_references: List[str] = field(default_factory=list)
    report_id: str = ""

    def __post_init__(self) -> None:
        if not self.report_id:
            self.report_id = (
                f"FAIR-{self.model_id}-"
                f"{self.assessment_date.strftime('%Y%m%d')}"
            )
        if not self.regulatory_references:
            self.regulatory_references = [
                "UK Equality Act 2010 — Protected Characteristics",
                "EU AI Act 2024 Article 10(5) — Training Data Bias Avoidance",
                "EU AI Act 2024 Recital 44 — Fundamental Rights Impact Assessment",
                "FCA Consumer Duty PS22/3 — Fair Outcomes Across Customer Segments",
                "PRA SS1/23 Section 3.2(e) — Benchmarking and Fairness Validation",
            ]

    @property
    def failing_tests(self) -> List[FairnessTestResult]:
        return [t for t in self.test_results if t.result == FairnessResult.FAIL]

    @property
    def review_required_tests(self) -> List[FairnessTestResult]:
        return [t for t in self.test_results if t.result == FairnessResult.REVIEW_REQUIRED]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "model_id": self.model_id,
            "model_name": self.model_name,
            "assessment_date": self.assessment_date.isoformat(),
            "overall_result": self.overall_result.value,
            "remediation_required": self.remediation_required,
            "test_count": len(self.test_results),
            "failing_tests": [t.to_dict() for t in self.failing_tests],
            "remediation_actions": self.remediation_actions,
            "regulatory_references": self.regulatory_references,
        }


# ---------------------------------------------------------------------------
# Fairness tests
# ---------------------------------------------------------------------------

def run_disparate_impact_test(
    group_stats: List[GroupStatistics],
    reference_group: str,
    characteristic: str,
) -> FairnessTestResult:
    """
    Disparate Impact Ratio test (4/5ths rule).

    Any group with an approval rate < 80% of the reference group's rate
    fails the 4/5ths rule and requires investigation and remediation.

    Args:
        group_stats: Statistics per demographic group.
        reference_group: Name of the reference group (typically majority group).
        characteristic: Name of the protected characteristic being tested.

    Returns:
        FairnessTestResult.
    """
    stats_by_name = {s.group_name: s for s in group_stats}

    if reference_group not in stats_by_name:
        raise ValueError(f"Reference group '{reference_group}' not found in group_stats.")

    reference_rate = stats_by_name[reference_group].approval_rate
    if reference_rate <= 0:
        raise ValueError("Reference group approval rate must be positive.")

    ratios = {}
    for name, stats in stats_by_name.items():
        if name == reference_group:
            ratios[name] = 1.0
        else:
            ratios[name] = stats.approval_rate / reference_rate

    failing_groups = [g for g, r in ratios.items() if g != reference_group and r < DISPARATE_IMPACT_THRESHOLD]
    worst_ratio = min(ratios.values())

    if failing_groups:
        result = FairnessResult.FAIL
        notes = (
            f"Disparate impact ratio below 0.80 (4/5ths rule) for groups: {failing_groups}. "
            f"Worst ratio: {worst_ratio:.3f}. "
            f"EU AI Act: fundamental rights impact assessment required. "
            f"UK Equality Act 2010: investigate potential indirect discrimination."
        )
    elif worst_ratio < 0.90:
        result = FairnessResult.REVIEW_REQUIRED
        notes = (
            f"Disparate impact ratio {worst_ratio:.3f} is within 4/5ths rule "
            f"but warrants monitoring. Enhanced review recommended."
        )
    else:
        result = FairnessResult.PASS
        notes = (
            f"All groups meet the 4/5ths rule (minimum ratio: {worst_ratio:.3f}). "
            f"No fairness concerns identified for {characteristic}."
        )

    return FairnessTestResult(
        metric=FairnessMetric.DISPARATE_IMPACT,
        characteristic=characteristic,
        reference_group=reference_group,
        result=result,
        worst_ratio=worst_ratio,
        failing_groups=failing_groups,
        group_values={name: round(r, 4) for name, r in ratios.items()},
        notes=notes,
    )


def run_demographic_parity_test(
    group_stats: List[GroupStatistics],
    reference_group: str,
    characteristic: str,
    tolerance: float = DEMOGRAPHIC_PARITY_TOLERANCE,
) -> FairnessTestResult:
    """
    Demographic Parity test: approval rates within tolerance across groups.

    Args:
        group_stats: Statistics per demographic group.
        reference_group: Name of the reference group.
        characteristic: Name of the protected characteristic.
        tolerance: Absolute tolerance in approval rate (default ±5pp).

    Returns:
        FairnessTestResult.
    """
    stats_by_name = {s.group_name: s for s in group_stats}

    if reference_group not in stats_by_name:
        raise ValueError(f"Reference group '{reference_group}' not found.")

    reference_rate = stats_by_name[reference_group].approval_rate
    gaps: Dict[str, float] = {}

    for name, stats in stats_by_name.items():
        if name != reference_group:
            gaps[name] = abs(stats.approval_rate - reference_rate)

    failing_groups = [g for g, gap in gaps.items() if gap > tolerance]
    max_gap = max(gaps.values()) if gaps else 0.0

    if failing_groups:
        result = FairnessResult.FAIL
        worst_ratio = 1.0 - max_gap
        notes = (
            f"Demographic parity gap exceeds ±{tolerance*100:.0f}pp tolerance for: "
            f"{failing_groups}. Maximum gap: {max_gap*100:.1f}pp."
        )
    else:
        result = FairnessResult.PASS
        worst_ratio = 1.0 - max_gap
        notes = (
            f"Demographic parity maintained: maximum gap {max_gap*100:.1f}pp "
            f"within ±{tolerance*100:.0f}pp tolerance."
        )

    approval_rates = {s.group_name: round(s.approval_rate, 4) for s in group_stats}

    return FairnessTestResult(
        metric=FairnessMetric.DEMOGRAPHIC_PARITY,
        characteristic=characteristic,
        reference_group=reference_group,
        result=result,
        worst_ratio=round(worst_ratio, 4),
        failing_groups=failing_groups,
        group_values=approval_rates,
        notes=notes,
    )


def run_equal_opportunity_test(
    group_stats: List[GroupStatistics],
    reference_group: str,
    characteristic: str,
) -> FairnessTestResult:
    """
    Equal Opportunity test: true positive rates within acceptable range.

    A model satisfies equal opportunity if the probability of a truly qualified
    applicant being approved is similar across groups.

    Args:
        group_stats: Statistics with true_positive_rate populated per group.
        reference_group: Reference group name.
        characteristic: Protected characteristic name.

    Returns:
        FairnessTestResult.

    Raises:
        ValueError: If true_positive_rate is not populated for any group.
    """
    for s in group_stats:
        if s.true_positive_rate is None:
            raise ValueError(
                f"true_positive_rate must be provided for group '{s.group_name}' "
                f"to run Equal Opportunity test."
            )

    stats_by_name = {s.group_name: s for s in group_stats}

    if reference_group not in stats_by_name:
        raise ValueError(f"Reference group '{reference_group}' not found.")

    reference_tpr = stats_by_name[reference_group].true_positive_rate
    if reference_tpr <= 0:
        raise ValueError("Reference group true positive rate must be positive.")

    ratios = {
        name: s.true_positive_rate / reference_tpr
        for name, s in stats_by_name.items()
        if name != reference_group
    }
    ratios[reference_group] = 1.0

    failing_groups = [g for g, r in ratios.items() if g != reference_group and r < DISPARATE_IMPACT_THRESHOLD]
    worst_ratio = min(ratios.values())

    if failing_groups:
        result = FairnessResult.FAIL
        notes = (
            f"Equal opportunity violated for groups: {failing_groups}. "
            f"True positive rates significantly lower than reference group. "
            f"UK Equality Act 2010: investigate indirect discrimination."
        )
    else:
        result = FairnessResult.PASS
        notes = f"Equal opportunity maintained across all groups (minimum ratio: {worst_ratio:.3f})."

    tpr_values = {name: round(s.true_positive_rate, 4) for name, s in stats_by_name.items()}

    return FairnessTestResult(
        metric=FairnessMetric.EQUAL_OPPORTUNITY,
        characteristic=characteristic,
        reference_group=reference_group,
        result=result,
        worst_ratio=worst_ratio,
        failing_groups=failing_groups,
        group_values=tpr_values,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Fairness assessment runner
# ---------------------------------------------------------------------------

class FairnessAssessor:
    """
    Orchestrates the full fairness assessment for a model.

    Runs disparate impact, demographic parity, and (optionally) equal
    opportunity tests across all configured protected characteristics.

    EU AI Act Article 10(5): bias monitoring and mitigation is mandatory
    for high-risk AI systems. Results stored as technical documentation
    per Article 11.
    """

    def run_assessment(
        self,
        model_id: str,
        model_name: str,
        characteristics: Dict[str, Tuple[List[GroupStatistics], str]],
        assessor: str = "Model Risk Function",
    ) -> FairnessReport:
        """
        Run a complete fairness assessment.

        Args:
            model_id: Model registration ID.
            model_name: Model name.
            characteristics: Dict of characteristic_name → (group_stats, reference_group).
            assessor: Name of the assessor (second-line independence).

        Returns:
            FairnessReport with all test results and overall result.
        """
        all_tests: List[FairnessTestResult] = []

        for char_name, (group_stats, reference_group) in characteristics.items():
            # Always run disparate impact
            all_tests.append(
                run_disparate_impact_test(group_stats, reference_group, char_name)
            )
            # Run demographic parity
            all_tests.append(
                run_demographic_parity_test(group_stats, reference_group, char_name)
            )
            # Run equal opportunity if true_positive_rate is available
            has_tpr = all(s.true_positive_rate is not None for s in group_stats)
            if has_tpr:
                all_tests.append(
                    run_equal_opportunity_test(group_stats, reference_group, char_name)
                )

        # Determine overall result
        failing = [t for t in all_tests if t.result == FairnessResult.FAIL]
        review = [t for t in all_tests if t.result == FairnessResult.REVIEW_REQUIRED]

        if failing:
            overall = FairnessResult.FAIL
            remediation_required = True
            remediation_actions = [
                f"Investigate and remediate disparate impact for: "
                f"{', '.join(set(t.characteristic for t in failing))}.",
                "Conduct root cause analysis of training data for identified groups.",
                "Engage AWB's Fair Lending team and legal counsel.",
                "Re-test after remediation; document findings for EU AI Act technical file.",
                "Report to Model Risk Committee within 10 business days.",
            ]
        elif review:
            overall = FairnessResult.REVIEW_REQUIRED
            remediation_required = False
            remediation_actions = [
                "Monitor flagged characteristics at each subsequent validation.",
                "Document review findings in model technical file.",
            ]
        else:
            overall = FairnessResult.PASS
            remediation_required = False
            remediation_actions = []

        return FairnessReport(
            model_id=model_id,
            model_name=model_name,
            assessor=assessor,
            test_results=all_tests,
            overall_result=overall,
            remediation_required=remediation_required,
            remediation_actions=remediation_actions,
        )
