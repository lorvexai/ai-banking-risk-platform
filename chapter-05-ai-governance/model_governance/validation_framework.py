"""
model_governance/validation_framework.py
AWB AI Governance Platform — Validation Framework
Chapter 5: Model Risk Management (PRA SS1/23)

Implements PRA SS1/23 Section 3 validation requirements:
- Independent validation of medium/high/critical risk models
- Standard validation tests: accuracy, stability, fairness, documentation
- ValidationReport as the formal output of each validation exercise

Test types:
- PERFORMANCE: Accuracy metrics (GINI, AUC, precision/recall)
- STABILITY: Population Stability Index (PSI), concept drift
- FAIRNESS: Demographic parity, equal opportunity (UK Equality Act 2010)
- DOCUMENTATION: ModelCard completeness, monitoring plan existence

PRA SS1/23 Section 3.2: validation must cover conceptual soundness,
data quality, outcome testing, sensitivity analysis, and benchmarking.
"""

from __future__ import annotations

import datetime
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ValidationResult(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    CONDITIONAL = "CONDITIONAL"   # Pass with conditions


class TestType(str, Enum):
    PERFORMANCE = "PERFORMANCE"
    STABILITY = "STABILITY"
    FAIRNESS = "FAIRNESS"
    DOCUMENTATION = "DOCUMENTATION"
    SENSITIVITY = "SENSITIVITY"
    BENCHMARKING = "BENCHMARKING"


class OverallResult(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    CONDITIONAL = "CONDITIONAL"


# ---------------------------------------------------------------------------
# Individual validation test
# ---------------------------------------------------------------------------

@dataclass
class ValidationTest:
    """
    Result of a single validation test.

    metric_value: The measured value (e.g. GINI = 0.62, PSI = 0.08)
    threshold: The pass/fail threshold (e.g. GINI ≥ 0.40, PSI ≤ 0.20)
    higher_is_better: True for GINI/AUC; False for PSI/error rates
    """
    test_name: str
    test_type: TestType
    result: ValidationResult
    metric_value: float
    threshold: float
    higher_is_better: bool = True
    notes: str = ""
    test_date: datetime.date = field(default_factory=datetime.date.today)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_name": self.test_name,
            "test_type": self.test_type.value,
            "result": self.result.value,
            "metric_value": round(self.metric_value, 6),
            "threshold": round(self.threshold, 6),
            "higher_is_better": self.higher_is_better,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    """
    Formal validation report per PRA SS1/23 Section 3.

    Produced by the independent model risk function (second line).
    Stored as audit evidence for 7 years (UK statutory minimum).
    Submitted to: Model Risk Committee → ERCC → Board (for HIGH/CRITICAL models).
    """
    model_id: str
    model_name: str
    validation_date: datetime.date
    validator_name: str                    # Second-line validator
    validator_team: str = ""
    test_results: List[ValidationTest] = field(default_factory=list)
    overall_result: OverallResult = OverallResult.PASS
    conditions: List[str] = field(default_factory=list)     # For CONDITIONAL result
    risk_rating_confirmed: str = ""         # Confirmed risk rating
    pra_ss1_23_findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    next_validation_due: Optional[datetime.date] = None
    report_id: str = ""

    def __post_init__(self) -> None:
        if not self.report_id:
            self.report_id = (
                f"VAL-{self.model_id}-"
                f"{self.validation_date.strftime('%Y%m%d')}"
            )

    @property
    def passed_tests(self) -> List[ValidationTest]:
        return [t for t in self.test_results if t.result == ValidationResult.PASS]

    @property
    def failed_tests(self) -> List[ValidationTest]:
        return [t for t in self.test_results if t.result == ValidationResult.FAIL]

    @property
    def pass_rate(self) -> float:
        if not self.test_results:
            return 0.0
        return len(self.passed_tests) / len(self.test_results)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "model_id": self.model_id,
            "model_name": self.model_name,
            "validation_date": self.validation_date.isoformat(),
            "validator_name": self.validator_name,
            "overall_result": self.overall_result.value,
            "pass_rate": round(self.pass_rate, 4),
            "test_count": len(self.test_results),
            "failed_tests": [t.test_name for t in self.failed_tests],
            "conditions": self.conditions,
            "risk_rating_confirmed": self.risk_rating_confirmed,
            "pra_ss1_23_findings": self.pra_ss1_23_findings,
            "recommendations": self.recommendations,
        }


# ---------------------------------------------------------------------------
# Standard test implementations
# ---------------------------------------------------------------------------

def run_gini_coefficient_test(
    gini_value: float,
    threshold: float = 0.40,
) -> ValidationTest:
    """
    GINI coefficient test for discriminatory power of credit scoring models.

    PRA SS1/23 Section 3.2(c): outcome testing for IRB/credit models.
    EBA GL/2017/07: minimum acceptable GINI for retail IRB models = 0.40.

    Args:
        gini_value: Measured Gini coefficient (0–1).
        threshold: Minimum acceptable Gini (default: 0.40).

    Returns:
        ValidationTest with PASS/FAIL result.
    """
    result = ValidationResult.PASS if gini_value >= threshold else ValidationResult.FAIL
    return ValidationTest(
        test_name="GINI Coefficient",
        test_type=TestType.PERFORMANCE,
        result=result,
        metric_value=gini_value,
        threshold=threshold,
        higher_is_better=True,
        notes=(
            f"GINI of {gini_value:.3f} {'meets' if result == ValidationResult.PASS else 'fails'} "
            f"EBA GL/2017/07 minimum of {threshold:.2f}."
        ),
    )


def run_psi_test(
    psi_value: float,
) -> ValidationTest:
    """
    Population Stability Index (PSI) test for input data stability.

    PSI thresholds (EBA / industry standard):
    - PSI < 0.10: Stable — GREEN
    - PSI 0.10–0.20: Monitoring required — AMBER
    - PSI > 0.20: Significant shift — RED — model may require revalidation

    Args:
        psi_value: Measured PSI value (non-negative).

    Returns:
        ValidationTest with PASS/CONDITIONAL/FAIL result based on PSI bands.
    """
    if psi_value < 0.10:
        result = ValidationResult.PASS
        notes = f"PSI {psi_value:.4f}: stable population (GREEN). No action required."
    elif psi_value <= 0.20:
        result = ValidationResult.CONDITIONAL
        notes = (
            f"PSI {psi_value:.4f}: moderate population shift (AMBER). "
            f"Investigate input distribution; enhanced monitoring required."
        )
    else:
        result = ValidationResult.FAIL
        notes = (
            f"PSI {psi_value:.4f}: significant population shift (RED). "
            f"Model performance may be degraded. Revalidation recommended."
        )

    return ValidationTest(
        test_name="Population Stability Index (PSI)",
        test_type=TestType.STABILITY,
        result=result,
        metric_value=psi_value,
        threshold=0.20,
        higher_is_better=False,
        notes=notes,
    )


def run_accuracy_test(
    accuracy_value: float,
    threshold: float = 0.80,
) -> ValidationTest:
    """
    Accuracy test for classification models.

    Args:
        accuracy_value: Model accuracy (0–1).
        threshold: Minimum acceptable accuracy (default: 0.80).

    Returns:
        ValidationTest with PASS/FAIL result.
    """
    result = ValidationResult.PASS if accuracy_value >= threshold else ValidationResult.FAIL
    return ValidationTest(
        test_name="Model Accuracy",
        test_type=TestType.PERFORMANCE,
        result=result,
        metric_value=accuracy_value,
        threshold=threshold,
        higher_is_better=True,
        notes=f"Accuracy of {accuracy_value:.3f} ({'' if result == ValidationResult.PASS else 'below '}threshold {threshold:.2f}).",
    )


def run_documentation_test(
    model_card,
    required_fields: Optional[List[str]] = None,
) -> ValidationTest:
    """
    Check that the ModelCard has all required documentation fields populated.

    PRA SS1/23 Section 3.2(a): validation must cover conceptual soundness
    documentation, including purpose, inputs, outputs, and limitations.

    Args:
        model_card: ModelCard instance to check.
        required_fields: List of attribute names to verify (optional).

    Returns:
        ValidationTest indicating documentation completeness.
    """
    if required_fields is None:
        required_fields = [
            "purpose", "inputs", "outputs", "limitations",
            "monitoring_plan", "owner",
        ]

    missing = []
    for f in required_fields:
        val = getattr(model_card, f, None)
        if val is None or (isinstance(val, (str, list)) and not val):
            missing.append(f)

    if missing:
        result = ValidationResult.FAIL
        notes = f"Missing documentation fields: {', '.join(missing)}."
    else:
        result = ValidationResult.PASS
        notes = "All required documentation fields are populated."

    score = (len(required_fields) - len(missing)) / len(required_fields)

    return ValidationTest(
        test_name="Documentation Completeness",
        test_type=TestType.DOCUMENTATION,
        result=result,
        metric_value=round(score, 4),
        threshold=1.0,
        higher_is_better=True,
        notes=notes,
    )


def run_demographic_parity_test(
    group_approval_rates: Dict[str, float],
    reference_group: str,
) -> ValidationTest:
    """
    Demographic parity test: checks that approval rates are similar across groups.

    Uses the 4/5ths rule (disparate impact ratio): any group with an approval
    rate < 80% of the reference group's rate triggers a FAIL.

    Protected characteristics per UK Equality Act 2010:
    age, disability, gender reassignment, marriage and civil partnership,
    pregnancy and maternity, race, religion or belief, sex, sexual orientation.

    Args:
        group_approval_rates: Dict of group_name → approval_rate (0–1).
        reference_group: The reference group for ratio calculation.

    Returns:
        ValidationTest with the minimum disparate impact ratio as metric_value.
    """
    if reference_group not in group_approval_rates:
        raise ValueError(f"Reference group '{reference_group}' not in group_approval_rates.")

    reference_rate = group_approval_rates[reference_group]
    if reference_rate <= 0:
        raise ValueError("Reference group approval rate must be positive.")

    ratios = {
        group: rate / reference_rate
        for group, rate in group_approval_rates.items()
        if group != reference_group
    }

    min_ratio = min(ratios.values()) if ratios else 1.0
    failing_groups = [g for g, r in ratios.items() if r < 0.80]

    if failing_groups:
        result = ValidationResult.FAIL
        notes = (
            f"Demographic parity FAIL: groups {failing_groups} have disparate "
            f"impact ratio < 0.80 (4/5ths rule). Minimum ratio: {min_ratio:.3f}. "
            f"EU AI Act: fundamental rights impact assessment required."
        )
    else:
        result = ValidationResult.PASS
        notes = (
            f"Demographic parity PASS: minimum disparate impact ratio {min_ratio:.3f} "
            f"≥ 0.80 (4/5ths rule). All groups within acceptable bounds."
        )

    return ValidationTest(
        test_name="Demographic Parity (4/5ths Rule)",
        test_type=TestType.FAIRNESS,
        result=result,
        metric_value=round(min_ratio, 4),
        threshold=0.80,
        higher_is_better=True,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Validation runner
# ---------------------------------------------------------------------------

class ModelValidator:
    """
    Orchestrates the standard AWB validation test suite for a model.

    PRA SS1/23 Section 3.1: validation must be independent of model development.
    This class is used by the second-line Model Risk function.
    """

    @staticmethod
    def run_standard_suite(
        model_card,
        gini: Optional[float] = None,
        psi: Optional[float] = None,
        accuracy: Optional[float] = None,
        group_approval_rates: Optional[Dict[str, float]] = None,
        reference_group: Optional[str] = None,
        validator_name: str = "Model Risk Function",
        validator_team: str = "AWB Second Line",
    ) -> ValidationReport:
        """
        Run the standard AWB validation test suite and produce a ValidationReport.

        Args:
            model_card: ModelCard instance to validate.
            gini: GINI coefficient (for credit models; optional).
            psi: Population Stability Index (optional).
            accuracy: Model accuracy (optional).
            group_approval_rates: Dict for fairness testing (optional).
            reference_group: Reference group for fairness test (optional).
            validator_name: Name of the validator.
            validator_team: Team performing validation.

        Returns:
            ValidationReport with all test results and overall result.
        """
        tests: List[ValidationTest] = []

        # Documentation completeness always run
        tests.append(run_documentation_test(model_card))

        if gini is not None:
            tests.append(run_gini_coefficient_test(gini))

        if psi is not None:
            tests.append(run_psi_test(psi))

        if accuracy is not None:
            tests.append(run_accuracy_test(accuracy))

        if group_approval_rates and reference_group:
            tests.append(run_demographic_parity_test(group_approval_rates, reference_group))

        # Determine overall result
        failed = [t for t in tests if t.result == ValidationResult.FAIL]
        conditional = [t for t in tests if t.result == ValidationResult.CONDITIONAL]

        if failed:
            overall = OverallResult.FAIL
            findings = [f"FAIL: {t.test_name} — {t.notes}" for t in failed]
        elif conditional:
            overall = OverallResult.CONDITIONAL
            findings = [f"CONDITIONAL: {t.test_name} — {t.notes}" for t in conditional]
        else:
            overall = OverallResult.PASS
            findings = []

        conditions = []
        if overall == OverallResult.CONDITIONAL:
            conditions = [
                "Implement enhanced monitoring for flagged metrics within 30 days.",
                "Report status to Model Risk Committee at next monthly meeting.",
            ]

        pra_findings = findings or ["No findings — model meets all validation standards."]

        # Set next validation date
        freq_months = getattr(model_card, "validation_frequency_months", 12)
        next_val_date = datetime.date.today() + datetime.timedelta(days=30 * freq_months)

        return ValidationReport(
            model_id=model_card.model_id,
            model_name=model_card.model_name,
            validation_date=datetime.date.today(),
            validator_name=validator_name,
            validator_team=validator_team,
            test_results=tests,
            overall_result=overall,
            conditions=conditions,
            risk_rating_confirmed=model_card.risk_rating.value,
            pra_ss1_23_findings=pra_findings,
            recommendations=(
                ["No remediation required."] if not failed else
                [f"Remediate: {t.test_name}" for t in failed]
            ),
            next_validation_due=next_val_date,
        )
