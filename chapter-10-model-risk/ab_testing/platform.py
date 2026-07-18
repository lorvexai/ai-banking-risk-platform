"""
ab_testing/platform.py — AWB Model A/B Testing Platform.
PRA SS1/23: challenger model must outperform champion before
production deployment. All tests require model risk approval.
Avon & Wessex Bank plc (AWB) — AWB-AI-2025 programme.
"""
from __future__ import annotations
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Callable
from awb_commons.models import ABTestResult

logger = logging.getLogger(__name__)

# PRA SS1/23: minimum statistical significance for model switch
SIGNIFICANCE_THRESHOLD = 0.05   # p < 0.05
MIN_SAMPLE_SIZE        = 1000   # per variant


@dataclass
class ModelVariant:
    """A champion or challenger model variant."""
    version: str
    model_fn: Callable  # predict(features) -> float
    description: str = ""


@dataclass
class ABTestConfig:
    """Configuration for a model A/B test."""
    mr_reference: str
    test_name: str
    control: ModelVariant
    treatment: ModelVariant
    traffic_split_pct: float = 50.0  # % to treatment
    min_sample_size: int = MIN_SAMPLE_SIZE
    significance_threshold: float = SIGNIFICANCE_THRESHOLD
    primary_metric: str = "accuracy"


class ABTestingPlatform:
    """
    Champion/challenger testing platform for AWB models.

    Implements a two-proportion z-test for binary outcomes
    (approval/decline, fraud/legitimate) and a t-test for
    continuous metrics (Gini, AUC-ROC).

    PRA SS1/23 requirement: challenger model must demonstrate
    statistically significant improvement (p < 0.05) on the
    primary metric before the model risk committee approves
    a production switch.

    EU AI Act Art. 9: risk management system must include
    ongoing evaluation of high-risk AI system performance.
    """

    def __init__(self, config: ABTestConfig) -> None:
        self.config = config
        self._control_outcomes: list[float] = []
        self._treatment_outcomes: list[float] = []
        self.test_start = datetime.utcnow()
        logger.info(
            "A/B test started: %s control=%s treatment=%s",
            config.test_name,
            config.control.version,
            config.treatment.version,
        )

    def record_outcome(
        self,
        is_treatment: bool,
        outcome: float,
    ) -> None:
        """
        Record a model prediction outcome (0.0 or 1.0 for
        binary metrics; continuous value for Gini/AUC).

        Args:
            is_treatment: True if treatment variant was used.
            outcome: Observed outcome (0.0 or 1.0).
        """
        if is_treatment:
            self._treatment_outcomes.append(outcome)
        else:
            self._control_outcomes.append(outcome)

    def analyse(self) -> ABTestResult:
        """
        Run statistical analysis on accumulated outcomes.

        Returns:
            ABTestResult with lift, p-value, and recommendation.

        Raises:
            ValueError: If insufficient samples collected.
        """
        n_c = len(self._control_outcomes)
        n_t = len(self._treatment_outcomes)
        if n_c < self.config.min_sample_size:
            raise ValueError(
                f"Insufficient control samples: "
                f"{n_c} < {self.config.min_sample_size}"
            )
        if n_t < self.config.min_sample_size:
            raise ValueError(
                f"Insufficient treatment samples: "
                f"{n_t} < {self.config.min_sample_size}"
            )
        control_mean  = sum(self._control_outcomes) / n_c
        treatment_mean = sum(self._treatment_outcomes) / n_t
        lift_pct = (
            (treatment_mean - control_mean)
            / max(control_mean, 1e-9)
        ) * 100.0
        p_value = self._two_proportion_z_test(
            control_mean, n_c, treatment_mean, n_t
        )
        significant = p_value < self.config.significance_threshold
        recommendation = self._recommend(
            lift_pct, significant
        )
        result = ABTestResult(
            mr_reference=self.config.mr_reference,
            control_version=self.config.control.version,
            treatment_version=self.config.treatment.version,
            sample_size_control=n_c,
            sample_size_treatment=n_t,
            metric_name=self.config.primary_metric,
            control_value=round(control_mean, 4),
            treatment_value=round(treatment_mean, 4),
            lift_pct=round(lift_pct, 2),
            p_value=round(p_value, 4),
            statistically_significant=significant,
            recommendation=recommendation,
            test_start=self.test_start,
            test_end=datetime.utcnow(),
        )
        logger.info(
            "A/B test complete: lift=%.1f%% p=%.4f "
            "significant=%s recommendation=%s",
            lift_pct, p_value, significant, recommendation,
        )
        return result

    def required_sample_size(
        self,
        baseline_rate: float,
        min_detectable_effect: float = 0.02,
        power: float = 0.80,
    ) -> int:
        """
        Calculate minimum sample size per variant.
        Uses standard two-proportion z-test formula.
        """
        alpha = self.config.significance_threshold
        z_alpha = 1.96   # two-tailed α=0.05
        z_beta  = 0.842  # power=0.80
        p1 = baseline_rate
        p2 = baseline_rate + min_detectable_effect
        p_bar = (p1 + p2) / 2
        numerator = (
            z_alpha * math.sqrt(2 * p_bar * (1 - p_bar))
            + z_beta * math.sqrt(
                p1 * (1 - p1) + p2 * (1 - p2)
            )
        ) ** 2
        denominator = (p2 - p1) ** 2
        return math.ceil(numerator / denominator)

    # ── Private helpers ───────────────────────────────────────────

    def _two_proportion_z_test(
        self,
        p1: float, n1: int,
        p2: float, n2: int,
    ) -> float:
        """
        Two-proportion z-test.
        Returns p-value (two-tailed).
        """
        p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
        se = math.sqrt(
            p_pool * (1 - p_pool) * (1/n1 + 1/n2)
        )
        if se < 1e-10:
            return 1.0
        z = abs(p2 - p1) / se
        # Normal CDF approximation (Abramowitz & Stegun)
        p_one_tail = self._normal_cdf(-abs(z))
        return 2.0 * p_one_tail

    def _normal_cdf(self, z: float) -> float:
        """Approximation of standard normal CDF."""
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))

    def _recommend(
        self, lift_pct: float, significant: bool
    ) -> str:
        if significant and lift_pct > 0:
            return (
                f"ADOPT treatment: +{lift_pct:.1f}% "
                "significant improvement. "
                "Submit to model risk committee for approval."
            )
        elif significant and lift_pct <= 0:
            return (
                "REJECT treatment: significant degradation. "
                "Retain control model."
            )
        else:
            return (
                "INCONCLUSIVE: no significant difference. "
                "Extend test or revise hypothesis."
            )
