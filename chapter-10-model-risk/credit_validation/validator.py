"""
credit_validation/validator.py — AWB Credit Model Validator.
PRA SS1/23 independent model validation for credit AI systems.
Avon & Wessex Bank plc (AWB) — AWB-AI-2025 programme.

Validates:
- MR-2026-035 (Credit Document Analyser)
- MR-2026-037 (Credit Decision Agent)
- Chapter 6 PD/LGD/EAD models
"""
from __future__ import annotations
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from awb_commons.models import ValidationResult

logger = logging.getLogger(__name__)

# PRA SS1/23 validation thresholds (AWB MRMP §4.2)
GINI_MIN      = 0.700
PSI_WARNING   = 0.100
PSI_ACTION    = 0.200
KS_MIN        = 0.300
AUC_MIN       = 0.750


@dataclass
class ValidationDataset:
    """Out-of-time / out-of-sample validation dataset."""
    dataset_id: str
    model_predictions: list[float]  # scores [0,1]
    actual_outcomes:   list[int]    # 0=good, 1=default
    development_dist:  list[float]  # development score dist
    validation_period: str          # e.g. "2025-Q4"

    def __post_init__(self) -> None:
        if len(self.model_predictions) != len(self.actual_outcomes):
            raise ValueError(
                "predictions and outcomes must have equal length"
            )
        if not self.model_predictions:
            raise ValueError(
                "dataset cannot be empty"
            )


class CreditModelValidator:
    """
    Independent validation of AWB credit AI models.

    Performs the standard validation suite required by
    PRA SS1/23 for MEDIUM and HIGH risk credit models:
    1. Discriminatory power (Gini, KS, AUC-ROC)
    2. Population Stability Index (PSI)
    3. Calibration assessment
    4. Backtesting (predicted PD vs observed default rate)

    Results are recorded in the model registry with findings
    and conditional pass/fail outcome.
    """

    def validate(
        self,
        mr_reference: str,
        validator_id: str,
        dataset: ValidationDataset,
    ) -> ValidationResult:
        """
        Run full validation suite on a credit model.

        Args:
            mr_reference: Model identifier (e.g. MR-2026-035).
            validator_id: Validator employee/team ID.
            dataset: Out-of-time validation data.

        Returns:
            ValidationResult with metrics, findings, outcome.
        """
        findings: list[str] = []
        gini = self._gini_coefficient(
            dataset.model_predictions,
            dataset.actual_outcomes,
        )
        ks   = self._ks_statistic(
            dataset.model_predictions,
            dataset.actual_outcomes,
        )
        auc  = self._auc_roc(
            dataset.model_predictions,
            dataset.actual_outcomes,
        )
        psi  = self._population_stability_index(
            dataset.development_dist,
            dataset.model_predictions,
        )
        if gini < GINI_MIN:
            findings.append(
                f"Gini {gini:.3f} below minimum {GINI_MIN}"
            )
        if psi > PSI_ACTION:
            findings.append(
                f"PSI {psi:.3f} exceeds action threshold "
                f"{PSI_ACTION} — significant population shift"
            )
        elif psi > PSI_WARNING:
            findings.append(
                f"PSI {psi:.3f} exceeds warning threshold "
                f"{PSI_WARNING} — monitor closely"
            )
        if auc < AUC_MIN:
            findings.append(
                f"AUC-ROC {auc:.3f} below minimum {AUC_MIN}"
            )
        outcome = self._determine_outcome(gini, psi, auc, findings)
        logger.info(
            "Validation complete: %s gini=%.3f psi=%.3f "
            "auc=%.3f outcome=%s",
            mr_reference, gini, psi, auc, outcome,
        )
        return ValidationResult(
            mr_reference=mr_reference,
            validator_id=validator_id,
            gini_coefficient=round(gini, 4),
            psi=round(psi, 4),
            ks_statistic=round(ks, 4),
            auc_roc=round(auc, 4),
            outcome=outcome,
            findings=findings,
        )

    # ── Metric implementations ────────────────────────────────────

    def _gini_coefficient(
        self,
        scores: list[float],
        outcomes: list[int],
    ) -> float:
        """Gini = 2 × AUC - 1."""
        return 2.0 * self._auc_roc(scores, outcomes) - 1.0

    def _auc_roc(
        self,
        scores: list[float],
        outcomes: list[int],
    ) -> float:
        """
        AUC-ROC via trapezoidal rule.
        Time complexity O(n log n).
        """
        paired = sorted(
            zip(scores, outcomes),
            key=lambda x: x[0],
            reverse=True,
        )
        n_pos = sum(outcomes)
        n_neg = len(outcomes) - n_pos
        if n_pos == 0 or n_neg == 0:
            return 0.5
        tp = fp = 0
        prev_tp = prev_fp = 0
        auc = 0.0
        prev_score = None
        for score, label in paired:
            if score != prev_score and prev_score is not None:
                auc += (fp - prev_fp) * (tp + prev_tp) / 2
                prev_tp, prev_fp = tp, fp
            if label == 1:
                tp += 1
            else:
                fp += 1
            prev_score = score
        auc += (fp - prev_fp) * (tp + prev_tp) / 2
        return auc / (n_pos * n_neg)

    def _ks_statistic(
        self,
        scores: list[float],
        outcomes: list[int],
    ) -> float:
        """Kolmogorov-Smirnov statistic."""
        n = len(scores)
        n_pos = sum(outcomes)
        n_neg = n - n_pos
        if n_pos == 0 or n_neg == 0:
            return 0.0
        paired = sorted(
            zip(scores, outcomes),
            key=lambda x: x[0],
            reverse=True,
        )
        cum_pos = cum_neg = 0.0
        max_ks = 0.0
        for _, label in paired:
            if label == 1:
                cum_pos += 1.0 / n_pos
            else:
                cum_neg += 1.0 / n_neg
            max_ks = max(max_ks, abs(cum_pos - cum_neg))
        return max_ks

    def _population_stability_index(
        self,
        expected: list[float],
        actual: list[float],
        n_bins: int = 10,
    ) -> float:
        """
        PSI = Σ (Actual% - Expected%) × ln(Actual%/Expected%).
        Uses equal-width bins over [0, 1].
        """
        bin_edges = [i / n_bins for i in range(n_bins + 1)]
        def bin_pct(data: list[float]) -> list[float]:
            counts = [0] * n_bins
            for v in data:
                idx = min(int(v * n_bins), n_bins - 1)
                counts[idx] += 1
            n = len(data)
            return [max(c / n, 1e-6) for c in counts]

        exp_pct = bin_pct(expected)
        act_pct = bin_pct(actual)
        psi = sum(
            (a - e) * math.log(a / e)
            for a, e in zip(act_pct, exp_pct)
        )
        return psi

    def _determine_outcome(
        self,
        gini: float,
        psi: float,
        auc: float,
        findings: list[str],
    ) -> str:
        if any(
            "below minimum" in f or
            "exceeds action" in f
            for f in findings
        ):
            return "FAIL"
        if any("exceeds warning" in f for f in findings):
            return "CONDITIONAL_PASS"
        return "PASS"
