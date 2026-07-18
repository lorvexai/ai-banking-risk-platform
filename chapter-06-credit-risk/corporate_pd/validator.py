"""AWB Corporate PD — CRR3 Art. 176 Validation Suite.

Runs the full validation suite for MR-2026-040:
  - Discrimination: AUC-ROC, AUC-PR, Gini, K-S
  - Calibration: Brier score, Hosmer-Lemeshow
  - Stability: Population Stability Index (PSI)
  - CRR3: long-run PD deviation (Art. 180)

Triggers mandatory recalibration if PSI > 0.20.
Results written to MR-2026-040 SS1/23 model card.
"""
from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import datetime
from typing import Optional

import numpy as np

from awb_commons.schemas import CreditFeatures, ValidationReport

log = logging.getLogger(__name__)


class PDModelValidator:
    """CRR3 Art. 176 full validation suite for MR-2026-040.

    Usage::

        validator = PDModelValidator()
        report = validator.validate(model, X_test, y_test, X_ref)
        if report.recalibrate_required:
            trigger_recalibration_workflow()
    """

    # ── AWB SS1/23 validation thresholds ─────────────────────────
    AUC_ROC_MIN          = ValidationReport.AUC_ROC_MIN
    AUC_PR_MIN           = ValidationReport.AUC_PR_MIN
    BRIER_MAX            = ValidationReport.BRIER_MAX
    HL_P_MIN             = ValidationReport.HL_P_MIN
    GINI_MIN             = ValidationReport.GINI_MIN
    KS_MIN               = ValidationReport.KS_MIN
    PSI_ALERT            = ValidationReport.PSI_ALERT
    PSI_RECAL            = ValidationReport.PSI_RECAL_TRIGGER
    PD_DEV_MAX_BPS       = ValidationReport.PD_DEVIATION_MAX_BPS

    def validate(
        self,
        model,
        X_test: list[CreditFeatures],
        y_test: np.ndarray,
        X_ref:  Optional[list[CreditFeatures]] = None,
        observed_default_rate: Optional[float] = None,
    ) -> ValidationReport:
        """Run the full CRR3 Art. 176 validation suite.

        Args:
            model: AWBCorporatePDModel instance.
            X_test: List of CreditFeatures (test set).
            y_test: Array of actual defaults (0/1).
            X_ref:  Reference distribution for PSI
                    (training set or prior period).
            observed_default_rate: Long-run average default
                    rate for CRR3 Art. 180 PD calibration check.

        Returns:
            ValidationReport with pass/fail and recalibration flag.
        """
        proba = np.array([
            m.pd_calibrated
            for m in [model.predict(x) for x in X_test]
        ])
        y = np.array(y_test)

        auc_roc  = self._auc_roc(y, proba)
        auc_pr   = self._auc_pr(y, proba)
        brier    = float(np.mean((proba - y) ** 2))
        hl_p     = self._hosmer_lemeshow(y, proba)
        gini     = 2 * auc_roc - 1
        ks       = self._ks_stat(y, proba)
        psi      = self._psi(proba, X_ref, model) if X_ref else 0.0

        # CRR3 Art. 180: PD calibration check
        model_pd_mean = float(proba.mean())
        pd_dev_bps = 0.0
        if observed_default_rate is not None:
            pd_dev_bps = abs(
                model_pd_mean - observed_default_rate
            ) * 10_000

        pass_all = all([
            auc_roc  >= self.AUC_ROC_MIN,
            auc_pr   >= self.AUC_PR_MIN,
            brier    <= self.BRIER_MAX,
            hl_p     >= self.HL_P_MIN,
            gini     >= self.GINI_MIN,
            ks       >= self.KS_MIN,
            psi      <  self.PSI_RECAL,
            pd_dev_bps <= self.PD_DEV_MAX_BPS,
        ])

        report = ValidationReport(
            model_id    = "MR-2026-040",
            validation_date = datetime.utcnow(),
            auc_roc     = round(auc_roc, 4),
            auc_pr      = round(auc_pr, 4),
            brier_score = round(brier, 4),
            hosmer_lemeshow_p = round(hl_p, 4),
            gini_coefficient  = round(gini, 4),
            ks_statistic      = round(ks, 4),
            psi               = round(psi, 4),
            long_run_pd_deviation_bps = round(pd_dev_bps, 2),
            pass_all          = pass_all,
            recalibrate_required = psi >= self.PSI_RECAL,
            details = {
                "auc_roc_threshold":  self.AUC_ROC_MIN,
                "psi_alert":          self.PSI_ALERT,
                "psi_recal_trigger":  self.PSI_RECAL,
                "n_test":             len(y),
                "default_rate_test":  float(y.mean()),
            },
        )

        if report.recalibrate_required:
            log.warning(
                "MR-2026-040 RECALIBRATION REQUIRED: PSI=%.3f > %.2f",
                psi, self.PSI_RECAL,
            )
        elif psi >= self.PSI_ALERT:
            log.warning(
                "MR-2026-040 PSI ALERT: PSI=%.3f > %.2f",
                psi, self.PSI_ALERT,
            )

        log.info(
            "Validation MR-2026-040: auc=%.3f brier=%.3f "
            "psi=%.3f pass=%s",
            auc_roc, brier, psi, pass_all,
        )
        return report

    # ── Metric implementations ────────────────────────────────────

    def _auc_roc(
        self, y: np.ndarray, proba: np.ndarray
    ) -> float:
        """Compute AUC-ROC using Mann-Whitney U statistic."""
        pos = proba[y == 1]
        neg = proba[y == 0]
        if len(pos) == 0 or len(neg) == 0:
            return 0.5
        u = sum(
            (p > n) + 0.5 * (p == n)
            for p in pos for n in neg
        )
        return u / (len(pos) * len(neg))

    def _auc_pr(
        self, y: np.ndarray, proba: np.ndarray
    ) -> float:
        """Compute area under precision-recall curve."""
        thresholds = np.linspace(0, 1, 101)[::-1]
        precisions, recalls = [1.0], [0.0]
        for t in thresholds:
            pred = (proba >= t).astype(int)
            tp = int(((pred == 1) & (y == 1)).sum())
            fp = int(((pred == 1) & (y == 0)).sum())
            fn = int(((pred == 0) & (y == 1)).sum())
            p = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precisions.append(p)
            recalls.append(r)
        # Trapezoidal rule
        recalls_arr    = np.array(recalls)
        precisions_arr = np.array(precisions)
        order = np.argsort(recalls_arr)
        return float(np.trapezoid(
            precisions_arr[order], recalls_arr[order]
        ))

    def _hosmer_lemeshow(
        self, y: np.ndarray, proba: np.ndarray, g: int = 10
    ) -> float:
        """Hosmer-Lemeshow goodness-of-fit p-value.

        H0: model is well calibrated (p > 0.05 → accept H0).
        """
        from scipy.stats import chi2

        # Sort by predicted probability
        order    = np.argsort(proba)
        y_s      = y[order]
        p_s      = proba[order]
        n        = len(y)
        size     = n // g
        hl_stat  = 0.0

        for i in range(g):
            start = i * size
            end   = (i + 1) * size if i < g - 1 else n
            y_g   = y_s[start:end]
            p_g   = p_s[start:end]
            o1    = y_g.sum()
            e1    = p_g.sum()
            o0    = len(y_g) - o1
            e0    = len(y_g) - e1
            if e1 > 0:
                hl_stat += (o1 - e1) ** 2 / e1
            if e0 > 0:
                hl_stat += (o0 - e0) ** 2 / e0

        p_val = 1 - chi2.cdf(hl_stat, df=g - 2)
        return float(p_val)

    def _ks_stat(
        self, y: np.ndarray, proba: np.ndarray
    ) -> float:
        """Kolmogorov-Smirnov separation statistic."""
        pos = np.sort(proba[y == 1])
        neg = np.sort(proba[y == 0])
        if len(pos) == 0 or len(neg) == 0:
            return 0.0
        all_vals = np.sort(np.unique(np.concatenate([pos, neg])))
        ks = 0.0
        for v in all_vals:
            cdf_pos = (pos <= v).mean()
            cdf_neg = (neg <= v).mean()
            ks = max(ks, abs(cdf_pos - cdf_neg))
        return float(ks)

    def _psi(
        self,
        current_proba: np.ndarray,
        X_ref: list[CreditFeatures],
        model,
        n_bins: int = 10,
    ) -> float:
        """Population Stability Index (PSI).

        PSI = Σ (actual% - expected%) × ln(actual% / expected%)
        < 0.10: stable | 0.10-0.20: alert | > 0.20: recalibrate
        """
        ref_proba = np.array([
            model.predict(x).pd_calibrated for x in X_ref
        ])
        bins = np.linspace(0, 1, n_bins + 1)
        eps  = 1e-6

        act_counts = np.histogram(current_proba, bins)[0]
        exp_counts = np.histogram(ref_proba, bins)[0]

        act_pct = act_counts / (len(current_proba) + eps)
        exp_pct = exp_counts / (len(ref_proba) + eps)

        act_pct = np.clip(act_pct, eps, None)
        exp_pct = np.clip(exp_pct, eps, None)

        psi = float(
            np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
        )
        return psi
