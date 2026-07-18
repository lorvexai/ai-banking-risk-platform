"""AWB Consumer Loan — FCA PS22/9 Fairness Monitor.

Model ID:  MR-2026-041 (part of consumer origination system)
Reg basis: FCA Consumer Duty PS22/9 (fair outcomes)
           Equality Act 2010 (protected characteristics)

Runs monthly on the prior month's origination decisions.
Alerts the Retail Compliance team if:
  - Demographic parity ratio < 0.80 for any group
  - Equalised odds difference > 5 percentage points
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

# FCA PS22/9 thresholds
PARITY_THRESHOLD       = 0.80   # min approval rate ratio
EQUALISED_ODDS_MAX_DIFF = 0.05  # max FNR/FPR gap between groups
MIN_GROUP_SIZE         = 100    # min applications for stat validity

MONITORED_SEGMENTS = [
    "age_band",
    "gender",
    "imd_decile",       # Index of Multiple Deprivation
    "employment_status",
]


@dataclass
class FairnessAlert:
    """Single fairness alert raised by the monitor."""
    segment: str
    metric: str          # parity_ratio | equalised_odds
    value: float
    threshold: float
    worst_group: str
    best_group: str
    month: str           # YYYY-MM


@dataclass
class FairnessReport:
    """Monthly FCA PS22/9 fairness report for MR-2026-041."""
    report_month: str
    total_decisions: int
    alerts: list[FairnessAlert] = field(default_factory=list)
    segment_results: dict = field(default_factory=dict)

    @property
    def has_alerts(self) -> bool:
        return len(self.alerts) > 0

    @property
    def alert_count(self) -> int:
        return len(self.alerts)


class FairnessMonitor:
    """FCA PS22/9 demographic parity and equalised-odds monitor.

    Runs monthly on the origination decisions DataFrame.

    Usage::

        monitor = FairnessMonitor()
        report = monitor.monthly_report(decisions_df, "2026-06")
        if report.has_alerts:
            notify_compliance(report)
    """

    def monthly_report(
        self,
        decisions: pd.DataFrame,
        month: str,
    ) -> FairnessReport:
        """Compute fairness metrics for a month of decisions.

        Args:
            decisions: DataFrame with columns:
                age_band, gender, imd_decile,
                employment_status, approved (bool),
                defaulted (bool, nullable — for outcome metrics)
            month: Report month in YYYY-MM format.

        Returns:
            FairnessReport with any PS22/9 alerts.
        """
        report = FairnessReport(
            report_month    = month,
            total_decisions = len(decisions),
        )

        for segment in MONITORED_SEGMENTS:
            if segment not in decisions.columns:
                log.warning("Segment %s not in data", segment)
                continue

            result = self._analyse_segment(
                decisions, segment, month, report
            )
            report.segment_results[segment] = result

        log.info(
            "PS22/9 report %s: %d decisions, %d alerts",
            month, len(decisions), report.alert_count,
        )
        return report

    # ── Segment analysis ──────────────────────────────────────────

    def _analyse_segment(
        self,
        decisions: pd.DataFrame,
        segment: str,
        month: str,
        report: FairnessReport,
    ) -> dict:
        """Compute parity ratio and equalised odds for a segment."""
        groups = (
            decisions
            .groupby(segment)["approved"]
            .agg(["sum", "count"])
            .rename(columns={"sum": "approved_n", "count": "total"})
        )
        # Drop groups below minimum size
        groups = groups[groups["total"] >= MIN_GROUP_SIZE]
        if groups.empty:
            return {"skipped": "insufficient_group_size"}

        groups["rate"] = groups["approved_n"] / groups["total"]
        max_rate  = groups["rate"].max()
        min_rate  = groups["rate"].min()
        ratio     = min_rate / max_rate if max_rate > 0 else 1.0
        best_grp  = str(groups["rate"].idxmax())
        worst_grp = str(groups["rate"].idxmin())

        result = {
            "parity_ratio":  round(ratio, 4),
            "approval_rates": groups["rate"].round(4).to_dict(),
            "alert":         ratio < PARITY_THRESHOLD,
        }

        if ratio < PARITY_THRESHOLD:
            alert = FairnessAlert(
                segment   = segment,
                metric    = "parity_ratio",
                value     = round(ratio, 4),
                threshold = PARITY_THRESHOLD,
                worst_group = worst_grp,
                best_group  = best_grp,
                month     = month,
            )
            report.alerts.append(alert)
            log.warning(
                "PS22/9 ALERT: %s parity_ratio=%.3f "
                "(worst=%s best=%s)",
                segment, ratio, worst_grp, best_grp,
            )

        return result

    def equalised_odds_check(
        self,
        decisions: pd.DataFrame,
        segment: str,
        month: str,
    ) -> dict:
        """Check equalised odds (FNR gap) across groups.

        Requires 'defaulted' column with observed outcomes.
        Only meaningful on cohorts with 12+ month history.
        """
        if "defaulted" not in decisions.columns:
            return {"skipped": "no_outcome_data"}

        # Restrict to decisions where outcome is known
        with_outcome = decisions.dropna(subset=["defaulted"])
        if len(with_outcome) < 200:
            return {"skipped": "insufficient_outcome_data"}

        # True positives: defaulted AND declined (correct)
        # False negatives: defaulted AND approved (missed defaults)
        groups = with_outcome.groupby(segment).apply(
            lambda g: pd.Series({
                "fnr": (
                    (g["defaulted"] & g["approved"]).sum()
                    / g["defaulted"].sum()
                    if g["defaulted"].sum() > 0 else 0.0
                ),
                "n": len(g),
            })
        )
        groups = groups[groups["n"] >= MIN_GROUP_SIZE]
        if len(groups) < 2:
            return {"skipped": "insufficient_groups"}

        fnr_max  = groups["fnr"].max()
        fnr_min  = groups["fnr"].min()
        fnr_diff = fnr_max - fnr_min

        return {
            "fnr_gap":       round(float(fnr_diff), 4),
            "fnr_by_group":  groups["fnr"].round(4).to_dict(),
            "alert":         fnr_diff > EQUALISED_ODDS_MAX_DIFF,
        }
