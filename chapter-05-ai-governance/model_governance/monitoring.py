"""
model_governance/monitoring.py
AWB AI Governance Platform — Ongoing Monitoring
Chapter 5: Model Risk Management (PRA SS1/23)

Implements PRA SS1/23 Section 4 ongoing monitoring requirements:
"Firms must implement ongoing monitoring of model performance."

PSI Alert thresholds (EBA / industry standard):
- PSI < 0.10  : GREEN  — stable; no action required
- PSI 0.10–0.20: AMBER  — moderate shift; investigate; enhanced monitoring
- PSI > 0.20  : RED    — significant shift; model review/revalidation required

DORA Article 11: AI system monitoring must detect anomalous patterns
that may indicate adversarial activity or model degradation.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class AlertLevel(str, Enum):
    """
    Traffic-light alert levels for model monitoring.
    Maps to PSI thresholds and action requirements.
    """
    GREEN = "GREEN"    # PSI < 0.10 — stable; routine monitoring
    AMBER = "AMBER"   # PSI 0.10–0.20 — investigate; enhanced monitoring
    RED = "RED"       # PSI > 0.20 — model review required; escalate to CRO


# ---------------------------------------------------------------------------
# PSI classification
# ---------------------------------------------------------------------------

PSI_GREEN_THRESHOLD = 0.10
PSI_RED_THRESHOLD = 0.20


def classify_psi(psi_value: float) -> AlertLevel:
    """
    Classify a PSI value into a traffic-light alert level.

    Args:
        psi_value: Population Stability Index (non-negative float).

    Returns:
        AlertLevel (GREEN, AMBER, or RED).

    Raises:
        ValueError: If psi_value is negative.
    """
    if psi_value < 0:
        raise ValueError(f"PSI value cannot be negative; got {psi_value}.")
    if psi_value < PSI_GREEN_THRESHOLD:
        return AlertLevel.GREEN
    if psi_value <= PSI_RED_THRESHOLD:
        return AlertLevel.AMBER
    return AlertLevel.RED


# ---------------------------------------------------------------------------
# Monitoring log entry
# ---------------------------------------------------------------------------

@dataclass
class MonitoringLogEntry:
    """
    A single monitoring observation for a model metric.

    Stored in AWB's model risk database; retained for 7 years (PRA SS1/23).
    Used to trend metrics over time and trigger escalations.
    """
    log_id: str
    model_id: str
    metric_name: str
    metric_value: float
    alert_level: AlertLevel
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.utcnow)
    threshold_green: Optional[float] = None
    threshold_red: Optional[float] = None
    notes: str = ""
    requires_action: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "log_id": self.log_id,
            "model_id": self.model_id,
            "metric_name": self.metric_name,
            "metric_value": round(self.metric_value, 6),
            "alert_level": self.alert_level.value,
            "timestamp": self.timestamp.isoformat(),
            "notes": self.notes,
            "requires_action": self.requires_action,
        }


# ---------------------------------------------------------------------------
# Model monitor
# ---------------------------------------------------------------------------

class ModelMonitor:
    """
    Tracks performance and stability metrics for a single model over time.

    Each model in production should have an associated ModelMonitor instance,
    configured with the metrics defined in the model's monitoring plan
    (PRA SS1/23 Section 4.1).

    Usage:
        monitor = ModelMonitor("MR-2026-036")
        monitor.record_psi(0.08)          # GREEN
        monitor.record_psi(0.15)          # AMBER — investigate
        monitor.record_psi(0.25)          # RED — escalate

        alerts = monitor.get_alerts(AlertLevel.AMBER)
    """

    def __init__(self, model_id: str, model_name: str = ""):
        self.model_id = model_id
        self.model_name = model_name
        self._log: List[MonitoringLogEntry] = []
        self._entry_counter = 0

    def _next_log_id(self) -> str:
        self._entry_counter += 1
        return f"MON-{self.model_id}-{self._entry_counter:06d}"

    def record_psi(
        self,
        psi_value: float,
        feature_name: str = "overall",
        notes: str = "",
    ) -> MonitoringLogEntry:
        """
        Record a PSI observation and classify into alert level.

        Args:
            psi_value: Measured PSI value.
            feature_name: Which feature/variable the PSI refers to.
            notes: Optional notes (e.g. reason for elevated PSI).

        Returns:
            MonitoringLogEntry with the classified alert level.
        """
        alert = classify_psi(psi_value)
        requires_action = alert in (AlertLevel.AMBER, AlertLevel.RED)

        entry = MonitoringLogEntry(
            log_id=self._next_log_id(),
            model_id=self.model_id,
            metric_name=f"PSI_{feature_name}",
            metric_value=psi_value,
            alert_level=alert,
            threshold_green=PSI_GREEN_THRESHOLD,
            threshold_red=PSI_RED_THRESHOLD,
            notes=notes or self._psi_notes(psi_value, alert, feature_name),
            requires_action=requires_action,
        )
        self._log.append(entry)
        return entry

    def record_metric(
        self,
        metric_name: str,
        metric_value: float,
        green_threshold: float,
        red_threshold: float,
        higher_is_better: bool = True,
        notes: str = "",
    ) -> MonitoringLogEntry:
        """
        Record an arbitrary model performance metric with configurable thresholds.

        Args:
            metric_name: Metric identifier (e.g. "GINI", "accuracy", "AUC").
            metric_value: Measured value.
            green_threshold: Threshold below which we escalate to AMBER/RED
                             (if higher_is_better=True) or above which we alert (False).
            red_threshold: Threshold for RED alert.
            higher_is_better: True for GINI/accuracy; False for error rates/PSI.
            notes: Optional notes.

        Returns:
            MonitoringLogEntry.
        """
        if higher_is_better:
            if metric_value >= green_threshold:
                alert = AlertLevel.GREEN
            elif metric_value >= red_threshold:
                alert = AlertLevel.AMBER
            else:
                alert = AlertLevel.RED
        else:
            if metric_value <= green_threshold:
                alert = AlertLevel.GREEN
            elif metric_value <= red_threshold:
                alert = AlertLevel.AMBER
            else:
                alert = AlertLevel.RED

        entry = MonitoringLogEntry(
            log_id=self._next_log_id(),
            model_id=self.model_id,
            metric_name=metric_name,
            metric_value=metric_value,
            alert_level=alert,
            threshold_green=green_threshold,
            threshold_red=red_threshold,
            notes=notes,
            requires_action=alert != AlertLevel.GREEN,
        )
        self._log.append(entry)
        return entry

    def get_all_logs(self) -> List[MonitoringLogEntry]:
        """Return all log entries, chronological order."""
        return list(self._log)

    def get_alerts(self, min_level: AlertLevel = AlertLevel.AMBER) -> List[MonitoringLogEntry]:
        """
        Return log entries at or above the specified alert level.

        Args:
            min_level: Minimum alert level to return (GREEN, AMBER, or RED).

        Returns:
            Filtered list of MonitoringLogEntry.
        """
        level_order = {AlertLevel.GREEN: 0, AlertLevel.AMBER: 1, AlertLevel.RED: 2}
        min_ord = level_order[min_level]
        return [e for e in self._log if level_order[e.alert_level] >= min_ord]

    def get_latest(self, metric_name: str) -> Optional[MonitoringLogEntry]:
        """Return the most recent entry for a given metric."""
        matching = [e for e in self._log if e.metric_name == metric_name]
        return matching[-1] if matching else None

    def has_red_alert(self) -> bool:
        """Returns True if any metric is currently in RED alert."""
        return any(e.alert_level == AlertLevel.RED for e in self._log)

    def summary(self) -> Dict[str, Any]:
        """Produce a monitoring summary for the Model Risk Committee report."""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "total_observations": len(self._log),
            "red_alerts": len([e for e in self._log if e.alert_level == AlertLevel.RED]),
            "amber_alerts": len([e for e in self._log if e.alert_level == AlertLevel.AMBER]),
            "green_observations": len([e for e in self._log if e.alert_level == AlertLevel.GREEN]),
            "requires_action": any(e.requires_action for e in self._log),
            "as_at": datetime.datetime.utcnow().isoformat(),
        }

    @staticmethod
    def _psi_notes(psi_value: float, alert: AlertLevel, feature: str) -> str:
        msgs = {
            AlertLevel.GREEN: f"PSI {psi_value:.4f} for {feature}: stable. No action.",
            AlertLevel.AMBER: (
                f"PSI {psi_value:.4f} for {feature}: moderate shift. "
                f"Investigate input data distribution. Enhanced monitoring active."
            ),
            AlertLevel.RED: (
                f"PSI {psi_value:.4f} for {feature}: significant shift. "
                f"Model performance may be degraded. Escalate to CRO. "
                f"Revalidation required (PRA SS1/23 Section 4.1)."
            ),
        }
        return msgs[alert]


# ---------------------------------------------------------------------------
# Multi-model monitoring registry
# ---------------------------------------------------------------------------

class MonitoringRegistry:
    """
    Registry of ModelMonitor instances for all production models.

    Used by the Model Risk function to:
    - Check all models for RED alerts at start of each business day
    - Produce the monthly Model Risk Committee monitoring report
    - Identify models requiring escalation to the CRO/Board
    """

    def __init__(self) -> None:
        self._monitors: Dict[str, ModelMonitor] = {}

    def register(self, model_id: str, model_name: str = "") -> ModelMonitor:
        """Register a new monitor for a model."""
        monitor = ModelMonitor(model_id, model_name)
        self._monitors[model_id] = monitor
        return monitor

    def get(self, model_id: str) -> ModelMonitor:
        if model_id not in self._monitors:
            raise KeyError(f"No monitor registered for model '{model_id}'.")
        return self._monitors[model_id]

    def get_all_red_alerts(self) -> List[Tuple[str, MonitoringLogEntry]]:
        """Return (model_id, entry) tuples for all current RED alerts."""
        alerts = []
        for model_id, monitor in self._monitors.items():
            for entry in monitor.get_alerts(AlertLevel.RED):
                alerts.append((model_id, entry))
        return alerts

    def daily_dashboard(self) -> List[Dict[str, Any]]:
        """Produce a daily monitoring dashboard for the Model Risk team."""
        return [m.summary() for m in self._monitors.values()]
