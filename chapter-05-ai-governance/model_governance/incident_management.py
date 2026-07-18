"""
model_governance/incident_management.py
AWB AI Governance Platform — AI Incident Management
Chapter 5: Model Risk Management (PRA SS1/23)

Implements AWB's AI Incident Management framework aligned with:
- PRA SS1/23 Section 4: ongoing monitoring and incident response
- FCA operational resilience (PS21/3): important business services
- EU AI Act 2024 Article 73: serious incident notification (15 days)
- DORA Article 17: ICT-related incident classification and reporting

Incident severity:
- P1: Material customer harm — FCA notification within 4 hours
- P2: Model performance degradation — Board notification within 24 hours
- P3: Monitoring alert — Risk Committee notification within 5 business days
- P4: Minor issue — standard change management; no regulatory notification

British English throughout.
"""

from __future__ import annotations

import datetime
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class IncidentSeverity(str, Enum):
    """
    AI Incident severity levels.

    Aligned with DORA incident classification and FCA operational resilience
    requirements for important business services.
    """
    P1 = "P1"  # Critical — material customer harm or regulatory breach
    P2 = "P2"  # Major — significant model degradation or data issue
    P3 = "P3"  # Moderate — monitoring alert or near-miss
    P4 = "P4"  # Minor — low-impact issue; standard remediation


class IncidentStatus(str, Enum):
    OPEN = "OPEN"
    INVESTIGATING = "INVESTIGATING"
    RESOLVED = "RESOLVED"
    CLOSED = "CLOSED"
    ESCALATED = "ESCALATED"


class NotificationRecipient(str, Enum):
    FCA = "FCA"
    PRA = "PRA"
    BOARD = "BOARD"
    ERCC = "ERCC"          # Enterprise Risk & Compliance Committee
    RISK_COMMITTEE = "RISK_COMMITTEE"
    CRO = "CRO"
    EU_AI_OFFICE = "EU_AI_OFFICE"  # EU AI Act Article 73 serious incidents


# ---------------------------------------------------------------------------
# Notification requirement lookup
# ---------------------------------------------------------------------------

SEVERITY_NOTIFICATION_MAP: Dict[IncidentSeverity, Dict[str, Any]] = {
    IncidentSeverity.P1: {
        "regulatory_notification_required": True,
        "notification_deadline_hours": 4,
        "recipients": [
            NotificationRecipient.FCA,
            NotificationRecipient.PRA,
            NotificationRecipient.BOARD,
            NotificationRecipient.CRO,
        ],
        "description": (
            "P1 — Critical: Material customer harm or regulatory breach. "
            "FCA/PRA notification within 4 hours. Board convened immediately."
        ),
    },
    IncidentSeverity.P2: {
        "regulatory_notification_required": False,
        "notification_deadline_hours": 24,
        "recipients": [
            NotificationRecipient.BOARD,
            NotificationRecipient.ERCC,
            NotificationRecipient.CRO,
        ],
        "description": (
            "P2 — Major: Significant model degradation or data quality issue. "
            "Board notification within 24 hours."
        ),
    },
    IncidentSeverity.P3: {
        "regulatory_notification_required": False,
        "notification_deadline_hours": 120,  # 5 business days
        "recipients": [
            NotificationRecipient.RISK_COMMITTEE,
            NotificationRecipient.CRO,
        ],
        "description": (
            "P3 — Moderate: Monitoring alert or near-miss. "
            "Risk Committee notification within 5 business days."
        ),
    },
    IncidentSeverity.P4: {
        "regulatory_notification_required": False,
        "notification_deadline_hours": None,  # Standard change management
        "recipients": [],
        "description": (
            "P4 — Minor: Low-impact issue. Standard change management process. "
            "No regulatory notification required."
        ),
    },
}

# EU AI Act serious incident threshold: any P1 or P2 incident involving
# a HIGH_RISK AI system requires notification to national authority within 15 days
EU_AI_ACT_NOTIFICATION_DAYS = 15


# ---------------------------------------------------------------------------
# AIIncident dataclass
# ---------------------------------------------------------------------------

@dataclass
class AIIncident:
    """
    Records a single AI model incident.

    Stored as regulatory audit evidence (7-year retention).
    P1 incidents are logged with FCA/PRA and included in the Annual Model
    Risk Report submitted to the Board under PRA SS1/23 Section 5.2.
    """
    # --- Identity ---
    incident_id: str = field(default_factory=lambda: f"INC-{uuid.uuid4().hex[:8].upper()}")
    model_id: str = ""
    model_name: str = ""

    # --- Classification ---
    severity: IncidentSeverity = IncidentSeverity.P4
    title: str = ""
    description: str = ""

    # --- Timeline ---
    detected_at: datetime.datetime = field(default_factory=datetime.datetime.utcnow)
    reported_at: Optional[datetime.datetime] = None
    resolved_at: Optional[datetime.datetime] = None
    notification_deadline: Optional[datetime.datetime] = None

    # --- Root Cause and Remediation ---
    root_cause: str = ""
    root_cause_category: str = ""  # DATA / MODEL / INFRASTRUCTURE / PROCESS / THIRD_PARTY
    remediation_actions: List[str] = field(default_factory=list)
    lessons_learned: str = ""

    # --- Regulatory ---
    regulatory_notification_required: bool = False
    eu_ai_act_serious_incident: bool = False
    notifications_sent: List[str] = field(default_factory=list)
    regulatory_ref: str = ""   # FCA/PRA notification reference number

    # --- Status ---
    status: IncidentStatus = IncidentStatus.OPEN
    assigned_to: str = ""
    escalated_to: str = ""

    def __post_init__(self) -> None:
        """Automatically set regulatory notification requirement based on severity."""
        sev_config = SEVERITY_NOTIFICATION_MAP.get(self.severity, {})
        self.regulatory_notification_required = sev_config.get(
            "regulatory_notification_required", False
        )

        # Set notification deadline
        deadline_hours = sev_config.get("notification_deadline_hours")
        if deadline_hours and self.detected_at:
            self.notification_deadline = self.detected_at + datetime.timedelta(
                hours=deadline_hours
            )

    @property
    def resolution_time_hours(self) -> Optional[float]:
        """Time to resolution in hours, or None if not yet resolved."""
        if self.resolved_at and self.detected_at:
            delta = self.resolved_at - self.detected_at
            return delta.total_seconds() / 3600
        return None

    @property
    def is_overdue_for_notification(self) -> bool:
        """Returns True if notification deadline has passed."""
        if not self.notification_deadline or not self.regulatory_notification_required:
            return False
        return datetime.datetime.utcnow() > self.notification_deadline

    @property
    def required_notification_recipients(self) -> List[NotificationRecipient]:
        """Return the notification recipients for this severity level."""
        config = SEVERITY_NOTIFICATION_MAP.get(self.severity, {})
        recipients = list(config.get("recipients", []))
        if self.eu_ai_act_serious_incident:
            if NotificationRecipient.EU_AI_OFFICE not in recipients:
                recipients.append(NotificationRecipient.EU_AI_OFFICE)
        return recipients

    def mark_resolved(
        self,
        root_cause: str,
        remediation_actions: List[str],
        lessons_learned: str = "",
    ) -> None:
        """Mark the incident as resolved with root cause and remediation."""
        self.root_cause = root_cause
        self.remediation_actions = remediation_actions
        self.lessons_learned = lessons_learned
        self.resolved_at = datetime.datetime.utcnow()
        self.status = IncidentStatus.RESOLVED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "model_id": self.model_id,
            "model_name": self.model_name,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "detected_at": self.detected_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "notification_deadline": (
                self.notification_deadline.isoformat() if self.notification_deadline else None
            ),
            "regulatory_notification_required": self.regulatory_notification_required,
            "eu_ai_act_serious_incident": self.eu_ai_act_serious_incident,
            "root_cause": self.root_cause,
            "remediation_actions": self.remediation_actions,
            "status": self.status.value,
            "resolution_time_hours": self.resolution_time_hours,
        }


# ---------------------------------------------------------------------------
# Severity classifier
# ---------------------------------------------------------------------------

def classify_incident_severity(
    customer_harm: bool = False,
    regulatory_breach: bool = False,
    performance_degraded: bool = False,
    data_quality_issue: bool = False,
    monitoring_alert_only: bool = False,
) -> IncidentSeverity:
    """
    Classify an AI incident severity based on observed impact.

    Decision logic:
    - Any customer harm or regulatory breach → P1
    - Performance degradation or data quality issue → P2
    - Monitoring alert only → P3
    - Otherwise → P4

    Args:
        customer_harm: True if customers experienced material harm.
        regulatory_breach: True if a regulatory obligation was breached.
        performance_degraded: True if model performance dropped materially.
        data_quality_issue: True if input data quality was materially impacted.
        monitoring_alert_only: True if this is a monitoring alert without confirmed impact.

    Returns:
        IncidentSeverity (P1–P4).
    """
    if customer_harm or regulatory_breach:
        return IncidentSeverity.P1
    if performance_degraded or data_quality_issue:
        return IncidentSeverity.P2
    if monitoring_alert_only:
        return IncidentSeverity.P3
    return IncidentSeverity.P4


# ---------------------------------------------------------------------------
# Incident log
# ---------------------------------------------------------------------------

class IncidentLog:
    """
    Central registry of AI incidents for AWB.

    Maintains a complete, auditable log of all AI model incidents.
    Provides reporting for:
    - Daily operational review (open P1/P2 incidents)
    - Monthly Risk Committee pack (P1–P3 summary)
    - Annual PRA Model Risk Report
    """

    def __init__(self) -> None:
        self._incidents: Dict[str, AIIncident] = {}

    def create_incident(
        self,
        model_id: str,
        model_name: str,
        severity: IncidentSeverity,
        title: str,
        description: str,
        eu_ai_act_serious_incident: bool = False,
        assigned_to: str = "",
    ) -> AIIncident:
        """
        Create and register a new incident.

        Returns:
            Newly created AIIncident.
        """
        incident = AIIncident(
            model_id=model_id,
            model_name=model_name,
            severity=severity,
            title=title,
            description=description,
            reported_at=datetime.datetime.utcnow(),
            eu_ai_act_serious_incident=eu_ai_act_serious_incident,
            assigned_to=assigned_to,
        )
        self._incidents[incident.incident_id] = incident
        return incident

    def get(self, incident_id: str) -> AIIncident:
        if incident_id not in self._incidents:
            raise KeyError(f"Incident '{incident_id}' not found.")
        return self._incidents[incident_id]

    def get_open(self) -> List[AIIncident]:
        return [i for i in self._incidents.values() if i.status == IncidentStatus.OPEN]

    def get_by_severity(self, severity: IncidentSeverity) -> List[AIIncident]:
        return [i for i in self._incidents.values() if i.severity == severity]

    def get_requiring_regulatory_notification(self) -> List[AIIncident]:
        return [i for i in self._incidents.values() if i.regulatory_notification_required]

    def get_overdue_notifications(self) -> List[AIIncident]:
        return [i for i in self._incidents.values() if i.is_overdue_for_notification]

    def summary(self) -> Dict[str, Any]:
        all_i = list(self._incidents.values())
        return {
            "total_incidents": len(all_i),
            "open": len(self.get_open()),
            "by_severity": {s.value: len(self.get_by_severity(s)) for s in IncidentSeverity},
            "regulatory_notifications_required": len(self.get_requiring_regulatory_notification()),
            "overdue_notifications": len(self.get_overdue_notifications()),
        }
