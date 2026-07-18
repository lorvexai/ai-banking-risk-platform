"""
awb_commons.audit — 7-year audit trail for PRA SS1/23 compliance.
Avon & Wessex Bank plc (AWB) — AWB-AI-2025 programme.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)


class AuditLogger:
    """
    Append-only audit trail satisfying PRA SS1/23 and
    7-year FCA retention requirement (COBS 9).

    In production, writes to PostgreSQL audit_log table.
    This stub logs to the Python logger for testing.
    """

    def __init__(self, model_id: str, dry_run: bool = False) -> None:
        self.model_id = model_id
        self.dry_run = dry_run

    def log_prediction(
        self,
        input_data: dict[str, Any],
        output_data: dict[str, Any],
        confidence: float,
        latency_ms: int,
    ) -> UUID:
        """
        Record a model prediction to the audit trail.

        Args:
            input_data: Sanitised input features (no PII).
            output_data: Model output and recommendation.
            confidence: Prediction confidence score.
            latency_ms: Inference latency in milliseconds.

        Returns:
            UUID of the audit record created.
        """
        record_id = uuid4()
        record = {
            "audit_id": str(record_id),
            "model_id": self.model_id,
            "input_hash": hash(
                json.dumps(input_data, sort_keys=True)
            ),
            "output": output_data,
            "confidence": confidence,
            "latency_ms": latency_ms,
            "logged_at": datetime.utcnow().isoformat(),
            "dry_run": self.dry_run,
        }
        logger.info("AUDIT: %s", json.dumps(record))
        return record_id

    def log_fraud_alert(
        self,
        alert_id: UUID,
        action_taken: str,
        reviewer_id: str | None = None,
    ) -> None:
        """Record human review outcome for a fraud alert."""
        record = {
            "event": "FRAUD_ALERT_ACTIONED",
            "alert_id": str(alert_id),
            "action": action_taken,
            "reviewer_id": reviewer_id,
            "logged_at": datetime.utcnow().isoformat(),
        }
        logger.info("AUDIT: %s", json.dumps(record))
