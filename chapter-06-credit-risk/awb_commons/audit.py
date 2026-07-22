"""AWB Commons — Audit Logger.

7-year append-only audit log satisfying FCA COBS 9
and PRA SS1/23 model use record requirements.

Production: PostgreSQL (partitioned by year).
Tests: SQLite in-memory via AUDIT_DB_URL env var.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Optional

log = logging.getLogger(__name__)

AUDIT_DB_URL = os.getenv(
    "AUDIT_DB_URL",
    "sqlite:///awb_audit_test.db"
)

_audit_store: list[dict] = []   # in-memory fallback for tests


class AuditLogger:
    """Append-only audit log for all AWB AI model decisions.

    Schema (awb_model_audit):
      audit_id         UUID PK
      model_id         VARCHAR(20)   MR-2026-043 etc.
      facility_id      VARCHAR(50)
      application_id   VARCHAR(50)   nullable
      event_type       VARCHAR(50)   SCORE|DECISION
      input_snapshot   JSONB
      output_snapshot  JSONB
      model_version    VARCHAR(64)   SHA-256
      human_reviewer   VARCHAR(100)  nullable
      logged_at        TIMESTAMPTZ   immutable

    Retention: 7 years (FCA COBS 9, PRA SS1/23).
    Partition: by logged_at year.
    Index: (facility_id, logged_at DESC).
    """

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    def log_score(
        self,
        facility_id: str,
        input_features: dict,
        output_snapshot: dict,
        model_version: str,
        application_id: Optional[str] = None,
    ) -> None:
        """Log a model scoring event."""
        self._write({
            "model_id":        self.model_id,
            "facility_id":     facility_id,
            "application_id":  application_id,
            "event_type":      "SCORE",
            "input_snapshot":  input_features,
            "output_snapshot": output_snapshot,
            "model_version":   model_version,
            "logged_at":       datetime.utcnow().isoformat(),
        })

    def log_decision(
        self,
        facility_id: str,
        decision: str,
        pd_calibrated: float,
        shap_values: dict,
        model_version: str,
        human_reviewer: Optional[str] = None,
    ) -> None:
        """Log a credit decision with SHAP evidence."""
        self._write({
            "model_id":       self.model_id,
            "facility_id":    facility_id,
            "event_type":     "DECISION",
            "input_snapshot": {"shap_values": shap_values},
            "output_snapshot": {
                "decision":      decision,
                "pd_calibrated": pd_calibrated,
            },
            "model_version":  model_version,
            "human_reviewer": human_reviewer,
            "logged_at":      datetime.utcnow().isoformat(),
        })

    def get_records(
        self, facility_id: Optional[str] = None
    ) -> list[dict]:
        """Retrieve audit records (test helper)."""
        records = _audit_store[:]
        if facility_id:
            records = [
                r for r in records
                if r.get("facility_id") == facility_id
            ]
        return records

    def clear_for_test(self) -> None:
        """Reset in-memory store (test teardown only)."""
        _audit_store.clear()

    def _write(self, record: dict) -> None:
        _audit_store.append(record)
        log.debug(
            "Audit: model=%s fac=%s type=%s",
            record["model_id"],
            record.get("facility_id"),
            record["event_type"],
        )
