"""
AWB Unified Audit Event Log
============================
Single cross-system audit table satisfying
FCA COBS 9 seven-year retention for all 23
AI systems in the AWB-AI-2025 programme.

Schema:
  event_id        UUID PK
  system_id       VARCHAR  (MR-2026-035 etc.)
  mr_reference    VARCHAR  (PRA SS1/23 registry ID)
  awb_customer_id VARCHAR  (universal join key)
  decision_type   VARCHAR
  input_hash      VARCHAR  (SHA-256 of input)
  output_hash     VARCHAR  (SHA-256 of output)
  confidence      NUMERIC
  human_reviewed  BOOLEAN
  logged_at       TIMESTAMPTZ

Usage:
    from awb_commons.audit import AuditLogger
    logger = AuditLogger()
    await logger.log_decision(event)
"""
from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

DDL = """
CREATE TABLE IF NOT EXISTS unified_audit_log (
    event_id        UUID PRIMARY KEY,
    system_id       VARCHAR(32)  NOT NULL,
    mr_reference    VARCHAR(32)  NOT NULL,
    awb_customer_id VARCHAR(64)  NOT NULL,
    decision_type   VARCHAR(64)  NOT NULL,
    input_hash      VARCHAR(64),
    output_hash     VARCHAR(64),
    confidence      NUMERIC(5,4),
    human_reviewed  BOOLEAN DEFAULT FALSE,
    logged_at       TIMESTAMPTZ  NOT NULL
        DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS
    idx_ual_customer
    ON unified_audit_log (awb_customer_id);

CREATE INDEX IF NOT EXISTS
    idx_ual_system
    ON unified_audit_log (system_id, logged_at);

CREATE INDEX IF NOT EXISTS
    idx_ual_logged_at
    ON unified_audit_log (logged_at);

COMMENT ON TABLE unified_audit_log IS
    'FCA COBS 9: 7-year retention. '
    'AWB-AI-2025 programme all 23 systems.';
"""

RETENTION_POLICY = """
-- Run monthly via pg_cron or Airflow
-- Purge records older than 7 years (2555 days)
DELETE FROM unified_audit_log
WHERE logged_at < NOW() - INTERVAL '2555 days';
"""


class AuditEvent(BaseModel):
    """Single audit event for any AI decision."""

    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )
    system_id: str = Field(
        ...,
        description="MR-2026-035 etc.",
        max_length=32,
    )
    mr_reference: str = Field(
        ...,
        description="PRA SS1/23 registry ID",
        max_length=32,
    )
    awb_customer_id: str = Field(
        ...,
        description="Universal cross-system key",
        max_length=64,
    )
    decision_type: str = Field(
        ...,
        description=(
            "credit_decision | aml_alert | "
            "capital_calc | rag_query"
        ),
        max_length=64,
    )
    input_payload: Optional[Any] = None
    output_payload: Optional[Any] = None
    confidence: Optional[float] = Field(
        default=None, ge=0.0, le=1.0
    )
    human_reviewed: bool = False
    logged_at: datetime = Field(
        default_factory=lambda: datetime.now(
            tz=timezone.utc
        )
    )

    @property
    def input_hash(self) -> Optional[str]:
        """SHA-256 hash of input payload."""
        if self.input_payload is None:
            return None
        raw = str(self.input_payload).encode()
        return hashlib.sha256(raw).hexdigest()

    @property
    def output_hash(self) -> Optional[str]:
        """SHA-256 hash of output payload."""
        if self.output_payload is None:
            return None
        raw = str(self.output_payload).encode()
        return hashlib.sha256(raw).hexdigest()


class AuditLogger:
    """
    Write audit events to unified_audit_log.

    Thread-safe; uses connection pool.
    All writes are synchronous for reliability
    (fire-and-forget not permitted for audit data).
    """

    INSERT_SQL = """
        INSERT INTO unified_audit_log (
            event_id,
            system_id,
            mr_reference,
            awb_customer_id,
            decision_type,
            input_hash,
            output_hash,
            confidence,
            human_reviewed,
            logged_at
        ) VALUES (
            %(event_id)s,
            %(system_id)s,
            %(mr_reference)s,
            %(awb_customer_id)s,
            %(decision_type)s,
            %(input_hash)s,
            %(output_hash)s,
            %(confidence)s,
            %(human_reviewed)s,
            %(logged_at)s
        )
    """

    def __init__(self, db_conn: Any) -> None:
        self._conn = db_conn

    def log_decision(
        self, event: AuditEvent
    ) -> str:
        """
        Write a single audit event.

        Returns event_id on success.
        Raises RuntimeError if write fails
        (audit writes must not be silently dropped).
        """
        try:
            cur = self._conn.cursor()
            cur.execute(
                self.INSERT_SQL,
                {
                    "event_id": event.event_id,
                    "system_id": event.system_id,
                    "mr_reference": (
                        event.mr_reference
                    ),
                    "awb_customer_id": (
                        event.awb_customer_id
                    ),
                    "decision_type": (
                        event.decision_type
                    ),
                    "input_hash": event.input_hash,
                    "output_hash": event.output_hash,
                    "confidence": event.confidence,
                    "human_reviewed": (
                        event.human_reviewed
                    ),
                    "logged_at": event.logged_at,
                },
            )
            self._conn.commit()
            log.info(
                "audit_event_written "
                "event_id=%s system=%s customer=%s",
                event.event_id,
                event.system_id,
                event.awb_customer_id,
            )
            return event.event_id
        except Exception as exc:
            self._conn.rollback()
            log.error(
                "audit_write_failed "
                "event_id=%s error=%s",
                event.event_id,
                exc,
            )
            raise RuntimeError(
                f"Audit write failed: {exc}"
            ) from exc

    def query_by_customer(
        self,
        awb_customer_id: str,
        limit: int = 100,
    ) -> list[dict]:
        """
        Retrieve audit events for a customer.

        Args:
            awb_customer_id: AWB universal key.
            limit: Max records to return.

        Returns:
            List of audit event dicts.
        """
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT
                event_id,
                system_id,
                mr_reference,
                decision_type,
                confidence,
                human_reviewed,
                logged_at
            FROM unified_audit_log
            WHERE awb_customer_id = %s
            ORDER BY logged_at DESC
            LIMIT %s
            """,
            (awb_customer_id, limit),
        )
        cols = [d[0] for d in cur.description]
        return [
            dict(zip(cols, row))
            for row in cur.fetchall()
        ]
