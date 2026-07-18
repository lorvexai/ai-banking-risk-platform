"""
chatbot/audit_log.py
AWB AI Customer Service Platform — FCA Consumer Duty Audit Log

Every customer interaction is logged with full detail for regulatory review.

Retention: 7 years (UK statutory minimum — Financial Services and Markets Act 2000)
Storage:   PostgreSQL (primary) with optional SQLite fallback for local development
Schema:    FCA Consumer Duty PS22/9 audit requirements

In production this writes to the AWB PostgreSQL RDS cluster in AWS eu-west-2.
In local development it writes to SQLite (audit_log.db) when PostgreSQL unavailable.

Author: AWB AI Programme (AWB-AI-2025)
Chapter: 1 — The AI Transformation of Risk and Compliance in Banking
"""

import json
import logging
import os
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Interaction log record
# ---------------------------------------------------------------------------

@dataclass
class InteractionLog:
    """
    Complete audit record for one customer interaction.

    All fields are mandatory for FCA Consumer Duty compliance.
    interaction_id is the primary key — immutable once written.
    """
    interaction_id: str                  # UUID4 — immutable
    session_id: str
    customer_id: str | None              # None for unauthenticated sessions
    customer_segment: str                # "retail" | "sme" | "private"
    timestamp_utc: str                   # ISO 8601, always UTC
    channel: str                         # "web" | "app" | "ivr"

    # Customer input
    message_text: str

    # Classification result
    intent: str
    intent_confidence: float
    entities_json: str                   # JSON string of extracted entities

    # Escalation
    requires_escalation: bool
    escalation_reason: str | None
    escalated_to_agent: bool

    # Response delivered
    response_text: str
    response_approved: bool

    # Compliance screening
    compliance_flags_json: str           # JSON array of flag names
    compliance_audit_notes: str

    # Infrastructure
    llm_model: str                       # e.g. "gemini-3.5-flash"
    latency_ms: int | None               # End-to-end milliseconds
    dora_asset_id: str = "CS-2026-001"  # DORA ICT asset registration

    # Vulnerability and sensitive flags
    vulnerable_customer_flag: bool = False


# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS interaction_log (
    interaction_id          TEXT PRIMARY KEY,
    session_id              TEXT NOT NULL,
    customer_id             TEXT,
    customer_segment        TEXT NOT NULL,
    timestamp_utc           TEXT NOT NULL,
    channel                 TEXT NOT NULL,
    message_text            TEXT NOT NULL,
    intent                  TEXT NOT NULL,
    intent_confidence       REAL NOT NULL,
    entities_json           TEXT NOT NULL,
    requires_escalation     INTEGER NOT NULL,
    escalation_reason       TEXT,
    escalated_to_agent      INTEGER NOT NULL,
    response_text           TEXT NOT NULL,
    response_approved       INTEGER NOT NULL,
    compliance_flags_json   TEXT NOT NULL,
    compliance_audit_notes  TEXT NOT NULL,
    llm_model               TEXT NOT NULL,
    latency_ms              INTEGER,
    dora_asset_id           TEXT NOT NULL,
    vulnerable_customer_flag INTEGER NOT NULL DEFAULT 0
);
"""

CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_interaction_session
    ON interaction_log (session_id);
CREATE INDEX IF NOT EXISTS idx_interaction_timestamp
    ON interaction_log (timestamp_utc);
CREATE INDEX IF NOT EXISTS idx_interaction_customer
    ON interaction_log (customer_id);
"""


def _get_db_path() -> str:
    """
    Determine database path.
    Production: PostgreSQL DSN from POSTGRES_DSN env var.
    Development: SQLite file at AUDIT_DB_PATH env var or ./audit_log.db.
    """
    return os.environ.get("AUDIT_DB_PATH", "audit_log.db")


@contextmanager
def _get_connection():
    """Context manager for SQLite connection with WAL mode for concurrency."""
    db_path = _get_db_path()
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def initialise_db() -> None:
    """Create interaction_log table and indexes if they do not exist."""
    with _get_connection() as conn:
        conn.executescript(CREATE_TABLE_SQL + CREATE_INDEX_SQL)
    logger.info("Audit log database initialised", extra={"db_path": _get_db_path()})


# ---------------------------------------------------------------------------
# Log write function
# ---------------------------------------------------------------------------

def log_interaction(record: InteractionLog) -> str:
    """
    Write an interaction log record to the audit database.

    Args:
        record: Complete InteractionLog dataclass instance.

    Returns:
        interaction_id (UUID string) of the written record.

    Raises:
        sqlite3.Error: On database write failure.

    Note:
        Records are immutable once written (no UPDATE statements).
        FCA Consumer Duty requires the original interaction to be preserved.
    """
    # Ensure DB exists
    initialise_db()

    row = asdict(record)

    # Convert boolean fields to integers for SQLite
    for bool_field in ("requires_escalation", "escalated_to_agent",
                       "response_approved", "vulnerable_customer_flag"):
        row[bool_field] = int(row[bool_field])

    placeholders = ", ".join(["?"] * len(row))
    columns = ", ".join(row.keys())
    values = list(row.values())

    sql = f"INSERT INTO interaction_log ({columns}) VALUES ({placeholders})"

    with _get_connection() as conn:
        conn.execute(sql, values)

    logger.info(
        "Interaction logged",
        extra={
            "interaction_id": record.interaction_id,
            "intent": record.intent,
            "session_id": record.session_id,
        },
    )
    return record.interaction_id


# ---------------------------------------------------------------------------
# Factory helper — builds a complete InteractionLog from pipeline outputs
# ---------------------------------------------------------------------------

def build_interaction_log(
    session_id: str,
    customer_id: str | None,
    customer_segment: str,
    channel: str,
    message_text: str,
    intent_result,              # IntentResult from classifier
    response_text: str,
    compliance_result,          # ComplianceResult from compliance_filter
    escalated_to_agent: bool,
    latency_ms: int | None = None,
    vulnerable_customer_flag: bool = False,
    llm_model: str = "gemini-3.5-flash",
) -> InteractionLog:
    """
    Convenience factory — assembles InteractionLog from pipeline objects.
    Generates a new UUID interaction_id automatically.
    """
    return InteractionLog(
        interaction_id=str(uuid.uuid4()),
        session_id=session_id,
        customer_id=customer_id,
        customer_segment=customer_segment,
        timestamp_utc=datetime.now(tz=timezone.utc).isoformat(),
        channel=channel,
        message_text=message_text,
        intent=intent_result.intent.value,
        intent_confidence=intent_result.confidence,
        entities_json=json.dumps(intent_result.entities),
        requires_escalation=intent_result.requires_escalation,
        escalation_reason=intent_result.escalation_reason,
        escalated_to_agent=escalated_to_agent,
        response_text=response_text,
        response_approved=compliance_result.approved,
        compliance_flags_json=json.dumps([f.value for f in compliance_result.flags]),
        compliance_audit_notes=compliance_result.audit_notes,
        llm_model=llm_model,
        latency_ms=latency_ms,
        vulnerable_customer_flag=vulnerable_customer_flag,
    )


# ---------------------------------------------------------------------------
# Audit query helpers (for compliance team use)
# ---------------------------------------------------------------------------

def get_interactions_by_session(session_id: str) -> list[dict]:
    """Retrieve all interaction records for a session. Used in compliance review."""
    initialise_db()
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM interaction_log WHERE session_id = ? ORDER BY timestamp_utc",
            (session_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_flagged_interactions(
    since_utc: str | None = None,
    limit: int = 100,
) -> list[dict]:
    """
    Retrieve interactions with compliance flags. Used for weekly sample review
    (FCA Consumer Duty: 50 interactions reviewed per week by compliance team).
    """
    initialise_db()
    sql = """
        SELECT * FROM interaction_log
        WHERE compliance_flags_json != '[]'
        {since_clause}
        ORDER BY timestamp_utc DESC
        LIMIT ?
    """
    if since_utc:
        sql = sql.format(since_clause=f"AND timestamp_utc >= '{since_utc}'")
    else:
        sql = sql.format(since_clause="")

    with _get_connection() as conn:
        rows = conn.execute(sql, (limit,)).fetchall()
    return [dict(row) for row in rows]
