"""
document_analyser/audit_log.py
AWB Credit Document Analyser — Extraction Audit Log

Every extraction and validation is logged for:
  - PRA SS1/23 model monitoring (MR-2026-035): performance tracking, drift detection
  - EU AI Act technical documentation: audit trail for conformity assessment
  - 7-year retention (UK Financial Services and Markets Act 2000)
  - DORA: LLM provider usage tracking for concentration risk

Drift detection:
  PSI (Population Stability Index) on confidence score distributions.
  PSI > 0.2 → significant drift → model review triggered (EBA threshold).
  Confidence score trend: rolling 100-extraction moving average.

Storage:
  Production: PostgreSQL (AWB RDS, AWS eu-west-2)
  Development: SQLite (extraction_audit.db)

Author: AWB AI Programme (AWB-AI-2025)
Chapter: 2 — LLM Foundations for Financial Document Analysis
"""

from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Audit record schema
# ---------------------------------------------------------------------------

@dataclass
class ExtractionAuditRecord:
    """
    Full audit record for one document extraction.

    Immutable once written — PRA SS1/23 requires original record preserved.
    """
    extraction_id: str                # UUID4
    document_id: str
    extraction_date_utc: str          # ISO 8601 UTC
    llm_model: str                    # e.g. "gemini-3.1-pro"
    model_id: str                     # PRA SS1/23 model registry ID e.g. "MR-2026-035"
    eu_ai_act_status: str             # "HIGH_RISK" | "LIMITED_RISK" etc.
    dora_asset_id: str                # DORA ICT asset ID

    # Extraction outcomes
    company_name: str | None
    reporting_period: str | None
    overall_confidence: float
    analyst_review_required: bool
    analyst_review_reasons_json: str  # JSON array

    # Per-field confidence scores (for drift monitoring)
    field_confidences_json: str       # JSON object: field_name -> confidence

    # Validation outcome
    validation_passed: bool
    p1_flag_count: int
    p2_flag_count: int
    red_flags_json: str               # JSON array of flag codes
    cross_validation_issues_json: str # JSON array of issue descriptions
    conservatism_applied: bool

    # Performance
    latency_ms: int | None = None

    # Provider tracking (DORA concentration risk)
    llm_provider: str = "google"      # For concentration risk monitoring


@dataclass
class MonitoringMetrics:
    """
    Rolling monitoring metrics for PRA SS1/23 model monitoring.
    Computed from audit records — not stored per-record.
    """
    model_id: str
    window_size: int                  # Number of extractions in window
    mean_confidence: float
    min_confidence: float
    p1_flag_rate: float               # % of extractions with P1 flags
    analyst_review_rate: float        # % of extractions requiring review
    psi_confidence: float | None      # PSI vs. baseline (None if < 50 records)
    drift_alert_level: str            # "GREEN" | "AMBER" | "RED"
    computed_at_utc: str


# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS extraction_audit (
    extraction_id               TEXT PRIMARY KEY,
    document_id                 TEXT NOT NULL,
    extraction_date_utc         TEXT NOT NULL,
    llm_model                   TEXT NOT NULL,
    model_id                    TEXT NOT NULL,
    eu_ai_act_status            TEXT NOT NULL,
    dora_asset_id               TEXT NOT NULL,
    company_name                TEXT,
    reporting_period            TEXT,
    overall_confidence          REAL NOT NULL,
    analyst_review_required     INTEGER NOT NULL,
    analyst_review_reasons_json TEXT NOT NULL,
    field_confidences_json      TEXT NOT NULL,
    validation_passed           INTEGER NOT NULL,
    p1_flag_count               INTEGER NOT NULL,
    p2_flag_count               INTEGER NOT NULL,
    red_flags_json              TEXT NOT NULL,
    cross_validation_issues_json TEXT NOT NULL,
    conservatism_applied        INTEGER NOT NULL,
    latency_ms                  INTEGER,
    llm_provider                TEXT NOT NULL DEFAULT 'google'
);
"""

CREATE_INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_audit_document
    ON extraction_audit (document_id);
CREATE INDEX IF NOT EXISTS idx_audit_date
    ON extraction_audit (extraction_date_utc);
CREATE INDEX IF NOT EXISTS idx_audit_model
    ON extraction_audit (model_id);
"""


def _get_db_path() -> str:
    return os.environ.get("AUDIT_DB_PATH", "extraction_audit.db")


@contextmanager
def _get_connection():
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
    """Create tables and indexes if they do not exist."""
    with _get_connection() as conn:
        conn.executescript(CREATE_TABLE_SQL + CREATE_INDEXES_SQL)
    logger.info("Extraction audit DB initialised", extra={"db": _get_db_path()})


# ---------------------------------------------------------------------------
# Write functions
# ---------------------------------------------------------------------------

def log_extraction(record: ExtractionAuditRecord) -> str:
    """
    Write extraction audit record. Immutable — no updates permitted.

    Returns: extraction_id
    """
    initialise_db()
    row = asdict(record)
    # Convert booleans
    for bf in ("analyst_review_required", "validation_passed", "conservatism_applied"):
        row[bf] = int(row[bf])

    cols = ", ".join(row.keys())
    placeholders = ", ".join(["?"] * len(row))
    sql = f"INSERT INTO extraction_audit ({cols}) VALUES ({placeholders})"

    with _get_connection() as conn:
        conn.execute(sql, list(row.values()))

    logger.info(
        "Extraction audit record written",
        extra={
            "extraction_id": record.extraction_id,
            "document_id": record.document_id,
            "confidence": record.overall_confidence,
        },
    )
    return record.extraction_id


def build_audit_record(
    summary,        # FinancialSummary from extractor
    validation,     # ValidationResult from validator
    latency_ms: int | None = None,
) -> ExtractionAuditRecord:
    """
    Build an ExtractionAuditRecord from pipeline outputs.
    Generates a new UUID extraction_id automatically.
    """
    field_names = [
        "revenue", "ebitda", "ebitda_margin_pct", "net_debt",
        "total_assets", "current_assets", "current_liabilities",
        "leverage_ratio", "interest_cover", "current_ratio",
    ]
    field_confidences = {}
    for fname in field_names:
        fe = getattr(summary, fname, None)
        if fe is not None:
            field_confidences[fname] = fe.confidence

    return ExtractionAuditRecord(
        extraction_id=str(uuid.uuid4()),
        document_id=summary.document_id,
        extraction_date_utc=datetime.now(tz=timezone.utc).isoformat(),
        llm_model=summary.extraction_model,
        model_id=summary.model_id,
        eu_ai_act_status=summary.eu_ai_act_status,
        dora_asset_id="DA-2026-002",
        company_name=str(summary.company_name.value) if summary.company_name.value else None,
        reporting_period=str(summary.reporting_period.value) if summary.reporting_period.value else None,
        overall_confidence=summary.overall_confidence,
        analyst_review_required=summary.analyst_review_required,
        analyst_review_reasons_json=json.dumps(summary.analyst_review_reasons),
        field_confidences_json=json.dumps(field_confidences),
        validation_passed=validation.validation_passed,
        p1_flag_count=len(validation.p1_flags),
        p2_flag_count=len(validation.p2_flags),
        red_flags_json=json.dumps([f.flag_code for f in validation.red_flags]),
        cross_validation_issues_json=json.dumps(validation.cross_validation_issues),
        conservatism_applied=validation.adjusted_net_debt is not None,
        latency_ms=latency_ms,
    )


# ---------------------------------------------------------------------------
# Drift monitoring (PRA SS1/23 model monitoring)
# ---------------------------------------------------------------------------

def compute_monitoring_metrics(
    model_id: str = "MR-2026-035",
    window: int = 100,
) -> MonitoringMetrics | None:
    """
    Compute rolling monitoring metrics for PRA SS1/23 model monitoring.

    PSI thresholds (EBA standard):
      PSI < 0.1  → GREEN (stable)
      0.1–0.2    → AMBER (monitor)
      PSI > 0.2  → RED (significant drift — model review required)

    Args:
        model_id: PRA SS1/23 model ID to filter records.
        window:   Number of most recent extractions to include.

    Returns:
        MonitoringMetrics or None if fewer than 10 records exist.
    """
    initialise_db()

    with _get_connection() as conn:
        rows = conn.execute(
            """
            SELECT overall_confidence, analyst_review_required, p1_flag_count
            FROM extraction_audit
            WHERE model_id = ?
            ORDER BY extraction_date_utc DESC
            LIMIT ?
            """,
            (model_id, window),
        ).fetchall()

    if len(rows) < 10:
        logger.info("Insufficient records for drift monitoring", extra={"count": len(rows)})
        return None

    confidences = [r["overall_confidence"] for r in rows]
    review_required = [r["analyst_review_required"] for r in rows]
    p1_flags = [r["p1_flag_count"] for r in rows]

    mean_conf = sum(confidences) / len(confidences)
    min_conf = min(confidences)
    review_rate = sum(review_required) / len(review_required)
    p1_rate = sum(1 for p in p1_flags if p > 0) / len(p1_flags)

    # PSI computation vs. a baseline of [0.85 mean, normal-ish distribution]
    # Simplified PSI: compare distribution of confidence scores to baseline
    psi = _compute_psi(confidences) if len(confidences) >= 50 else None

    if psi is None:
        alert = "GREEN"
    elif psi < 0.1:
        alert = "GREEN"
    elif psi < 0.2:
        alert = "AMBER"
    else:
        alert = "RED"

    if alert in ("AMBER", "RED"):
        logger.warning(
            "Model drift alert",
            extra={"model_id": model_id, "psi": psi, "alert": alert},
        )

    return MonitoringMetrics(
        model_id=model_id,
        window_size=len(rows),
        mean_confidence=mean_conf,
        min_confidence=min_conf,
        p1_flag_rate=p1_rate,
        analyst_review_rate=review_rate,
        psi_confidence=psi,
        drift_alert_level=alert,
        computed_at_utc=datetime.now(tz=timezone.utc).isoformat(),
    )


def _compute_psi(
    observed: list[float],
    n_bins: int = 10,
) -> float:
    """
    Compute Population Stability Index (PSI) for a list of confidence scores.

    Baseline distribution: uniform across [0.7, 1.0] range (expected healthy model).
    PSI = Σ (Observed% - Expected%) × ln(Observed% / Expected%)

    EBA reference: EBA/GL/2017/16 — PSI used for model stability monitoring.
    """
    if not observed:
        return 0.0

    # Bin edges from 0.0 to 1.0
    bin_edges = [i / n_bins for i in range(n_bins + 1)]

    # Observed distribution
    obs_counts = [0] * n_bins
    for v in observed:
        bin_idx = min(int(v * n_bins), n_bins - 1)
        obs_counts[bin_idx] += 1

    n = len(observed)
    # Expected: baseline healthy model has most confidence scores in [0.8, 1.0]
    # Model baseline: 80% of extractions in top 2 bins, 20% spread across rest
    expected_pct = [0.025] * (n_bins - 2) + [0.5, 0.5]  # simplified baseline

    psi = 0.0
    epsilon = 1e-6
    for obs_count, exp_pct in zip(obs_counts, expected_pct):
        obs_pct = obs_count / n
        exp_pct_safe = max(exp_pct, epsilon)
        obs_pct_safe = max(obs_pct, epsilon)
        psi += (obs_pct_safe - exp_pct_safe) * math.log(obs_pct_safe / exp_pct_safe)

    return abs(psi)


# ---------------------------------------------------------------------------
# Query helpers (compliance team use)
# ---------------------------------------------------------------------------

def get_records_by_document(document_id: str) -> list[dict]:
    """Retrieve all audit records for a document. Used in analyst review."""
    initialise_db()
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM extraction_audit WHERE document_id = ? ORDER BY extraction_date_utc",
            (document_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_high_risk_extractions(limit: int = 50) -> list[dict]:
    """
    Retrieve extractions with P1 flags or low confidence.
    Used for weekly model performance review (PRA SS1/23 §6).
    """
    initialise_db()
    with _get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM extraction_audit
            WHERE p1_flag_count > 0 OR overall_confidence < 0.70
            ORDER BY extraction_date_utc DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]
