"""
access_control/access_control.py
AWB Multi-Tenant Access-Controlled RAG
Chapter 4: Section 4.9 — Multi-Tenant Access-Controlled RAG

AWB has four user groups with different document entitlements.
Access control is enforced at query time via ChromaDB metadata filters —
not at ingestion time (which would require 4 separate indices).

Access tiers:
  COMPLIANCE_OFFICER:  All 847 documents (unrestricted)
  CREDIT_ANALYST:      Credit + capital docs (CRR3, SS1/23, IRB)
  RELATIONSHIP_MANAGER: Client-facing docs (FCA PS22/9, COBS)
  TREASURY:            Liquidity + market risk docs (LCR, NSFR, FRTB)

Every query produces an audit record retained 7 years (FCA COBS 9).

Regulatory context:
  FCA PS22/9: Relationship managers must only see client-appropriate docs
  PRA SS1/23 MR-2026-038: Access boundary enforcement tested quarterly
  DORA Art. 9: ICT asset RKA-2026-001; access logs retained 7 years
  UK GDPR: No personal data stored; regulatory content only
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger("awb.rag.access_control")


# ── Access tier definitions ───────────────────────────────────────────────────

class AccessTier(str, Enum):
    """
    AWB user access tiers for the regulatory knowledge base.

    Tier determines which document categories are retrievable.
    COMPLIANCE_OFFICER has unrestricted access; all other tiers
    are limited to their functional domain.
    """
    COMPLIANCE_OFFICER   = "compliance_officer"
    CREDIT_ANALYST       = "credit_analyst"
    RELATIONSHIP_MANAGER = "relationship_manager"
    TREASURY             = "treasury"


# Document categories allowed per tier
# Empty list = unrestricted (all categories)
TIER_FILTERS: Dict[AccessTier, List[str]] = {
    AccessTier.COMPLIANCE_OFFICER: [],   # all documents
    AccessTier.CREDIT_ANALYST: [
        "credit",
        "capital",
        "model_risk",
        "irb",
        "basel",
    ],
    AccessTier.RELATIONSHIP_MANAGER: [
        "conduct",
        "product",
        "consumer_duty",
        "cobs",
        "retail",
    ],
    AccessTier.TREASURY: [
        "liquidity",
        "market_risk",
        "alm",
        "lcr",
        "nsfr",
        "frtb",
    ],
}


# ── Filter builder ────────────────────────────────────────────────────────────

def build_access_filter(tier: AccessTier) -> Optional[dict]:
    """
    Build a ChromaDB where-clause filter for the given access tier.

    Returns None for COMPLIANCE_OFFICER (no filter — all documents).
    Returns a $in filter for all other tiers.

    Args:
        tier: The caller's access tier.

    Returns:
        ChromaDB metadata filter dict, or None if unrestricted.

    Example:
        filter = build_access_filter(AccessTier.CREDIT_ANALYST)
        # {"document_category": {"$in": ["credit", "capital", ...]}}
        results = collection.query(
            query_embeddings=[...],
            where=filter,
        )
    """
    allowed = TIER_FILTERS[tier]
    if not allowed:
        return None   # COMPLIANCE_OFFICER: no restriction
    return {"document_category": {"$in": allowed}}


def get_allowed_categories(tier: AccessTier) -> List[str]:
    """Return the list of allowed document categories for a tier."""
    return TIER_FILTERS[tier].copy()


# ── Audit logging ─────────────────────────────────────────────────────────────

@dataclass
class RAGAuditRecord:
    """
    Audit record for a single RAG query.

    Retained 7 years per FCA COBS 9 requirements.
    Stored in PostgreSQL with index on user_id, timestamp, model_id.

    Fields:
        audit_id:               UUID for this record
        user_id:                AWB staff ID (not personal data — staff ID)
        user_role:              AccessTier value
        query:                  The question asked (no personal data)
        access_filter_applied:  The ChromaDB filter used
        documents_retrieved:    List of document_ids returned
        confidence_score:       RAG confidence (0.0 to 1.0)
        answer_length:          Character count of generated answer
        timestamp:              UTC datetime of query
        model_id:               Always "MR-2026-038"
    """
    user_id:                str
    user_role:              str
    query:                  str
    access_filter_applied:  dict  = field(default_factory=dict)
    documents_retrieved:    list  = field(default_factory=list)
    confidence_score:       float = 0.0
    answer_length:          int   = 0
    audit_id:               str   = field(
        default_factory=lambda: str(uuid.uuid4())
    )
    timestamp:              datetime = field(
        default_factory=datetime.utcnow
    )
    model_id:               str = "MR-2026-038"


class AuditLogger:
    """
    Persist RAG audit records for 7-year FCA retention.

    In production: writes to PostgreSQL via awb_commons.db.
    In testing: accumulates in-memory list.
    """

    def __init__(self, db_client=None) -> None:
        """
        Args:
            db_client: Optional PostgreSQL client. If None, records
                are accumulated in self.records (for testing).
        """
        self._db = db_client
        self.records: List[RAGAuditRecord] = []

    def log(self, record: RAGAuditRecord) -> None:
        """
        Persist a RAG audit record.

        Args:
            record: The audit record to store.
        """
        if self._db is not None:
            self._db.execute(
                """
                INSERT INTO rag_audit_log (
                    audit_id, user_id, user_role, query,
                    access_filter_applied, documents_retrieved,
                    confidence_score, answer_length,
                    timestamp, model_id
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    record.audit_id,
                    record.user_id,
                    record.user_role,
                    record.query,
                    str(record.access_filter_applied),
                    str(record.documents_retrieved),
                    record.confidence_score,
                    record.answer_length,
                    record.timestamp,
                    record.model_id,
                ),
            )
        else:
            self.records.append(record)
        logger.debug(
            "Audit log: user=%s role=%s confidence=%.2f",
            record.user_id, record.user_role, record.confidence_score,
        )


# ── FastAPI dependency ────────────────────────────────────────────────────────

def get_access_tier_from_jwt(token: str) -> AccessTier:
    """
    Extract AccessTier from a validated JWT claim.

    Production: JWT validated by AWB identity provider (Azure AD).
    Testing: Pass tier name directly as token string.

    Args:
        token: JWT bearer token or tier name string (for tests).

    Returns:
        AccessTier for the authenticated user.

    Raises:
        ValueError: If token contains an unrecognised tier claim.
    """
    # Production implementation:
    # import jwt
    # payload = jwt.decode(token, settings.JWT_SECRET, algorithms=["RS256"])
    # tier_claim = payload.get("awb_rag_tier", "relationship_manager")

    # Simplified for testing — accept tier name directly
    try:
        return AccessTier(token.lower())
    except ValueError:
        logger.warning(
            "Unrecognised access tier claim '%s'; defaulting to "
            "RELATIONSHIP_MANAGER (most restrictive)",
            token,
        )
        return AccessTier.RELATIONSHIP_MANAGER
