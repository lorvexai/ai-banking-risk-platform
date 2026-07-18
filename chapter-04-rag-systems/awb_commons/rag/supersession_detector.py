"""
awb_commons/rag/supersession_detector.py
AWB Document Lifecycle Management — Supersession Detection
Chapter 4: Section 4.7 — Document Freshness and Lifecycle Management

Root cause of the £12M capital near-miss (Section 4.1):
  AWB's corpus contained DRAFT and FINAL CRR3 documents with no version
  tagging. The RAG system retrieved the 2022 draft EBA consultation
  paper instead of the 2024 final CRR3 standard, producing a wrong
  risk weight for a £48M specialised lending exposure.

This module implements the lifecycle management system built in response:
  - Four document states: DRAFT / CONSULTATION / FINAL / SUPERSEDED
  - Automatic supersession detection on new FINAL publications
  - State change audit log (PRA SS1/23 traceability)
  - Freshness monitoring: flag documents > 18 months without review

Integration:
  - Chapter 2 Regulatory Intelligence Monitor publishes new documents
  - This detector marks superseded documents within 24 hours
  - Section 4.4 RKA filters SUPERSEDED documents at query time

Regulatory context:
  PRA SS1/23 MR-2026-038: audit trail required for all corpus changes
  FCA PS22/9: stale regulatory citations are a consumer duty risk
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional

logger = logging.getLogger("awb.rag.supersession")


# ── Document lifecycle states ─────────────────────────────────────────────────

class DocumentStatus(str, Enum):
    """
    Regulatory document lifecycle state.

    Retrieval behaviour by state:
      FINAL:        Default retrieval — highest priority
      DRAFT:        Excluded from default retrieval
      CONSULTATION: Returned only if explicitly requested
      SUPERSEDED:   Excluded from retrieval; archived
    """
    DRAFT        = "DRAFT"
    CONSULTATION = "CONSULTATION"
    FINAL        = "FINAL"
    SUPERSEDED   = "SUPERSEDED"


# States excluded from default RAG retrieval
NON_RETRIEVABLE_STATES = {
    DocumentStatus.DRAFT,
    DocumentStatus.SUPERSEDED,
}

# Flag documents not reviewed within this threshold
FRESHNESS_THRESHOLD_DAYS = 548  # 18 months


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class SupersessionResult:
    """Result of a supersession detection run."""
    new_document_id:  str
    superseded_ids:   List[str]
    effective_date:   datetime
    detection_method: str = "semantic_similarity"

    @property
    def superseded_count(self) -> int:
        return len(self.superseded_ids)


@dataclass
class StateChangeRecord:
    """
    Audit record for a document state change.
    Retained 7 years per PRA SS1/23 requirements.
    """
    document_id:   str
    from_status:   DocumentStatus
    to_status:     DocumentStatus
    changed_at:    datetime = field(default_factory=datetime.utcnow)
    changed_by:    str = "awb.rag.supersession_detector"
    reason:        str = ""
    triggered_by:  Optional[str] = None   # document_id that caused change


@dataclass
class FreshnessAlert:
    """Document flagged as potentially stale."""
    document_id:   str
    document_name: str
    last_reviewed: datetime
    days_since_review: int
    assigned_to:   str = "compliance-team@awb.co.uk"


# ── Main detector class ───────────────────────────────────────────────────────

class SupersessionDetector:
    """
    Detect when a new FINAL publication supersedes older documents.

    AWB target: mark superseded documents within 24 hours of a new
    FINAL publication being detected by the Regulatory Intelligence
    Monitor (Chapter 2, Section 2.4).

    Usage:
        detector = SupersessionDetector()
        result = detector.detect_and_mark(
            new_doc_id="CRR3_final_2024",
            corpus_client=chroma_client,
        )
        print(f"Marked {result.superseded_count} documents SUPERSEDED")
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        audit_log: Optional[list] = None,
    ) -> None:
        """
        Args:
            similarity_threshold: Cosine similarity above which two
                documents are considered to cover the same regulation.
                AWB default: 0.85.
            audit_log: Optional external list for state change records.
                If None, records are logged only via logger.
        """
        self.similarity_threshold = similarity_threshold
        self._audit_log: List[StateChangeRecord] = (
            audit_log if audit_log is not None else []
        )

    def detect_and_mark(
        self,
        new_doc_id: str,
        corpus_client,
    ) -> SupersessionResult:
        """
        Identify and mark documents superseded by new_doc_id.

        Steps:
          1. Retrieve embedding for new_doc_id
          2. Query corpus for semantically similar FINAL/CONSULTATION docs
          3. Mark matches with older effective_date as SUPERSEDED
          4. Write audit records for each state change

        Args:
            new_doc_id: ID of the newly ingested FINAL document.
            corpus_client: ChromaDB collection client supporting
                .get(ids), .update(ids, metadatas), .query(embeddings).

        Returns:
            SupersessionResult with list of superseded document IDs.

        Raises:
            KeyError: If new_doc_id is not found in the corpus.
        """
        logger.info("Running supersession detection for %s", new_doc_id)

        superseded_ids = self._find_superseded(new_doc_id, corpus_client)

        for doc_id in superseded_ids:
            self._mark_superseded(
                doc_id=doc_id,
                triggered_by=new_doc_id,
                corpus_client=corpus_client,
            )

        result = SupersessionResult(
            new_document_id=new_doc_id,
            superseded_ids=superseded_ids,
            effective_date=datetime.utcnow(),
        )
        logger.info(
            "Supersession complete: %s -> %d superseded",
            new_doc_id, result.superseded_count,
        )
        return result

    def run_freshness_audit(
        self,
        corpus_client,
        as_of: Optional[datetime] = None,
    ) -> List[FreshnessAlert]:
        """
        Flag FINAL documents not reviewed in > 18 months.

        AWB compliance team reviews flagged documents to confirm
        they remain current. Run weekly via scheduled task.

        Args:
            corpus_client: ChromaDB collection client.
            as_of: Reference date for freshness calculation.
                Defaults to datetime.utcnow().

        Returns:
            List of FreshnessAlert for documents needing review.
        """
        cutoff = (as_of or datetime.utcnow()) - timedelta(
            days=FRESHNESS_THRESHOLD_DAYS
        )
        results = corpus_client.get(
            where={"status": DocumentStatus.FINAL.value},
            include=["metadatas"],
        )
        alerts: List[FreshnessAlert] = []
        for meta in results.get("metadatas", []):
            last_rev_str = meta.get("last_reviewed", "")
            if not last_rev_str:
                continue
            try:
                last_rev = datetime.fromisoformat(last_rev_str)
            except ValueError:
                continue
            if last_rev < cutoff:
                days = (datetime.utcnow() - last_rev).days
                alerts.append(FreshnessAlert(
                    document_id=meta.get("document_id", ""),
                    document_name=meta.get("document_name", ""),
                    last_reviewed=last_rev,
                    days_since_review=days,
                ))
        logger.info(
            "Freshness audit: %d documents flagged for review",
            len(alerts),
        )
        return alerts

    # ── Private helpers ───────────────────────────────────────────────────────

    def _find_superseded(
        self,
        new_doc_id: str,
        corpus_client,
    ) -> List[str]:
        """
        Find existing documents likely superseded by new_doc_id.
        Uses semantic similarity + effective_date comparison.
        """
        new_docs = corpus_client.get(
            ids=[new_doc_id], include=["embeddings", "metadatas"]
        )
        if not new_docs["ids"]:
            raise KeyError(f"Document not found in corpus: {new_doc_id}")

        new_embedding = new_docs["embeddings"][0]
        new_meta = new_docs["metadatas"][0]
        new_date_str = new_meta.get("effective_date", "")

        candidates = corpus_client.query(
            query_embeddings=[new_embedding],
            n_results=20,
            where={
                "status": {
                    "$in": [
                        DocumentStatus.FINAL.value,
                        DocumentStatus.CONSULTATION.value,
                    ]
                }
            },
            include=["metadatas", "distances"],
        )
        superseded = []
        for cand_id, dist, meta in zip(
            candidates["ids"][0],
            candidates["distances"][0],
            candidates["metadatas"][0],
        ):
            if cand_id == new_doc_id:
                continue
            similarity = 1 - dist   # ChromaDB returns L2; approx cosine
            if similarity < self.similarity_threshold:
                continue
            cand_date_str = meta.get("effective_date", "")
            if self._is_older(cand_date_str, new_date_str):
                superseded.append(cand_id)
        return superseded

    def _mark_superseded(
        self,
        doc_id: str,
        triggered_by: str,
        corpus_client,
    ) -> None:
        """Update document status to SUPERSEDED and write audit record."""
        current = corpus_client.get(ids=[doc_id], include=["metadatas"])
        current_status_str = current["metadatas"][0].get(
            "status", DocumentStatus.FINAL.value
        )
        current_status = DocumentStatus(current_status_str)

        corpus_client.update(
            ids=[doc_id],
            metadatas=[{
                "status": DocumentStatus.SUPERSEDED.value,
                "superseded_by": triggered_by,
                "superseded_at": datetime.utcnow().isoformat(),
            }],
        )
        record = StateChangeRecord(
            document_id=doc_id,
            from_status=current_status,
            to_status=DocumentStatus.SUPERSEDED,
            reason=f"Superseded by {triggered_by}",
            triggered_by=triggered_by,
        )
        self._audit_log.append(record)
        logger.info(
            "Marked %s SUPERSEDED (triggered by %s)",
            doc_id, triggered_by,
        )

    @staticmethod
    def _is_older(date_a: str, date_b: str) -> bool:
        """Return True if date_a is strictly before date_b."""
        if not date_a or not date_b:
            return False
        try:
            return (
                datetime.fromisoformat(date_a)
                < datetime.fromisoformat(date_b)
            )
        except ValueError:
            return False
