"""
tests/test_ch04_new_modules.py
Tests for Chapter 4 new modules:
  - HybridRouter (Section 4.6)
  - SupersessionDetector (Section 4.7)
  - AWBRagasEvaluator (Section 4.8)
  - AccessControl / AuditLogger (Section 4.9)

All tests use mocks — no live API keys or database required.
Run with: pytest tests/test_ch04_new_modules.py -v

Test sections:
  TestHybridRouter          (12 tests)
  TestSupersessionDetector  (10 tests)
  TestAWBRagasEvaluator     (8 tests)
  TestAccessControl         (10 tests)
  TestAuditLogger           (5 tests)
  Total: 45 tests
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timedelta
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from awb_commons.rag.hybrid_router import (
    HybridRouter,
    QueryType,
    ClassificationResult,
    _heuristic_classify,
)
from awb_commons.rag.supersession_detector import (
    SupersessionDetector,
    DocumentStatus,
    StateChangeRecord,
    FreshnessAlert,
    NON_RETRIEVABLE_STATES,
    FRESHNESS_THRESHOLD_DAYS,
)
from evaluation.ragas_evaluator import (
    AWBRagasEvaluator,
    RAGASThresholds,
    RAGASScores,
    ValidationResult,
    QAPair,
)
from access_control.access_control import (
    AccessTier,
    TIER_FILTERS,
    build_access_filter,
    get_allowed_categories,
    RAGAuditRecord,
    AuditLogger,
    get_access_tier_from_jwt,
)


# ─────────────────────────────────────────────────────────────────────────────
# Section 4.6 — HybridRouter
# ─────────────────────────────────────────────────────────────────────────────

class TestHybridRouter:
    """Tests for query classification into DOCUMENT / DATA / HYBRID."""

    def test_heuristic_document_query(self):
        """Regulatory definition queries classify as DOCUMENT."""
        result = _heuristic_classify(
            "What is the CRR3 Article 153 slotting treatment?"
        )
        assert result == QueryType.DOCUMENT

    def test_heuristic_hybrid_query(self):
        """Compliance check queries classify as HYBRID."""
        result = _heuristic_classify(
            "Is our current LCR compliant with CRR3 minimum requirements? Is our ratio above minimum?"
        )
        assert result == QueryType.HYBRID

    def test_heuristic_data_query(self):
        """Operational data queries classify as DATA."""
        result = _heuristic_classify(
            "What was our LCR ratio last tuesday as of last week?"
        )
        assert result == QueryType.DATA

    def test_heuristic_ambiguous_returns_none(self):
        """Ambiguous short queries return None (fall through to LLM)."""
        result = _heuristic_classify("leverage")
        assert result is None

    def test_router_classify_document(self):
        """Router returns DOCUMENT for regulatory text queries."""
        router = HybridRouter()
        result = router.classify(
            "What are PRA SS1/23 model validation requirements?"
        )
        assert isinstance(result, ClassificationResult)
        assert result.query_type == QueryType.DOCUMENT

    def test_router_classify_hybrid(self):
        """Router returns HYBRID for compliance threshold checks."""
        router = HybridRouter()
        result = router.classify(
            "Is our leverage ratio above the CRR3 Art.429 minimum threshold today?"
        )
        assert result.query_type == QueryType.HYBRID

    def test_router_classify_hybrid_lcr(self):
        """LCR compliance query classified as HYBRID."""
        router = HybridRouter()
        result = router.classify(
            "Is our LCR ratio above the minimum requirements today? Are we compliant?"
        )
        assert result.query_type == QueryType.HYBRID

    def test_router_latency_populated(self):
        """Classification result includes latency_ms."""
        router = HybridRouter()
        result = router.classify("What is FRTB?")
        assert result.latency_ms >= 0.0

    def test_router_empty_query_raises(self):
        """Empty query raises ValueError."""
        router = HybridRouter()
        with pytest.raises(ValueError, match="must not be empty"):
            router.classify("")

    def test_router_whitespace_query_raises(self):
        """Whitespace-only query raises ValueError."""
        router = HybridRouter()
        with pytest.raises(ValueError):
            router.classify("   ")

    def test_router_no_llm_defaults_to_document(self, caplog):
        """Ambiguous query defaults to DOCUMENT when no LLM client."""
        router = HybridRouter(use_heuristic_first=False)
        result = router.classify("regulatory")
        assert result.query_type == QueryType.DOCUMENT

    def test_router_with_mock_llm_client(self):
        """LLM client is called for ambiguous queries."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = (
            "TYPE: hybrid\nREASON: needs both regulatory text and AWB data"
        )
        router = HybridRouter(use_heuristic_first=False)
        result = router.classify("check compliance", llm_client=mock_llm)
        assert result.query_type == QueryType.HYBRID
        mock_llm.generate.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# Section 4.7 — SupersessionDetector
# ─────────────────────────────────────────────────────────────────────────────

def _make_corpus_client(
    new_doc: dict,
    candidates: List[dict],
) -> MagicMock:
    """Build a mock ChromaDB corpus client."""
    client = MagicMock()
    client.get.return_value = {
        "ids": [new_doc["id"]],
        "embeddings": [[0.1] * 768],
        "metadatas": [new_doc["meta"]],
    }
    client.query.return_value = {
        "ids": [[c["id"] for c in candidates]],
        "distances": [[c["distance"] for c in candidates]],
        "metadatas": [[c["meta"] for c in candidates]],
    }
    return client


class TestSupersessionDetector:
    """Tests for document lifecycle and supersession management."""

    def _old_doc(self, doc_id: str) -> dict:
        return {
            "id": doc_id,
            "distance": 0.05,   # high similarity
            "meta": {
                "status": DocumentStatus.FINAL.value,
                "effective_date": "2022-01-01T00:00:00",
                "document_id": doc_id,
                "document_name": f"Regulation {doc_id}",
            },
        }

    def _new_doc_meta(self) -> dict:
        return {
            "id": "CRR3_final_2024",
            "meta": {
                "effective_date": "2024-11-01T00:00:00",
            },
        }

    def test_detects_superseded_document(self):
        """Older similar document is correctly identified as superseded."""
        audit_log = []
        detector = SupersessionDetector(
            similarity_threshold=0.85, audit_log=audit_log
        )
        old = self._old_doc("EBA_consultation_2022")
        corpus = _make_corpus_client(self._new_doc_meta(), [old])

        result = detector.detect_and_mark("CRR3_final_2024", corpus)

        assert "EBA_consultation_2022" in result.superseded_ids
        assert result.superseded_count == 1

    def test_audit_record_created(self):
        """State change audit record is created for each supersession."""
        audit_log = []
        detector = SupersessionDetector(audit_log=audit_log)
        old = self._old_doc("EBA_consultation_2022")
        corpus = _make_corpus_client(self._new_doc_meta(), [old])

        detector.detect_and_mark("CRR3_final_2024", corpus)

        assert len(audit_log) == 1
        record = audit_log[0]
        assert isinstance(record, StateChangeRecord)
        assert record.to_status == DocumentStatus.SUPERSEDED
        assert record.triggered_by == "CRR3_final_2024"

    def test_new_doc_not_in_corpus_raises(self):
        """KeyError raised when new document not found in corpus."""
        detector = SupersessionDetector()
        corpus = MagicMock()
        corpus.get.return_value = {"ids": [], "embeddings": [], "metadatas": []}

        with pytest.raises(KeyError):
            detector.detect_and_mark("nonexistent_doc", corpus)

    def test_dissimilar_document_not_superseded(self):
        """Document with low similarity is not marked as superseded."""
        detector = SupersessionDetector(similarity_threshold=0.85)
        dissimilar = {
            "id": "unrelated_doc",
            "distance": 0.50,   # low similarity (1 - 0.50 = 0.50)
            "meta": {
                "status": DocumentStatus.FINAL.value,
                "effective_date": "2020-01-01T00:00:00",
                "document_id": "unrelated_doc",
                "document_name": "Unrelated Doc",
            },
        }
        corpus = _make_corpus_client(self._new_doc_meta(), [dissimilar])

        result = detector.detect_and_mark("CRR3_final_2024", corpus)
        assert result.superseded_count == 0

    def test_newer_document_not_superseded(self):
        """Document with later effective date is NOT superseded."""
        detector = SupersessionDetector()
        newer = {
            "id": "later_doc",
            "distance": 0.05,
            "meta": {
                "status": DocumentStatus.FINAL.value,
                "effective_date": "2025-06-01T00:00:00",  # later
                "document_id": "later_doc",
                "document_name": "Later Doc",
            },
        }
        corpus = _make_corpus_client(self._new_doc_meta(), [newer])
        result = detector.detect_and_mark("CRR3_final_2024", corpus)
        assert result.superseded_count == 0

    def test_non_retrievable_states_correct(self):
        """DRAFT and SUPERSEDED are excluded from retrieval."""
        assert DocumentStatus.DRAFT in NON_RETRIEVABLE_STATES
        assert DocumentStatus.SUPERSEDED in NON_RETRIEVABLE_STATES
        assert DocumentStatus.FINAL not in NON_RETRIEVABLE_STATES
        assert DocumentStatus.CONSULTATION not in NON_RETRIEVABLE_STATES

    def test_freshness_threshold_18_months(self):
        """Freshness threshold is 548 days (~18 months)."""
        assert FRESHNESS_THRESHOLD_DAYS == 548

    def test_freshness_audit_flags_stale_document(self):
        """Documents not reviewed in > 18 months are flagged."""
        detector = SupersessionDetector()
        stale_date = (
            datetime.utcnow() - timedelta(days=600)
        ).isoformat()
        corpus = MagicMock()
        corpus.get.return_value = {
            "metadatas": [{
                "status": DocumentStatus.FINAL.value,
                "document_id": "old_doc",
                "document_name": "Old Regulation",
                "last_reviewed": stale_date,
            }]
        }
        alerts = detector.run_freshness_audit(corpus)
        assert len(alerts) == 1
        assert alerts[0].document_id == "old_doc"
        assert alerts[0].days_since_review > 548

    def test_freshness_audit_skips_recent_doc(self):
        """Recently reviewed documents are not flagged."""
        detector = SupersessionDetector()
        recent_date = (
            datetime.utcnow() - timedelta(days=30)
        ).isoformat()
        corpus = MagicMock()
        corpus.get.return_value = {
            "metadatas": [{
                "status": DocumentStatus.FINAL.value,
                "document_id": "recent_doc",
                "document_name": "Recent Regulation",
                "last_reviewed": recent_date,
            }]
        }
        alerts = detector.run_freshness_audit(corpus)
        assert len(alerts) == 0

    def test_supersession_result_attributes(self):
        """SupersessionResult has correct attributes."""
        detector = SupersessionDetector()
        old = self._old_doc("old_doc")
        corpus = _make_corpus_client(self._new_doc_meta(), [old])
        result = detector.detect_and_mark("CRR3_final_2024", corpus)
        assert result.new_document_id == "CRR3_final_2024"
        assert isinstance(result.effective_date, datetime)


# ─────────────────────────────────────────────────────────────────────────────
# Section 4.8 — AWBRagasEvaluator
# ─────────────────────────────────────────────────────────────────────────────

def _make_test_set_file(n: int = 10) -> str:
    """Write a temporary JSON test set file."""
    categories = ["credit", "capital", "reporting"]
    pairs = [
        {
            "question": f"What is regulatory requirement {i}?",
            "expected_answer": f"Requirement {i} states ...",
            "category": categories[i % 3],
            "question_id": f"q{i:03d}",
        }
        for i in range(n)
    ]
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    )
    json.dump(pairs, tmp)
    tmp.close()
    return tmp.name


class TestAWBRagasEvaluator:
    """Tests for RAGAS evaluation harness."""

    def test_mock_validation_passes(self):
        """Mock validation returns PASS with correct model_id."""
        evaluator = AWBRagasEvaluator(use_mock=True)
        path = _make_test_set_file()
        result = evaluator.validate("MR-2026-038", path)
        assert result.passed is True
        assert result.model_id == "MR-2026-038"

    def test_mock_scores_above_thresholds(self):
        """Mock scores all exceed AWB validation thresholds."""
        evaluator = AWBRagasEvaluator(use_mock=True)
        path = _make_test_set_file()
        result = evaluator.validate("MR-2026-038", path)
        t = RAGASThresholds()
        assert result.scores.faithfulness >= t.faithfulness
        assert result.scores.answer_relevancy >= t.answer_relevancy
        assert result.scores.context_precision >= t.context_precision
        assert result.scores.context_recall >= t.context_recall

    def test_validation_result_has_test_set_size(self):
        """ValidationResult records the number of test questions."""
        evaluator = AWBRagasEvaluator(use_mock=True)
        path = _make_test_set_file(n=15)
        result = evaluator.validate("MR-2026-038", path)
        assert result.test_set_size == 15

    def test_validation_result_summary_format(self):
        """Summary string includes model_id and key metrics."""
        evaluator = AWBRagasEvaluator(use_mock=True)
        path = _make_test_set_file()
        result = evaluator.validate("MR-2026-038", path)
        summary = result.summary()
        assert "MR-2026-038" in summary
        assert "PASS" in summary
        assert "faithfulness" in summary

    def test_failing_threshold_produces_failure_reason(self):
        """ValidationResult failure_reason is set when score is below threshold."""
        # Use custom thresholds that mock scores will not meet
        strict = RAGASThresholds(
            faithfulness=0.99,
            answer_relevancy=0.99,
            context_precision=0.99,
            context_recall=0.99,
        )
        evaluator = AWBRagasEvaluator(
            thresholds=strict, use_mock=True
        )
        path = _make_test_set_file()
        result = evaluator.validate("MR-2026-038", path)
        assert result.passed is False
        assert result.failure_reason is not None
        assert "below threshold" in result.failure_reason

    def test_empty_test_set_raises(self):
        """Empty test set file raises ValueError."""
        evaluator = AWBRagasEvaluator(use_mock=True)
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )
        json.dump([], tmp)
        tmp.close()
        with pytest.raises(ValueError, match="empty or unreadable"):
            evaluator.validate("MR-2026-038", tmp.name)

    def test_missing_test_set_raises(self):
        """Non-existent test set path raises ValueError."""
        evaluator = AWBRagasEvaluator(use_mock=True)
        with pytest.raises(ValueError):
            evaluator.validate("MR-2026-038", "/nonexistent/path.json")

    def test_sampling_returns_smaller_file(self):
        """5% sample produces fewer QA pairs than full test set."""
        evaluator = AWBRagasEvaluator(use_mock=True)
        path = _make_test_set_file(n=100)
        sample_path = evaluator.sample_for_monitoring(path, sample_rate=0.05)
        with open(sample_path) as f:
            sample = json.load(f)
        assert len(sample) < 100
        assert len(sample) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Section 4.9 — Access Control
# ─────────────────────────────────────────────────────────────────────────────

class TestAccessControl:
    """Tests for multi-tenant access tier enforcement."""

    def test_compliance_officer_no_filter(self):
        """COMPLIANCE_OFFICER gets no filter (unrestricted)."""
        f = build_access_filter(AccessTier.COMPLIANCE_OFFICER)
        assert f is None

    def test_credit_analyst_filter_includes_credit(self):
        """CREDIT_ANALYST filter includes credit and capital."""
        f = build_access_filter(AccessTier.CREDIT_ANALYST)
        assert f is not None
        categories = f["document_category"]["$in"]
        assert "credit" in categories
        assert "capital" in categories

    def test_relationship_manager_filter(self):
        """RELATIONSHIP_MANAGER filter includes consumer_duty but not credit."""
        f = build_access_filter(AccessTier.RELATIONSHIP_MANAGER)
        assert f is not None
        categories = f["document_category"]["$in"]
        assert "consumer_duty" in categories
        assert "credit" not in categories

    def test_treasury_filter(self):
        """TREASURY filter includes liquidity docs but not credit."""
        f = build_access_filter(AccessTier.TREASURY)
        assert f is not None
        categories = f["document_category"]["$in"]
        assert "liquidity" in categories
        assert "lcr" in categories
        assert "credit" not in categories

    def test_relationship_manager_cannot_retrieve_pra_supervisory_statements(self):
        """
        RELATIONSHIP_MANAGER must not have access to PRA supervisory docs.
        Key test required by session prompt — access boundary enforcement.
        """
        allowed = get_allowed_categories(AccessTier.RELATIONSHIP_MANAGER)
        # PRA supervisory statements are categorised as "model_risk" or "supervision"
        assert "model_risk" not in allowed
        assert "supervision" not in allowed
        # Only client-facing categories permitted
        for cat in allowed:
            assert cat in ["conduct", "product", "consumer_duty", "cobs", "retail"]

    def test_credit_analyst_cannot_retrieve_aml_docs(self):
        """CREDIT_ANALYST must not have access to AML documents."""
        allowed = get_allowed_categories(AccessTier.CREDIT_ANALYST)
        assert "aml" not in allowed
        assert "kyc" not in allowed

    def test_all_tiers_defined(self):
        """All four access tiers have defined filter sets."""
        for tier in AccessTier:
            assert tier in TIER_FILTERS

    def test_jwt_extraction_compliance_officer(self):
        """JWT extraction returns correct tier for compliance_officer."""
        tier = get_access_tier_from_jwt("compliance_officer")
        assert tier == AccessTier.COMPLIANCE_OFFICER

    def test_jwt_extraction_unknown_defaults_to_rm(self):
        """Unknown tier claim defaults to RELATIONSHIP_MANAGER (most restrictive)."""
        tier = get_access_tier_from_jwt("unknown_tier_xyz")
        assert tier == AccessTier.RELATIONSHIP_MANAGER

    def test_get_allowed_categories_returns_copy(self):
        """get_allowed_categories returns a copy (mutation-safe)."""
        cats1 = get_allowed_categories(AccessTier.CREDIT_ANALYST)
        cats1.append("injected_category")
        cats2 = get_allowed_categories(AccessTier.CREDIT_ANALYST)
        assert "injected_category" not in cats2


# ─────────────────────────────────────────────────────────────────────────────
# Section 4.9 — AuditLogger
# ─────────────────────────────────────────────────────────────────────────────

class TestAuditLogger:
    """Tests for 7-year FCA COBS 9 audit logging."""

    def _make_record(self, user_id: str = "AWB001") -> RAGAuditRecord:
        return RAGAuditRecord(
            user_id=user_id,
            user_role=AccessTier.CREDIT_ANALYST.value,
            query="What is CRR3 Article 153?",
            access_filter_applied={"document_category": {"$in": ["credit"]}},
            documents_retrieved=["CRR3_final_2024"],
            confidence_score=0.92,
            answer_length=345,
        )

    def test_record_logged_in_memory(self):
        """Audit record is stored in logger.records when no db client."""
        audit = AuditLogger()
        record = self._make_record()
        audit.log(record)
        assert len(audit.records) == 1
        assert audit.records[0].user_id == "AWB001"

    def test_record_has_uuid(self):
        """Each audit record has a unique UUID."""
        r1 = self._make_record("AWB001")
        r2 = self._make_record("AWB002")
        assert r1.audit_id != r2.audit_id

    def test_record_model_id_always_mr_2026_038(self):
        """model_id is always MR-2026-038 (PRA SS1/23 traceability)."""
        record = self._make_record()
        assert record.model_id == "MR-2026-038"

    def test_multiple_records_stored(self):
        """Multiple records are all stored."""
        audit = AuditLogger()
        for i in range(5):
            audit.log(self._make_record(f"AWB{i:03d}"))
        assert len(audit.records) == 5

    def test_db_client_called_on_log(self):
        """DB client execute() is called when db_client provided."""
        mock_db = MagicMock()
        audit = AuditLogger(db_client=mock_db)
        audit.log(self._make_record())
        mock_db.execute.assert_called_once()
        # In-memory records list should be empty when db client is provided
        assert len(audit.records) == 0
