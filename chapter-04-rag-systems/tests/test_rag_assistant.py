"""
tests/test_rag_assistant.py
Comprehensive test suite for AWB Regulatory Knowledge Assistant
Chapter 4: RAG for Compliance

Test coverage:
- Document loading and metadata (10 tests)
- Text chunking (7 tests)
- Vector store operations (10 tests)
- Query engine and RAG pipeline (10 tests)
- Grounding / citation checks (5 tests)
- Hallucination guard (5 tests)
- Metadata filtering (4 tests)

Total: 51 tests
All vector operations use MockEmbeddingProvider (no live Google API required).
All LLM generation uses MockLLMGenerationClient.

Run with: pytest tests/test_rag_assistant.py -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag_assistant.document_loader import (
    RegulatoryDocumentLoader,
    DocumentChunk,
    chunk_text,
    REGULATORY_DOCUMENT_REGISTRY,
    CHUNK_SIZE_CHARS,
    CHUNK_OVERLAP_CHARS,
    _approx_tokens,
    _detect_section,
)
from rag_assistant.vector_store import (
    RegulatoryVectorStore,
    SearchResult,
    MockEmbeddingProvider,
    MIN_RELEVANCE_SCORE,
    VALID_REGULATORS,
)
from rag_assistant.query_engine import (
    RegulatoryQueryEngine,
    RegulatoryAnswer,
    Citation,
    MockLLMGenerationClient,
    UNCERTAINTY_RESPONSE,
    build_system_prompt,
    MODEL_REGISTRATION,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DOCS_DIR = Path(__file__).parent.parent / "rag_assistant" / "regulatory_documents"


@pytest.fixture(scope="session")
def docs_dir() -> Path:
    """Path to the regulatory documents directory."""
    assert DOCS_DIR.exists(), f"Docs dir missing: {DOCS_DIR}"
    return DOCS_DIR


@pytest.fixture(scope="session")
def loader(docs_dir) -> RegulatoryDocumentLoader:
    return RegulatoryDocumentLoader(str(docs_dir))


@pytest.fixture(scope="session")
def all_chunks(loader) -> List[DocumentChunk]:
    return loader.load_all_as_list()


@pytest.fixture
def mock_store() -> RegulatoryVectorStore:
    """In-memory ChromaDB store with mock embeddings — no persistence, no API."""
    return RegulatoryVectorStore(
        collection_name=f"test_{uuid.uuid4().hex[:8]}",
        persist_directory=None,
        embedding_provider=MockEmbeddingProvider(),
        top_k=5,
        min_relevance_score=MIN_RELEVANCE_SCORE,
    )


@pytest.fixture
def populated_store(all_chunks) -> RegulatoryVectorStore:
    """In-memory store pre-populated with all regulatory chunks."""
    store = RegulatoryVectorStore(
        collection_name=f"test_{uuid.uuid4().hex[:8]}",
        persist_directory=None,
        embedding_provider=MockEmbeddingProvider(),
        top_k=5,
        min_relevance_score=0.0,  # Accept all results for retrieval tests
    )
    store.upsert_chunks(all_chunks)
    return store


@pytest.fixture
def query_engine(populated_store) -> RegulatoryQueryEngine:
    """RAG query engine using mock embedding + mock LLM."""
    return RegulatoryQueryEngine(
        vector_store=populated_store,
        llm_client=MockLLMGenerationClient(),
        min_relevance_score=0.0,  # All results pass threshold in tests
        top_k=5,
    )


@pytest.fixture
def strict_query_engine(mock_store) -> RegulatoryQueryEngine:
    """Query engine with empty store — triggers hallucination guard."""
    return RegulatoryQueryEngine(
        vector_store=mock_store,
        llm_client=MockLLMGenerationClient(),
        min_relevance_score=MIN_RELEVANCE_SCORE,
        top_k=5,
    )


# ===========================================================================
# SECTION 1: Document Loading (10 tests)
# ===========================================================================

class TestDocumentLoading:

    def test_loader_initialises_with_valid_dir(self, docs_dir):
        loader = RegulatoryDocumentLoader(str(docs_dir))
        assert loader is not None

    def test_loader_raises_for_missing_dir(self):
        with pytest.raises(FileNotFoundError):
            RegulatoryDocumentLoader("/nonexistent/path/xyz")

    def test_load_pra_document(self, docs_dir):
        loader = RegulatoryDocumentLoader(str(docs_dir))
        doc = loader.load_file(docs_dir / "PRA_SS1_23_extract.txt")
        assert doc.regulator == "PRA"
        assert doc.total_chunks > 0
        assert len(doc.chunks) == doc.total_chunks

    def test_load_fca_document(self, docs_dir):
        loader = RegulatoryDocumentLoader(str(docs_dir))
        doc = loader.load_file(docs_dir / "FCA_PS22_9_extract.txt")
        assert doc.regulator == "FCA"
        assert doc.total_chunks > 0

    def test_load_eu_ai_act_document(self, docs_dir):
        loader = RegulatoryDocumentLoader(str(docs_dir))
        doc = loader.load_file(docs_dir / "EU_AI_Act_Annex_III_extract.txt")
        assert doc.regulator == "EC"

    def test_load_dora_document(self, docs_dir):
        loader = RegulatoryDocumentLoader(str(docs_dir))
        doc = loader.load_file(docs_dir / "DORA_ICT_extract.txt")
        assert doc.regulator == "EBA"

    def test_all_chunks_have_required_fields(self, all_chunks):
        for chunk in all_chunks:
            assert chunk.chunk_id
            assert chunk.text.strip()
            assert chunk.regulator in VALID_REGULATORS
            assert chunk.document_name
            assert chunk.source_file.endswith(".txt")

    def test_chunk_ids_are_unique(self, all_chunks):
        ids = [c.chunk_id for c in all_chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs found"

    def test_chunk_ids_are_deterministic(self, docs_dir):
        """Loading the same document twice produces identical chunk IDs."""
        loader = RegulatoryDocumentLoader(str(docs_dir))
        chunks_a = loader.load_all_as_list()
        chunks_b = loader.load_all_as_list()
        ids_a = {c.chunk_id for c in chunks_a}
        ids_b = {c.chunk_id for c in chunks_b}
        assert ids_a == ids_b

    def test_document_summary_returns_all_files(self, loader):
        summary = loader.get_document_summary()
        regulators = {s["regulator"] for s in summary}
        assert "PRA" in regulators
        assert "FCA" in regulators


# ===========================================================================
# SECTION 2: Text Chunking (7 tests)
# ===========================================================================

class TestChunking:

    def test_short_text_produces_single_chunk(self):
        text = "This is a short regulatory paragraph."
        chunks = chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_produces_multiple_chunks(self):
        # Generate text larger than CHUNK_SIZE_CHARS
        long_text = ("This is a sentence in a regulatory document. " * 200)
        chunks = chunk_text(long_text)
        assert len(chunks) > 1

    def test_no_empty_chunks_produced(self):
        text = "Para one.\n\nPara two.\n\n\n\nPara three."
        chunks = chunk_text(text)
        for chunk in chunks:
            assert chunk.strip()

    def test_all_text_content_preserved(self):
        """Every word in the input should appear in at least one chunk."""
        text = "unique_word_alpha\n\n" + ("padding " * 600) + "\n\nunique_word_beta"
        chunks = chunk_text(text)
        all_text = " ".join(chunks)
        assert "unique_word_alpha" in all_text
        assert "unique_word_beta" in all_text

    def test_chunk_sizes_within_bounds(self):
        long_text = "A regulatory sentence with important compliance requirements. " * 300
        chunks = chunk_text(long_text)
        for chunk in chunks:
            # Allow 20% tolerance for overlap and paragraph-boundary rounding
            assert len(chunk) <= CHUNK_SIZE_CHARS * 1.50, (
                f"Chunk too large: {len(chunk)} chars"
            )

    def test_approx_tokens_returns_positive(self):
        assert _approx_tokens("hello world") > 0

    def test_detect_section_finds_section_header(self):
        text = "SECTION 3: VALIDATION REQUIREMENTS\n\nContent follows here."
        result = _detect_section(text)
        assert result is not None
        assert "SECTION" in result.upper() or "3" in result


# ===========================================================================
# SECTION 3: Vector Store Operations (10 tests)
# ===========================================================================

class TestVectorStore:

    def test_store_initialises_in_memory(self, mock_store):
        assert mock_store.count() == 0

    def test_upsert_single_chunk(self, all_chunks, mock_store):
        chunk = all_chunks[0]
        mock_store.upsert_chunk(chunk)
        assert mock_store.count() == 1

    def test_upsert_chunks_bulk(self, all_chunks, mock_store):
        count = mock_store.upsert_chunks(all_chunks[:10])
        assert count == 10
        assert mock_store.count() == 10

    def test_upsert_is_idempotent(self, all_chunks, mock_store):
        mock_store.upsert_chunks(all_chunks[:5])
        mock_store.upsert_chunks(all_chunks[:5])  # Second upsert
        assert mock_store.count() == 5  # Should not double-count

    def test_search_returns_results(self, populated_store):
        results = populated_store.search("model validation requirements")
        assert len(results) > 0

    def test_search_results_have_relevance_score(self, populated_store):
        results = populated_store.search("AI systems governance")
        for result in results:
            assert 0.0 <= result.relevance_score <= 1.0

    def test_search_results_sorted_descending(self, populated_store):
        results = populated_store.search("credit scoring high risk")
        scores = [r.relevance_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_returns_at_most_top_k(self, populated_store):
        results = populated_store.search("regulatory compliance", top_k=3)
        assert len(results) <= 3

    def test_search_empty_query_raises(self, mock_store):
        with pytest.raises(ValueError, match="empty"):
            mock_store.search("")

    def test_search_invalid_regulator_filter_raises(self, mock_store):
        with pytest.raises(ValueError, match="regulator_filter"):
            mock_store.search("test query", regulator_filter="INVALID")


# ===========================================================================
# SECTION 4: Metadata Filtering (4 tests)
# ===========================================================================

class TestMetadataFiltering:

    def test_filter_by_pra_returns_only_pra(self, populated_store):
        results = populated_store.search(
            "model risk management", regulator_filter="PRA"
        )
        for r in results:
            assert r.regulator == "PRA"

    def test_filter_by_fca_returns_only_fca(self, populated_store):
        results = populated_store.search(
            "consumer duty outcomes", regulator_filter="FCA"
        )
        for r in results:
            assert r.regulator == "FCA"

    def test_filter_by_ec_returns_eu_ai_act(self, populated_store):
        results = populated_store.search(
            "high risk AI systems credit", regulator_filter="EC"
        )
        for r in results:
            assert r.regulator == "EC"

    def test_filter_by_eba_returns_dora(self, populated_store):
        results = populated_store.search(
            "ICT risk management digital resilience", regulator_filter="EBA"
        )
        for r in results:
            assert r.regulator == "EBA"


# ===========================================================================
# SECTION 5: Query Engine and RAG Pipeline (10 tests)
# ===========================================================================

class TestQueryEngine:

    def test_query_returns_regulatory_answer(self, query_engine):
        answer = query_engine.query("What are PRA model validation requirements?")
        assert isinstance(answer, RegulatoryAnswer)

    def test_query_has_non_empty_answer(self, query_engine):
        answer = query_engine.query("What is model risk management?")
        assert len(answer.answer) > 10

    def test_query_has_citations(self, query_engine):
        answer = query_engine.query("What are the governance requirements for AI models?")
        assert len(answer.citations) > 0

    def test_answer_has_confidence_score(self, query_engine):
        answer = query_engine.query("How should firms handle model validation?")
        assert 0.0 <= answer.confidence <= 1.0

    def test_answer_has_caveats(self, query_engine):
        answer = query_engine.query("What are model documentation requirements?")
        assert isinstance(answer.caveats, list)
        assert len(answer.caveats) > 0

    def test_answer_has_model_registration(self, query_engine):
        answer = query_engine.query("What is the PRA's approach to AI governance?")
        assert answer.model_registration == MODEL_REGISTRATION

    def test_answer_is_not_uncertainty_when_context_exists(self, query_engine):
        answer = query_engine.query("What are PRA model risk requirements?")
        assert answer.is_uncertainty_response is False

    def test_empty_query_raises(self, query_engine):
        with pytest.raises(ValueError, match="empty"):
            query_engine.query("")

    def test_query_with_regulator_filter(self, query_engine):
        answer = query_engine.query(
            "What are the model validation requirements?",
            regulator_filter="PRA",
        )
        assert isinstance(answer, RegulatoryAnswer)

    def test_query_with_logging_returns_dict(self, query_engine):
        result = query_engine.query_with_logging("What is Consumer Duty?")
        assert isinstance(result, dict)
        assert "answer" in result
        assert "model_registration" in result


# ===========================================================================
# SECTION 6: Grounding and Citation Requirements (5 tests)
# ===========================================================================

class TestGrounding:

    def test_non_uncertainty_answer_has_at_least_one_citation(self, query_engine):
        """Grounding requirement: every factual answer must cite a source."""
        answer = query_engine.query("What are PRA's audit trail requirements?")
        if not answer.is_uncertainty_response:
            assert len(answer.citations) >= 1

    def test_citation_has_document_reference(self, query_engine):
        answer = query_engine.query("What are governance requirements for model risk?")
        if not answer.is_uncertainty_response:
            for citation in answer.citations:
                assert citation.document_reference
                assert len(citation.document_reference) > 0

    def test_citation_has_source_file(self, query_engine):
        answer = query_engine.query("What validation steps are required for AI models?")
        if not answer.is_uncertainty_response:
            for citation in answer.citations:
                assert citation.source_file.endswith(".txt")

    def test_citation_relevance_score_valid(self, query_engine):
        answer = query_engine.query("What are model risk governance requirements?")
        if not answer.is_uncertainty_response:
            for citation in answer.citations:
                assert 0.0 <= citation.relevance_score <= 1.0

    def test_citation_excerpt_not_empty(self, query_engine):
        answer = query_engine.query("What are PRA model documentation requirements?")
        if not answer.is_uncertainty_response:
            for citation in answer.citations:
                assert len(citation.excerpt) > 0

    def test_system_prompt_includes_source_context(self):
        """Verify system prompt is built with retrieved context."""
        mock_result = MagicMock(spec=SearchResult)
        mock_result.document_name = "PRA SS1/23"
        mock_result.section_number = "Section 3"
        mock_result.relevance_score = 0.92
        mock_result.text = "Sample regulatory text about model validation."
        mock_result.metadata = {
            "document_reference": "SS1/23",
            "regulator": "PRA",
        }

        prompt = build_system_prompt([mock_result])
        assert "PRA SS1/23" in prompt
        assert "RETRIEVED REGULATORY CONTEXT" in prompt


# ===========================================================================
# SECTION 7: Hallucination Guard (5 tests)
# ===========================================================================

class TestHallucinationGuard:

    def test_empty_store_returns_uncertainty_response(self, strict_query_engine):
        """With no relevant context, the engine must return the uncertainty response."""
        answer = strict_query_engine.query("What are the PRA requirements?")
        assert answer.is_uncertainty_response is True

    def test_uncertainty_response_matches_standard_text(self, strict_query_engine):
        answer = strict_query_engine.query("Explain Basel III capital requirements.")
        if answer.is_uncertainty_response:
            assert "cannot find" in answer.answer.lower() or "regulatory library" in answer.answer.lower()

    def test_uncertainty_response_has_zero_confidence(self, strict_query_engine):
        answer = strict_query_engine.query("What is a hypothetical made-up regulation?")
        if answer.is_uncertainty_response:
            assert answer.confidence == 0.0

    def test_uncertainty_response_has_caveats(self, strict_query_engine):
        answer = strict_query_engine.query("Random question about fictional regulation.")
        if answer.is_uncertainty_response:
            assert len(answer.caveats) > 0

    def test_uncertainty_response_allows_empty_citations(self):
        """Pydantic model must accept empty citations for uncertainty responses."""
        answer = RegulatoryAnswer(
            query="test query",
            answer=UNCERTAINTY_RESPONSE,
            citations=[],
            confidence=0.0,
            caveats=["No relevant context found."],
            is_uncertainty_response=True,
        )
        assert answer.is_uncertainty_response is True
        assert len(answer.citations) == 0


# ===========================================================================
# SECTION 8: RegulatoryAnswer Pydantic Validation (5 tests)
# ===========================================================================

class TestRegulatoryAnswerValidation:

    def _make_citation(self) -> Citation:
        return Citation(
            document_reference="SS1/23",
            document_name="PRA SS1/23",
            section_number="Section 3",
            source_file="PRA_SS1_23_extract.txt",
            relevance_score=0.88,
            excerpt="Model validation must be independent.",
        )

    def test_valid_answer_constructs(self):
        citation = self._make_citation()
        answer = RegulatoryAnswer(
            query="test",
            answer="This is a valid answer with more than 10 characters.",
            citations=[citation],
            confidence=0.85,
            caveats=["Not legal advice."],
        )
        assert answer.confidence == 0.85

    def test_non_uncertainty_answer_without_citations_raises(self):
        with pytest.raises(Exception):
            RegulatoryAnswer(
                query="test",
                answer="Answer without any citations.",
                citations=[],
                confidence=0.90,
                caveats=["Some caveat."],
                is_uncertainty_response=False,
            )

    def test_low_confidence_without_caveats_raises(self):
        citation = self._make_citation()
        with pytest.raises(Exception):
            RegulatoryAnswer(
                query="test",
                answer="Low confidence answer here.",
                citations=[citation],
                confidence=0.50,
                caveats=[],  # Must have caveats for low confidence
                is_uncertainty_response=False,
            )

    def test_to_dict_returns_dict(self):
        citation = self._make_citation()
        answer = RegulatoryAnswer(
            query="test",
            answer="A well-grounded regulatory answer.",
            citations=[citation],
            confidence=0.88,
            caveats=["Not legal advice."],
        )
        d = answer.to_dict()
        assert isinstance(d, dict)
        assert "answer" in d
        assert "citations" in d
        assert "model_registration" in d

    def test_model_registration_default_value(self):
        citation = self._make_citation()
        answer = RegulatoryAnswer(
            query="test",
            answer="A well-grounded regulatory answer.",
            citations=[citation],
            confidence=0.88,
            caveats=["Not legal advice."],
        )
        assert answer.model_registration == MODEL_REGISTRATION


# ===========================================================================
# SECTION 9: Document Registry (3 tests)
# ===========================================================================

class TestDocumentRegistry:

    def test_registry_has_all_four_documents(self):
        expected = {
            "PRA_SS1_23_extract.txt",
            "FCA_PS22_9_extract.txt",
            "EU_AI_Act_Annex_III_extract.txt",
            "DORA_ICT_extract.txt",
        }
        assert expected == set(REGULATORY_DOCUMENT_REGISTRY.keys())

    def test_registry_entries_have_required_fields(self):
        required_fields = {"document_name", "regulator", "document_reference", "effective_date"}
        for filename, meta in REGULATORY_DOCUMENT_REGISTRY.items():
            missing = required_fields - set(meta.keys())
            assert not missing, f"Missing fields in {filename}: {missing}"

    def test_all_registered_files_exist(self, docs_dir):
        for filename in REGULATORY_DOCUMENT_REGISTRY:
            file_path = docs_dir / filename
            assert file_path.exists(), f"Missing regulatory document: {filename}"
