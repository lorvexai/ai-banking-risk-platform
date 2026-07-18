"""
tests/test_graph_rag.py
Tests for GraphRAGEngine (Section 4.9.4 — GraphRAG).

All tests use mocks — no live Neo4j instance or API key required.
Run with: pytest tests/test_graph_rag.py -v

Test sections:
  TestClassifier            (6 tests)  — _requires_graph_hop routing
  TestVectorEntryPoints     (3 tests)  — pgvector entry-point retrieval
  TestGraphTraversal        (5 tests)  — bounded Cypher traversal + Neo4jGraphStore
  TestContextFusion         (3 tests)  — vector+graph context merge
  TestGraphRAGEngineQuery   (8 tests)  — end-to-end query() behaviour
  Total: 25 tests
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag_assistant.graph_rag import (
    CYPHER_CONTROL_LINEAGE,
    CYPHER_SUPERSESSION_IMPACT,
    MAX_GRAPH_HOPS,
    MAX_TRAVERSAL_NODES,
    EdgeType,
    GraphHopStatus,
    GraphNode,
    GraphRAGEngine,
    GraphRAGResult,
    GraphTraversalHop,
    Neo4jGraphStore,
    NodeType,
)


def _vector_hit(reference="PRA SS1/23 §4.2", score=0.88, clause_id="clause-001"):
    """Build a fake pgvector search-result object matching RegulatoryVectorStore's shape."""
    return SimpleNamespace(
        text="Sample regulatory passage text.",
        document_name="PRA SS1/23",
        section_number="4.2",
        relevance_score=score,
        metadata={"document_reference": reference, "clause_id": clause_id},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Classifier
# ─────────────────────────────────────────────────────────────────────────────

class TestClassifier:
    """Tests for the _requires_graph_hop per-query routing decision."""

    def setup_method(self):
        self.engine = GraphRAGEngine()

    def test_single_document_lookup_does_not_need_graph(self):
        """Plain factual lookups should not trigger graph traversal."""
        assert self.engine._requires_graph_hop("What is the LCR minimum under CRR3?") is False

    def test_control_lineage_query_needs_graph(self):
        """Control-to-clause lineage queries should trigger graph traversal."""
        assert self.engine._requires_graph_hop(
            "Which controls implement PRA SS1/23 §4.2?"
        ) is True

    def test_supersession_impact_query_needs_graph(self):
        """Superseded-regulation impact queries should trigger graph traversal."""
        assert self.engine._requires_graph_hop(
            "Which controls need to change if CRR3 Article 122a is amended?"
        ) is True

    def test_filing_traceability_query_needs_graph(self):
        """Queries about which filings a system feeds should trigger graph traversal."""
        assert self.engine._requires_graph_hop(
            "Which filings does the MR-2026-038 system feed?"
        ) is True

    def test_cross_regulator_query_needs_graph(self):
        """Cross-regulator reference queries should trigger graph traversal."""
        assert self.engine._requires_graph_hop(
            "Is this PRA clause cross-referenced by an EBA guideline?"
        ) is True

    def test_case_insensitive_signal_match(self):
        """Classifier signal matching is case-insensitive."""
        assert self.engine._requires_graph_hop("Which CONTROLS implement this clause?") is True


# ─────────────────────────────────────────────────────────────────────────────
# Vector entry-point retrieval (Step 1)
# ─────────────────────────────────────────────────────────────────────────────

class TestVectorEntryPoints:
    """Tests for pgvector entry-point retrieval feeding the graph traversal."""

    def test_no_vector_store_returns_empty(self):
        engine = GraphRAGEngine(vector_store=None)
        assert engine._find_entry_points("any query") == []

    def test_entry_points_filtered_by_relevance_threshold(self):
        store = MagicMock()
        store.search.return_value = [_vector_hit(score=0.95), _vector_hit(score=0.40)]
        engine = GraphRAGEngine(vector_store=store)
        entries = engine._find_entry_points("query")
        assert len(entries) == 1
        assert entries[0]["score"] == 0.95

    def test_vector_store_exception_handled_gracefully(self):
        store = MagicMock()
        store.search.side_effect = RuntimeError("connection refused")
        engine = GraphRAGEngine(vector_store=store)
        assert engine._find_entry_points("query") == []


# ─────────────────────────────────────────────────────────────────────────────
# Bounded graph traversal (Step 2)
# ─────────────────────────────────────────────────────────────────────────────

class TestGraphTraversal:
    """Tests for Neo4jGraphStore and the bounded-depth Cypher traversal."""

    def test_graph_store_without_driver_returns_empty(self):
        store = Neo4jGraphStore(driver=None)
        assert store.run_cypher(CYPHER_CONTROL_LINEAGE, {}) == []

    def test_graph_store_runs_cypher_via_driver_session(self):
        mock_record = {"ctrl": {"id": "AWB-CTL-0142", "node_type": "Control"}}
        mock_session = MagicMock()
        mock_session.run.return_value = [mock_record]
        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        store = Neo4jGraphStore(driver=mock_driver)
        records = store.run_cypher(CYPHER_CONTROL_LINEAGE, {"entry_clause_id": "clause-001"})
        assert records == [mock_record]

    def test_graph_store_handles_driver_exception(self):
        mock_driver = MagicMock()
        mock_driver.session.side_effect = RuntimeError("Neo4j unavailable")
        store = Neo4jGraphStore(driver=mock_driver)
        assert store.run_cypher(CYPHER_CONTROL_LINEAGE, {}) == []

    def test_traversal_bounded_to_max_hops(self):
        """At most MAX_GRAPH_HOPS traversal hops are issued, one per entry point."""
        graph_store = MagicMock()
        graph_store.run_cypher.return_value = []
        engine = GraphRAGEngine(graph_store=graph_store)
        entry_points = [
            {"clause_id": f"clause-{i}", "reference": f"ref-{i}"} for i in range(10)
        ]
        hops = engine._traverse_graph(entry_points, "which controls implement this clause?")
        assert len(hops) == MAX_GRAPH_HOPS

    def test_supersession_query_selects_supersession_template(self):
        graph_store = MagicMock()
        graph_store.run_cypher.return_value = []
        engine = GraphRAGEngine(graph_store=graph_store)
        entry_points = [{"clause_id": "reg-001", "reference": "CRR3 Art.122a"}]
        engine._traverse_graph(entry_points, "which controls need to change if this is amended?")
        called_cypher = graph_store.run_cypher.call_args[0][0]
        assert called_cypher.strip() == CYPHER_SUPERSESSION_IMPACT.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Context fusion (Step 3)
# ─────────────────────────────────────────────────────────────────────────────

class TestContextFusion:
    """Tests for fusing vector passages and graph facts into one context block."""

    def setup_method(self):
        self.engine = GraphRAGEngine()

    def test_fusion_includes_vector_block(self):
        entry_points = [{"reference": "PRA SS1/23 §4.2", "section": "4.2",
                          "score": 0.9, "text": "Sample text"}]
        fused = self.engine._fuse_context(entry_points, [])
        assert "[VECTOR" in fused
        assert "Sample text" in fused

    def test_fusion_includes_graph_block(self):
        node = GraphNode(node_id="AWB-CTL-0142", node_type=NodeType.CONTROL,
                          properties={"name": "AWB-CTL-0142"})
        hop = GraphTraversalHop(hop_index=0, cypher="MATCH...", entry_node="clause-001",
                                 nodes_found=[node], status=GraphHopStatus.SUCCESS)
        fused = self.engine._fuse_context([], [hop])
        assert "[GRAPH] Control AWB-CTL-0142" in fused

    def test_fusion_empty_when_no_evidence(self):
        assert self.engine._fuse_context([], []) == ""


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end query()
# ─────────────────────────────────────────────────────────────────────────────

class TestGraphRAGEngineQuery:
    """End-to-end tests for GraphRAGEngine.query()."""

    def test_single_document_query_skips_graph_traversal(self):
        store = MagicMock()
        store.search.return_value = [_vector_hit()]
        graph_store = MagicMock()
        engine = GraphRAGEngine(vector_store=store, graph_store=graph_store)
        result = engine.query("What is the LCR minimum under CRR3?")
        graph_store.run_cypher.assert_not_called()
        assert result.used_graph_traversal is False
        assert result.total_graph_hops == 0

    def test_lineage_query_invokes_graph_traversal(self):
        store = MagicMock()
        store.search.return_value = [_vector_hit()]
        graph_store = MagicMock()
        graph_store.run_cypher.return_value = [
            {"ctrl": {"id": "AWB-CTL-0142", "node_type": "Control"}}
        ]
        engine = GraphRAGEngine(vector_store=store, graph_store=graph_store)
        result = engine.query("Which controls implement PRA SS1/23 §4.2?")
        graph_store.run_cypher.assert_called()
        assert result.total_graph_hops >= 1
        assert result.used_graph_traversal is True

    def test_force_graph_overrides_classifier(self):
        store = MagicMock()
        store.search.return_value = [_vector_hit()]
        graph_store = MagicMock()
        graph_store.run_cypher.return_value = []
        engine = GraphRAGEngine(vector_store=store, graph_store=graph_store)
        result = engine.query("What is FRTB?", force_graph=True)
        graph_store.run_cypher.assert_called()

    def test_no_entry_points_skips_traversal_even_if_signalled(self):
        """If no vector entry points are found, graph traversal is skipped (Section 4.9.4
        fallback: 'supplements, does not replace' — never traverse without an anchor)."""
        store = MagicMock()
        store.search.return_value = []
        graph_store = MagicMock()
        engine = GraphRAGEngine(vector_store=store, graph_store=graph_store)
        result = engine.query("Which controls implement this clause?")
        graph_store.run_cypher.assert_not_called()
        assert result.total_graph_hops == 0

    def test_run_id_is_unique_per_query(self):
        engine = GraphRAGEngine()
        r1 = engine.query("What is FRTB?")
        r2 = engine.query("What is FRTB?")
        assert r1.run_id != r2.run_id

    def test_result_is_graphrag_result_instance(self):
        engine = GraphRAGEngine()
        result = engine.query("What is FRTB?")
        assert isinstance(result, GraphRAGResult)

    def test_no_evidence_at_all_returns_fallback_answer(self):
        engine = GraphRAGEngine(vector_store=None, graph_store=Neo4jGraphStore(driver=None))
        result = engine.query("What is FRTB?")
        assert "Insufficient" in result.answer
        assert result.confidence == 0.0

    def test_citations_include_both_vector_and_graph_sources(self):
        store = MagicMock()
        store.search.return_value = [_vector_hit()]
        graph_store = MagicMock()
        graph_store.run_cypher.return_value = [
            {"ctrl": {"id": "AWB-CTL-0142", "node_type": "Control"}}
        ]
        engine = GraphRAGEngine(vector_store=store, graph_store=graph_store)
        result = engine.query("Which controls implement PRA SS1/23 §4.2?")
        source_types = {c["source_type"] for c in result.citations}
        assert "vector" in source_types
        assert "graph" in source_types

    def test_to_dict_round_trips_key_fields(self):
        engine = GraphRAGEngine()
        result = engine.query("What is FRTB?")
        d = result.to_dict()
        assert d["run_id"] == result.run_id
        assert d["model_registration"] == "MR-2026-038"


# ─────────────────────────────────────────────────────────────────────────────
# Schema sanity checks
# ─────────────────────────────────────────────────────────────────────────────

class TestSchemaConstants:
    """Sanity checks on the node/edge schema described in Section 4.9.4."""

    def test_five_node_types_defined(self):
        assert {n.value for n in NodeType} == {
            "Regulation", "Clause", "Control", "System", "Filing"
        }

    def test_four_edge_types_defined(self):
        assert {e.value for e in EdgeType} == {
            "IMPLEMENTS", "SUPERSEDES", "FEEDS", "CROSS_REFERENCES"
        }

    def test_cypher_templates_respect_node_limit(self):
        assert f"LIMIT {MAX_TRAVERSAL_NODES}" in CYPHER_CONTROL_LINEAGE
        assert f"LIMIT {MAX_TRAVERSAL_NODES}" in CYPHER_SUPERSESSION_IMPACT
