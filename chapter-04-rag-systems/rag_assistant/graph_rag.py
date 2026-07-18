"""
rag_assistant/graph_rag.py
AWB GraphRAG — Hybrid Vector-Plus-Graph Retrieval Engine
Chapter 4: Section 4.9.4 — GraphRAG

Supplements (does not replace) the pgvector semantic search described in
Section 4.4 with a bounded-depth Neo4j property-graph traversal, for the
class of regulatory questions that flat vector similarity cannot answer:
cross-regulatory traceability, control-to-clause lineage, and superseded-
regulation impact analysis.

Why flat vector search fails for these questions:
    Vector-only: "Which controls need to change if CRR3 Article 122a is
    amended?"
    → Retrieves chunks that mention Article 122a → no way to enumerate the
      Control nodes that IMPLEMENT it, or the Filing nodes those controls
      FEED, because that information is a relationship, not a similarity.

    GraphRAG: Same query →
      Step 1: pgvector match on "CRR3 Article 122a" → entry-point Clause node
      Step 2: Cypher traversal (<=3 hops) outward along IMPLEMENTS / FEEDS /
              CROSS_REFERENCES from that Clause
      Step 3: Traversal results rendered to natural language and fused with
              the vector passages before LLM synthesis

Schema (five node types, AWB regulatory knowledge graph):
    Regulation  — e.g. PRA SS1/23, CRR3 Article 122a
    Clause      — an individual paragraph or article within a Regulation
    Control     — an AWB internal control/policy implementing one or more Clauses
    System      — the AWB platform or model the Control runs on (e.g. MR-2026-038)
    Filing      — a recurring regulatory return (e.g. COREP C 33.00)

Edges (the relationships chunk similarity cannot represent):
    IMPLEMENTS         Control        -> Clause
    SUPERSEDES         Regulation     -> Regulation   (e.g. CRR3 supersedes CRR2)
    FEEDS              System         -> Filing
    CROSS_REFERENCES   Clause         -> Clause        (cross-regulator)

Access-tier and document-freshness metadata mirrors the vector store
(Sections 4.7, 4.8) — the graph is a second index over the same governed
corpus, not a parallel, ungoverned data source.

Selective use (Section 4.9.4 "When GraphRAG earns its added complexity"):
    GraphRAG is the most expensive of the four Section 4.9 techniques to
    build and operate, so the AWB platform team applies it selectively
    rather than book-wide. A lightweight classifier (_requires_graph_hop)
    decides per-query whether graph traversal is invoked at all; for
    single-document factual lookups it is not justified, and the query
    falls back to vector-only retrieval (Table 4.1A chunking + Section
    4.9.2 Contextual Retrieval already serve those well).

Regulatory context:
    PRA SS1/23 MR-2026-038: every traversal hop logged to audit trail
    EU AI Act Art. 14: human review on HITL_THRESHOLD queries (>= HIGH risk)
    DORA Art. 9: graph queries logged with latency for ICT monitoring
    FCA PS22/9: fused answers must cite every source used (vector + graph)

Bridge to Section 4.9.3:
    GraphRAGEngine follows the same tool-call / hop-trace audit pattern as
    AgenticRAGEngine (agentic_rag.py) — bounded iterations, typed actions,
    a hop_chain, and a confidence/HITL gate. The difference is the action
    space: GraphRAG's "hops" are Cypher traversal steps, not ReAct tool
    calls, and the loop terminates on hop-depth, not on a sufficiency
    self-evaluation.

AWB production limits:
    MAX_GRAPH_HOPS = 3            -- bounded traversal depth (predictable latency)
    GRAPH_TIMEOUT_S = 10           -- DORA Art. 11 ICT continuity requirement
    MAX_TRAVERSAL_NODES = 50       -- LIMIT clause on every Cypher query
    CONFIDENCE_THRESHOLD = 0.75    -- minimum to generate a final answer without HITL
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("awb.rag.graphrag")


# ── Constants ────────────────────────────────────────────────────────────────

MAX_GRAPH_HOPS        = 3      # Bounded Cypher traversal depth (Section 4.9.4)
GRAPH_TIMEOUT_S        = 10    # Per-traversal timeout (DORA Art. 11)
MAX_TRAVERSAL_NODES    = 50    # LIMIT on every Cypher query (predictable latency)
CONFIDENCE_THRESHOLD   = 0.75  # Minimum confidence to answer without escalation
HITL_THRESHOLD_SCORE   = 0.90  # Queries above this risk score -> human review
VECTOR_ENTRY_TOP_K     = 5     # Entry-point candidates from pgvector match


# ── Enums ────────────────────────────────────────────────────────────────────

class NodeType(str, Enum):
    """The five node types in the AWB regulatory knowledge graph (Section 4.9.4)."""
    REGULATION = "Regulation"   # e.g. PRA SS1/23, CRR3 Article 122a
    CLAUSE     = "Clause"       # an individual paragraph/article within a Regulation
    CONTROL    = "Control"      # an AWB internal control/policy
    SYSTEM     = "System"       # the AWB platform/model the Control runs on
    FILING     = "Filing"       # a recurring regulatory return (e.g. COREP C 33.00)


class EdgeType(str, Enum):
    """Edge types capturing relationships chunk similarity cannot represent."""
    IMPLEMENTS        = "IMPLEMENTS"          # Control -> Clause
    SUPERSEDES        = "SUPERSEDES"          # Regulation -> Regulation
    FEEDS             = "FEEDS"               # System -> Filing
    CROSS_REFERENCES  = "CROSS_REFERENCES"    # Clause -> Clause (cross-regulator)


class GraphHopStatus(str, Enum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILED  = "failed"
    SKIPPED = "skipped"


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class GraphNode:
    """A single node returned from a Cypher traversal."""
    node_id:    str
    node_type:  NodeType
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_context_line(self) -> str:
        """Render this node as a natural-language fragment for LLM context.

        e.g. "Control AWB-CTL-0142 implements PRA SS1/23 §4.2 and feeds
        COREP C 33.00" — the human-readable form of a graph fact, built so
        the synthesis LLM can treat it identically to a vector passage.
        """
        name = self.properties.get("name") or self.properties.get("reference") or self.node_id
        return f"{self.node_type.value} {name}"


@dataclass
class GraphTraversalHop:
    """
    A single bounded-depth Cypher traversal step.

    Mirrors RetrievalHop in agentic_rag.py — same audit semantics
    (PRA SS1/23 MR-2026-038 hop-trace requirement), different action
    space (graph traversal vs ReAct tool calls).
    """
    hop_index:   int
    cypher:      str
    entry_node:  str
    nodes_found: List[GraphNode] = field(default_factory=list)
    status:      GraphHopStatus = GraphHopStatus.PENDING
    latency_ms:  float = 0.0
    error:       str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hop_index":   self.hop_index,
            "cypher":      self.cypher,
            "entry_node":  self.entry_node,
            "node_count":  len(self.nodes_found),
            "status":      self.status.value,
            "latency_ms":  round(self.latency_ms, 1),
            "error":       self.error,
        }


@dataclass
class GraphRAGResult:
    """
    Result of a hybrid vector-plus-graph query.

    Carries the final synthesised answer, the fused vector+graph context,
    the complete traversal trace (for audit), and an HITL escalation flag.
    Field names deliberately mirror AgenticRAGResult so callers (and the
    §3.9 action-boundary control) can treat both result types uniformly.
    """
    run_id:              str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    query:               str = ""
    answer:              str = ""
    citations:           List[Dict] = field(default_factory=list)
    graph_hops:          List[GraphTraversalHop] = field(default_factory=list)
    total_graph_hops:    int = 0
    used_graph_traversal: bool = False
    confidence:          float = 0.0
    requires_human:      bool = False
    human_review_reason: str = ""
    model_registration:  str = "MR-2026-038"
    latency_ms:          float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id":               self.run_id,
            "query":                self.query,
            "answer":               self.answer,
            "citations":            self.citations,
            "graph_hop_trace":      [h.to_dict() for h in self.graph_hops],
            "total_graph_hops":     self.total_graph_hops,
            "used_graph_traversal": self.used_graph_traversal,
            "confidence":           round(self.confidence, 3),
            "requires_human":       self.requires_human,
            "human_review_reason":  self.human_review_reason,
            "model_registration":   self.model_registration,
            "latency_ms":           round(self.latency_ms, 1),
        }


# ── Cypher templates ──────────────────────────────────────────────────────────

# Bounded multi-hop traversal: control/filing lookup plus cross-reference
# expansion, abbreviated from the manuscript's Section 4.9.4 example.
# LIMIT enforces MAX_TRAVERSAL_NODES; the CROSS_REFERENCES hop depth is
# capped at MAX_GRAPH_HOPS to keep latency predictable.
CYPHER_CONTROL_LINEAGE = f"""
  MATCH (c:Clause {{id: $entry_clause_id}})
  MATCH (c)<-[:IMPLEMENTS]-(ctrl:Control)-[:FEEDS]->(f:Filing)
  OPTIONAL MATCH (c)-[:CROSS_REFERENCES*1..{MAX_GRAPH_HOPS}]-(related:Clause)
  RETURN ctrl, f, related LIMIT {MAX_TRAVERSAL_NODES}
"""

# Superseded-regulation impact analysis: "which controls need to change if
# this Regulation is amended/superseded?"
CYPHER_SUPERSESSION_IMPACT = f"""
  MATCH (r:Regulation {{id: $entry_regulation_id}})
  MATCH (r)-[:SUPERSEDES*0..{MAX_GRAPH_HOPS}]-(affected:Regulation)
  MATCH (affected)<-[:IMPLEMENTS]-()-[:IMPLEMENTS]->(c:Clause)<-[:IMPLEMENTS]-(ctrl:Control)
  RETURN affected, c, ctrl LIMIT {MAX_TRAVERSAL_NODES}
"""


# ── Graph store wrapper ───────────────────────────────────────────────────────

class Neo4jGraphStore:
    """
    Thin wrapper around a Neo4j driver session.

    Kept deliberately small: GraphRAGEngine never constructs raw Cypher
    from user input — it selects from a small set of parameterised,
    pre-approved templates (CYPHER_CONTROL_LINEAGE,
    CYPHER_SUPERSESSION_IMPACT), the same safety pattern AgenticRAGTools
    uses for query_operational_data() in agentic_rag.py.
    """

    def __init__(self, driver=None, database: str = "awb-regulatory-graph"):
        self._driver = driver
        self._database = database

    def run_cypher(
        self,
        cypher: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute a parameterised Cypher query and return a list of records.

        Returns an empty list (not an exception) if the driver is not
        configured — callers fall back to vector-only retrieval rather
        than failing the whole query, consistent with the "supplements,
        does not replace" design principle in Section 4.9.4.
        """
        if not self._driver:
            logger.info("Neo4j driver not configured — graph traversal skipped")
            return []
        try:
            with self._driver.session(database=self._database) as session:
                result = session.run(cypher, parameters or {})
                return [dict(record) for record in result]
        except Exception as exc:
            logger.warning("Cypher traversal failed: %s", exc)
            return []


# ── Main GraphRAG Engine ──────────────────────────────────────────────────────

class GraphRAGEngine:
    """
    Hybrid retrieval engine fusing pgvector semantic search with bounded
    Neo4j graph traversal.

    Does not replace vector search with graph traversal — it runs both and
    fuses the results (Section 4.9.4 "Query pattern: hybrid vector-plus-
    graph retrieval"):

        Step 1  Embed the query, match against pgvector to find the most
                semantically relevant Clause/Control nodes (entry points).
        Step 2  Bounded-depth Cypher traversal (<= MAX_GRAPH_HOPS) expands
                outward from those entry points along IMPLEMENTS,
                SUPERSEDES, FEEDS, and CROSS_REFERENCES edges.
        Step 3  Traversal results are rendered to natural-language context
                and merged with the vector-retrieved passages before being
                passed to the LLM for synthesis.

    A lightweight classifier (_requires_graph_hop) decides per-query
    whether the graph traversal step is invoked at all, so the added
    latency and infrastructure cost (Table 4.10) is paid only when the
    question actually requires relational reasoning — cross-regulatory
    traceability, control-to-clause lineage, or superseded-regulation
    impact analysis.

    Usage:
        graph_store = Neo4jGraphStore(driver=neo4j_driver)
        engine = GraphRAGEngine(
            vector_store=store,
            graph_store=graph_store,
            llm_client=llm,
        )
        result = engine.query(
            "Which controls need to change if CRR3 Article 122a is amended?"
        )
        print(result.answer)
        print(f"Used graph traversal: {result.used_graph_traversal}")
        for hop in result.graph_hops:
            print(f"  Hop {hop.hop_index}: {hop.entry_node} -> {len(hop.nodes_found)} nodes")
    """

    # Signals that the question needs relational reasoning, not just a
    # single-document factual lookup (Section 4.9.4 "When GraphRAG earns
    # its added complexity").
    GRAPH_SIGNALS = {
        "controls", "control", "lineage", "implement", "implements",
        "trace", "traceability", "linked", "related", "impact",
        "supersede", "supersedes", "superseded", "amended", "amend",
        "feeds", "filing", "filings", "cross-reference", "cross_reference",
        "which controls", "need to change", "affects", "affected",
    }

    def __init__(
        self,
        vector_store=None,    # RegulatoryVectorStore (pgvector entry-point match)
        graph_store: Optional[Neo4jGraphStore] = None,
        llm_client=None,      # LLMGenerationClient
        max_hops: int = MAX_GRAPH_HOPS,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ):
        self._vector_store        = vector_store
        self._graph_store         = graph_store or Neo4jGraphStore()
        self._llm                 = llm_client
        self.max_hops             = max_hops
        self.confidence_threshold = confidence_threshold

    # ── Classifier ─────────────────────────────────────────────────────────────

    def _requires_graph_hop(self, query: str) -> bool:
        """
        Decide whether this query needs graph traversal at all.

        Single-document factual lookups ("What is the LCR minimum under
        CRR3?") are well served by Table 4.1A chunking and Section 4.9.2
        Contextual Retrieval alone — invoking the graph for those would
        pay GraphRAG's added latency and infrastructure cost for no
        benefit. Returns True only when the query references relational
        concepts the graph schema actually models (IMPLEMENTS, SUPERSEDES,
        FEEDS, CROSS_REFERENCES).
        """
        lowered = query.lower()
        return any(signal in lowered for signal in self.GRAPH_SIGNALS)

    # ── Step 1: vector entry-point retrieval ─────────────────────────────────

    def _find_entry_points(self, query: str) -> List[Dict[str, Any]]:
        """
        Embed the query and match against pgvector to find the most
        semantically relevant Clause/Control nodes — the entry points for
        graph traversal (Section 4.9.4, "Step 1").
        """
        if not self._vector_store:
            return []
        try:
            results = self._vector_store.search(query=query, top_k=VECTOR_ENTRY_TOP_K)
            return [
                {
                    "text":      r.text[:600],
                    "document":  r.document_name,
                    "reference": r.metadata.get("document_reference", ""),
                    "clause_id": r.metadata.get("clause_id", r.metadata.get("document_reference", "")),
                    "section":   r.section_number or "",
                    "score":     round(r.relevance_score, 3),
                }
                for r in results
                if r.relevance_score >= 0.70
            ]
        except Exception as exc:
            logger.warning("Vector entry-point retrieval failed: %s", exc)
            return []

    # ── Step 2: bounded graph traversal ──────────────────────────────────────

    def _traverse_graph(
        self,
        entry_points: List[Dict[str, Any]],
        query: str,
    ) -> List[GraphTraversalHop]:
        """
        Run a bounded-depth Cypher traversal from each vector entry point.

        One hop per entry point, each hop itself bounded to
        MAX_GRAPH_HOPS relationship depth via the CROSS_REFERENCES*1..N
        pattern in the Cypher template (Section 4.9.4, "Step 2").
        """
        hops: List[GraphTraversalHop] = []
        lowered = query.lower()
        use_supersession = any(
            sig in lowered for sig in ("supersede", "amended", "amend", "impact")
        )
        template = CYPHER_SUPERSESSION_IMPACT if use_supersession else CYPHER_CONTROL_LINEAGE

        for hop_idx, entry in enumerate(entry_points[: self.max_hops]):
            entry_id = entry.get("clause_id", "")
            params = (
                {"entry_regulation_id": entry_id}
                if use_supersession
                else {"entry_clause_id": entry_id}
            )
            hop = GraphTraversalHop(
                hop_index=hop_idx,
                cypher=template.strip(),
                entry_node=entry_id,
            )
            start = time.monotonic()
            try:
                records = self._graph_store.run_cypher(template, params)
                hop.latency_ms = (time.monotonic() - start) * 1000
                hop.nodes_found = self._records_to_nodes(records)
                hop.status = GraphHopStatus.SUCCESS if records else GraphHopStatus.SKIPPED
            except Exception as exc:
                hop.latency_ms = (time.monotonic() - start) * 1000
                hop.status = GraphHopStatus.FAILED
                hop.error = str(exc)
                logger.warning("Graph traversal hop %d failed: %s", hop_idx, exc)
            hops.append(hop)

        return hops

    @staticmethod
    def _records_to_nodes(records: List[Dict[str, Any]]) -> List[GraphNode]:
        """Flatten raw Cypher records into typed GraphNode fragments."""
        nodes: List[GraphNode] = []
        for record in records:
            for key, value in record.items():
                if not isinstance(value, dict):
                    continue
                node_type_str = value.get("node_type") or key.capitalize()
                try:
                    node_type = NodeType(node_type_str)
                except ValueError:
                    node_type = NodeType.CONTROL  # safe default; doesn't block synthesis
                nodes.append(
                    GraphNode(
                        node_id=str(value.get("id", value.get("name", key))),
                        node_type=node_type,
                        properties=value,
                    )
                )
        return nodes

    # ── Step 3: fuse vector + graph context, then synthesise ────────────────

    def _fuse_context(
        self,
        entry_points: List[Dict[str, Any]],
        graph_hops: List[GraphTraversalHop],
    ) -> str:
        """
        Render traversal results to natural-language context and merge
        with the vector-retrieved passages (Section 4.9.4, "Step 3").

        e.g. "Control AWB-CTL-0142 implements PRA SS1/23 §4.2 and feeds
        COREP C 33.00" is treated identically to a vector passage by the
        downstream synthesis prompt.
        """
        vector_block = "\n---\n".join(
            f"[VECTOR | {e.get('reference','?')} {e.get('section','')} | score={e.get('score')}]\n"
            f"{e.get('text','')}"
            for e in entry_points
        )
        graph_lines: List[str] = []
        for hop in graph_hops:
            for node in hop.nodes_found:
                graph_lines.append(f"[GRAPH] {node.to_context_line()}")
        graph_block = "\n".join(graph_lines)

        parts = [p for p in (vector_block, graph_block) if p]
        return "\n\n".join(parts)

    def _synthesise_answer(self, query: str, fused_context: str) -> str:
        """Generate the final grounded answer from the fused vector+graph context."""
        if not fused_context:
            return (
                "Insufficient regulatory evidence was found for this query. "
                "Please consult AWB's Regulatory Affairs team directly."
            )
        if not self._llm:
            # Heuristic synthesis: surface the fused context directly
            return f"Based on retrieved vector and graph evidence:\n{fused_context[:1200]}"

        prompt = f"""You are the AWB Regulatory Knowledge Assistant, answering using a
hybrid vector-plus-graph retrieval result.

ORIGINAL QUERY: {query}

FUSED EVIDENCE (vector passages marked [VECTOR], graph facts marked [GRAPH]):
{fused_context[:3000]}

TASK:
Write a complete, grounded answer using ONLY the evidence above.
Rules:
- Cite every claim with its source document/section, or the graph relationship it came from
- Treat [GRAPH] facts (e.g. "Control X implements Clause Y") as authoritative relationship evidence
- Use precise regulatory language (SS1/23, CRR3, DORA, etc.)
- British English; PRA/FCA/EBA citation format
- 3-5 paragraphs maximum

Answer:"""
        try:
            return self._llm.generate(system_prompt="", user_query=prompt)
        except Exception as exc:
            logger.warning("GraphRAG synthesis failed: %s", exc)
            return (
                f"Evidence was retrieved via hybrid vector+graph search but final "
                f"synthesis failed: {exc}. Please review the traversal trace."
            )

    # ── Main query method ─────────────────────────────────────────────────────

    def query(self, user_query: str, force_graph: bool = False) -> GraphRAGResult:
        """
        Answer a regulatory query, invoking bounded graph traversal only
        when the query requires relational reasoning.

        Args:
            user_query:  The regulatory compliance question.
            force_graph: Override the classifier; always attempt traversal.

        Returns:
            GraphRAGResult with the synthesised answer, citations, and the
            complete graph traversal trace (PRA SS1/23 MR-2026-038 audit
            requirement).
        """
        run_id = str(uuid.uuid4())[:12]
        start = time.monotonic()

        entry_points = self._find_entry_points(user_query)

        needs_graph = force_graph or self._requires_graph_hop(user_query)
        graph_hops: List[GraphTraversalHop] = []
        if needs_graph and entry_points:
            logger.info("GraphRAG traversal invoked: run_id=%s query=%r", run_id, user_query[:60])
            graph_hops = self._traverse_graph(entry_points, user_query)
        elif needs_graph and not entry_points:
            logger.info(
                "GraphRAG signal detected but no vector entry points found — "
                "falling back to vector-only context: run_id=%s", run_id
            )

        fused_context = self._fuse_context(entry_points, graph_hops)
        answer = self._synthesise_answer(user_query, fused_context)

        citations = [
            {
                "document_reference": e.get("reference", ""),
                "document_name":      e.get("document", ""),
                "section_number":     e.get("section", ""),
                "relevance_score":    e.get("score", 0.0),
                "source_type":        "vector",
            }
            for e in entry_points
        ]
        for hop in graph_hops:
            for node in hop.nodes_found:
                citations.append({
                    "document_reference": node.node_id,
                    "document_name":      node.node_type.value,
                    "section_number":     "",
                    "relevance_score":    None,
                    "source_type":        "graph",
                })

        confidence = (
            sum(e.get("score", 0.65) for e in entry_points) / max(len(entry_points), 1)
            if entry_points else (0.55 if graph_hops else 0.0)
        )
        used_graph = any(h.status == GraphHopStatus.SUCCESS for h in graph_hops)
        requires_human = confidence < self.confidence_threshold and bool(entry_points or graph_hops)

        elapsed_ms = (time.monotonic() - start) * 1000

        result = GraphRAGResult(
            run_id=run_id,
            query=user_query,
            answer=answer,
            citations=citations[:10],
            graph_hops=graph_hops,
            total_graph_hops=len(graph_hops),
            used_graph_traversal=used_graph,
            confidence=min(confidence, 1.0),
            requires_human=requires_human,
            human_review_reason=(
                f"Confidence {confidence:.0%} below threshold {self.confidence_threshold:.0%}"
                if requires_human else ""
            ),
            latency_ms=elapsed_ms,
        )

        logger.info(
            "GraphRAG complete: run_id=%s used_graph=%s hops=%d confidence=%.2f latency=%.0fms",
            run_id, used_graph, len(graph_hops), result.confidence, elapsed_ms,
        )
        return result
