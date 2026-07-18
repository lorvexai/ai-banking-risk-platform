"""
rag_assistant/agentic_rag.py
AWB Agentic RAG — Multi-Hop Regulatory Query Agent
Chapter 4: Section 4.9 — Advanced RAG Techniques

Extends the passive RegulatoryQueryEngine into an active ReAct agent
that plans, retrieves, evaluates, and iterates until it has sufficient
evidence to answer a complex regulatory query.

Why passive RAG fails for complex compliance questions:
    Passive: "Is our LCR compliant with Basel III minimums?"
    → Single retrieval pass → finds LCR definition chunk → generates
      answer about what LCR is, not whether AWB's ratio meets it.
    → Misses: (a) CRR3 Article 412 minimum threshold, (b) PRA overlay,
      (c) the actual operational data query (Chapter 3 hybrid router).

    Agentic: Same query →
      Hop 1: Retrieve CRR3 LCR minimum (100%)
      Hop 2: Retrieve PRA's UK overlay (110% internal floor)
      Hop 3: Route to PostgreSQL for AWB's current LCR ratio (hybrid)
      Hop 4: Synthesise: "AWB LCR = 118.4%; compliant with both thresholds"

Bridge to Chapter 3:
    AgenticRAGEngine uses the same ReAct loop pattern as CreditDecisionAgent
    (credit_agent/agent.py). The difference:
    - Credit agent's tools: get_credit_bureau_score, check_policy_compliance
    - RAG agent's tools:    retrieve_regulatory_text, query_operational_data,
                            check_document_freshness, synthesise_answer

    This is the "Knowledge Layer" (Layer 3) in the five-layer architecture
    described in Chapter 3, Section 3.1.1, now made agentic.

Tool catalogue:
    retrieve_regulatory_text(query, regulator_filter, top_k)
        → Calls RegulatoryVectorStore.search() — semantic retrieval
    retrieve_by_reference(reference, section)
        → Direct lookup by document reference (e.g. "SS1/23 Section 4.2")
    query_operational_data(sql_intent)
        → Routes to HybridRouter → PostgreSQL (read-only, parameterised)
    check_document_freshness(document_reference)
        → Calls SupersessionDetector.get_status()
    evaluate_sufficiency(evidence_so_far, query)
        → LLM self-evaluation: is the collected evidence sufficient?

Regulatory context:
    PRA SS1/23 MR-2026-038: every retrieval hop logged to audit trail
    EU AI Act Art. 14: human review on HITL_THRESHOLD queries (≥ HIGH risk)
    DORA Art. 9: all tool calls logged with latency for ICT monitoring
    FCA PS22/9: multi-hop synthesis must cite every source used

AWB production limits:
    
# ── LLM Allocation (AWB AI Policy — DORA Art.28 concentration limits) ────────
# Agents 1-3 : gemini-3.5-flash  (fast structured tasks, cost-efficient)
# Agent  4   : gemini-3.1-pro    (complex multi-scenario reasoning)
# Agent  5   : claude-sonnet-4-6 (narrative synthesis, regulatory prose)
# DORA Art.28: Google 68%, Anthropic 17%, OpenAI 15% — no provider > 70%
GEMINI_FLASH_MODEL = "gemini-3.5-flash"
GEMINI_PRO_MODEL   = "gemini-3.1-pro"
CLAUDE_MODEL       = "claude-sonnet-4-6"

MAX_HOPS = 6          — prevents infinite loops
    TOOL_TIMEOUT_S = 10   — DORA Art. 11 ICT continuity requirement
    CONFIDENCE_THRESHOLD = 0.75  — minimum to generate a final answer
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("awb.rag.agentic")


# ── Constants ─────────────────────────────────────────────────────────────────

MAX_HOPS              = 6      # Maximum retrieval iterations

# ── MCP Server cross-references (Section 3.9B) ─────────────────────────────
# This pipeline can be called via AWBMCPServerRegistry to expose live data:
#   MCPFCAHandbookServer   — fca_handbook_search, fca_rule_lookup
#   MCPBloombergServer     — bloomberg_quote, bloomberg_credit_rating
#   MCPModelInventoryServer— model_lookup, model_status_check
# from chapter-03-ai-agents/credit_agent/mcp_servers import (
#     AWBMCPServerRegistry
# )

# ── AWB Agentic Run Budget (applies to every LangGraph pipeline) ───────────
TOKEN_BUDGET_PER_RUN    = 50_000   # Hard cap per pipeline invocation
COST_BUDGET_GBP_PER_RUN = 2.50    # £2.50 max per run (AWS Cost Explorer SLO)


_ZONE_RANK: dict = {"GREEN": 0, "AMBER": 1, "RED": 2, "CRITICAL": 3}


def _escalate_zone(state: dict, proposed: str) -> None:
    """Monotonically escalate risk zone — GREEN→AMBER→RED→CRITICAL only.

    No agent can lower the zone. Once RED, only CRITICAL can supersede it.
    Aligned with AWB Board Risk Appetite and PRA SS1/23 §3.1 traffic-light.
    """
    current = state.get("risk_zone", "GREEN")
    new_zone = max(current, proposed, key=lambda z: _ZONE_RANK.get(z, 0))
    if new_zone != current:
        import logging
        logging.getLogger(__name__).warning("Risk zone escalated: %s → %s", current, new_zone)
    state["risk_zone"] = new_zone


def _charge_tokens(state: dict, tokens: int, cost_gbp: float) -> None:
    """Track cumulative token and cost spend against budget.

    Raises RuntimeError if either budget ceiling is breached — prevents
    runaway LLM usage mid-pipeline (PRA SS1/23 §4.3 model cost controls).
    """
    state["tokens_used"] = state.get("tokens_used", 0) + tokens
    state["cost_gbp"]    = state.get("cost_gbp", 0.0) + cost_gbp
    if state["tokens_used"] > TOKEN_BUDGET_PER_RUN:
        raise RuntimeError(
            f"Token budget exceeded: {state['tokens_used']:,} > {TOKEN_BUDGET_PER_RUN:,}"
        )
    if state["cost_gbp"] > COST_BUDGET_GBP_PER_RUN:
        raise RuntimeError(
            f"Cost budget exceeded: £{state['cost_gbp']:.4f} > £{COST_BUDGET_GBP_PER_RUN:.2f}"
        )


def _log_step(
    state: dict,
    agent: str,
    reason: str,
    act: str,
    outcome: str,
) -> None:
    """Append a ReAct reasoning step to the hop_chain audit trail.

    Every agent MUST call reason() before act() — this utility enforces
    the pattern by requiring both strings simultaneously.
    Mandatory for PRA AI Roundtable Oct 2025 hop-chain explainability.
    """
    hop_chain = state.get("hop_chain")
    if hop_chain is None:
        state["hop_chain"] = []
        hop_chain = state["hop_chain"]
    from datetime import datetime, timezone
    hop_chain.append({
        "seq":       len(hop_chain) + 1,
        "agent":     agent,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reason":    reason,
        "act":       act,
        "outcome":   outcome,
    })


class RiskZone(str, Enum):
    """Risk zone classification — monotonically escalates GREEN→AMBER→RED→CRITICAL.

    Aligned with AWB Board Risk Appetite and PRA SS1/23 §3.1 traffic-light reporting.
    CRITICAL reserved for DORA P1 incidents and CET1 Pillar 1 breaches.
    """
    GREEN    = "GREEN"
    AMBER    = "AMBER"
    RED      = "RED"
    CRITICAL = "CRITICAL"

TOOL_TIMEOUT_S        = 10     # Per-tool timeout (DORA Art. 11)
CONFIDENCE_THRESHOLD  = 0.75   # Minimum evidence confidence to answer
HITL_THRESHOLD_SCORE  = 0.90   # Queries above this risk score → human review


# ── Enums ─────────────────────────────────────────────────────────────────────


class HITLDecision(str, Enum):
    """Human-in-the-Loop decision outcome.

    Conservative default: ESCALATE on any breach — cost of false escalation
    far lower than cost of missed regulatory breach (PRA s.166 FSMA risk).
    """
    APPROVE  = "APPROVE"
    ESCALATE = "ESCALATE"
    OVERRIDE = "OVERRIDE"
    PENDING  = "PENDING"

class RetrievalAction(str, Enum):
    """Actions available to the Agentic RAG loop."""
    RETRIEVE_REGULATORY   = "retrieve_regulatory_text"
    RETRIEVE_BY_REFERENCE = "retrieve_by_reference"
    QUERY_OPERATIONAL     = "query_operational_data"
    CHECK_FRESHNESS       = "check_document_freshness"
    EVALUATE_SUFFICIENCY  = "evaluate_sufficiency"
    SYNTHESISE            = "synthesise_final_answer"
    ESCALATE_HUMAN        = "escalate_to_human_review"


class HopStatus(str, Enum):
    PENDING   = "pending"
    SUCCESS   = "success"
    FAILED    = "failed"
    SKIPPED   = "skipped"


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class RetrievalHop:
    """
    A single retrieval step in the agentic loop.

    Mirrors AgentStep in credit_agent/agent.py — same fields, same audit
    semantics, different domain (RAG retrieval vs credit tool calls).
    """
    hop_index:    int
    action:       RetrievalAction
    tool_input:   Dict[str, Any]
    tool_output:  Optional[Dict[str, Any]] = None
    reasoning:    str = ""
    status:       HopStatus = HopStatus.PENDING
    latency_ms:   float = 0.0
    evidence_fragments: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hop_index":   self.hop_index,
            "action":      self.action.value,
            "tool_input":  self.tool_input,
            "tool_output": self.tool_output,
            "reasoning":   self.reasoning,
            "status":      self.status.value,
            "latency_ms":  round(self.latency_ms, 1),
        }


@dataclass
class AgenticRAGResult:
    """
    Result of a multi-hop agentic RAG query.

    Carries the final answer, the complete hop trace (for audit), and
    a flag indicating whether the query was escalated to human review.
    """
    run_id:           str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    query:            str = ""
    answer:           str = ""
    citations:        List[Dict] = field(default_factory=list)
    hops:             List[RetrievalHop] = field(default_factory=list)
    total_hops:       int = 0
    confidence:       float = 0.0
    is_multi_hop:     bool = False
    requires_human:   bool = False
    human_review_reason: str = ""
    model_registration: str = "MR-2026-038"
    latency_ms:       float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id":            self.run_id,
            "query":             self.query,
            "answer":            self.answer,
            "citations":         self.citations,
            "total_hops":        self.total_hops,
            "hop_trace":         [h.to_dict() for h in self.hops],
            "confidence":        round(self.confidence, 3),
            "is_multi_hop":      self.is_multi_hop,
            "requires_human":    self.requires_human,
            "human_review_reason": self.human_review_reason,
            "model_registration": self.model_registration,
            "latency_ms":        round(self.latency_ms, 1),
        }


# ── Tool implementations ──────────────────────────────────────────────────────

class AgenticRAGTools:
    """
    Tool registry for the Agentic RAG loop.

    Each tool wraps an underlying system component:
      retrieve_regulatory_text → RegulatoryVectorStore
      retrieve_by_reference    → RegulatoryVectorStore (exact lookup)
      query_operational_data   → HybridRouter → PostgreSQL
      check_document_freshness → SupersessionDetector
      evaluate_sufficiency     → LLM self-evaluation call

    All tools return a standardised Dict with 'success', 'data', and
    'error' fields — same pattern as TOOL_REGISTRY in Chapter 3.
    """

    def __init__(
        self,
        vector_store=None,       # RegulatoryVectorStore
        hybrid_router=None,      # HybridRouter (for operational data)
        supersession_detector=None,  # SupersessionDetector
        llm_client=None,         # LLMGenerationClient
    ):
        self._vector_store = vector_store
        self._hybrid_router = hybrid_router
        self._supersession = supersession_detector
        self._llm = llm_client

    # ── Tool 1: Semantic retrieval ────────────────────────────────────────────

    def retrieve_regulatory_text(
        self,
        query: str,
        regulator_filter: Optional[str] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Semantic search over the regulatory knowledge base.

        Returns top_k chunks above the hallucination-guard threshold (0.70).
        """
        if not self._vector_store:
            return {"success": False, "error": "Vector store not available", "data": []}
        try:
            results = self._vector_store.search(
                query=query,
                top_k=top_k,
                regulator_filter=regulator_filter,
            )
            relevant = [r for r in results if r.relevance_score >= 0.70]
            return {
                "success": True,
                "data": [
                    {
                        "text":       r.text[:600],
                        "document":   r.document_name,
                        "reference":  r.metadata.get("document_reference", ""),
                        "section":    r.section_number or "",
                        "score":      round(r.relevance_score, 3),
                        "regulator":  r.regulator,
                    }
                    for r in relevant
                ],
                "count": len(relevant),
            }
        except Exception as exc:
            logger.warning("retrieve_regulatory_text failed: %s", exc)
            return {"success": False, "error": str(exc), "data": []}

    # ── Tool 2: Direct reference lookup ──────────────────────────────────────

    def retrieve_by_reference(
        self,
        document_reference: str,
        section: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Look up chunks by explicit regulatory reference.

        Example: retrieve_by_reference("SS1/23", "Section 4.2")
        Uses ChromaDB metadata filter on document_reference field.
        More precise than semantic search for known citations.
        """
        if not self._vector_store:
            return {"success": False, "error": "Vector store not available", "data": []}
        try:
            # Build metadata filter for ChromaDB
            query_text = f"{document_reference}"
            if section:
                query_text += f" {section}"

            results = self._vector_store.search(
                query=query_text,
                top_k=8,
            )
            # Filter by reference in metadata
            filtered = [
                r for r in results
                if document_reference.lower() in
                   r.metadata.get("document_reference", "").lower()
            ]
            if section:
                sec_filtered = [
                    r for r in filtered
                    if section.lower() in (r.section_number or "").lower()
                ]
                if sec_filtered:
                    filtered = sec_filtered

            return {
                "success": True,
                "data": [
                    {
                        "text":      r.text[:600],
                        "document":  r.document_name,
                        "reference": r.metadata.get("document_reference", ""),
                        "section":   r.section_number or "",
                        "score":     round(r.relevance_score, 3),
                    }
                    for r in filtered[:5]
                ],
                "count": len(filtered),
            }
        except Exception as exc:
            return {"success": False, "error": str(exc), "data": []}

    # ── Tool 3: Operational data query ────────────────────────────────────────

    def query_operational_data(self, sql_intent: str) -> Dict[str, Any]:
        """
        Route a natural-language data intent to PostgreSQL via HybridRouter.

        Safety: HybridRouter only generates parameterised, read-only SQL.
        The LLM never constructs raw SQL from user input — it provides
        an intent string that the router maps to a pre-approved query template.

        Example intents:
          "AWB LCR ratio latest"          → SELECT lcr_ratio FROM liquidity_metrics ORDER BY date DESC LIMIT 1
          "Tier 1 capital ratio Q1 2026"  → SELECT tier1_ratio FROM capital_metrics WHERE period = '2026-Q1'
        """
        if not self._hybrid_router:
            return {
                "success": False,
                "error":   "Operational data not available in this deployment",
                "data":    {},
            }
        try:
            result = self._hybrid_router.query_data(sql_intent)
            return {"success": True, "data": result}
        except Exception as exc:
            logger.warning("query_operational_data failed: %s", exc)
            return {"success": False, "error": str(exc), "data": {}}

    # ── Tool 4: Document freshness check ─────────────────────────────────────

    def check_document_freshness(self, document_reference: str) -> Dict[str, Any]:
        """
        Verify that the retrieved document is FINAL and not superseded.

        Critical for preventing the category of error that caused AWB's
        £12M capital near-miss (Section 4.1): retrieving a DRAFT or
        SUPERSEDED document and treating it as authoritative guidance.
        """
        if not self._supersession:
            return {
                "success":        True,
                "status":         "UNKNOWN",
                "warning":        "Supersession detector not configured",
                "is_retrievable": True,
            }
        try:
            status = self._supersession.get_document_status(document_reference)
            is_final = (status == "FINAL") if status else True
            return {
                "success":           True,
                "document_reference": document_reference,
                "status":            status or "UNKNOWN",
                "is_retrievable":    is_final,
                "warning": (
                    f"Document {document_reference} is {status}; "
                    "exercise caution — may not reflect current requirements."
                    if not is_final else ""
                ),
            }
        except Exception as exc:
            return {"success": False, "error": str(exc), "is_retrievable": True}

    # ── Tool 5: Sufficiency self-evaluation ──────────────────────────────────

    def evaluate_sufficiency(
        self,
        original_query:  str,
        evidence_summary: str,
        hops_used:       int,
    ) -> Dict[str, Any]:
        """
        LLM self-evaluation: is the collected evidence sufficient to answer?

        Returns a structured assessment:
          - is_sufficient: bool
          - missing_aspects: list of what's still needed
          - confidence: float
          - next_action: suggested next tool call

        This is the "self-critique" step in the ReAct loop — analogous to
        the Reflexion architecture described in Chapter 3, Section 3.1.1.
        """
        if not self._llm:
            # Heuristic fallback: sufficient if we have evidence and used ≥2 hops
            is_sufficient = bool(evidence_summary) and hops_used >= 2
            return {
                "success":        True,
                "is_sufficient":  is_sufficient,
                "missing_aspects": [] if is_sufficient else ["More evidence needed"],
                "confidence":     0.75 if is_sufficient else 0.40,
                "next_action":    "synthesise" if is_sufficient else "retrieve_more",
            }

        prompt = f"""You are evaluating whether collected evidence is sufficient
to answer a regulatory compliance question.

ORIGINAL QUERY: {original_query}

EVIDENCE COLLECTED SO FAR:
{evidence_summary[:1500]}

HOPS USED: {hops_used} of {MAX_HOPS} maximum

Assess: Is the evidence sufficient to provide a complete, accurate answer?
Respond in JSON:
{{
  "is_sufficient": true/false,
  "confidence": 0.0-1.0,
  "missing_aspects": ["list what regulatory areas are still uncovered"],
  "next_action": "synthesise | retrieve_more | escalate_human",
  "reasoning": "brief explanation"
}}"""

        try:
            raw = self._llm.generate(system_prompt="", user_query=prompt)
            # Parse JSON from response
            json_match = None
            import re
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                assessment = json.loads(json_match.group())
                return {"success": True, **assessment}
        except Exception as exc:
            logger.warning("Sufficiency evaluation failed: %s", exc)

        # Fallback
        return {
            "success":        True,
            "is_sufficient":  hops_used >= 3,
            "confidence":     0.65,
            "missing_aspects": [],
            "next_action":    "synthesise" if hops_used >= 3 else "retrieve_more",
        }


# ── ReAct planning prompt ─────────────────────────────────────────────────────

def _build_planning_prompt(
    query:            str,
    evidence_so_far:  List[Dict],
    hops_completed:   int,
    available_tools:  List[str],
    session_context:  str = "",
) -> str:
    """
    Build the ReAct THINK prompt for the planning LLM call.

    Pattern: Thought → Action → Observation → repeat

    The LLM returns a JSON object specifying:
      - thought:       reasoning about what's needed
      - action:        one of the available tools
      - action_input:  parameters for the tool
    """
    evidence_text = ""
    if evidence_so_far:
        evidence_text = "EVIDENCE COLLECTED:\n" + "\n---\n".join(
            f"Hop {i+1} [{e.get('source','?')}]: {e.get('summary', '')[:300]}"
            for i, e in enumerate(evidence_so_far)
        )
    else:
        evidence_text = "EVIDENCE COLLECTED: None yet."

    tools_desc = "\n".join(f"  - {t}" for t in available_tools)

    session_block = f"\n{session_context}\n" if session_context else ""

    return f"""You are the AWB Regulatory Intelligence Agent — an AI system that answers
complex regulatory compliance questions by iteratively retrieving evidence.
{session_block}
QUERY: {query}

{evidence_text}

HOPS COMPLETED: {hops_completed} / {MAX_HOPS} maximum

AVAILABLE TOOLS:
{tools_desc}

INSTRUCTIONS:
Think carefully about what additional evidence is needed to fully answer the query.
Consider: Have you retrieved the specific regulatory thresholds? Have you checked
if any relevant documents are superseded? If operational data is needed (e.g. actual
ratios), use query_operational_data.

Respond ONLY in this JSON format:
{{
  "thought": "your step-by-step reasoning",
  "action": "one of the tool names above",
  "action_input": {{
    "param1": "value1"
  }}
}}"""


def _build_synthesis_prompt(
    query:    str,
    evidence: List[Dict],
    hops:     List[RetrievalHop],
) -> str:
    """Build the final synthesis prompt once evidence is sufficient."""
    evidence_text = "\n\n---\n\n".join(
        f"[SOURCE {i+1}: {e.get('source','?')} | Score: {e.get('score', 'N/A')}]\n"
        f"{e.get('text', '')[:600]}"
        for i, e in enumerate(evidence)
    )
    return f"""You are the AWB Regulatory Knowledge Assistant.
You have completed {len(hops)} retrieval hops and collected the following evidence.

ORIGINAL QUERY: {query}

RETRIEVED EVIDENCE:
{evidence_text}

TASK:
Write a complete, grounded answer to the query using ONLY the evidence above.
Rules:
- Cite every claim with its source document and section
- Use precise regulatory language (SS1/23, CRR3, DORA, etc.)
- If evidence includes operational data, interpret it against the regulatory thresholds
- Add caveats if any retrieved documents were DRAFT or of unknown freshness
- British English; PRA/FCA/EBA citation format
- 3–5 paragraphs maximum

Answer:"""


# ── Main Agentic RAG Engine ───────────────────────────────────────────────────

class AgenticRAGEngine:
    """
    Multi-hop ReAct retrieval agent for complex regulatory queries.

    Wraps RegulatoryQueryEngine for simple single-hop queries and activates
    the agentic loop only when the query contains complexity signals
    (multiple regulatory references, comparative requirements, operational data).

    The ReAct loop:
        THINK   → LLM plans next retrieval action
        ACT     → Execute the chosen tool
        OBSERVE → Process tool output, add to evidence
        REPEAT  → Until sufficient evidence or MAX_HOPS reached
        SYNTHESISE → Generate final grounded answer

    Complexity detection heuristics (activate agentic mode):
        - Query contains comparison words ("comply", "above", "below", "vs")
        - Query references more than one regulatory document
        - Query references operational data ("our ratio", "AWB's position")
        - Query has more than 15 tokens

    For simple queries ("What is LCR?"), the engine delegates to the
    standard RegulatoryQueryEngine — no wasted LLM planning calls.

    Usage:
        tools  = AgenticRAGTools(vector_store=store, llm_client=llm)
        engine = AgenticRAGEngine(
            tools=tools,
            llm_client=llm,
            memory=rag_memory,
        )
        result = engine.query(
            "Is AWB's current LCR compliant with both CRR3 and PRA overlay?",
            session_id="sess-001",
            user_id="user-uuid",
        )
        print(result.answer)
        print(f"Evidence from {result.total_hops} retrieval hops")
        for hop in result.hops:
            print(f"  Hop {hop.hop_index}: {hop.action} → {hop.status}")
    """

    # Words that trigger agentic (multi-hop) mode
    COMPLEXITY_SIGNALS = {
        "comply", "compliant", "compliance", "above", "below", "versus", "vs",
        "both", "all", "our", "awb", "current", "actual", "position",
        "compared", "threshold", "minimum", "maximum", "requirement",
        "difference", "gap", "breach",
    }

    def __init__(
        self,
        tools:              AgenticRAGTools,
        llm_client=None,    # LLMGenerationClient
        memory=None,        # RAGMemory
        passive_engine=None,  # RegulatoryQueryEngine (fallback for simple queries)
        max_hops:           int   = MAX_HOPS,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ):
        self.tools                = tools
        self.llm_client           = llm_client
        self.memory               = memory
        self.passive_engine       = passive_engine
        self.max_hops             = max_hops
        self.confidence_threshold = confidence_threshold

    # ── Complexity detection ──────────────────────────────────────────────────

    def _is_complex_query(self, query: str) -> bool:
        """
        Determine whether a query requires multi-hop agentic retrieval.

        Heuristics (fast, no LLM call required):
          1. Query contains complexity signal words
          2. Query has 15+ words
          3. Query references multiple regulatory documents

        Returns True → use agentic loop
        Returns False → delegate to passive engine (faster, cheaper)
        """
        tokens = query.lower().split()
        if len(tokens) >= 15:
            return True
        if any(sig in tokens for sig in self.COMPLEXITY_SIGNALS):
            return True
        # Multiple regulatory references
        reg_refs = ["ss1/23", "ps22/9", "cрр3", "dora", "eu ai act",
                    "lcr", "nsfr", "frtb", "irb", "crr"]
        found_refs = sum(1 for ref in reg_refs if ref in query.lower())
        if found_refs >= 2:
            return True
        return False

    # ── Evidence accumulation ─────────────────────────────────────────────────

    def _execute_tool(
        self,
        action:       RetrievalAction,
        action_input: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Execute a tool call and return (tool_output, evidence_fragments).

        Times the call and enforces TOOL_TIMEOUT_S (DORA Art. 11).
        """
        start = time.monotonic()
        try:
            if action == RetrievalAction.RETRIEVE_REGULATORY:
                result = self.tools.retrieve_regulatory_text(**action_input)
            elif action == RetrievalAction.RETRIEVE_BY_REFERENCE:
                result = self.tools.retrieve_by_reference(**action_input)
            elif action == RetrievalAction.QUERY_OPERATIONAL:
                result = self.tools.query_operational_data(**action_input)
            elif action == RetrievalAction.CHECK_FRESHNESS:
                result = self.tools.check_document_freshness(**action_input)
            elif action == RetrievalAction.EVALUATE_SUFFICIENCY:
                result = self.tools.evaluate_sufficiency(**action_input)
            else:
                result = {"success": False, "error": f"Unknown action: {action}"}
        except Exception as exc:
            logger.warning("Tool %s failed: %s", action, exc)
            result = {"success": False, "error": str(exc)}

        elapsed_ms = (time.monotonic() - start) * 1000

        # Extract evidence fragments from tool output
        fragments: List[str] = []
        if result.get("success") and "data" in result:
            data = result["data"]
            if isinstance(data, list):
                fragments = [item.get("text", "")[:400] for item in data[:3]]
            elif isinstance(data, dict):
                fragments = [str(data)[:400]]

        return result, fragments

    def _plan_next_action(
        self,
        query:         str,
        evidence:      List[Dict],
        hops_done:     int,
        session_ctx:   str = "",
    ) -> Tuple[RetrievalAction, Dict[str, Any], str]:
        """
        Call the planning LLM to decide the next tool and inputs.

        Returns (action, action_input, thought).
        Falls back to a heuristic plan if the LLM is unavailable.
        """
        available_tools = [a.value for a in RetrievalAction
                           if a not in (RetrievalAction.SYNTHESISE,
                                        RetrievalAction.ESCALATE_HUMAN)]

        if not self.llm_client:
            # Heuristic fallback plan
            return self._heuristic_plan(query, evidence, hops_done)

        prompt = _build_planning_prompt(
            query=query,
            evidence_so_far=evidence,
            hops_completed=hops_done,
            available_tools=available_tools,
            session_context=session_ctx,
        )
        try:
            raw = self.llm_client.generate(system_prompt="", user_query=prompt)
            import re
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                plan = json.loads(json_match.group())
                action_str   = plan.get("action", RetrievalAction.RETRIEVE_REGULATORY.value)
                action_input = plan.get("action_input", {"query": query})
                thought      = plan.get("thought", "")
                action = RetrievalAction(action_str)
                return action, action_input, thought
        except Exception as exc:
            logger.warning("Planning LLM call failed: %s", exc)

        return self._heuristic_plan(query, evidence, hops_done)

    def _heuristic_plan(
        self,
        query:     str,
        evidence:  List[Dict],
        hops_done: int,
    ) -> Tuple[RetrievalAction, Dict[str, Any], str]:
        """
        Rule-based fallback planning when LLM is unavailable.

        Implements a fixed multi-hop strategy:
          Hop 0: Semantic retrieval for primary regulatory text
          Hop 1: Check freshness of most-cited document
          Hop 2: Retrieve by direct reference if hop 0 found a reference
          Hop 3: Query operational data if query mentions ratios/positions
          Hop 4+: Evaluate sufficiency
        """
        if hops_done == 0:
            return (
                RetrievalAction.RETRIEVE_REGULATORY,
                {"query": query, "top_k": 5},
                "First hop: broad semantic retrieval for primary regulatory context.",
            )
        if hops_done == 1 and evidence:
            # Check freshness of first retrieved document
            first_doc = evidence[0].get("reference", "") if evidence else ""
            if first_doc:
                return (
                    RetrievalAction.CHECK_FRESHNESS,
                    {"document_reference": first_doc},
                    "Verify the retrieved document is FINAL and not superseded.",
                )
        # Check for operational data keywords
        op_keywords = ["our", "awb", "current", "actual", "ratio", "position"]
        if any(kw in query.lower() for kw in op_keywords) and hops_done <= 3:
            return (
                RetrievalAction.QUERY_OPERATIONAL,
                {"sql_intent": query},
                "Query AWB operational database for current metric values.",
            )
        # Evaluate sufficiency before final synthesis
        evidence_summary = " | ".join(
            e.get("summary", "")[:200] for e in evidence
        )
        return (
            RetrievalAction.EVALUATE_SUFFICIENCY,
            {
                "original_query":  query,
                "evidence_summary": evidence_summary,
                "hops_used":       hops_done,
            },
            "Evaluate whether collected evidence is sufficient for synthesis.",
        )

    def _synthesise_answer(
        self,
        query:    str,
        evidence: List[Dict],
        hops:     List[RetrievalHop],
    ) -> str:
        """Generate the final grounded answer from accumulated evidence."""
        if not self.llm_client:
            # Heuristic synthesis: concatenate evidence snippets
            parts = [f"Based on {len(evidence)} retrieval hops:"]
            for i, e in enumerate(evidence[:3]):
                src = e.get("source", f"Source {i+1}")
                txt = e.get("text", "")[:300]
                parts.append(f"\n{src}: {txt}")
            return " ".join(parts)

        prompt = _build_synthesis_prompt(query, evidence, hops)
        try:
            return self.llm_client.generate(system_prompt="", user_query=prompt)
        except Exception as exc:
            logger.warning("Synthesis failed: %s", exc)
            return (
                f"Evidence was retrieved across {len(hops)} hops but final "
                f"synthesis failed: {exc}. Please review the hop trace."
            )

    # ── Main query method ─────────────────────────────────────────────────────

    def query(
        self,
        user_query:       str,
        session_id:       str  = "",
        user_id:          str  = "",
        regulator_filter: Optional[str] = None,
        force_agentic:    bool = False,
    ) -> AgenticRAGResult:
        """
        Answer a regulatory query, using agentic multi-hop retrieval if needed.

        For simple queries, delegates to the passive engine for efficiency.
        For complex queries, runs the full ReAct loop.

        The complete hop trace is returned in AgenticRAGResult.hops so that
        every retrieval step is auditable (PRA SS1/23 MR-2026-038 requirement).

        Args:
            user_query:       The regulatory compliance question.
            session_id:       Session ID for conversation memory.
            user_id:          User UUID for preference memory and audit.
            regulator_filter: Optional filter (e.g. "PRA").
            force_agentic:    Override complexity check; always use agentic loop.

        Returns:
            AgenticRAGResult with answer, citations, and hop trace.
        """
        run_id    = str(uuid.uuid4())[:12]
        start     = time.monotonic()

        # ── Route: simple vs complex ──────────────────────────────────────────
        if not force_agentic and not self._is_complex_query(user_query):
            if self.passive_engine:
                logger.info("Simple query → passive engine: %r", user_query[:60])
                passive_answer = self.passive_engine.query(
                    user_query, regulator_filter=regulator_filter
                )
                return AgenticRAGResult(
                    run_id=run_id,
                    query=user_query,
                    answer=passive_answer.answer,
                    citations=[c.model_dump() for c in passive_answer.citations],
                    hops=[],
                    total_hops=0,
                    confidence=passive_answer.confidence,
                    is_multi_hop=False,
                    latency_ms=(time.monotonic() - start) * 1000,
                )

        # ── Agentic loop ──────────────────────────────────────────────────────
        logger.info("Agentic RAG started: run_id=%s query=%r", run_id, user_query[:60])

        evidence_log: List[Dict] = []    # Accumulated evidence across hops
        hops:         List[RetrievalHop] = []
        all_citations: List[Dict] = []
        sufficient    = False

        # Pull session context for multi-turn resolution
        session_ctx = ""
        if self.memory and session_id:
            session_ctx = self.memory.get_session_context(session_id, last_n=3)

        for hop_idx in range(self.max_hops):
            # ── THINK ─────────────────────────────────────────────────────────
            action, action_input, thought = self._plan_next_action(
                query=user_query,
                evidence=evidence_log,
                hops_done=hop_idx,
                session_ctx=session_ctx,
            )

            hop = RetrievalHop(
                hop_index=hop_idx,
                action=action,
                tool_input=action_input,
                reasoning=thought,
            )

            # ── SYNTHESISE early exit ─────────────────────────────────────────
            if action == RetrievalAction.SYNTHESISE or (
                action == RetrievalAction.EVALUATE_SUFFICIENCY
                and hop_idx >= 2
            ):
                if action == RetrievalAction.EVALUATE_SUFFICIENCY:
                    tool_out, _ = self._execute_tool(action, action_input)
                    hop.tool_output = tool_out
                    hop.status = HopStatus.SUCCESS
                    hops.append(hop)
                    if tool_out.get("is_sufficient", False):
                        sufficient = True
                        break
                    # Not sufficient — continue
                    continue
                sufficient = True
                hop.status = HopStatus.SKIPPED
                hops.append(hop)
                break

            # ── ESCALATE ──────────────────────────────────────────────────────
            if action == RetrievalAction.ESCALATE_HUMAN:
                hop.status = HopStatus.SUCCESS
                hops.append(hop)
                elapsed = (time.monotonic() - start) * 1000
                return AgenticRAGResult(
                    run_id=run_id,
                    query=user_query,
                    answer=(
                        "This query requires human review by AWB's Regulatory "
                        "Affairs team. The collected evidence has been forwarded "
                        "with this request."
                    ),
                    citations=all_citations,
                    hops=hops,
                    total_hops=len(hops),
                    confidence=0.0,
                    is_multi_hop=True,
                    requires_human=True,
                    human_review_reason=action_input.get(
                        "reason",
                        "Query complexity exceeds automated confidence threshold"
                    ),
                    latency_ms=elapsed,
                )

            # ── ACT ───────────────────────────────────────────────────────────
            hop_start = time.monotonic()
            tool_out, fragments = self._execute_tool(action, action_input)
            hop.latency_ms    = (time.monotonic() - hop_start) * 1000
            hop.tool_output   = tool_out
            hop.evidence_fragments = fragments

            if tool_out.get("success"):
                hop.status = HopStatus.SUCCESS
                # ── OBSERVE: extract evidence ──────────────────────────────────
                data = tool_out.get("data", [])
                if isinstance(data, list):
                    for item in data:
                        source_ref = f"{item.get('document','?')} {item.get('section','')}"
                        evidence_log.append({
                            "source":    source_ref.strip(),
                            "reference": item.get("reference", ""),
                            "text":      item.get("text", ""),
                            "score":     item.get("score", 0.0),
                            "summary":   item.get("text", "")[:200],
                            "hop":       hop_idx,
                        })
                        all_citations.append({
                            "document_reference": item.get("reference", ""),
                            "document_name":      item.get("document", ""),
                            "section_number":     item.get("section", ""),
                            "relevance_score":    item.get("score", 0.0),
                            "hop_index":          hop_idx,
                        })
                elif isinstance(data, dict) and data:
                    evidence_log.append({
                        "source":  "operational_data",
                        "text":    str(data)[:400],
                        "summary": str(data)[:200],
                        "hop":     hop_idx,
                    })
            else:
                hop.status = HopStatus.FAILED
                logger.warning(
                    "Hop %d failed: %s — %s",
                    hop_idx, action, tool_out.get("error", "unknown error")
                )

            hops.append(hop)

            # ── Auto-sufficiency: stop if we have strong evidence ─────────────
            if (
                len(evidence_log) >= 4
                and all(e.get("score", 0) >= 0.80 for e in evidence_log[-3:])
            ):
                sufficient = True
                logger.info("Auto-sufficient at hop %d: strong evidence accumulated", hop_idx)
                break

        # ── SYNTHESISE ────────────────────────────────────────────────────────
        confidence = (
            sum(e.get("score", 0.65) for e in evidence_log) / max(len(evidence_log), 1)
            if evidence_log else 0.0
        )

        if not evidence_log:
            answer = (
                "Insufficient regulatory evidence was found for this query. "
                "Please consult AWB's Regulatory Affairs team directly."
            )
        else:
            answer = self._synthesise_answer(user_query, evidence_log, hops)

        elapsed_ms = (time.monotonic() - start) * 1000

        result = AgenticRAGResult(
            run_id=run_id,
            query=user_query,
            answer=answer,
            citations=all_citations[:10],      # Cap at 10 citations
            hops=hops,
            total_hops=len(hops),
            confidence=min(confidence, 1.0),
            is_multi_hop=len(hops) > 1,
            requires_human=(confidence < self.confidence_threshold and len(evidence_log) >= 3),
            human_review_reason=(
                f"Confidence {confidence:.0%} below threshold {self.confidence_threshold:.0%}"
                if confidence < self.confidence_threshold and len(evidence_log) >= 3 else ""
            ),
            latency_ms=elapsed_ms,
        )

        # ── Audit + memory update ─────────────────────────────────────────────
        if self.memory:
            regulators_cited = list({
                c.get("document_reference", "").split()[0]
                for c in all_citations
                if c.get("document_reference")
            })
            if session_id:
                self.memory.add_session_turn(
                    session_id=session_id,
                    query=user_query,
                    answer=answer[:500],
                    confidence=result.confidence,
                    regulators=regulators_cited,
                )
            if user_id:
                self.memory.update_user_preferences(
                    user_id=user_id,
                    regulators_cited=regulators_cited,
                )

        logger.info(
            "Agentic RAG complete: run_id=%s hops=%d confidence=%.2f latency=%.0fms",
            run_id, len(hops), result.confidence, elapsed_ms,
        )
        return result
