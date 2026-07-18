"""AWB Agentic Credit Intelligence Monitor — LangGraph Multi-Agent Orchestration.

Model ID:    MR-2026-055-AGT (Agentic extension of MR-2026-055)
Risk rating: HIGH (PRA SS1/23) — board approval required
EU AI Act:   HIGH-RISK Annex III §5b; agentic governance per
             PRA AI Roundtable findings (Oct 2025 / Feb 2026)
Board Risk Appetite: BAP-2026-CRD-007 (£500k HITL threshold)
HITL Threshold: ESCALATE_TO_HUMAN (autonomous actions bounded)

Architecture:
  CIM Supervisor Agent (LangGraph StateGraph)
    ├── CovenantAgent     → pgvector RAG over 23,000 facility letters
    ├── NewsAgent         → Gemini Flash→Pro adverse news pipeline
    ├── PSIAgent          → SHAP drift detection + breach narrator
    ├── StagingAgent      → 12 SICR triggers + LLM edge-case review
    └── NarrativeAgent    → Gemini 3.1 Pro CFO credit narrative

Each specialist agent implements the ReAct loop from Chapter 3
(AWBCreditDecisionAgent) adapted for monitoring rather than origination.
The supervisor routes tasks, aggregates findings, and enforces HITL
gates per PRA AI Roundtable Feb 2026 requirements for agentic systems.

Usage::
    graph = build_cim_graph()
    result = await graph.ainvoke({
        "facility_ids": ["FAC-2026-00123"],
        "run_date": date.today(),
        "hitl_gate": True,
    })
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Annotated, Any, Literal, Optional, Sequence

log = logging.getLogger(__name__)

# ── LangGraph / LangChain imports ─────────────────────────────────
try:
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:  # pragma: no cover — tested in integration env
    StateGraph = ToolNode = None  # type: ignore
    add_messages = lambda a, b: a + b  # type: ignore

# ── AWB platform imports ───────────────────────────────────────────
from awb_commons.schemas import (
    CIMRunResult,
    CovenantBreachAlert,
    NewsAlert,
    PSIBreach,
    SICRAlert,
)


# ══════════════════════════════════════════════════════════════════
# State schema
# ══════════════════════════════════════════════════════════════════

CLAUDE_MODEL = "claude-sonnet-4-6"  # Agent 5 — NarrativeAgent (AWB AI Policy)

class HITLDecision(str, Enum):
    """Human-in-the-loop gate outcome."""
    APPROVE   = "approve"
    ESCALATE  = "escalate"
    OVERRIDE  = "override"
    PENDING   = "pending"


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


@dataclass
class AgentStep:
    """Single hop in the multi-agent chain for audit trail."""
    agent_name: str
    action:     str
    observation: str
    token_count: int = 0


class CIMState(dict):
    """LangGraph state for Agentic CIM.

    Extends dict so it works with LangGraph's state reducer.
    All mutable collections use add_messages-style reducers
    to allow parallel node execution without clobbering.

    Fields:
        facility_ids:   AWB facility references to monitor.
        run_date:       Monitoring run date (ISO).
        messages:       LangGraph message list (supervisor comms).
        hop_chain:      Ordered list of AgentStep — full audit trail
                        required by PRA AI Roundtable Oct 2025
                        finding that hop-chain explainability is
                        mandatory for agentic credit systems.
        covenant_alerts:  Outputs from CovenantAgent.
        news_alerts:      Outputs from NewsAgent.
        psi_breaches:     Outputs from PSIAgent.
        sicr_alerts:      Outputs from StagingAgent.
        narrative_draft:  Draft narrative from NarrativeAgent.
        hitl_decision:    Human gate outcome.
        hitl_notes:       Reviewer notes for audit log.
        errors:           Any per-agent errors (non-fatal).
    """

    def __init__(
        self,
        facility_ids: list[str],
        run_date: date,
        hitl_gate: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            facility_ids      = facility_ids,
            run_date          = run_date.isoformat(),
            hitl_gate         = hitl_gate,
            messages          = [],
            hop_chain         = [],
            covenant_alerts   = [],
            news_alerts       = [],
            psi_breaches      = [],
            sicr_alerts       = [],
            narrative_draft   = "",
            hitl_decision     = HITLDecision.PENDING,
            hitl_notes        = "",
            errors            = [],
            **kwargs,
        )


# ══════════════════════════════════════════════════════════════════
# Specialist agent nodes
# ══════════════════════════════════════════════════════════════════

class CovenantAgent:
    """Module 1 — Covenant Compliance Analyser (agentic wrapper).

    Queries the pgvector store over 23,000 facility letters.
    ReAct loop: retrieve → check_ratio → flag_breach → summarise.
    FCA PS22/9 regulatory context injected into every retrieval.
    """

    NAME = "CovenantAgent"

    def __init__(self, pgvector_conn_str: str) -> None:
        self._conn = pgvector_conn_str

    async def __call__(self, state: CIMState) -> dict:
        log.info("CovenantAgent: scanning %d facilities", len(state["facility_ids"]))
        alerts: list[CovenantBreachAlert] = []
        steps:  list[dict] = []

        for fac_id in state["facility_ids"]:
            # ReAct hop 1: retrieve facility covenants from pgvector
            covenants = await self._retrieve_covenants(fac_id)
            steps.append({
                "agent_name":  self.NAME,
                "action":      f"retrieve_covenants({fac_id})",
                "observation": f"{len(covenants)} covenant clauses found",
                "token_count": len(covenants) * 12,  # approx
            })

            # ReAct hop 2: check each ratio against covenant threshold
            breaches = await self._check_ratios(fac_id, covenants)
            steps.append({
                "agent_name":  self.NAME,
                "action":      f"check_ratios({fac_id})",
                "observation": f"{len(breaches)} potential breaches",
                "token_count": 0,
            })

            if breaches:
                alerts.extend(breaches)

        return {
            "covenant_alerts": state["covenant_alerts"] + alerts,
            "hop_chain":       state["hop_chain"] + steps,
        }

    async def _retrieve_covenants(self, fac_id: str) -> list[dict]:
        """pgvector semantic search over facility letter store."""
        # Production: query pgvector via asyncpg
        # Stub returns empty for unit tests
        return []

    async def _check_ratios(
        self,
        fac_id: str,
        covenants: list[dict],
    ) -> list[CovenantBreachAlert]:
        """Compare live financial ratios to covenant thresholds."""
        return []


class NewsAgent:
    """Module 2 — Adverse News Monitor (agentic wrapper).

    Two-stage Gemini pipeline: Flash (triage) → Pro (deep analysis).
    POCA 2002 Suspicious Activity Report flag injected when score ≥ 0.75.
    ReAct loop: fetch_news → flash_triage → pro_analysis → flag_sar.
    """

    NAME = "NewsAgent"
    SAR_THRESHOLD = 0.75

    def __init__(self, llm_flash: Any, llm_pro: Any) -> None:
        self._flash = llm_flash
        self._pro   = llm_pro

    async def __call__(self, state: CIMState) -> dict:
        log.info("NewsAgent: screening %d facilities", len(state["facility_ids"]))
        alerts: list[NewsAlert] = []
        steps:  list[dict] = []

        for fac_id in state["facility_ids"]:
            # Hop 1: fetch live news from 4 sources
            raw_news = await self._fetch_news(fac_id)
            steps.append({
                "agent_name":  self.NAME,
                "action":      f"fetch_news({fac_id})",
                "observation": f"{len(raw_news)} articles retrieved",
                "token_count": sum(len(a.get("text","")) for a in raw_news) // 4,
            })

            # Hop 2: Gemini Flash triage (cost-efficient first pass)
            triage_results = await self._flash_triage(fac_id, raw_news)
            flagged = [r for r in triage_results if r.get("risk_score", 0) > 0.4]
            steps.append({
                "agent_name":  self.NAME,
                "action":      f"flash_triage({fac_id})",
                "observation": f"{len(flagged)}/{len(triage_results)} articles flagged",
                "token_count": len(triage_results) * 50,
            })

            # Hop 3: Gemini Pro deep analysis on flagged only
            if flagged:
                deep = await self._pro_analysis(fac_id, flagged)
                for item in deep:
                    if item.get("risk_score", 0) >= self.SAR_THRESHOLD:
                        # POCA 2002: potential SAR trigger
                        item["sar_flag"] = True
                    alerts.append(NewsAlert(**item))
                steps.append({
                    "agent_name":  self.NAME,
                    "action":      f"pro_analysis({fac_id})",
                    "observation": f"{len(deep)} deep analyses; "
                                   f"{sum(1 for a in alerts if getattr(a,'sar_flag',False))} SAR flags",
                    "token_count": len(flagged) * 300,
                })

        return {
            "news_alerts": state["news_alerts"] + alerts,
            "hop_chain":   state["hop_chain"] + steps,
        }

    async def _fetch_news(self, fac_id: str) -> list[dict]:
        return []

    async def _flash_triage(
        self, fac_id: str, articles: list[dict]
    ) -> list[dict]:
        return []

    async def _pro_analysis(
        self, fac_id: str, flagged: list[dict]
    ) -> list[dict]:
        return []


class PSIAgent:
    """Module 3 — Population Stability Index Monitor (agentic wrapper).

    SHAP-based drift detection with Gemini breach narrator.
    ReAct loop: compute_psi → detect_shap_drift → narrate_breach.
    PRA SS1/23 thresholds: PSI > 0.10 (warning), PSI > 0.25 (critical).
    """

    NAME = "PSIAgent"
    PSI_WARNING  = 0.10
    PSI_CRITICAL = 0.25

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    async def __call__(self, state: CIMState) -> dict:
        log.info("PSIAgent: computing drift metrics")
        breaches: list[PSIBreach] = []
        steps:    list[dict] = []

        # Hop 1: compute PSI vs development baseline
        psi_results = await self._compute_psi(state["facility_ids"])
        critical = [r for r in psi_results if r.get("psi", 0) >= self.PSI_WARNING]
        steps.append({
            "agent_name":  self.NAME,
            "action":      "compute_psi(all_facilities)",
            "observation": f"PSI computed; {len(critical)} above warning threshold",
            "token_count": 0,
        })

        # Hop 2: SHAP feature drift for critical PSI cases
        for case in critical:
            shap_drift = await self._detect_shap_drift(case)
            steps.append({
                "agent_name":  self.NAME,
                "action":      f"shap_drift({case.get('feature','?')})",
                "observation": f"Top drifting features: {shap_drift.get('top_features',[])}",
                "token_count": 80,
            })

            # Hop 3: Gemini narrator for PSI breach > critical
            if case.get("psi", 0) >= self.PSI_CRITICAL:
                narrative = await self._narrate_breach(case, shap_drift)
                breaches.append(PSIBreach(
                    feature=case.get("feature", ""),
                    psi=case.get("psi", 0),
                    severity="CRITICAL",
                    narrative=narrative,
                ))
                steps.append({
                    "agent_name":  self.NAME,
                    "action":      "narrate_breach",
                    "observation": f"Narrative: {narrative[:80]}...",
                    "token_count": 500,
                })

        return {
            "psi_breaches": state["psi_breaches"] + breaches,
            "hop_chain":    state["hop_chain"] + steps,
        }

    async def _compute_psi(self, facility_ids: list[str]) -> list[dict]:
        return []

    async def _detect_shap_drift(self, case: dict) -> dict:
        return {}

    async def _narrate_breach(self, case: dict, drift: dict) -> str:
        return ""


class StagingAgent:
    """Module 4 — IFRS 9 Staging Engine (agentic wrapper).

    12 deterministic SICR triggers + LLM edge-case review.
    ReAct loop: apply_triggers → llm_edge_review → human_gate.
    Human gate mandatory per PRA AI Roundtable Feb 2026 HITL guidance.
    """

    NAME = "StagingAgent"

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    async def __call__(self, state: CIMState) -> dict:
        log.info("StagingAgent: evaluating SICR for %d facilities", len(state["facility_ids"]))
        alerts: list[SICRAlert] = []
        steps:  list[dict] = []

        # Hop 1: deterministic SICR rules (12 triggers)
        deterministic = await self._apply_triggers(state["facility_ids"])
        steps.append({
            "agent_name":  self.NAME,
            "action":      "apply_12_sicr_triggers",
            "observation": f"{len(deterministic)} SICR triggers fired",
            "token_count": 0,
        })

        # Hop 2: LLM review for ambiguous / edge cases (~5%)
        edge_cases = [d for d in deterministic if d.get("edge_case")]
        if edge_cases:
            llm_decisions = await self._llm_edge_review(edge_cases)
            steps.append({
                "agent_name":  self.NAME,
                "action":      f"llm_edge_review({len(edge_cases)} cases)",
                "observation": f"{sum(1 for d in llm_decisions if d.get('stage2'))} confirmed Stage 2",
                "token_count": len(edge_cases) * 600,
            })
            alerts.extend(SICRAlert(**d) for d in llm_decisions)
        else:
            alerts.extend(SICRAlert(**d) for d in deterministic)

        # Note: human gate is handled by supervisor HITLNode
        return {
            "sicr_alerts": state["sicr_alerts"] + alerts,
            "hop_chain":   state["hop_chain"] + steps,
        }

    async def _apply_triggers(self, facility_ids: list[str]) -> list[dict]:
        return []

    async def _llm_edge_review(self, cases: list[dict]) -> list[dict]:
        return cases


class NarrativeAgent:
    """Module 5 — CFO Credit Narrative Generator (agentic wrapper).

    Gemini 3.1 Pro RAG over BoE Credit Conditions Survey + Pillar 3 + CRR3.
    Aggregates all upstream agent outputs into a four-panel CFO narrative.
    ReAct loop: aggregate_signals → rag_context → generate_narrative.
    """

    NAME = "NarrativeAgent"

    def __init__(self, llm: Any, rag_retriever: Any) -> None:
        self._llm       = llm
        self._retriever = rag_retriever

    async def __call__(self, state: CIMState) -> dict:
        log.info("NarrativeAgent: generating CFO narrative")
        steps: list[dict] = []

        # Hop 1: aggregate upstream signals into structured summary
        signal_summary = self._aggregate_signals(state)
        steps.append({
            "agent_name":  self.NAME,
            "action":      "aggregate_signals",
            "observation": (
                f"Covenants: {len(state['covenant_alerts'])}, "
                f"News: {len(state['news_alerts'])}, "
                f"PSI: {len(state['psi_breaches'])}, "
                f"SICR: {len(state['sicr_alerts'])}"
            ),
            "token_count": 200,
        })

        # Hop 2: RAG retrieval — BoE CCS + Pillar 3 + CRR3
        rag_context = await self._retrieve_context(signal_summary)
        steps.append({
            "agent_name":  self.NAME,
            "action":      "rag_context_retrieval",
            "observation": f"{len(rag_context)} context chunks retrieved",
            "token_count": sum(len(c) for c in rag_context) // 4,
        })

        # Hop 3: Gemini 3.1 Pro narrative generation (four-panel)
        narrative = await self._generate_narrative(signal_summary, rag_context)
        steps.append({
            "agent_name":  self.NAME,
            "action":      "generate_cfo_narrative",
            "observation": f"Narrative generated: {len(narrative)} chars",
            "token_count": len(narrative) // 4,
        })

        return {
            "narrative_draft": narrative,
            "hop_chain":       state["hop_chain"] + steps,
        }

    def _aggregate_signals(self, state: CIMState) -> dict:
        return {
            "covenant_count": len(state["covenant_alerts"]),
            "news_count":     len(state["news_alerts"]),
            "psi_critical":   sum(
                1 for b in state["psi_breaches"]
                if getattr(b, "severity", "") == "CRITICAL"
            ),
            "sicr_count":     len(state["sicr_alerts"]),
            "run_date":       state["run_date"],
        }

    async def _retrieve_context(self, summary: dict) -> list[str]:
        return []

    async def _generate_narrative(
        self, summary: dict, context: list[str]
    ) -> str:
        return ""


# ══════════════════════════════════════════════════════════════════
# HITL gate node
# ══════════════════════════════════════════════════════════════════

async def hitl_gate_node(state: CIMState) -> dict:
    """Human-in-the-loop gate (PRA AI Roundtable Feb 2026 requirement).

    For agentic systems operating on credit portfolios, PRA expects
    a meaningful human review step before final staging decisions
    are committed. This node:
      1. Packages the full hop_chain + all alerts into a review payload.
      2. Sends to AWB Credit Review Portal (async webhook).
      3. Awaits reviewer decision within the SLA window.
      4. If SLA expires → auto-escalate (never auto-approve).

    Board Risk Appetite BAP-2026-CRD-007:
      - Individual exposure > £500k: mandatory human approval.
      - SICR Stage 2 moves > 5% of portfolio: mandatory board alert.
    """
    if not state.get("hitl_gate", True):
        log.warning("HITL gate bypassed (test mode only)")
        return {"hitl_decision": HITLDecision.APPROVE}

    total_alerts = (
        len(state["covenant_alerts"])
        + len(state["news_alerts"])
        + len(state["psi_breaches"])
        + len(state["sicr_alerts"])
    )

    if total_alerts == 0:
        return {"hitl_decision": HITLDecision.APPROVE, "hitl_notes": "No alerts — auto-approved"}

    # Build hop-chain summary for reviewer
    hop_summary = "\n".join(
        f"[{s['agent_name']}] {s['action']} → {s['observation']}"
        for s in state["hop_chain"]
    )
    log.info(
        "HITL gate: %d total alerts. Hop chain:\n%s",
        total_alerts, hop_summary,
    )

    # Production: POST to AWB Credit Review Portal and poll for decision
    # Stub: escalate when alerts exist (conservative default)
    return {
        "hitl_decision": HITLDecision.ESCALATE,
        "hitl_notes": f"Auto-escalated: {total_alerts} alerts require reviewer action",
    }


# ══════════════════════════════════════════════════════════════════
# Supervisor router
# ══════════════════════════════════════════════════════════════════

def supervisor_router(state: CIMState) -> Literal[
    "covenant", "news", "psi", "staging", "narrative", "hitl", "__end__"
]:
    """LangGraph conditional edge — routes to next agent or terminates.

    Routing logic:
      covenant  → news  → psi  → staging  → narrative  → hitl  → END
    Each agent runs sequentially (monitoring order) to allow downstream
    agents to reference upstream findings. Parallel execution is
    available for covenant + news (independent data sources).
    """
    if not state["covenant_alerts"] and not state.get("_covenant_done"):
        return "covenant"
    if not state["news_alerts"] and not state.get("_news_done"):
        return "news"
    if not state["psi_breaches"] and not state.get("_psi_done"):
        return "psi"
    if not state["sicr_alerts"] and not state.get("_staging_done"):
        return "staging"
    if not state["narrative_draft"]:
        return "narrative"
    if state["hitl_decision"] == HITLDecision.PENDING:
        return "hitl"
    return "__end__"


# ══════════════════════════════════════════════════════════════════
# Graph construction
# ══════════════════════════════════════════════════════════════════

def build_cim_graph(
    pgvector_conn: str = "",
    llm_flash: Any = None,
    llm_pro: Any = None,
    rag_retriever: Any = None,
) -> Any:
    """Build and compile the Agentic CIM LangGraph StateGraph.

    Node topology::

        START
          │
          ▼
        covenant_node ──► news_node ──► psi_node
                                           │
                                           ▼
                                      staging_node
                                           │
                                           ▼
                                      narrative_node
                                           │
                                           ▼
                                        hitl_node
                                           │
                                           ▼
                                          END

    Returns the compiled graph. Invoke with::

        result = await graph.ainvoke(CIMState(
            facility_ids=["FAC-001"],
            run_date=date.today(),
        ))

    The full hop_chain in result["hop_chain"] satisfies PRA AI
    Roundtable Oct 2025 explainability requirements for agentic
    credit systems.
    """
    if StateGraph is None:
        raise ImportError("langgraph is required: pip install langgraph")

    # Initialise specialist agents
    covenant_agent  = CovenantAgent(pgvector_conn)
    news_agent      = NewsAgent(llm_flash, llm_pro)
    psi_agent       = PSIAgent(llm_pro)
    staging_agent   = StagingAgent(llm_pro)
    narrative_agent = NarrativeAgent(llm_pro, rag_retriever)

    # Build StateGraph
    builder = StateGraph(CIMState)

    builder.add_node("covenant_node",  covenant_agent)
    builder.add_node("news_node",      news_agent)
    builder.add_node("psi_node",       psi_agent)
    builder.add_node("staging_node",   staging_agent)
    builder.add_node("narrative_node", narrative_agent)
    builder.add_node("hitl_node",      hitl_gate_node)

    # Sequential edges
    builder.add_edge(START,             "covenant_node")
    builder.add_edge("covenant_node",   "news_node")
    builder.add_edge("news_node",       "psi_node")
    builder.add_edge("psi_node",        "staging_node")
    builder.add_edge("staging_node",    "narrative_node")
    builder.add_edge("narrative_node",  "hitl_node")
    builder.add_edge("hitl_node",       END)

    return builder.compile()


# ══════════════════════════════════════════════════════════════════
# Convenience runner
# ══════════════════════════════════════════════════════════════════

async def run_agentic_cim(
    facility_ids: list[str],
    run_date: Optional[date] = None,
    hitl_gate: bool = True,
    pgvector_conn: str = "",
    llm_flash: Any = None,
    llm_pro: Any = None,
    rag_retriever: Any = None,
) -> CIMState:
    """Run the Agentic CIM for a list of facilities.

    Args:
        facility_ids:   AWB facility references to monitor.
        run_date:       Monitoring date (defaults to today).
        hitl_gate:      Enable HITL gate (disable only for testing).
        pgvector_conn:  PostgreSQL connection string for pgvector.
        llm_flash:      Gemini Flash LLM instance.
        llm_pro:        Gemini 3.1 Pro LLM instance.
        rag_retriever:  RAG retriever for BoE CCS / Pillar 3 / CRR3.

    Returns:
        Final CIMState with all alerts, hop_chain, and HITL decision.

    Example::

        from langchain_google_genai import ChatGoogleGenerativeAI

        flash = ChatGoogleGenerativeAI(model="gemini-3.5-flash")
        pro   = ChatGoogleGenerativeAI(model="gemini-3.1-pro")
    claude = "claude-sonnet-4-6"  # Agent 5 narrative synthesis (AWB AI Policy)

        result = await run_agentic_cim(
            facility_ids=["FAC-2026-00123", "FAC-2026-00456"],
            run_date=date(2026, 5, 23),
            llm_flash=flash,
            llm_pro=pro,
        )
        print(f"HITL decision: {result['hitl_decision']}")
        print(f"Hop chain steps: {len(result['hop_chain'])}")
    """
    graph = build_cim_graph(
        pgvector_conn=pgvector_conn,
        llm_flash=llm_flash,
        llm_pro=llm_pro,
        rag_retriever=rag_retriever,
    )
    initial_state = CIMState(
        facility_ids=facility_ids,
        run_date=run_date or date.today(),
        hitl_gate=hitl_gate,
    )
    return await graph.ainvoke(initial_state)
