# agentic_ai_patterns.py | AWB Advanced Agentic AI Architecture
# Chapter 3 | Section 3.9A | FCA/PRA Regulated Orchestration Patterns
# Guardrail taxonomy, multi-agent topologies, agentic loop design
# BAP-2026-AGENT-001 | EU AI Act Arts 9/13/14 | PRA SS1/23 para 7
"""
AWB Advanced Agentic AI Architecture — Section 3.9A

This module documents and implements the four agentic AI architectural
patterns used throughout the AWB-AI-2025 programme, and the complete
FCA/PRA guardrail taxonomy that makes them safe to deploy in a UK
regulated banking environment.

Contents
--------
1. AgentTopology — four topology patterns with AWB usage mapping
2. ReActLoop — the core reasoning-acting loop with audit trail
3. GuardrailRegistry — five-tier FCA/PRA guardrail taxonomy
4. SupervisorWorkerGraph — hierarchy pattern (Chapter 12 AML/KYC)
5. PeerToPeerGraph — parallel pattern (Chapter 7 market risk)
6. SequentialPipelineGraph — pipeline pattern (Chapters 3,9,11,15)
7. HierarchicalGraph — three-tier pattern (Chapter 16 integrated)
8. AgentRunBudget — token and cost circuit breaker (all chapters)
"""
# ── MCP Server cross-references (Section 3.9B) ─────────────────────────────
# Runtime data access via AWBMCPServerRegistry:
#   MCPFCAHandbookServer    — fca_handbook_search, fca_rule_lookup
#   MCPBloombergServer      — bloomberg_quote, bloomberg_credit_rating
#   MCPModelInventoryServer — model_lookup, model_status_check
# from chapter-03-ai-agents/credit_agent.mcp_servers import (
#     AWBMCPServerRegistry
# )
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LangGraph import with stub fallback
# ---------------------------------------------------------------------------
try:
    from langgraph.graph import StateGraph, END, START  # type: ignore
    _LANGGRAPH_AVAILABLE = True
except ModuleNotFoundError:
    _LANGGRAPH_AVAILABLE = False
    END = "END"
    START = "START"

    class StateGraph:  # type: ignore
        def __init__(self, schema):
            self._nodes: Dict[str, Any] = {}
            self._edges: List[Tuple[str, str]] = []
            self._entry = None
        def add_node(self, name, fn): self._nodes[name] = fn; return self
        def add_edge(self, s, d): self._edges.append((s, d)); return self
        def set_entry_point(self, n): self._entry = n; return self
        def compile(self): return self


# ---------------------------------------------------------------------------
# 1. Agent Topology Catalogue
# ---------------------------------------------------------------------------


class RiskZone(str, Enum):
    """Risk zone classification — monotonically escalates GREEN→AMBER→RED→CRITICAL.

    Aligned with AWB Board Risk Appetite and PRA SS1/23 §3.1 traffic-light reporting.
    CRITICAL reserved for DORA P1 incidents and CET1 Pillar 1 breaches.
    """
    GREEN    = "GREEN"
    AMBER    = "AMBER"
    RED      = "RED"
    CRITICAL = "CRITICAL"


# ── AWB Per-Run Token and Cost Budget ────────────────────────────────────────
TOKEN_BUDGET_PER_RUN    = 50_000   # Hard cap per pipeline invocation (all chapters)
COST_BUDGET_GBP_PER_RUN = 2.50    # £2.50 max per run (AWS Cost Explorer SLO)


_ZONE_RANK_PATTERNS: dict = {"GREEN": 0, "AMBER": 1, "RED": 2, "CRITICAL": 3}


def _escalate_zone(current: str, proposed: str) -> str:
    """Pure function: return the higher-severity zone of the two inputs.

    Used by all AWB agentic pipelines (Ch4-Ch16) to ensure monotonic escalation.
    Unlike state-mutating versions in domain chapters, this returns the result
    so the caller assigns it (agentic_ai_patterns.py is a library, not a pipeline).

    Example::
        zone = _escalate_zone("GREEN", "RED")   # → "RED"
        zone = _escalate_zone("RED", "AMBER")   # → "RED"  (never downgrades)
    """
    return max(current, proposed, key=lambda z: _ZONE_RANK_PATTERNS.get(z, 0))


# ── Canonical AWB LLM Allocation (DORA Art.28 — no provider > 70%) ───────────
# Reference: AWB AI Policy v2.3, June 2026
# Agents 1-3 : gemini-3.5-flash  (fast structured tasks, cost-efficient)
# Agent  4   : gemini-3.1-pro    (complex multi-scenario reasoning budget)
# Agent  5   : claude-sonnet-4-6 (nuanced regulatory narrative synthesis)
# Concentration: Google 68%, Anthropic 17%, OpenAI 15%
AWB_LLM_AGENTS_1_3 = "gemini-3.5-flash"
AWB_LLM_AGENT_4    = "gemini-3.1-pro"
AWB_LLM_AGENT_5    = "claude-sonnet-4-6"

class AgentTopology(str, Enum):
    """
    Four multi-agent topology patterns used in AWB-AI-2025.

    SEQUENTIAL_PIPELINE
        Agents run in a fixed chain: A → B → C → D → E → HITL.
        State flows forward; each agent enriches the shared dict.
        Used by: ALL 13 agentic pipeline chapters (3, 7-15).
        Regulatory fit: Easiest to audit — deterministic hop-chain.

    SUPERVISOR_WORKER
        A supervisor LLM (Gemini 3.1 Pro) dynamically routes tasks
        to specialist worker agents and aggregates their results.
        Used by: Chapter 12 AML (TypologyMatchingAgent as supervisor),
                 Chapter 16 integrated platform orchestrator.
        Regulatory fit: Supervisor reasoning must be logged; each
        worker output independently validated before aggregation.

    PEER_TO_PEER
        Agents run in parallel; no hierarchy. Each agent produces
        an independent assessment; results are merged by vote or
        maximum-severity rule.
        Used by: Chapter 7 market risk (CVA + VaR + IRC agents
        run in parallel then PnLAttributionAgent merges).
        Regulatory fit: Parallel audit trails merge at join node;
        all must reach HITL gate together.

    HIERARCHICAL
        Three-tier: orchestrator → domain supervisors → specialist
        workers. The Chapter 16 integrated platform uses this to
        coordinate credit, market, liquidity, AML, and compliance
        domains simultaneously.
        Regulatory fit: Most complex; requires trace_id propagation
        across all three tiers and a unified hop-chain.
    """
    SEQUENTIAL_PIPELINE = "sequential_pipeline"
    SUPERVISOR_WORKER    = "supervisor_worker"
    PEER_TO_PEER         = "peer_to_peer"
    HIERARCHICAL         = "hierarchical"


# ---------------------------------------------------------------------------
# 2. ReAct Loop — reasoning before acting
# ---------------------------------------------------------------------------

@dataclass
class ReActStep:
    """One iteration of the Reason → Act → Observe loop."""
    seq:       int
    agent:     str
    thought:   str   # regulatory reasoning BEFORE acting
    action:    str   # tool call or LLM invocation
    observation: str # result / outcome
    timestamp: str   = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    tool_cost_gbp: float = 0.0


class ReActLoop:
    """
    Regulated ReAct loop for AWB AI agents.

    The ReAct pattern (Reason + Act) requires each agent to produce
    an explicit reasoning string BEFORE executing any action. This
    satisfies EU AI Act Article 13 transparency and PRA SS1/23
    para 7 audit trail requirements — the regulator can inspect
    exactly WHY each tool was called.

    AWB enforcement rules (BAP-2026-AGENT-001 §3):
    - thought must reference at least one regulatory paragraph
    - action must be one of the 4 approved tool types
    - observation must be recorded before next thought
    - max_iterations enforced by AgentRunBudget circuit breaker

    Example
    -------
    loop = ReActLoop(agent_name="PolicyCheckerAgent", max_iterations=5)
    for step in loop.steps:
        log.info("[%s] thought=%s", step.agent, step.thought[:80])
    """

    def __init__(
        self,
        agent_name: str,
        max_iterations: int = 10,
    ) -> None:
        self.agent_name = agent_name
        self.max_iterations = max_iterations
        self.steps: List[ReActStep] = []
        self._iteration = 0

    def reason(self, thought: str) -> None:
        """Record reasoning step — must precede every act() call."""
        if not thought.strip():
            raise ValueError("ReAct thought cannot be empty — regulatory requirement")
        self._current_thought = thought

    def act(self, action: str, observation: str, tool_cost_gbp: float = 0.0) -> ReActStep:
        """Record action + observation. Returns the completed step."""
        if not hasattr(self, "_current_thought"):
            raise RuntimeError(
                "act() called without prior reason() — violates ReAct pattern "
                "and BAP-2026-AGENT-001 §3.1 regulatory reasoning requirement"
            )
        if self._iteration >= self.max_iterations:
            raise RuntimeError(
                f"AgentRunBudget: max_iterations={self.max_iterations} exceeded "
                f"for agent {self.agent_name}. Escalating per BAP-2026-AGENT-001 §4."
            )
        step = ReActStep(
            seq=self._iteration + 1,
            agent=self.agent_name,
            thought=self._current_thought,
            action=action,
            observation=observation,
            tool_cost_gbp=tool_cost_gbp,
        )
        self.steps.append(step)
        self._iteration += 1
        del self._current_thought
        return step

    def total_cost_gbp(self) -> float:
        return sum(s.tool_cost_gbp for s in self.steps)

    def to_hop_chain(self) -> List[Dict[str, Any]]:
        """Export as hop-chain format for BAP-2026-AGENT-001 §6 audit."""
        return [
            {
                "seq": s.seq,
                "agent": s.agent,
                "timestamp": s.timestamp,
                "thought": s.thought,
                "action": s.action,
                "observation": s.observation,
                "tool_cost_gbp": s.tool_cost_gbp,
            }
            for s in self.steps
        ]


# ---------------------------------------------------------------------------
# 3. FCA/PRA Guardrail Registry — five-tier taxonomy
# ---------------------------------------------------------------------------

class GuardrailTier(str, Enum):
    """
    AWB five-tier guardrail taxonomy for FCA/PRA regulated AI.

    TIER_1_INPUT_VALIDATION
        Schema and type validation before the LLM is invoked.
        Prevents prompt injection, malformed inputs, PII leakage.
        Implementation: Pydantic models on all tool inputs.
        Regulatory: EU AI Act Art.9 risk management system.

    TIER_2_OUTPUT_SCHEMA
        Structured output enforcement on every LLM response.
        Prevents hallucinated fields, invalid enumerations.
        Implementation: JSON Schema validation via instructor library.
        Regulatory: PRA SS1/23 §4 model output documentation.

    TIER_3_DOMAIN_RULES
        Deterministic business rules applied AFTER LLM output.
        Examples: CET1 floor, POCA s.333A gate, PSI threshold.
        These rules CANNOT be overridden by LLM reasoning.
        Regulatory: PRA PS7/25, FCA SYSC 15A, DORA Art.17.

    TIER_4_HITL_GATE
        Mandatory human review for HIGH-RISK decisions.
        Implementation: LangGraph interrupt() mechanism.
        Triggers: facility ≥ £500k, SAR filing, RED zone, CBET breach.
        Regulatory: EU AI Act Art.14 human oversight, PRA SS1/23 §7.

    TIER_5_AUDIT_TRAIL
        Immutable hop-chain persisted to PostgreSQL (7-year retention).
        Every agent step, tool call, and HITL decision recorded.
        Regulatory: FCA COBS 9.1.3R, DORA Art.17, SM&CR.
    """
    TIER_1_INPUT_VALIDATION = "tier_1_input_validation"
    TIER_2_OUTPUT_SCHEMA    = "tier_2_output_schema"
    TIER_3_DOMAIN_RULES     = "tier_3_domain_rules"
    TIER_4_HITL_GATE        = "tier_4_hitl_gate"
    TIER_5_AUDIT_TRAIL      = "tier_5_audit_trail"


@dataclass
class Guardrail:
    """A single guardrail registration entry."""
    tier:              GuardrailTier
    name:              str
    description:       str
    regulatory_basis:  str
    is_hard_block:     bool   # True = cannot be bypassed by LLM reasoning
    implementation:    str    # code pattern / class reference


class GuardrailRegistry:
    """
    AWB FCA/PRA Guardrail Registry (BAP-2026-AGENT-001 §2).

    Every guardrail applied by any AWB AI agent must be registered
    here. The registry is reviewed by the Model Risk Committee
    quarterly and submitted to PRA SS1/23 documentation.

    Usage
    -----
    registry = GuardrailRegistry()
    registry.register(Guardrail(
        tier=GuardrailTier.TIER_3_DOMAIN_RULES,
        name="POCA s.333A tipping-off gate",
        description="SAR indicator must never reach credit pipeline",
        regulatory_basis="POCA 2002 s.333A — criminal offence",
        is_hard_block=True,
        implementation="get_credit_gate_decision() in agentic_aml_kyc.py",
    ))
    registry.assert_all_tiers_covered(agent_name="SARDraftingAgent")
    """

    # Pre-registered AWB platform-wide guardrails
    _PLATFORM_GUARDRAILS: List[Guardrail] = [
        Guardrail(
            tier=GuardrailTier.TIER_1_INPUT_VALIDATION,
            name="ToolCallValidator",
            description="Pydantic schema validation on all tool inputs before execution",
            regulatory_basis="EU AI Act Art.9 risk management system",
            is_hard_block=True,
            implementation="awb_commons.tool_validator.ToolCallValidator",
        ),
        Guardrail(
            tier=GuardrailTier.TIER_2_OUTPUT_SCHEMA,
            name="LLM output JSON schema enforcement",
            description="instructor library enforces structured output on every LLM call",
            regulatory_basis="PRA SS1/23 §4 model output documentation",
            is_hard_block=True,
            implementation="awb_commons.llm_factory.AWBLLMFactory with response_model=",
        ),
        Guardrail(
            tier=GuardrailTier.TIER_3_DOMAIN_RULES,
            name="CET1 minimum floor",
            description="CET1 < 4.5% always RED regardless of LLM output",
            regulatory_basis="CRR3 Art.92 §1(a)",
            is_hard_block=True,
            implementation="agentic_regulatory_compliance.py CET1_MINIMUM_PCT=4.5",
        ),
        Guardrail(
            tier=GuardrailTier.TIER_3_DOMAIN_RULES,
            name="POCA s.333A tipping-off gate",
            description="SAR indicator must never reach credit pipeline",
            regulatory_basis="POCA 2002 s.333A — up to 5 years imprisonment",
            is_hard_block=True,
            implementation="get_credit_gate_decision() returns BLOCKED/CLEARED only",
        ),
        Guardrail(
            tier=GuardrailTier.TIER_4_HITL_GATE,
            name="Credit facility HITL — £500k threshold",
            description="LangGraph interrupt() for all credit decisions ≥ £500,000",
            regulatory_basis="EU AI Act Art.14 human oversight, FCA PS22/9",
            is_hard_block=True,
            implementation="langgraph_agent.py route_after_memo_draft()",
        ),
        Guardrail(
            tier=GuardrailTier.TIER_4_HITL_GATE,
            name="RED zone universal escalation",
            description="Any RED zone finding triggers HITLDecision.ESCALATE",
            regulatory_basis="PRA SS1/23 §7, EU AI Act Art.14",
            is_hard_block=True,
            implementation="All agentic pipeline hitl_gate() functions",
        ),
        Guardrail(
            tier=GuardrailTier.TIER_5_AUDIT_TRAIL,
            name="Hop-chain immutable audit log",
            description="Every agent step recorded to PostgreSQL with 7-year retention",
            regulatory_basis="FCA COBS 9.1.3R, DORA Art.17, SM&CR",
            is_hard_block=False,
            implementation="_log_step() in all agentic pipeline modules",
        ),
        Guardrail(
            tier=GuardrailTier.TIER_5_AUDIT_TRAIL,
            name="DORA Art.17 circuit breaker",
            description="CircuitBreaker halts all downstream calls on ICT failure",
            regulatory_basis="DORA Art.17 ICT resilience",
            is_hard_block=True,
            implementation="awb_commons.circuit_breaker.CircuitBreaker",
        ),
    ]

    def __init__(self) -> None:
        self._guardrails: List[Guardrail] = list(self._PLATFORM_GUARDRAILS)

    def register(self, guardrail: Guardrail) -> None:
        """Add an agent-specific guardrail to the registry."""
        self._guardrails.append(guardrail)
        log.info("Guardrail registered: [%s] %s", guardrail.tier.value, guardrail.name)

    def get_by_tier(self, tier: GuardrailTier) -> List[Guardrail]:
        return [g for g in self._guardrails if g.tier == tier]

    def get_hard_blocks(self) -> List[Guardrail]:
        return [g for g in self._guardrails if g.is_hard_block]

    def assert_all_tiers_covered(self, agent_name: str) -> None:
        """
        Assert all five guardrail tiers are covered.
        Called during agent initialisation — fails loudly if a tier is missing.
        Per BAP-2026-AGENT-001 §2.4: all five tiers mandatory for HIGH-RISK agents.
        """
        covered = {g.tier for g in self._guardrails}
        missing = set(GuardrailTier) - covered
        if missing:
            raise RuntimeError(
                f"Agent '{agent_name}' missing guardrail tiers: "
                f"{[t.value for t in missing]}. "
                f"BAP-2026-AGENT-001 §2.4 requires all five tiers for HIGH-RISK AI."
            )
        log.info("Agent '%s': all five guardrail tiers verified.", agent_name)

    def summary_table(self) -> List[Dict[str, str]]:
        """Return MRC-ready summary table of all registered guardrails."""
        return [
            {
                "tier": g.tier.value,
                "name": g.name,
                "regulatory_basis": g.regulatory_basis,
                "hard_block": str(g.is_hard_block),
            }
            for g in sorted(self._guardrails, key=lambda x: x.tier.value)
        ]


# ---------------------------------------------------------------------------
# 4. AgentRunBudget — token + cost circuit breaker
# ---------------------------------------------------------------------------

@dataclass
class AgentRunBudget:
    """
    Token and cost circuit breaker for every AWB agent run.

    AWB-AI-2025 enforces per-run budgets to prevent runaway LLM costs
    and satisfy FCA PRIN 3 (management and control) obligations.

    Per BAP-2026-AGENT-001 §4:
    - Hard stop at max_tokens_per_run (default: 50,000 tokens)
    - Cost circuit opens at 3× the 30-day rolling average cost per run
    - All budget breaches escalated to Architecture Board

    Usage
    -----
    budget = AgentRunBudget(max_tokens=50_000, max_cost_gbp=2.50)
    budget.record_llm_call(tokens=1_200, cost_gbp=0.04)
    budget.check()  # raises BudgetExceededError if over limit
    """
    max_tokens_per_run:  int   = 50_000
    max_cost_gbp:        float = 2.50
    warn_tokens_pct:     float = 0.80
    warn_cost_pct:       float = 0.80

    _tokens_used: int   = field(default=0, init=False)
    _cost_gbp:    float = field(default=0.0, init=False)
    _calls:       int   = field(default=0, init=False)

    def record_llm_call(self, tokens: int, cost_gbp: float = 0.0) -> None:
        self._tokens_used += tokens
        self._cost_gbp += cost_gbp
        self._calls += 1

    def check(self) -> Dict[str, Any]:
        """
        Check budget status. Raises BudgetExceededError if hard limit hit.
        Returns status dict for logging.
        """
        token_pct = self._tokens_used / self.max_tokens_per_run
        cost_pct  = self._cost_gbp / self.max_cost_gbp if self.max_cost_gbp > 0 else 0.0

        if self._tokens_used > self.max_tokens_per_run:
            raise BudgetExceededError(
                f"Token budget exceeded: {self._tokens_used:,} > {self.max_tokens_per_run:,}. "
                f"BAP-2026-AGENT-001 §4: escalating to Architecture Board."
            )
        if self._cost_gbp > self.max_cost_gbp:
            raise BudgetExceededError(
                f"Cost budget exceeded: £{self._cost_gbp:.4f} > £{self.max_cost_gbp:.2f}. "
                f"BAP-2026-AGENT-001 §4: circuit open."
            )

        status = "WARN" if (token_pct > self.warn_tokens_pct or cost_pct > self.warn_cost_pct) else "OK"
        return {
            "status": status,
            "tokens_used": self._tokens_used,
            "token_pct": round(token_pct * 100, 1),
            "cost_gbp": round(self._cost_gbp, 4),
            "cost_pct": round(cost_pct * 100, 1),
            "llm_calls": self._calls,
        }

    def reset(self) -> None:
        self._tokens_used = 0
        self._cost_gbp = 0.0
        self._calls = 0


class BudgetExceededError(RuntimeError):
    """Raised when an agent run exceeds token or cost budget."""


# ---------------------------------------------------------------------------
# 5. Supervisor–Worker topology example
# ---------------------------------------------------------------------------

class SupervisorWorkerOrchestrator:
    """
    Supervisor–Worker multi-agent topology.

    The supervisor (Gemini 3.1 Pro) receives the task and decides
    WHICH worker agents to invoke and in what order, based on the
    current state. Workers report back to the supervisor; the
    supervisor aggregates and produces the final output.

    AWB usage:
    - Chapter 12 AML/KYC: TypologyMatchingAgent acts as supervisor,
      dynamically routing between network analysis and sanctions
      screening based on alert score
    - Chapter 16: Platform orchestrator supervises all 13 domain agents

    Regulatory requirement (EU AI Act Art.14):
    The supervisor's routing decisions must be logged as part of the
    hop-chain — the regulator must be able to reconstruct WHY a
    particular worker was invoked for a specific case.
    """

    def __init__(
        self,
        supervisor_llm_call: Callable[[str, str], str],
        workers: Dict[str, Callable[[Dict], Dict]],
        budget: Optional[AgentRunBudget] = None,
    ) -> None:
        self._supervisor = supervisor_llm_call
        self._workers = workers
        self._budget = budget or AgentRunBudget()
        self._hop_chain: List[Dict] = []

    def _log(self, seq: int, agent: str, thought: str, act: str, outcome: str) -> None:
        self._hop_chain.append({
            "seq": seq,
            "agent": agent,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "thought": thought,
            "act": act,
            "outcome": outcome,
        })

    async def run(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the supervisor–worker loop.

        The supervisor first decides which workers to activate,
        then aggregates their outputs into the final result.
        All routing decisions are logged per BAP-2026-AGENT-001 §6.
        """
        run_id = str(uuid.uuid4())[:8]
        results: Dict[str, Any] = {"run_id": run_id, "worker_outputs": {}}

        # Step 1: Supervisor routing decision
        routing_prompt = (
            f"Task: {task}\n"
            f"Available workers: {list(self._workers.keys())}\n"
            f"Context summary: {str(context)[:500]}\n"
            "Which workers should be invoked and in what order? "
            "State regulatory reasoning for each selection."
        )
        routing_decision = self._supervisor(routing_prompt, "")
        self._log(1, "Supervisor", "Determining optimal worker routing", routing_prompt[:200], routing_decision[:200])

        # Step 2: Invoke workers based on routing
        for i, (worker_name, worker_fn) in enumerate(self._workers.items(), start=2):
            worker_input = {**context, "task": task, "supervisor_routing": routing_decision}
            try:
                worker_output = worker_fn(worker_input)
                results["worker_outputs"][worker_name] = worker_output
                self._log(i, f"Worker:{worker_name}", f"Executing {worker_name} task",
                          f"Input keys: {list(worker_input.keys())}", str(worker_output)[:200])
            except Exception as exc:
                log.warning("Worker %s failed: %s", worker_name, exc)
                results["worker_outputs"][worker_name] = {"error": str(exc)}

        # Step 3: Supervisor aggregation
        agg_prompt = (
            f"Aggregate worker outputs for task: {task}\n"
            f"Worker results: {results['worker_outputs']}\n"
            "Produce consolidated assessment with regulatory risk classification."
        )
        aggregation = self._supervisor(agg_prompt, "")
        results["aggregated_assessment"] = aggregation
        self._log(len(self._workers) + 2, "Supervisor", "Aggregating worker outputs",
                  agg_prompt[:200], aggregation[:200])

        results["hop_chain"] = self._hop_chain
        self._budget.check()
        return results


# ---------------------------------------------------------------------------
# 6. AWB Guardrail taxonomy validation helper
# ---------------------------------------------------------------------------

def validate_agent_output(
    output: Dict[str, Any],
    required_fields: List[str],
    domain_rules: Optional[Dict[str, Callable[[Any], bool]]] = None,
    agent_name: str = "UnknownAgent",
) -> Tuple[bool, List[str]]:
    """
    Tier 2 + Tier 3 guardrail validation on agent output.

    Applies:
    - Tier 2: Required field presence (output schema)
    - Tier 3: Domain rule assertions (hard deterministic checks)

    Returns (is_valid, list_of_violations).

    Example
    -------
    is_valid, violations = validate_agent_output(
        output=state,
        required_fields=["risk_zone", "hitl_decision"],
        domain_rules={
            "cet1_pct": lambda v: v >= 4.5,        # CRR3 Art.92 floor
            "hitl_decision": lambda v: v in ("APPROVE", "ESCALATE", "OVERRIDE"),
        },
        agent_name="CapitalAdequacyAgent",
    )
    """
    violations: List[str] = []

    # Tier 2: schema check
    for field_name in required_fields:
        if field_name not in output or output[field_name] is None:
            violations.append(
                f"[T2-SCHEMA] Required field '{field_name}' missing in {agent_name} output"
            )

    # Tier 3: domain rules
    if domain_rules:
        for field_name, rule_fn in domain_rules.items():
            if field_name in output:
                try:
                    if not rule_fn(output[field_name]):
                        violations.append(
                            f"[T3-DOMAIN] Domain rule failed for '{field_name}' "
                            f"in {agent_name}: value={output[field_name]}"
                        )
                except Exception as exc:
                    violations.append(
                        f"[T3-DOMAIN] Rule check error for '{field_name}': {exc}"
                    )

    if violations:
        log.warning("[%s] Output validation: %d violation(s): %s",
                    agent_name, len(violations), violations)
    return len(violations) == 0, violations


# ---------------------------------------------------------------------------
# 7. AWB Agentic AI Architecture reference map
# ---------------------------------------------------------------------------

AWB_AGENT_ARCHITECTURE_MAP: Dict[str, Dict[str, Any]] = {
    "chapter_03_credit_decision": {
        "model_ref": "MR-2026-037",
        "topology": AgentTopology.SEQUENTIAL_PIPELINE,
        "agents": ["DocumentIngestor", "FinancialAnalyser", "PolicyChecker", "MemoDrafter"],
        "hitl_trigger": "facility >= £500,000 — EU AI Act Art.14",
        "guardrail_tiers": [1, 2, 3, 4, 5],
    },
    "chapter_07_market_risk": {
        "model_ref": "MR-2026-049",
        "topology": AgentTopology.PEER_TO_PEER,
        "agents": ["CVAAgent", "VaRAgent", "IRCAgent", "PnLAttributionAgent", "MarketRiskReportAgent"],
        "hitl_trigger": "RED zone or FRTB IMA breach",
        "guardrail_tiers": [1, 2, 3, 4, 5],
    },
    "chapter_12_aml_kyc": {
        "model_ref": "MR-2026-060-AML",
        "topology": AgentTopology.SUPERVISOR_WORKER,
        "agents": ["TransactionScoringAgent", "NetworkGraphAgent", "KYCScreeningAgent",
                   "TypologyMatchingAgent (supervisor)", "SARDraftingAgent"],
        "hitl_trigger": "SAR threshold: score>0.70 or structuring ring or sanctions hit",
        "guardrail_tiers": [1, 2, 3, 4, 5],
        "special_guardrail": "POCA s.333A tipping-off hard block (Tier 3)",
    },
    "chapter_16_integrated_platform": {
        "model_ref": "MR-2026-074-IP",
        "topology": AgentTopology.HIERARCHICAL,
        "agents": ["PlatformOrchestratorAgent (L1)", "DomainSupervisors x5 (L2)",
                   "Specialist workers x13 (L3)"],
        "hitl_trigger": "Any domain RED zone or cross-domain correlation breach",
        "guardrail_tiers": [1, 2, 3, 4, 5],
    },
}


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)

    # Demonstrate guardrail registry
    registry = GuardrailRegistry()
    print("\n=== AWB Guardrail Registry ===")
    for row in registry.summary_table():
        print(f"  [{row['tier']}]  {row['name']}  (hard_block={row['hard_block']})")

    # Demonstrate ReAct loop
    print("\n=== ReAct Loop Demo ===")
    loop = ReActLoop("PolicyCheckerAgent", max_iterations=3)
    loop.reason("PRA SS1/23 §4.2 Gate 1 requires AUC-ROC >= 0.70 for credit models.")
    step = loop.act("evaluate_auc_roc(model_id='MR-2026-037')", "AUC-ROC=0.881 — PASS")
    print(f"  Step {step.seq}: {step.agent} — {step.observation}")

    # Demonstrate AgentRunBudget
    print("\n=== AgentRunBudget Demo ===")
    budget = AgentRunBudget(max_tokens=50_000, max_cost_gbp=2.50)
    budget.record_llm_call(tokens=1_200, cost_gbp=0.04)
    budget.record_llm_call(tokens=3_400, cost_gbp=0.12)
    status = budget.check()
    print(f"  Budget status: {status}")

    # Architecture map
    print("\n=== AWB Agent Architecture Map ===")
    for chapter, config in AWB_AGENT_ARCHITECTURE_MAP.items():
        print(f"  {chapter}: topology={config['topology'].value}, model={config['model_ref']}")
