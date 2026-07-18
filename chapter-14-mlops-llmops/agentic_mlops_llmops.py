# agentic_mlops_llmops.py | AWB Agentic MLOps/LLMOps Pipeline
# Chapter 14 | MR-2026-062-MLO | PRA SS1/23 para 7 human oversight
# Five-agent LangGraph StateGraph: model drift → CI/CD validation →
# prompt governance → LLMOps orchestration → MLOps report
# BAP-2026-MLO-001 | EU AI Act Arts 9/13/14 | FCA PRIN 11
"""
AWB Agentic MLOps/LLMOps Investigation Pipeline (MR-2026-062-MLO)

Addresses the governance question: once a model or LLM prompt change is
proposed, how does the AI system reason transparently through drift
detection, SS1/23 validation gates, RAGAS quality monitoring, and token
budget governance to a PRA-auditable deployment decision, with every step
logged for PRA SS1/23 para 7 and EU AI Act Article 13 audit purposes?

Agent topology (LangGraph StateGraph):
  START → model_drift → cicd_validation → prompt_governance
        → llmops_orchestration → mlops_report → hitl → END

LLM allocation:
  Agents 1-3: Gemini 3.5 Flash (fast structured MLOps tasks)
  Agent 4:    Gemini 3.1 Pro  (synthesis + rollback decisions)
  Agent 5:    Claude Sonnet 4.6 (regulatory narrative + audit report)
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
import uuid
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LangGraph import with _SequentialStub fallback
# ---------------------------------------------------------------------------
try:
    from langgraph.graph import StateGraph, END, START  # type: ignore
    _LANGGRAPH_AVAILABLE = True
except ModuleNotFoundError:
    _LANGGRAPH_AVAILABLE = False

    class _SequentialStub:
        """Fallback sequential executor when LangGraph is not installed."""
        def __init__(self, state_class):
            self._state_class = state_class
            self._nodes: Dict[str, Any] = {}
            self._edges: List[Tuple[str, str]] = []
            self._entry: Optional[str] = None

        def add_node(self, name: str, fn) -> None:
            self._nodes[name] = fn

        def add_edge(self, src: str, dst: str) -> None:
            self._edges.append((src, dst))

        def set_entry_point(self, name: str) -> None:
            self._entry = name

        def compile(self):
            return self

        async def ainvoke(self, state: dict) -> dict:
            visited = set()
            current = self._entry
            successors = {s: d for s, d in self._edges}
            while current and current not in ("__end__", END if _LANGGRAPH_AVAILABLE else "END"):
                if current in visited:
                    break
                visited.add(current)
                if current in self._nodes:
                    result = self._nodes[current](state)
                    if asyncio.iscoroutine(result):
                        result = await result
                    if isinstance(result, dict):
                        state.update(result)
                current = successors.get(current)
            return state

    class StateGraph:  # type: ignore
        def __init__(self, state_class):
            self._stub = _SequentialStub(state_class)
        def add_node(self, name, fn):
            self._stub.add_node(name, fn)
            return self
        def add_edge(self, src, dst):
            self._stub.add_edge(src, dst)
            return self
        def set_entry_point(self, name):
            self._stub.set_entry_point(name)
            return self
        def compile(self):
            return self._stub

    END = "END"
    START = "START"

# ---------------------------------------------------------------------------
# LLM provider stubs (replace with real SDK calls in production)
# ---------------------------------------------------------------------------
try:
    import google.generativeai as genai  # type: ignore
    _GEMINI_AVAILABLE = True
except ModuleNotFoundError:
    _GEMINI_AVAILABLE = False

try:
    import anthropic  # type: ignore
    _ANTHROPIC_AVAILABLE = True
except ModuleNotFoundError:
    _ANTHROPIC_AVAILABLE = False


def _call_gemini_flash(prompt: str, context: str = "") -> str:
    """Invoke Gemini 3.5 Flash for fast structured MLOps tasks."""
    if _GEMINI_AVAILABLE:
        model = genai.GenerativeModel("gemini-3.5-flash")
        response = model.generate_content(f"{prompt}\n\nContext:\n{context}")
        return response.text
    full = f"{prompt}\n\nContext:\n{context}" if context else prompt
    words = full.split()[:30]
    return f"[Gemini-Flash stub] {' '.join(words)}..."


def _call_gemini_pro(prompt: str, context: str = "") -> str:
    """Invoke Gemini 3.1 Pro for synthesis and rollback decisions."""
    if _GEMINI_AVAILABLE:
        model = genai.GenerativeModel("gemini-3.1-pro")
        response = model.generate_content(f"{prompt}\n\nContext:\n{context}")
        return response.text
    full = f"{prompt}\n\nContext:\n{context}" if context else prompt
    words = full.split()[:30]
    return f"[Gemini-Pro stub] {' '.join(words)}..."


def _call_claude_sonnet(prompt: str, context: str = "") -> str:
    """Invoke Claude Sonnet 4.6 for regulatory narrative and audit report."""
    if _ANTHROPIC_AVAILABLE:
        client = anthropic.Anthropic()
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": f"{prompt}\n\nContext:\n{context}" if context else prompt,
            }],
        )
        return message.content[0].text
    full = f"{prompt}\n\nContext:\n{context}" if context else prompt
    words = full.split()[:30]
    return f"[Claude-Sonnet stub] {' '.join(words)}..."


# ---------------------------------------------------------------------------
# Domain constants — MLOps/LLMOps thresholds
# ---------------------------------------------------------------------------

# Population Stability Index thresholds (PRA SS1/23 §4 monitoring)
PSI_WARN_THRESHOLD: float = 0.10      # minor shift — enhanced monitoring
PSI_BREACH_THRESHOLD: float = 0.20    # significant shift — retraining trigger

# SHAP feature importance drift thresholds
SHAP_DRIFT_WARN_PCT: float = 0.15     # 15% mean absolute change in top-10 features
SHAP_DRIFT_BREACH_PCT: float = 0.25   # 25% — immediate model review

# PRA SS1/23 4-gate CI/CD validation
SS123_GATES_REQUIRED: int = 4         # all four gates must pass for deployment
SS123_MIN_AUC_ROC: float = 0.70       # Gate 1: discriminatory power
SS123_MAX_GINI_DELTA: float = 0.05    # Gate 2: max Gini drop vs. production
SS123_MAX_PSI: float = 0.20           # Gate 3: population stability
SS123_MIN_OUTPUT_FLOOR_2026: float = 0.55  # Gate 4: CRR3 Art.92a output floor 2026

# RAGAS LLM quality monitoring thresholds
RAGAS_SAMPLE_RATE: float = 0.05       # 5% production sampling
RAGAS_FAITHFULNESS_ALERT: float = 0.85
RAGAS_FAITHFULNESS_ROLLBACK: float = 0.80   # auto-rollback below this
RAGAS_ANSWER_REL_ALERT: float = 0.75
RAGAS_WINDOW_HOURS: int = 3           # rolling evaluation window
RAGAS_ROLLBACK_SLA_MINUTES: int = 15  # target rollback SLA

# Prompt versioning governance
PROMPT_VERSION_MAX_AGE_DAYS: int = 90     # stale prompt review trigger
AB_TEST_MIN_DAYS_MAJOR: int = 14          # MAJOR change A/B test minimum
AB_TEST_MIN_DAYS_MINOR: int = 7           # MINOR change A/B test minimum

# Token budget governance (BAP-2026-MLO-001)
TOKEN_BUDGET_PER_RUN    = 50_000   # Hard cap per single pipeline invocation
COST_BUDGET_GBP_PER_RUN = 2.50    # £2.50 max per run
TOKEN_BUDGET_DAILY_LIMIT: int = 2_000_000    # total across all AWB LLM services
TOKEN_BUDGET_WARN_PCT: float = 0.80          # 80% consumed — alert
TOKEN_BUDGET_RED_PCT: float = 0.95           # 95% consumed — circuit open

# Churn model production statistics (MR-2026-053)
CHURN_MODEL_DAILY_CALLS: int = 4_200         # inference calls/day
RAGAS_DAILY_SAMPLES: int = 210               # 4,200 × 5%
CHURN_AUC_ROC_PRODUCTION: float = 0.847
CHURN_GINI_PRODUCTION: float = 0.694
CHURN_PRECISION_AT_30: float = 0.34          # precision@30%

# Infrastructure model counts
AWB_DEPLOYED_MODELS: int = 12                # models in MLflow registry production
AWB_LLM_SERVICES: int = 9                    # LLM-backed services in AWB platform

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class RiskZone(str, Enum):
    GREEN = "GREEN"
    AMBER = "AMBER"
    RED = "RED"


class HITLDecision(str, Enum):
    APPROVE = "APPROVE"
    ESCALATE = "ESCALATE"
    OVERRIDE = "OVERRIDE"
    PENDING = "PENDING"


class DriftSeverity(str, Enum):
    STABLE = "STABLE"
    MINOR = "MINOR"
    SIGNIFICANT = "SIGNIFICANT"
    CRITICAL = "CRITICAL"


class DeploymentDecision(str, Enum):
    DEPLOY = "DEPLOY"
    HOLD = "HOLD"
    ROLLBACK = "ROLLBACK"
    RETRAIN = "RETRAIN"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ModelDriftResult:
    """PSI and SHAP drift assessment for a deployed model."""
    model_id: str
    model_name: str
    psi_score: float
    psi_severity: DriftSeverity
    shap_drift_pct: float
    shap_severity: DriftSeverity
    top_drifted_features: List[str]
    auc_roc_current: float
    auc_roc_delta: float          # vs. training baseline
    gini_current: float
    retraining_recommended: bool
    regulatory_flag: str          # PRA SS1/23 classification


@dataclass
class SS123GateResult:
    """Single PRA SS1/23 validation gate result."""
    gate_number: int
    gate_name: str
    passed: bool
    metric_value: float
    threshold: float
    detail: str
    regulatory_ref: str


@dataclass
class CICDValidationResult:
    """Full 4-gate SS1/23 CI/CD validation outcome."""
    model_id: str
    run_id: str
    gate_results: List[SS123GateResult]
    all_gates_passed: bool
    deployment_decision: DeploymentDecision
    mrc_review_required: bool
    ab_test_days_required: int


@dataclass
class RAGASWindowStats:
    """Rolling-window RAGAS statistics for one LLM service."""
    service_id: str
    window_hours: int
    sample_count: int
    mean_faithfulness: float
    mean_answer_relevancy: float
    mean_context_precision: float
    rollback_triggered: bool
    rollback_reason: str


@dataclass
class PromptVersionStatus:
    """Status of a prompt version in the AWB registry."""
    service_id: str
    current_version: str
    change_type: str             # MAJOR / MINOR / PATCH
    ab_test_days_elapsed: int
    ab_test_days_required: int
    stale: bool
    promotion_ready: bool


@dataclass
class TokenBudgetStatus:
    """Daily token budget consumption status."""
    service_id: str
    tokens_consumed_today: int
    daily_limit: int
    consumption_pct: float
    risk_zone: RiskZone
    projected_overage: bool


@dataclass
class AgentStep:
    """Immutable hop-chain record per BAP-2026-MLO-001 §6."""
    seq: int
    agent: str
    timestamp: str
    reason: str
    act: str
    outcome: str


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

class MLOpsLLMOpsState(dict):
    """
    Shared state for the AWB MLOps/LLMOps Assessment Pipeline.

    Inherits from dict for LangGraph compatibility.

    Keys
    ----
    run_id             : str  — unique pipeline execution ID
    run_date           : str  — ISO date of assessment
    trigger_event      : str  — what triggered this run
    model_inputs       : dict — model performance metrics from MLflow
    cicd_inputs        : dict — CI/CD run metadata and gate metrics
    ragas_inputs       : dict — RAGAS evaluation results per LLM service
    prompt_inputs      : dict — prompt registry state
    token_inputs       : dict — token consumption by service
    drift_results      : list — ModelDriftResult per deployed model
    validation_results : list — CICDValidationResult per pending deployment
    ragas_results      : list — RAGASWindowStats per LLM service
    prompt_statuses    : list — PromptVersionStatus per service
    token_statuses     : list — TokenBudgetStatus per service
    overall_risk_zone  : str  — monotonically escalated aggregate
    hitl_decision      : str  — HITLDecision value
    deployment_summary : dict — deployment/rollback decisions
    executive_summary  : str  — ArchitectureReportAgent narrative (300 words)
    risk_narrative     : str  — detailed risk findings with regulatory mapping
    action_items       : list — prioritised remediation steps
    hop_chain          : list — AgentStep records (immutable audit trail)
    errors             : list — non-fatal processing errors
    """


def _log_step(state: MLOpsLLMOpsState, seq: int, agent: str,
               reason: str, act: str, outcome: str) -> None:
    """Append an immutable AgentStep to the hop-chain."""
    step = AgentStep(
        seq=seq,
        agent=agent,
        timestamp=datetime.utcnow().isoformat() + "Z",
        reason=reason,
        act=act,
        outcome=outcome,
    )
    state.setdefault("hop_chain", []).append(vars(step))


def _escalate_zone(current: RiskZone, new: RiskZone) -> RiskZone:
    """Risk zone escalation is monotonically upward only."""
    zone_rank = {RiskZone.GREEN: 0, RiskZone.AMBER: 1, RiskZone.RED: 2}
    return max(current, new, key=lambda z: zone_rank.get(z, 0))


# ---------------------------------------------------------------------------
# Agent 1 — ModelDriftAgent (Gemini Flash)
# ---------------------------------------------------------------------------

def model_drift_agent(state: MLOpsLLMOpsState) -> MLOpsLLMOpsState:
    """
    Agent 1: ModelDriftAgent
    LLM: Gemini 3.5 Flash

    Evaluates PSI and SHAP feature importance drift for all deployed models
    in the AWB MLflow registry. Applies PRA SS1/23 §4 monitoring obligations.
    Flags models requiring retraining or immediate model review.
    """
    reason = (
        "ReAct reasoning: PSI > 0.10 indicates minor population shift; PSI > 0.20 "
        "indicates significant shift requiring mandatory retraining per PRA SS1/23 "
        "para 4.3. SHAP drift > 25% in top-10 features indicates model assumptions "
        "may no longer hold. Computed on monthly T24 transaction extract vs. "
        "training baseline of 180,000 AWB retail accounts."
    )

    model_inputs = state.get("model_inputs", {})
    models = model_inputs.get("models", [])

    drift_results: List[ModelDriftResult] = []
    overall_zone = RiskZone.GREEN

    llm_context = f"""
You are evaluating PSI and SHAP drift for {len(models)} deployed AWB models.
PSI thresholds: WARN={PSI_WARN_THRESHOLD}, BREACH={PSI_BREACH_THRESHOLD}.
SHAP drift thresholds: WARN={SHAP_DRIFT_WARN_PCT*100}%, BREACH={SHAP_DRIFT_BREACH_PCT*100}%.
Regulatory: PRA SS1/23 §4 monitoring, SS1/23 HIGH-risk models require monthly revalidation.
For each model provide: psi_severity, shap_severity, retraining_recommended, regulatory_flag.
Model data: {models}
"""
    llm_response = _call_gemini_flash(
        "Assess model drift and provide structured JSON risk classification for each model.",
        llm_context
    )
    log.debug("ModelDriftAgent LLM response: %s", llm_response[:200])

    for m in models:
        model_id = m.get("model_id", "UNKNOWN")
        psi = float(m.get("psi_score", 0.0))
        shap_drift = float(m.get("shap_drift_pct", 0.0))
        auc_current = float(m.get("auc_roc_current", 0.0))
        auc_baseline = float(m.get("auc_roc_baseline", auc_current))
        gini_current = float(m.get("gini_current", 0.0))

        # PSI severity classification
        if psi >= PSI_BREACH_THRESHOLD:
            psi_severity = DriftSeverity.SIGNIFICANT
            zone = RiskZone.RED
        elif psi >= PSI_WARN_THRESHOLD:
            psi_severity = DriftSeverity.MINOR
            zone = RiskZone.AMBER
        else:
            psi_severity = DriftSeverity.STABLE
            zone = RiskZone.GREEN

        # SHAP drift severity
        if shap_drift >= SHAP_DRIFT_BREACH_PCT:
            shap_severity = DriftSeverity.SIGNIFICANT
            zone = _escalate_zone(zone, RiskZone.RED)
        elif shap_drift >= SHAP_DRIFT_WARN_PCT:
            shap_severity = DriftSeverity.MINOR
            zone = _escalate_zone(zone, RiskZone.AMBER)
        else:
            shap_severity = DriftSeverity.STABLE

        auc_delta = auc_current - auc_baseline
        retraining = psi_severity == DriftSeverity.SIGNIFICANT or shap_severity == DriftSeverity.SIGNIFICANT

        # PRA SS1/23 regulatory flag
        ss1_23_risk = m.get("ss1_23_risk_rating", "UNKNOWN")
        if retraining and ss1_23_risk in ("HIGH", "CRITICAL"):
            regulatory_flag = "PRA SS1/23 §4.6 mandatory revalidation — MLRO and CRO notification required"
        elif psi_severity == DriftSeverity.MINOR:
            regulatory_flag = "PRA SS1/23 §4.3 enhanced monitoring — next validation cycle prioritised"
        else:
            regulatory_flag = "PRA SS1/23 §4.1 routine monitoring — within tolerance"

        result = ModelDriftResult(
            model_id=model_id,
            model_name=m.get("model_name", model_id),
            psi_score=psi,
            psi_severity=psi_severity,
            shap_drift_pct=shap_drift,
            shap_severity=shap_severity,
            top_drifted_features=m.get("top_drifted_features", []),
            auc_roc_current=auc_current,
            auc_roc_delta=auc_delta,
            gini_current=gini_current,
            retraining_recommended=retraining,
            regulatory_flag=regulatory_flag,
        )
        drift_results.append(result)
        overall_zone = _escalate_zone(overall_zone, zone)

    state["drift_results"] = [vars(r) for r in drift_results]
    state["overall_risk_zone"] = overall_zone.value

    retraining_count = sum(1 for r in drift_results if r.retraining_recommended)
    outcome = (
        f"Assessed {len(drift_results)} models. "
        f"{retraining_count} require retraining. Zone: {overall_zone.value}."
    )
    _log_step(state, 1, "ModelDriftAgent", reason,
              f"PSI+SHAP drift assessment for {len(drift_results)} models", outcome)
    return state


# ---------------------------------------------------------------------------
# Agent 2 — CICDValidationAgent (Gemini Flash)
# ---------------------------------------------------------------------------

def cicd_validation_agent(state: MLOpsLLMOpsState) -> MLOpsLLMOpsState:
    """
    Agent 2: CICDValidationAgent
    LLM: Gemini Flash

    Runs the PRA SS1/23 four-gate mandatory validation suite for each model
    pending deployment. Gates: (1) discriminatory power AUC-ROC, (2) Gini
    coefficient delta vs. production, (3) PSI population stability,
    (4) CRR3 Art.92a output floor compliance (2026 phase-in: 55%).
    """
    reason = (
        "ReAct reasoning: PRA SS1/23 para 4.1 requires documented validation gates "
        "before any model deployment. Gate 1 (AUC-ROC ≥ 0.70) ensures discriminatory "
        "power. Gate 2 (Gini delta ≤ 5%) prevents silent performance degradation. "
        "Gate 3 (PSI ≤ 0.20) ensures population stability. Gate 4 (output floor ≥ 55% "
        "for 2026) enforces CRR3 Art.92a capital floor phase-in. Failure at any gate "
        "halts deployment and requires MRC review per BAP-2026-MLO-001 §4."
    )

    cicd_inputs = state.get("cicd_inputs", {})
    pending_deployments = cicd_inputs.get("pending_deployments", [])
    current_zone = RiskZone(state.get("overall_risk_zone", RiskZone.GREEN.value))

    validation_results: List[CICDValidationResult] = []

    llm_context = f"""
You are validating {len(pending_deployments)} model deployments against PRA SS1/23 4-gate criteria.
Gate 1: AUC-ROC >= {SS123_MIN_AUC_ROC} (discriminatory power)
Gate 2: Gini delta <= {SS123_MAX_GINI_DELTA} vs production (performance stability)
Gate 3: PSI <= {SS123_MAX_PSI} (population stability, CRR3)
Gate 4: Output floor >= {SS123_MIN_OUTPUT_FLOOR_2026} (CRR3 Art.92a 2026 phase-in)
For each deployment determine: all_gates_passed, deployment_decision (DEPLOY/HOLD/ROLLBACK/RETRAIN), mrc_review_required.
Deployments: {pending_deployments}
"""
    llm_response = _call_gemini_flash(
        "Evaluate CI/CD validation gates and recommend deployment decisions.",
        llm_context
    )
    log.debug("CICDValidationAgent LLM response: %s", llm_response[:200])

    for dep in pending_deployments:
        model_id = dep.get("model_id", "UNKNOWN")
        run_id = dep.get("run_id", str(uuid.uuid4())[:8])

        gates: List[SS123GateResult] = []

        # Gate 1: AUC-ROC
        auc = float(dep.get("auc_roc", 0.0))
        g1 = SS123GateResult(
            gate_number=1,
            gate_name="Discriminatory Power — AUC-ROC",
            passed=auc >= SS123_MIN_AUC_ROC,
            metric_value=auc,
            threshold=SS123_MIN_AUC_ROC,
            detail=f"AUC-ROC={auc:.3f} vs minimum {SS123_MIN_AUC_ROC}",
            regulatory_ref="PRA SS1/23 §4.2 Gate 1",
        )
        gates.append(g1)

        # Gate 2: Gini delta
        gini_delta = abs(float(dep.get("gini_delta", 0.0)))
        g2 = SS123GateResult(
            gate_number=2,
            gate_name="Performance Stability — Gini Delta",
            passed=gini_delta <= SS123_MAX_GINI_DELTA,
            metric_value=gini_delta,
            threshold=SS123_MAX_GINI_DELTA,
            detail=f"Gini delta={gini_delta:.3f} vs maximum {SS123_MAX_GINI_DELTA}",
            regulatory_ref="PRA SS1/23 §4.2 Gate 2",
        )
        gates.append(g2)

        # Gate 3: PSI population stability
        psi = float(dep.get("psi_score", 0.0))
        g3 = SS123GateResult(
            gate_number=3,
            gate_name="Population Stability — PSI",
            passed=psi <= SS123_MAX_PSI,
            metric_value=psi,
            threshold=SS123_MAX_PSI,
            detail=f"PSI={psi:.3f} vs maximum {SS123_MAX_PSI}",
            regulatory_ref="PRA SS1/23 §4.2 Gate 3 / CRR3 population stability",
        )
        gates.append(g3)

        # Gate 4: CRR3 Art.92a output floor 2026
        output_floor = float(dep.get("output_floor_pct", 0.0))
        g4 = SS123GateResult(
            gate_number=4,
            gate_name="CRR3 Art.92a Output Floor — 2026 Phase-in",
            passed=output_floor >= SS123_MIN_OUTPUT_FLOOR_2026,
            metric_value=output_floor,
            threshold=SS123_MIN_OUTPUT_FLOOR_2026,
            detail=f"Output floor={output_floor:.2%} vs minimum {SS123_MIN_OUTPUT_FLOOR_2026:.0%} (2026 phase-in)",
            regulatory_ref="CRR3 Art.92a output floor 2026 phase-in schedule",
        )
        gates.append(g4)

        all_passed = all(g.passed for g in gates)
        failed_count = sum(1 for g in gates if not g.passed)

        # Deployment decision
        if all_passed:
            decision = DeploymentDecision.DEPLOY
            mrc_required = False
            ab_days = 0
        elif failed_count == 1 and gates[0].passed:  # Gate 1 passes, minor failures
            decision = DeploymentDecision.HOLD
            mrc_required = True
            ab_days = AB_TEST_MIN_DAYS_MINOR
        else:
            decision = DeploymentDecision.HOLD
            mrc_required = True
            ab_days = AB_TEST_MIN_DAYS_MAJOR

        val_result = CICDValidationResult(
            model_id=model_id,
            run_id=run_id,
            gate_results=gates,
            all_gates_passed=all_passed,
            deployment_decision=decision,
            mrc_review_required=mrc_required,
            ab_test_days_required=ab_days,
        )
        validation_results.append(val_result)

        if not all_passed:
            current_zone = _escalate_zone(current_zone, RiskZone.AMBER)
        if mrc_required and failed_count > 1:
            current_zone = _escalate_zone(current_zone, RiskZone.RED)

    state["validation_results"] = [
        {
            "model_id": r.model_id,
            "run_id": r.run_id,
            "all_gates_passed": r.all_gates_passed,
            "deployment_decision": r.deployment_decision.value,
            "mrc_review_required": r.mrc_review_required,
            "ab_test_days_required": r.ab_test_days_required,
            "gate_results": [vars(g) for g in r.gate_results],
        }
        for r in validation_results
    ]
    state["overall_risk_zone"] = current_zone.value

    hold_count = sum(1 for r in validation_results if r.deployment_decision != DeploymentDecision.DEPLOY)
    outcome = (
        f"Validated {len(validation_results)} deployments. "
        f"{hold_count} on hold. Zone: {current_zone.value}."
    )
    _log_step(state, 2, "CICDValidationAgent", reason,
              f"PRA SS1/23 4-gate validation for {len(validation_results)} deployments", outcome)
    return state


# ---------------------------------------------------------------------------
# Agent 3 — PromptGovernanceAgent (Gemini Flash)
# ---------------------------------------------------------------------------

def prompt_governance_agent(state: MLOpsLLMOpsState) -> MLOpsLLMOpsState:
    """
    Agent 3: PromptGovernanceAgent
    LLM: Gemini Flash

    Monitors RAGAS quality metrics across AWB LLM services, validates prompt
    version A/B test completion, checks token budget consumption, and triggers
    auto-rollback when faithfulness drops below 0.80 per RAGAS auto-rollback
    SLA of 15 minutes (BAP-2026-MLO-001 §5).
    """
    reason = (
        "ReAct reasoning: RAGAS faithfulness < 0.80 constitutes an LLM quality "
        "breach requiring auto-rollback within 15 minutes per BAP-2026-MLO-001 §5. "
        "Prompt version promotion requires completed A/B test (MAJOR: 14 days, "
        "MINOR: 7 days) and RAGAS faithfulness > 0.85. Token budget at 95% triggers "
        "circuit open to prevent cost overrun per EU AI Act Art.9 risk management "
        "system and FCA PRIN 11 operational resilience."
    )

    ragas_inputs = state.get("ragas_inputs", {})
    prompt_inputs = state.get("prompt_inputs", {})
    token_inputs = state.get("token_inputs", {})
    current_zone = RiskZone(state.get("overall_risk_zone", RiskZone.GREEN.value))

    llm_context = f"""
You are assessing LLMOps governance for {AWB_LLM_SERVICES} AWB LLM services.
RAGAS thresholds: faithfulness alert={RAGAS_FAITHFULNESS_ALERT}, rollback={RAGAS_FAITHFULNESS_ROLLBACK}
Token budget: daily limit={TOKEN_BUDGET_DAILY_LIMIT:,} tokens, warn={TOKEN_BUDGET_WARN_PCT:.0%}, red={TOKEN_BUDGET_RED_PCT:.0%}
A/B test requirements: MAJOR={AB_TEST_MIN_DAYS_MAJOR} days, MINOR={AB_TEST_MIN_DAYS_MINOR} days
For each service determine: rollback_triggered, promotion_ready, token risk_zone.
RAGAS data: {ragas_inputs}
Prompt registry: {prompt_inputs}
Token data: {token_inputs}
"""
    llm_response = _call_gemini_flash(
        "Assess RAGAS quality, prompt governance, and token budget status.",
        llm_context
    )
    log.debug("PromptGovernanceAgent LLM response: %s", llm_response[:200])

    # Process RAGAS window statistics
    ragas_results: List[RAGASWindowStats] = []
    for svc in ragas_inputs.get("services", []):
        service_id = svc.get("service_id", "UNKNOWN")
        sample_count = int(svc.get("sample_count", 0))
        mean_faith = float(svc.get("mean_faithfulness", 1.0))
        mean_ans_rel = float(svc.get("mean_answer_relevancy", 1.0))
        mean_ctx_prec = float(svc.get("mean_context_precision", 1.0))

        rollback = mean_faith < RAGAS_FAITHFULNESS_ROLLBACK
        rollback_reason = ""
        if rollback:
            rollback_reason = (
                f"Faithfulness={mean_faith:.3f} below rollback threshold "
                f"{RAGAS_FAITHFULNESS_ROLLBACK}. Auto-rollback SLA: "
                f"{RAGAS_ROLLBACK_SLA_MINUTES} minutes per BAP-2026-MLO-001 §5."
            )
            current_zone = _escalate_zone(current_zone, RiskZone.RED)
        elif mean_faith < RAGAS_FAITHFULNESS_ALERT:
            current_zone = _escalate_zone(current_zone, RiskZone.AMBER)

        ragas_results.append(RAGASWindowStats(
            service_id=service_id,
            window_hours=RAGAS_WINDOW_HOURS,
            sample_count=sample_count,
            mean_faithfulness=mean_faith,
            mean_answer_relevancy=mean_ans_rel,
            mean_context_precision=mean_ctx_prec,
            rollback_triggered=rollback,
            rollback_reason=rollback_reason,
        ))

    # Process prompt version status
    prompt_statuses: List[PromptVersionStatus] = []
    for pv in prompt_inputs.get("versions", []):
        service_id = pv.get("service_id", "UNKNOWN")
        change_type = pv.get("change_type", "PATCH")
        ab_elapsed = int(pv.get("ab_test_days_elapsed", 0))
        version_age_days = int(pv.get("version_age_days", 0))

        required_days = (
            AB_TEST_MIN_DAYS_MAJOR if change_type == "MAJOR"
            else AB_TEST_MIN_DAYS_MINOR if change_type == "MINOR"
            else 0
        )
        promotion_ready = ab_elapsed >= required_days
        stale = version_age_days > PROMPT_VERSION_MAX_AGE_DAYS

        if stale:
            current_zone = _escalate_zone(current_zone, RiskZone.AMBER)

        prompt_statuses.append(PromptVersionStatus(
            service_id=service_id,
            current_version=pv.get("current_version", "unknown"),
            change_type=change_type,
            ab_test_days_elapsed=ab_elapsed,
            ab_test_days_required=required_days,
            stale=stale,
            promotion_ready=promotion_ready,
        ))

    # Process token budget status
    token_statuses: List[TokenBudgetStatus] = []
    for tsvc in token_inputs.get("services", []):
        service_id = tsvc.get("service_id", "UNKNOWN")
        consumed = int(tsvc.get("tokens_consumed_today", 0))
        pct = consumed / TOKEN_BUDGET_DAILY_LIMIT if TOKEN_BUDGET_DAILY_LIMIT > 0 else 0.0

        if pct >= TOKEN_BUDGET_RED_PCT:
            t_zone = RiskZone.RED
            current_zone = _escalate_zone(current_zone, RiskZone.RED)
        elif pct >= TOKEN_BUDGET_WARN_PCT:
            t_zone = RiskZone.AMBER
            current_zone = _escalate_zone(current_zone, RiskZone.AMBER)
        else:
            t_zone = RiskZone.GREEN

        token_statuses.append(TokenBudgetStatus(
            service_id=service_id,
            tokens_consumed_today=consumed,
            daily_limit=TOKEN_BUDGET_DAILY_LIMIT,
            consumption_pct=pct,
            risk_zone=t_zone,
            projected_overage=(pct > 1.0),
        ))

    state["ragas_results"] = [vars(r) for r in ragas_results]
    state["prompt_statuses"] = [vars(p) for p in prompt_statuses]
    state["token_statuses"] = [vars(t) for t in token_statuses]
    state["overall_risk_zone"] = current_zone.value

    rollback_count = sum(1 for r in ragas_results if r.rollback_triggered)
    stale_count = sum(1 for p in prompt_statuses if p.stale)
    outcome = (
        f"Assessed {len(ragas_results)} RAGAS windows, {len(prompt_statuses)} prompt versions, "
        f"{len(token_statuses)} token budgets. Rollbacks triggered: {rollback_count}. "
        f"Stale prompts: {stale_count}. Zone: {current_zone.value}."
    )
    _log_step(state, 3, "PromptGovernanceAgent", reason,
              "RAGAS quality + prompt version + token budget assessment", outcome)
    return state


# ---------------------------------------------------------------------------
# Agent 4 — LLMOpsOrchestrationAgent (Gemini 3.1 Pro)
# ---------------------------------------------------------------------------

def llmops_orchestration_agent(state: MLOpsLLMOpsState) -> MLOpsLLMOpsState:
    """
    Agent 4: LLMOpsOrchestrationAgent
    LLM: Gemini 3.1 Pro

    Synthesises drift findings, CI/CD gate results, and LLMOps quality metrics
    into a consolidated deployment summary and rollback decision matrix.
    Determines which models to deploy, hold, or retrain; which LLM services
    require rollback; and whether the AWB MLOps platform is in a healthy state
    for PRA SS1/23 para 7 sign-off.
    """
    reason = (
        "ReAct reasoning: Gemini 3.1 Pro synthesises all three fast-agent findings "
        "into a deployment decision matrix. Per PRA SS1/23 para 7, the MLRO and CRO "
        "must approve any HIGH-risk model deployment or rollback. EU AI Act Art.14 "
        "requires human oversight for HIGH-RISK AI system changes. BAP-2026-MLO-001 "
        "§6 requires the hop-chain to record the full decision rationale."
    )

    current_zone = RiskZone(state.get("overall_risk_zone", RiskZone.GREEN.value))
    drift_results = state.get("drift_results", [])
    validation_results = state.get("validation_results", [])
    ragas_results = state.get("ragas_results", [])
    prompt_statuses = state.get("prompt_statuses", [])
    token_statuses = state.get("token_statuses", [])

    llm_context = f"""
Synthesise MLOps/LLMOps health across:
- {len(drift_results)} model drift assessments (retraining flags, PSI/SHAP severity)
- {len(validation_results)} CI/CD validation runs (4-gate SS1/23 results)
- {len(ragas_results)} RAGAS quality windows (faithfulness, answer relevancy)
- {len(prompt_statuses)} prompt version governance statuses
- {len(token_statuses)} token budget statuses
Overall risk zone: {current_zone.value}

Drift: {drift_results}
Validation: {validation_results}
RAGAS: {ragas_results}
Prompts: {prompt_statuses}
Tokens: {token_statuses}

Produce a deployment_summary with:
1. models_to_deploy: list of model_ids cleared for production
2. models_on_hold: list with gate failure reasons
3. models_to_retrain: list requiring retraining
4. llm_rollbacks: list of service_ids requiring rollback
5. platform_health_verdict: HEALTHY/DEGRADED/CRITICAL
6. regulatory_sign_off_required: list of regulatory actions needed
"""
    llm_response = _call_gemini_pro(
        "Produce a consolidated deployment decision matrix for AWB MLOps platform.",
        llm_context
    )

    # Build deployment summary from validated findings
    models_to_deploy = [
        r["model_id"] for r in validation_results
        if r.get("all_gates_passed") and r.get("deployment_decision") == DeploymentDecision.DEPLOY.value
    ]
    models_on_hold = [
        r["model_id"] for r in validation_results
        if r.get("deployment_decision") in (DeploymentDecision.HOLD.value, "HOLD")
    ]
    models_to_retrain = [
        r["model_id"] for r in drift_results
        if r.get("retraining_recommended")
    ]
    llm_rollbacks = [
        r["service_id"] for r in ragas_results
        if r.get("rollback_triggered")
    ]
    mrc_required = [
        r["model_id"] for r in validation_results
        if r.get("mrc_review_required")
    ]

    # Platform health verdict
    if current_zone == RiskZone.RED:
        health_verdict = "CRITICAL"
    elif current_zone == RiskZone.AMBER:
        health_verdict = "DEGRADED"
    else:
        health_verdict = "HEALTHY"

    deployment_summary = {
        "models_to_deploy": models_to_deploy,
        "models_on_hold": models_on_hold,
        "models_to_retrain": models_to_retrain,
        "llm_rollbacks": llm_rollbacks,
        "mrc_review_required": mrc_required,
        "platform_health_verdict": health_verdict,
        "overall_zone": current_zone.value,
        "llm_synthesis": llm_response[:500],
    }
    state["deployment_summary"] = deployment_summary
    state["overall_risk_zone"] = current_zone.value

    outcome = (
        f"Deploy: {len(models_to_deploy)}, Hold: {len(models_on_hold)}, "
        f"Retrain: {len(models_to_retrain)}, LLM rollbacks: {len(llm_rollbacks)}. "
        f"Platform: {health_verdict}."
    )
    _log_step(state, 4, "LLMOpsOrchestrationAgent", reason,
              "Synthesise deployment decision matrix via Gemini 3.1 Pro", outcome)
    return state


# ---------------------------------------------------------------------------
# Agent 5 — MLOpsReportAgent (Claude Sonnet 4.6)
# ---------------------------------------------------------------------------

def mlops_report_agent(state: MLOpsLLMOpsState) -> MLOpsLLMOpsState:
    """
    Agent 5: MLOpsReportAgent
    LLM: Claude Sonnet 4.6

    Generates the formal MLOps/LLMOps health report for the AWB Model Risk
    Committee and CRO. Produces three outputs: executive_summary (300 words),
    risk_narrative (full regulatory mapping), and action_items (prioritised
    remediation steps). Satisfies PRA SS1/23 para 7 audit trail, EU AI Act
    Art.13 transparency, and BAP-2026-MLO-001 §7 reporting obligations.
    """
    reason = (
        "ReAct reasoning: Claude Sonnet 4.6 generates the formal regulatory report. "
        "PRA SS1/23 para 7 requires human-readable audit documentation for all HIGH-risk "
        "model deployment decisions. EU AI Act Art.13 requires transparency logs for "
        "HIGH-RISK AI system changes. BAP-2026-MLO-001 §7 mandates MRC-reportable "
        "summary within 2 business days of any RED zone finding."
    )

    current_zone = RiskZone(state.get("overall_risk_zone", RiskZone.GREEN.value))
    deployment_summary = state.get("deployment_summary", {})
    drift_results = state.get("drift_results", [])
    validation_results = state.get("validation_results", [])
    ragas_results = state.get("ragas_results", [])

    report_prompt = f"""
Generate an AWB MLOps/LLMOps Health Report for the Model Risk Committee.

Platform status: {deployment_summary.get('platform_health_verdict', 'UNKNOWN')}
Overall risk zone: {current_zone.value}
Run date: {state.get('run_date', date.today().isoformat())}
Trigger: {state.get('trigger_event', 'scheduled assessment')}

Findings summary:
- Models assessed for drift: {len(drift_results)}
- Models to retrain: {len(deployment_summary.get('models_to_retrain', []))}
- CI/CD validations run: {len(validation_results)}
- Models cleared for deployment: {len(deployment_summary.get('models_to_deploy', []))}
- Models on hold: {len(deployment_summary.get('models_on_hold', []))}
- LLM services requiring rollback: {len(deployment_summary.get('llm_rollbacks', []))}

Regulatory context: PRA SS1/23 §4-7, EU AI Act Arts 9/13/14,
BAP-2026-MLO-001, CRR3 Art.92a output floor 2026 phase-in (55%).

Produce:
1. executive_summary: 300-word MRC-ready executive summary
2. risk_narrative: Detailed findings with regulatory paragraph references
3. action_items: Numbered list of prioritised remediation steps with owners and deadlines
"""
    llm_response = _call_claude_sonnet(report_prompt)

    # Parse structured output from Claude (in production use structured output)
    sections = llm_response.split("\n\n") if llm_response else []
    exec_summary = sections[0] if sections else f"Platform health: {deployment_summary.get('platform_health_verdict', 'UNKNOWN')}. Zone: {current_zone.value}."
    risk_narrative = sections[1] if len(sections) > 1 else f"Risk zone {current_zone.value}. {len(drift_results)} models assessed."
    action_items_raw = sections[2] if len(sections) > 2 else "1. Review all RED zone findings with CRO."

    # Parse action items into list
    action_items = [
        line.strip() for line in action_items_raw.split("\n")
        if line.strip() and (line.strip()[0].isdigit() or line.strip().startswith("-"))
    ]
    if not action_items:
        action_items = [action_items_raw.strip()]

    state["executive_summary"] = exec_summary
    state["risk_narrative"] = risk_narrative
    state["action_items"] = action_items

    outcome = (
        f"MLOps report generated. Executive summary: {len(exec_summary)} chars. "
        f"Action items: {len(action_items)}. Platform: "
        f"{deployment_summary.get('platform_health_verdict', 'UNKNOWN')}."
    )
    _log_step(state, 5, "MLOpsReportAgent", reason,
              "Generate MRC MLOps/LLMOps report via Claude Sonnet 4.6", outcome)
    return state


# ---------------------------------------------------------------------------
# HITL gate — Human-in-the-Loop decision
# ---------------------------------------------------------------------------

def hitl_gate(state: MLOpsLLMOpsState) -> MLOpsLLMOpsState:
    """
    HITL Gate: Human-in-the-Loop deployment approval.

    PRA SS1/23 para 7 and EU AI Act Art.14 require human sign-off before
    deploying or rolling back HIGH-risk AI systems.

    APPROVE only when:
    - All CI/CD validation gates passed for all pending deployments
    - No RAGAS rollbacks triggered
    - No models require immediate retraining (PSI ≥ 0.20)
    - Overall zone is GREEN

    Any breach → ESCALATE to MRC, MLRO, and CRO.
    """
    current_zone = RiskZone(state.get("overall_risk_zone", RiskZone.GREEN.value))
    deployment_summary = state.get("deployment_summary", {})

    has_rollbacks = len(deployment_summary.get("llm_rollbacks", [])) > 0
    has_holds = len(deployment_summary.get("models_on_hold", [])) > 0
    has_retrains = len(deployment_summary.get("models_to_retrain", [])) > 0
    platform_critical = deployment_summary.get("platform_health_verdict") == "CRITICAL"

    if (current_zone == RiskZone.GREEN
            and not has_rollbacks
            and not has_holds
            and not has_retrains):
        decision = HITLDecision.APPROVE
    else:
        decision = HITLDecision.ESCALATE

    state["hitl_decision"] = decision.value

    breaches = []
    if has_rollbacks:
        breaches.append(f"LLM rollbacks: {deployment_summary.get('llm_rollbacks')}")
    if has_holds:
        breaches.append(f"Models on hold: {deployment_summary.get('models_on_hold')}")
    if has_retrains:
        breaches.append(f"Retraining required: {deployment_summary.get('models_to_retrain')}")
    if platform_critical:
        breaches.append("Platform health: CRITICAL")

    _log_step(
        state, 6, "HITLGate",
        "PRA SS1/23 §7 + EU AI Act Art.14 human oversight requirement",
        "Evaluate HITL deployment approval",
        f"Decision: {decision.value}. Zone: {current_zone.value}. "
        + (f"Breaches: {breaches}" if breaches else "All clear."),
    )
    return state


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _build_graph() -> Any:
    """Build and compile the AWB MLOps/LLMOps LangGraph StateGraph."""
    graph = StateGraph(MLOpsLLMOpsState)

    graph.add_node("model_drift", model_drift_agent)
    graph.add_node("cicd_validation", cicd_validation_agent)
    graph.add_node("prompt_governance", prompt_governance_agent)
    graph.add_node("llmops_orchestration", llmops_orchestration_agent)
    graph.add_node("mlops_report", mlops_report_agent)
    graph.add_node("hitl", hitl_gate)

    graph.add_edge(START, "model_drift")
    graph.add_edge("model_drift", "cicd_validation")
    graph.add_edge("cicd_validation", "prompt_governance")
    graph.add_edge("prompt_governance", "llmops_orchestration")
    graph.add_edge("llmops_orchestration", "mlops_report")
    graph.add_edge("mlops_report", "hitl")
    graph.add_edge("hitl", END)

    graph.set_entry_point("model_drift")
    return graph.compile()


_AWB_MLOPS_GRAPH = None


def _get_graph():
    global _AWB_MLOPS_GRAPH
    if _AWB_MLOPS_GRAPH is None:
        _AWB_MLOPS_GRAPH = _build_graph()
    return _AWB_MLOPS_GRAPH


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_agentic_mlops_llmops(
    run_date: str,
    trigger_event: str,
    model_inputs: Dict[str, Any],
    cicd_inputs: Dict[str, Any],
    ragas_inputs: Dict[str, Any],
    prompt_inputs: Dict[str, Any],
    token_inputs: Optional[Dict[str, Any]] = None,
) -> MLOpsLLMOpsState:
    """
    Run the AWB Agentic MLOps/LLMOps Pipeline (MR-2026-062-MLO).

    Args:
        run_date:       ISO date string (e.g. "2026-05-24")
        trigger_event:  What triggered this run (e.g. "scheduled_daily",
                        "model_deployment_request", "ragas_alert")
        model_inputs:   Dict with key "models" — list of ModelDriftInput dicts
                        each containing: model_id, model_name, psi_score,
                        shap_drift_pct, auc_roc_current, auc_roc_baseline,
                        gini_current, top_drifted_features, ss1_23_risk_rating
        cicd_inputs:    Dict with key "pending_deployments" — list of dicts
                        each containing: model_id, run_id, auc_roc, gini_delta,
                        psi_score, output_floor_pct
        ragas_inputs:   Dict with key "services" — list of RAGAS window dicts
                        each containing: service_id, sample_count,
                        mean_faithfulness, mean_answer_relevancy,
                        mean_context_precision
        prompt_inputs:  Dict with key "versions" — list of prompt version dicts
                        each containing: service_id, current_version,
                        change_type, ab_test_days_elapsed, version_age_days
        token_inputs:   Optional dict with key "services" — list of token dicts
                        each containing: service_id, tokens_consumed_today

    Returns:
        MLOpsLLMOpsState with all fields populated including:
        - drift_results, validation_results, ragas_results
        - deployment_summary, hitl_decision
        - executive_summary, risk_narrative, action_items
        - hop_chain (6 steps: 5 agents + HITL)
    """
    run_id = str(uuid.uuid4())
    log.info("Starting MR-2026-062-MLO run_id=%s date=%s trigger=%s",
             run_id, run_date, trigger_event)

    initial_state = MLOpsLLMOpsState(
        run_id=run_id,
        run_date=run_date,
        trigger_event=trigger_event,
        model_inputs=model_inputs,
        cicd_inputs=cicd_inputs,
        ragas_inputs=ragas_inputs,
        prompt_inputs=prompt_inputs,
        token_inputs=token_inputs or {"services": []},
        drift_results=[],
        validation_results=[],
        ragas_results=[],
        prompt_statuses=[],
        token_statuses=[],
        overall_risk_zone=RiskZone.GREEN.value,
        hitl_decision=HITLDecision.PENDING.value,
        deployment_summary={},
        executive_summary="",
        risk_narrative="",
        action_items=[],
        hop_chain=[],
        errors=[],
    )

    graph = _get_graph()

    try:
        if hasattr(graph, "ainvoke"):
            final_state = await graph.ainvoke(initial_state)
        else:
            final_state = graph.ainvoke(initial_state)
            if asyncio.iscoroutine(final_state):
                final_state = await final_state
    except Exception as exc:  # pylint: disable=broad-except
        log.exception("MLOps pipeline error: %s", exc)
        initial_state.setdefault("errors", []).append(str(exc))
        initial_state["hitl_decision"] = HITLDecision.ESCALATE.value
        return initial_state

    log.info(
        "MR-2026-062-MLO complete: zone=%s hitl=%s hop_chain_steps=%d",
        final_state.get("overall_risk_zone"),
        final_state.get("hitl_decision"),
        len(final_state.get("hop_chain", [])),
    )
    return MLOpsLLMOpsState(final_state)


# ---------------------------------------------------------------------------
# AWB Q1 2026 demonstration run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    # AWB production model portfolio — Q1 2026 snapshot
    demo_model_inputs = {
        "models": [
            {
                "model_id": "MR-2026-053",
                "model_name": "AWB Customer Churn XGBoost v2.1",
                "psi_score": 0.07,               # stable — below warn threshold
                "shap_drift_pct": 0.11,           # minor drift
                "auc_roc_current": CHURN_AUC_ROC_PRODUCTION,
                "auc_roc_baseline": 0.851,
                "gini_current": CHURN_GINI_PRODUCTION,
                "top_drifted_features": ["balance_slope_90d", "app_login_freq_90d"],
                "ss1_23_risk_rating": "LOW",
            },
            {
                "model_id": "MR-2026-037",
                "model_name": "AWB Credit Decision LightGBM v3.4",
                "psi_score": 0.13,               # minor shift — AMBER
                "shap_drift_pct": 0.08,
                "auc_roc_current": 0.881,
                "auc_roc_baseline": 0.889,
                "gini_current": 0.762,
                "top_drifted_features": ["debt_to_income", "employment_stability"],
                "ss1_23_risk_rating": "HIGH",
            },
        ]
    }

    # Pending CI/CD deployments
    demo_cicd_inputs = {
        "pending_deployments": [
            {
                "model_id": "MR-2026-053-v2.2",
                "run_id": "mlflow-run-8a3f9c",
                "auc_roc": 0.852,
                "gini_delta": 0.003,             # within 5% tolerance
                "psi_score": 0.08,               # stable
                "output_floor_pct": 0.57,        # above 55% 2026 floor
            }
        ]
    }

    # RAGAS monitoring — 3-hour rolling window
    demo_ragas_inputs = {
        "services": [
            {
                "service_id": "MR-2026-035",   # Credit Risk RAG
                "sample_count": 52,             # ~210/day × 3hr/24hr
                "mean_faithfulness": 0.91,      # healthy — above 0.85 alert
                "mean_answer_relevancy": 0.87,
                "mean_context_precision": 0.83,
            },
            {
                "service_id": "MR-2026-064",   # AML Typologies RAG
                "sample_count": 18,
                "mean_faithfulness": 0.88,
                "mean_answer_relevancy": 0.84,
                "mean_context_precision": 0.79,
            },
        ]
    }

    # Prompt registry status
    demo_prompt_inputs = {
        "versions": [
            {
                "service_id": "MR-2026-035",
                "current_version": "2.1.0",
                "change_type": "MINOR",
                "ab_test_days_elapsed": 9,       # > 7 days required
                "version_age_days": 45,
            },
            {
                "service_id": "MR-2026-037",
                "current_version": "1.3.2",
                "change_type": "PATCH",
                "ab_test_days_elapsed": 0,
                "version_age_days": 23,
            },
        ]
    }

    # Token budget consumption
    demo_token_inputs = {
        "services": [
            {
                "service_id": "MR-2026-035",
                "tokens_consumed_today": 1_240_000,  # 62% of 2M daily limit
            },
            {
                "service_id": "MR-2026-037",
                "tokens_consumed_today": 380_000,    # 19% — well within budget
            },
        ]
    }

    result = asyncio.run(run_agentic_mlops_llmops(
        run_date="2026-05-24",
        trigger_event="scheduled_daily_q1_2026_review",
        model_inputs=demo_model_inputs,
        cicd_inputs=demo_cicd_inputs,
        ragas_inputs=demo_ragas_inputs,
        prompt_inputs=demo_prompt_inputs,
        token_inputs=demo_token_inputs,
    ))

    print("\n" + "=" * 70)
    print("AWB MLOps/LLMOps Pipeline (MR-2026-062-MLO) — Q1 2026 Review")
    print("=" * 70)
    print(f"Run ID:          {result.get('run_id')}")
    print(f"Overall Zone:    {result.get('overall_risk_zone')}")
    print(f"HITL Decision:   {result.get('hitl_decision')}")
    print(f"Platform Health: {result.get('deployment_summary', {}).get('platform_health_verdict')}")
    print(f"Hop-chain steps: {len(result.get('hop_chain', []))}")
    print("\nDeployment Summary:")
    ds = result.get("deployment_summary", {})
    print(f"  Deploy:   {ds.get('models_to_deploy', [])}")
    print(f"  On Hold:  {ds.get('models_on_hold', [])}")
    print(f"  Retrain:  {ds.get('models_to_retrain', [])}")
    print(f"  Rollback: {ds.get('llm_rollbacks', [])}")
    print("\nHop-chain:")
    for step in result.get("hop_chain", []):
        print(f"  Step {step['seq']}: [{step['agent']}] {step['outcome']}")
    print("\nExecutive Summary (first 300 chars):")
    print(result.get("executive_summary", "")[:300])
    print("=" * 70)
