# agentic_data_infrastructure.py | AWB Agentic Data Infrastructure Pipeline
# Chapter 15 | MR-2026-063-DI | BCBS 239 / UK GDPR / PRA SS1/23
# Five-agent LangGraph StateGraph: data quality → BCBS 239 compliance →
# feature store health → data governance → data infra report
# BAP-2026-DI-001 | EU AI Act Arts 9/10/13 | FCA COBS 9
"""
AWB Agentic Data Infrastructure Assessment Pipeline (MR-2026-063-DI)

Addresses the governance question: how does the platform continuously
assess its own data estate health across five dimensions — Great
Expectations data quality, BCBS 239 eleven-principle compliance, feature
store training-serving skew, UK GDPR retention policy adherence, and
Customer 360 identity resolution freshness — with every finding
documented in an immutable hop-chain audit trail for PRA SS1/23 para 7
and BCBS 239 Principle 1 (governance) audit review?

Agent topology (LangGraph StateGraph):
  START → data_quality → bcbs239_compliance → feature_store_health
        → data_governance → data_infra_report → hitl → END

LLM allocation:
  Agents 1-3: Gemini 3.5 Flash (fast structured data health tasks)
  Agent 4:    Gemini 3.1 Pro  (governance synthesis + GDPR decisions)
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
from datetime import datetime, date, timedelta
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
            while current and current not in ("__end__", "END"):
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
# LLM provider stubs
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
    if _GEMINI_AVAILABLE:
        model = genai.GenerativeModel("gemini-3.5-flash")
        response = model.generate_content(f"{prompt}\n\nContext:\n{context}")
        return response.text
    full = f"{prompt}\n\nContext:\n{context}" if context else prompt
    return f"[Gemini-Flash stub] {' '.join(full.split()[:30])}..."


def _call_gemini_pro(prompt: str, context: str = "") -> str:
    if _GEMINI_AVAILABLE:
        model = genai.GenerativeModel("gemini-3.1-pro")
        response = model.generate_content(f"{prompt}\n\nContext:\n{context}")
        return response.text
    full = f"{prompt}\n\nContext:\n{context}" if context else prompt
    return f"[Gemini-Pro stub] {' '.join(full.split()[:30])}..."


def _call_claude_sonnet(prompt: str, context: str = "") -> str:
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
    return f"[Claude-Sonnet stub] {' '.join(full.split()[:30])}..."


# ---------------------------------------------------------------------------
# Domain constants — Data Infrastructure thresholds
# ---------------------------------------------------------------------------

# BCBS 239 compliance thresholds
BCBS239_MIN_SCORE_PCT: float = 90.0      # PRA target per principle
BCBS239_ESCALATION_PCT: float = 80.0    # below this → CRO + Board Risk Committee
BCBS239_REMEDIATION_DAYS: int = 90       # PRA: 90-day remediation plan required
BCBS239_PRINCIPLES_COUNT: int = 11       # Basel 2013/2022 — 11 principles

# AWB Q1 2026 BCBS 239 scores (from bcbs239_monitor.py)
AWB_BCBS239_SCORES: Dict[str, float] = {
    "P1-Governance": 92.0,
    "P2-DataArchitecture": 88.0,
    "P3-Accuracy": 94.0,
    "P4-Completeness": 91.0,
    "P5-Timeliness": 96.0,
    "P6-Adaptability": 85.0,
    "P7-AccuracyReporting": 92.0,
    "P8-Comprehensiveness": 89.0,
    "P9-Clarity": 86.0,
    "P10-Frequency": 93.0,
    "P11-Distribution": 74.0,   # Gap: Ch16 target 90%
}
AWB_BCBS239_OVERALL_Q1_2026: float = 92.0  # weighted average (pre-ERDW: 52%)

# Great Expectations data quality thresholds
DQ_COMPLETENESS_MIN_PCT: float = 99.5    # credit decision records
DQ_ACCURACY_MIN_PCT: float = 99.9        # BCBS 239 P3 Accuracy
DQ_TIMELINESS_MAX_HOURS: float = 4.0     # intraday risk positions
DQ_UNIQUENESS_MIN_PCT: float = 99.99     # AWB_CUSTOMER_ID deduplication

# Feature store health thresholds
FEATURE_SKEW_WARN_PCT: float = 0.02      # 2% training-serving divergence → alert
FEATURE_SKEW_RED_PCT: float = 0.05       # 5% → model retrain trigger
REDIS_CACHE_TTL_SECONDS: int = 30        # Customer 360 freshness SLA
REDIS_HIT_RATE_MIN_PCT: float = 95.0     # sub-5ms serving target
FEATURE_VERSION_DRIFT_DAYS: int = 7      # version mismatch alert threshold

# UK GDPR / DPA 2018 retention policy (from retention_policy.py)
RETENTION_CREDIT_DECISIONS_YEARS: int = 7    # FCA COBS 9.1.3R
RETENTION_SAR_RECORDS_YEARS: int = 5         # MLR 2017 Regulation 40
RETENTION_MODEL_OUTPUTS_YEARS: int = 7       # PRA SS1/23 Section 4
RETENTION_AUDIT_LOGS_YEARS: int = 7          # FCA COBS 9 / DORA Art.17
RETENTION_KYC_RECORDS_YEARS: int = 7         # MLR 2017 Reg 40 / POCA 2002
RETENTION_TRAINING_DATASETS_YEARS: int = 7   # PRA SS1/23
SAR_SLA_HOURS: int = 720                     # 30 days per UK GDPR Art.15

# Customer 360 platform thresholds
C360_FRESHNESS_SLA_MINUTES: int = 30     # profile update lag max
C360_IDENTITY_MATCH_MIN_PCT: float = 98.0  # AWB_CUSTOMER_ID resolution rate
C360_DOMAINS_COUNT: int = 8              # unified domain count

# AWB data estate scale (Q1 2026)
AWB_TOTAL_RECORDS_BILLIONS: float = 2.3      # total records under management
AWB_ERDW_TABLES: int = 847                   # ERDW table count post-remediation
AWB_SECTION166_COST_GBP: int = 680_000       # pre-ERDW Section 166 cost
AWB_ERDW_REMEDIATION_COST_GBP: int = 680_000 # one-off ERDW build cost

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class RiskZone(str, Enum):
    GREEN = "GREEN"
    AMBER = "AMBER"
    RED = "RED"



# ── AWB Per-Run Token and Cost Budget ────────────────────────────────────────
TOKEN_BUDGET_PER_RUN    = 50_000   # Hard cap per pipeline invocation (all chapters)
COST_BUDGET_GBP_PER_RUN = 2.50    # £2.50 max per run (AWS Cost Explorer SLO)

class HITLDecision(str, Enum):
    APPROVE = "APPROVE"
    ESCALATE = "ESCALATE"
    OVERRIDE = "OVERRIDE"
    PENDING = "PENDING"


class DataQualitySeverity(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


class BCBSSeverity(str, Enum):
    COMPLIANT = "COMPLIANT"       # >= 90%
    AMBER = "AMBER"               # 80-89%
    ESCALATE = "ESCALATE"         # < 80% → CRO + Board Risk Committee


class RetentionStatus(str, Enum):
    COMPLIANT = "COMPLIANT"
    OVERDUE = "OVERDUE"           # data retained beyond policy
    UNDER_RETAINED = "UNDER_RETAINED"  # data deleted before retention period


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class DataQualityCheckResult:
    """Great Expectations check result for one dataset."""
    dataset_id: str
    dataset_name: str
    completeness_pct: float
    accuracy_pct: float
    timeliness_hours: float
    uniqueness_pct: float
    overall_severity: DataQualitySeverity
    failed_expectations: List[str]
    regulatory_ref: str


@dataclass
class BCBS239PrincipleScore:
    """Assessment of one BCBS 239 principle."""
    principle: str
    score_pct: float
    severity: BCBSSeverity
    gap_pct: float
    gap_actions: List[str]
    cro_escalation_required: bool
    remediation_deadline: str


@dataclass
class FeatureStoreHealthResult:
    """Training-serving skew and cache health for one feature set."""
    feature_set_id: str
    feature_version: str
    skew_pct: float
    risk_zone: RiskZone
    redis_hit_rate_pct: float
    redis_serving_p50_ms: float
    version_drift_days: int
    retrain_recommended: bool


@dataclass
class RetentionComplianceResult:
    """UK GDPR / DPA 2018 retention policy compliance for one data category."""
    data_category: str
    retention_years_policy: int
    oldest_record_years: float
    status: RetentionStatus
    s3_object_lock_active: bool
    next_deletion_run: str
    regulatory_basis: str


@dataclass
class Customer360HealthResult:
    """Customer 360 platform freshness and identity resolution health."""
    domain: str
    freshness_minutes: float
    freshness_ok: bool
    records_count: int
    identity_match_rate_pct: float
    identity_ok: bool


@dataclass
class AgentStep:
    """Immutable hop-chain record per BAP-2026-DI-001 §6."""
    seq: int
    agent: str
    timestamp: str
    reason: str
    act: str
    outcome: str


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

class DataInfraState(dict):
    """
    Shared state for the AWB Data Infrastructure Assessment Pipeline.

    Keys
    ----
    run_id                  : str
    run_date                : str
    trigger_event           : str
    dq_inputs               : dict  — Great Expectations check results per dataset
    bcbs239_inputs          : dict  — BCBS 239 principle scores
    feature_store_inputs    : dict  — feature store health metrics
    governance_inputs       : dict  — retention policy + GDPR + SAR status
    c360_inputs             : dict  — Customer 360 domain freshness
    dq_results              : list  — DataQualityCheckResult per dataset
    bcbs239_results         : list  — BCBS239PrincipleScore per principle
    feature_store_results   : list  — FeatureStoreHealthResult per feature set
    retention_results       : list  — RetentionComplianceResult per category
    c360_results            : list  — Customer360HealthResult per domain
    overall_risk_zone       : str   — monotonically escalated aggregate
    data_governance_summary : dict  — synthesised governance findings
    hitl_decision           : str   — HITLDecision value
    executive_summary       : str   — 300-word CRO-ready narrative
    risk_narrative          : str   — full regulatory mapping
    action_items            : list  — prioritised remediation steps
    hop_chain               : list  — AgentStep records
    errors                  : list  — non-fatal processing errors
    """


def _log_step(state: DataInfraState, seq: int, agent: str,
               reason: str, act: str, outcome: str) -> None:
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
    zone_rank = {RiskZone.GREEN: 0, RiskZone.AMBER: 1, RiskZone.RED: 2}
    return max(current, new, key=lambda z: zone_rank.get(z, 0))


# ---------------------------------------------------------------------------
# Agent 1 — DataQualityAgent (Gemini Flash)
# ---------------------------------------------------------------------------

def data_quality_agent(state: DataInfraState) -> DataInfraState:
    """
    Agent 1: DataQualityAgent
    LLM: Gemini 3.5 Flash

    Evaluates Great Expectations data quality checks across all AWB data
    zones (raw, curated, analytics). Applies BCBS 239 P3 (Accuracy),
    P4 (Completeness), and P5 (Timeliness) thresholds. Flags datasets
    breaching PRA SS1/23 Section 4 data quality requirements.
    """
    reason = (
        "ReAct reasoning: BCBS 239 P3 (Accuracy ≥ 99.9%), P4 (Completeness ≥ 99.5%), "
        "and P5 (Timeliness ≤ 4 hours for intraday risk positions) are minimum PRA "
        "expectations. Great Expectations suites run nightly against all 847 ERDW "
        "tables. Failures below uniqueness threshold 99.99% risk duplicate "
        "AWB_CUSTOMER_ID records which triggered the £680K PRA Section 166 skilled "
        "person review before the ERDW remediation in 2024."
    )

    dq_inputs = state.get("dq_inputs", {})
    datasets = dq_inputs.get("datasets", [])
    overall_zone = RiskZone.GREEN

    llm_context = f"""
Evaluate data quality for {len(datasets)} AWB datasets.
Thresholds: completeness >= {DQ_COMPLETENESS_MIN_PCT}%, accuracy >= {DQ_ACCURACY_MIN_PCT}%,
timeliness <= {DQ_TIMELINESS_MAX_HOURS}h, uniqueness >= {DQ_UNIQUENESS_MIN_PCT}%.
Regulatory: BCBS 239 P3/P4/P5, PRA SS1/23 §4, FCA COBS 9.
Dataset data: {datasets}
"""
    llm_response = _call_gemini_flash(
        "Assess data quality and classify severity for each dataset.",
        llm_context
    )
    log.debug("DataQualityAgent LLM: %s", llm_response[:200])

    dq_results: List[DataQualityCheckResult] = []
    for ds in datasets:
        dataset_id = ds.get("dataset_id", "UNKNOWN")
        completeness = float(ds.get("completeness_pct", 100.0))
        accuracy = float(ds.get("accuracy_pct", 100.0))
        timeliness = float(ds.get("timeliness_hours", 0.0))
        uniqueness = float(ds.get("uniqueness_pct", 100.0))

        failed: List[str] = []
        zone = RiskZone.GREEN

        if completeness < DQ_COMPLETENESS_MIN_PCT:
            failed.append(f"completeness={completeness:.2f}% < {DQ_COMPLETENESS_MIN_PCT}%")
            zone = _escalate_zone(zone, RiskZone.AMBER)
        if accuracy < DQ_ACCURACY_MIN_PCT:
            failed.append(f"accuracy={accuracy:.2f}% < {DQ_ACCURACY_MIN_PCT}%")
            zone = _escalate_zone(zone, RiskZone.RED)
        if timeliness > DQ_TIMELINESS_MAX_HOURS:
            failed.append(f"timeliness={timeliness:.1f}h > {DQ_TIMELINESS_MAX_HOURS}h")
            zone = _escalate_zone(zone, RiskZone.AMBER)
        if uniqueness < DQ_UNIQUENESS_MIN_PCT:
            failed.append(f"uniqueness={uniqueness:.3f}% < {DQ_UNIQUENESS_MIN_PCT}%")
            zone = _escalate_zone(zone, RiskZone.RED)

        if zone == RiskZone.RED:
            severity = DataQualitySeverity.FAIL
        elif zone == RiskZone.AMBER:
            severity = DataQualitySeverity.WARN
        else:
            severity = DataQualitySeverity.PASS

        # Regulatory reference mapping
        if "credit" in dataset_id.lower() or "decision" in dataset_id.lower():
            reg_ref = "BCBS 239 P3/P4/P5, PRA SS1/23 §4, FCA COBS 9.1.3R"
        elif "risk" in dataset_id.lower() or "position" in dataset_id.lower():
            reg_ref = "BCBS 239 P3/P5, CRR3 Art.430 reporting accuracy"
        elif "kyc" in dataset_id.lower() or "aml" in dataset_id.lower():
            reg_ref = "BCBS 239 P3, MLR 2017 Reg 28, POCA 2002 s.330"
        else:
            reg_ref = "BCBS 239 P3/P4/P5"

        result = DataQualityCheckResult(
            dataset_id=dataset_id,
            dataset_name=ds.get("dataset_name", dataset_id),
            completeness_pct=completeness,
            accuracy_pct=accuracy,
            timeliness_hours=timeliness,
            uniqueness_pct=uniqueness,
            overall_severity=severity,
            failed_expectations=failed,
            regulatory_ref=reg_ref,
        )
        dq_results.append(result)
        overall_zone = _escalate_zone(overall_zone, zone)

    state["dq_results"] = [vars(r) for r in dq_results]
    state["overall_risk_zone"] = overall_zone.value

    fail_count = sum(1 for r in dq_results if r.overall_severity == DataQualitySeverity.FAIL)
    warn_count = sum(1 for r in dq_results if r.overall_severity == DataQualitySeverity.WARN)
    outcome = (
        f"Assessed {len(dq_results)} datasets. "
        f"FAIL: {fail_count}, WARN: {warn_count}. Zone: {overall_zone.value}."
    )
    _log_step(state, 1, "DataQualityAgent", reason,
              f"Great Expectations quality checks for {len(dq_results)} datasets", outcome)
    return state


# ---------------------------------------------------------------------------
# Agent 2 — BCBS239ComplianceAgent (Gemini Flash)
# ---------------------------------------------------------------------------

def bcbs239_compliance_agent(state: DataInfraState) -> DataInfraState:
    """
    Agent 2: BCBS239ComplianceAgent
    LLM: Gemini Flash

    Evaluates all 11 BCBS 239 principles against current AWB scores.
    Any principle below 90% is flagged AMBER. Any principle below 80%
    triggers CRO and Board Risk Committee escalation with a 90-day
    remediation plan per PRA supervisory expectation (ERDW-2026-001).
    """
    reason = (
        "ReAct reasoning: BCBS 239 P11 (Distribution) at 74% is below both the 80% "
        "escalation threshold and the 90% PRA target. AWB pre-ERDW score was 52% "
        "triggering a PRA Section 166 skilled person review costing £680,000. The "
        "current 92% weighted average represents post-remediation performance. "
        "P11 gap (16pp below target) requires Chapter 16 integrated platform remediation."
    )

    bcbs239_inputs = state.get("bcbs239_inputs", {})
    scores = bcbs239_inputs.get("scores", AWB_BCBS239_SCORES)
    current_zone = RiskZone(state.get("overall_risk_zone", RiskZone.GREEN.value))

    llm_context = f"""
Evaluate BCBS 239 compliance for all 11 principles.
Threshold: compliant >= {BCBS239_MIN_SCORE_PCT}%, escalation < {BCBS239_ESCALATION_PCT}%.
CRO escalation required for any principle below {BCBS239_ESCALATION_PCT}%.
90-day remediation plan per ERDW-2026-001.
AWB Q1 2026 scores: {scores}
"""
    llm_response = _call_gemini_flash(
        "Assess BCBS 239 principle compliance and flag escalations.",
        llm_context
    )
    log.debug("BCBS239ComplianceAgent LLM: %s", llm_response[:200])

    bcbs239_results: List[BCBS239PrincipleScore] = []
    today_str = date.today().isoformat()

    for principle, score in scores.items():
        score = float(score)
        gap = max(0.0, BCBS239_MIN_SCORE_PCT - score)

        if score >= BCBS239_MIN_SCORE_PCT:
            severity = BCBSSeverity.COMPLIANT
            zone = RiskZone.GREEN
            cro_required = False
        elif score >= BCBS239_ESCALATION_PCT:
            severity = BCBSSeverity.AMBER
            zone = RiskZone.AMBER
            cro_required = False
        else:
            severity = BCBSSeverity.ESCALATE
            zone = RiskZone.RED
            cro_required = True

        current_zone = _escalate_zone(current_zone, zone)

        # Remediation deadline
        if cro_required:
            deadline = (date.today() + timedelta(days=BCBS239_REMEDIATION_DAYS)).isoformat()
        else:
            deadline = "N/A"

        # Gap actions from known AWB data
        gap_actions: List[str] = []
        if principle == "P11-Distribution":
            gap_actions = [
                "Implement automated COREP report distribution via Chapter 16 integrated platform",
                "Deploy Airflow DAG for P11 scheduled distribution to all report consumers",
                "Target: 90% by Q3 2026 per BAP-2026-DI-001 §8",
            ]
        elif principle == "P6-Adaptability":
            gap_actions = [
                "Build BCBS 239 P6 ad hoc query layer on ERDW analytics zone",
                "Enable self-service risk data aggregation for MLRO and CRO teams",
            ]
        elif principle == "P2-DataArchitecture":
            gap_actions = [
                "Complete ERDW data lineage documentation from T24 to COREP cell",
                "Publish data catalogue with business glossary for 847 ERDW tables",
            ]

        bcbs239_results.append(BCBS239PrincipleScore(
            principle=principle,
            score_pct=score,
            severity=severity,
            gap_pct=gap,
            gap_actions=gap_actions,
            cro_escalation_required=cro_required,
            remediation_deadline=deadline,
        ))

    state["bcbs239_results"] = [vars(r) for r in bcbs239_results]
    state["overall_risk_zone"] = current_zone.value

    non_compliant = sum(1 for r in bcbs239_results if r.severity != BCBSSeverity.COMPLIANT)
    escalations = sum(1 for r in bcbs239_results if r.cro_escalation_required)
    weighted_avg = sum(r.score_pct for r in bcbs239_results) / len(bcbs239_results)
    outcome = (
        f"Assessed {BCBS239_PRINCIPLES_COUNT} BCBS 239 principles. "
        f"Weighted avg: {weighted_avg:.1f}%. Non-compliant: {non_compliant}. "
        f"CRO escalations: {escalations}. Zone: {current_zone.value}."
    )
    _log_step(state, 2, "BCBS239ComplianceAgent", reason,
              f"BCBS 239 11-principle assessment (ERDW-2026-001)", outcome)
    return state


# ---------------------------------------------------------------------------
# Agent 3 — FeatureStoreHealthAgent (Gemini Flash)
# ---------------------------------------------------------------------------

def feature_store_health_agent(state: DataInfraState) -> DataInfraState:
    """
    Agent 3: FeatureStoreHealthAgent
    LLM: Gemini Flash

    Monitors training-serving skew across all AWB feature sets,
    validates Redis cache hit rates and sub-5ms serving SLAs,
    checks feature version alignment between model training runs
    and production serving (FTS-2026-001 / PRA SS1/23 §4).
    """
    reason = (
        "ReAct reasoning: Training-serving skew above 5% indicates the model is receiving "
        "different features in production than it was trained on, undermining AUC-ROC "
        "performance claims made during PRA SS1/23 validation. Redis cache hit rate below "
        "95% indicates latency degradation beyond the sub-5ms serving SLA. Feature version "
        "drift above 7 days signals a decoupling between MLflow model registry and the "
        "feature store that violates PRA SS1/23 §4.2 model documentation requirements."
    )

    fs_inputs = state.get("feature_store_inputs", {})
    feature_sets = fs_inputs.get("feature_sets", [])
    current_zone = RiskZone(state.get("overall_risk_zone", RiskZone.GREEN.value))

    llm_context = f"""
Assess feature store health for {len(feature_sets)} feature sets.
Skew thresholds: warn={FEATURE_SKEW_WARN_PCT*100}%, red={FEATURE_SKEW_RED_PCT*100}%.
Redis hit rate minimum: {REDIS_HIT_RATE_MIN_PCT}%.
Version drift threshold: {FEATURE_VERSION_DRIFT_DAYS} days.
Feature sets: {feature_sets}
"""
    llm_response = _call_gemini_flash(
        "Assess feature store health and flag skew, cache, and version issues.",
        llm_context
    )
    log.debug("FeatureStoreHealthAgent LLM: %s", llm_response[:200])

    fs_results: List[FeatureStoreHealthResult] = []
    for fs in feature_sets:
        fs_id = fs.get("feature_set_id", "UNKNOWN")
        skew = float(fs.get("skew_pct", 0.0))
        redis_hit = float(fs.get("redis_hit_rate_pct", 100.0))
        p50_ms = float(fs.get("redis_p50_ms", 2.0))
        version_drift = int(fs.get("version_drift_days", 0))

        # Zone classification
        if skew >= FEATURE_SKEW_RED_PCT:
            zone = RiskZone.RED
        elif skew >= FEATURE_SKEW_WARN_PCT:
            zone = RiskZone.AMBER
        else:
            zone = RiskZone.GREEN

        if redis_hit < REDIS_HIT_RATE_MIN_PCT:
            zone = _escalate_zone(zone, RiskZone.AMBER)
        if version_drift > FEATURE_VERSION_DRIFT_DAYS:
            zone = _escalate_zone(zone, RiskZone.AMBER)

        retrain = skew >= FEATURE_SKEW_RED_PCT

        fs_results.append(FeatureStoreHealthResult(
            feature_set_id=fs_id,
            feature_version=fs.get("feature_version", FEATURE_VERSION if False else "v2.1.0"),
            skew_pct=skew,
            risk_zone=zone,
            redis_hit_rate_pct=redis_hit,
            redis_serving_p50_ms=p50_ms,
            version_drift_days=version_drift,
            retrain_recommended=retrain,
        ))
        current_zone = _escalate_zone(current_zone, zone)

    state["feature_store_results"] = [vars(r) for r in fs_results]
    state["overall_risk_zone"] = current_zone.value

    skew_issues = sum(1 for r in fs_results if r.risk_zone != RiskZone.GREEN)
    outcome = (
        f"Assessed {len(fs_results)} feature sets. "
        f"Skew/cache issues: {skew_issues}. Zone: {current_zone.value}."
    )
    _log_step(state, 3, "FeatureStoreHealthAgent", reason,
              f"Feature store skew + cache health for {len(fs_results)} feature sets", outcome)
    return state


# ---------------------------------------------------------------------------
# Agent 4 — DataGovernanceAgent (Gemini 3.1 Pro)
# ---------------------------------------------------------------------------

def data_governance_agent(state: DataInfraState) -> DataInfraState:
    """
    Agent 4: DataGovernanceAgent
    LLM: Gemini 3.1 Pro

    Synthesises UK GDPR / DPA 2018 retention policy compliance,
    Customer 360 platform freshness and identity resolution health,
    and SAR processing SLA adherence. Produces a consolidated
    data governance summary with regulatory risk mapping.
    """
    reason = (
        "ReAct reasoning: Gemini 3.1 Pro synthesises retention compliance "
        "(seven-year baseline unifying FCA COBS 9, MLR 2017, POCA 2002, "
        "PRA SS1/23, and DORA), Customer 360 freshness (30-minute profile "
        "update SLA), and SAR processing (30-day UK GDPR Art.15 SLA). "
        "The unified seven-year retention policy is operationally correct: "
        "maintaining five different periods for data needed together in "
        "regulatory investigations is fragile. AWS S3 Object Lock enforces "
        "immutability for all audit and model output data."
    )

    governance_inputs = state.get("governance_inputs", {})
    c360_inputs = state.get("c360_inputs", {})
    current_zone = RiskZone(state.get("overall_risk_zone", RiskZone.GREEN.value))

    # Process retention compliance
    retention_categories = governance_inputs.get("retention_categories", [])
    retention_results: List[RetentionComplianceResult] = []

    for cat in retention_categories:
        cat_name = cat.get("data_category", "UNKNOWN")
        policy_years = int(cat.get("retention_years_policy", 7))
        oldest_years = float(cat.get("oldest_record_years", 0.0))
        s3_lock = bool(cat.get("s3_object_lock_active", True))
        next_deletion = cat.get("next_deletion_run", "N/A")
        reg_basis = cat.get("regulatory_basis", "FCA COBS 9")

        if oldest_years > policy_years:
            status = RetentionStatus.OVERDUE
            current_zone = _escalate_zone(current_zone, RiskZone.RED)
        elif oldest_years < policy_years * 0.5 and oldest_years > 0:
            status = RetentionStatus.COMPLIANT
        else:
            status = RetentionStatus.COMPLIANT

        if not s3_lock:
            # S3 Object Lock disabled — audit trail integrity at risk
            current_zone = _escalate_zone(current_zone, RiskZone.AMBER)

        retention_results.append(RetentionComplianceResult(
            data_category=cat_name,
            retention_years_policy=policy_years,
            oldest_record_years=oldest_years,
            status=status,
            s3_object_lock_active=s3_lock,
            next_deletion_run=next_deletion,
            regulatory_basis=reg_basis,
        ))

    # Process Customer 360 health
    c360_results: List[Customer360HealthResult] = []
    for domain in c360_inputs.get("domains", []):
        dom_name = domain.get("domain", "UNKNOWN")
        freshness = float(domain.get("freshness_minutes", 0.0))
        record_count = int(domain.get("records_count", 0))
        match_rate = float(domain.get("identity_match_rate_pct", 100.0))

        freshness_ok = freshness <= C360_FRESHNESS_SLA_MINUTES
        identity_ok = match_rate >= C360_IDENTITY_MATCH_MIN_PCT

        if not freshness_ok:
            current_zone = _escalate_zone(current_zone, RiskZone.AMBER)
        if not identity_ok:
            current_zone = _escalate_zone(current_zone, RiskZone.RED)

        c360_results.append(Customer360HealthResult(
            domain=dom_name,
            freshness_minutes=freshness,
            freshness_ok=freshness_ok,
            records_count=record_count,
            identity_match_rate_pct=match_rate,
            identity_ok=identity_ok,
        ))

    # SAR SLA check
    sar_backlog = governance_inputs.get("sar_backlog_count", 0)
    sar_overdue = governance_inputs.get("sar_overdue_count", 0)
    if sar_overdue > 0:
        current_zone = _escalate_zone(current_zone, RiskZone.RED)

    # LLM synthesis
    llm_context = f"""
Synthesise data governance findings:
- Retention categories: {len(retention_results)} assessed
- Overdue retention: {sum(1 for r in retention_results if r.status == RetentionStatus.OVERDUE)}
- S3 Object Lock inactive: {sum(1 for r in retention_results if not r.s3_object_lock_active)}
- Customer 360 domains: {len(c360_results)} assessed
- Freshness breaches: {sum(1 for r in c360_results if not r.freshness_ok)}
- Identity match failures: {sum(1 for r in c360_results if not r.identity_ok)}
- SAR backlog: {sar_backlog}, overdue: {sar_overdue}
Overall zone: {current_zone.value}
BCBS 239 results: {state.get('bcbs239_results', [])}
"""
    llm_response = _call_gemini_pro(
        "Produce a consolidated data governance summary with UK GDPR and BCBS 239 risk mapping.",
        llm_context
    )

    state["retention_results"] = [vars(r) for r in retention_results]
    state["c360_results"] = [vars(r) for r in c360_results]
    state["overall_risk_zone"] = current_zone.value

    data_governance_summary = {
        "retention_categories_assessed": len(retention_results),
        "retention_overdue_count": sum(1 for r in retention_results if r.status == RetentionStatus.OVERDUE),
        "s3_lock_inactive_count": sum(1 for r in retention_results if not r.s3_object_lock_active),
        "c360_domains_assessed": len(c360_results),
        "c360_freshness_breaches": sum(1 for r in c360_results if not r.freshness_ok),
        "c360_identity_failures": sum(1 for r in c360_results if not r.identity_ok),
        "sar_backlog": sar_backlog,
        "sar_overdue": sar_overdue,
        "overall_zone": current_zone.value,
        "llm_synthesis": llm_response[:500],
    }
    state["data_governance_summary"] = data_governance_summary

    outcome = (
        f"Retention: {len(retention_results)} categories, "
        f"{data_governance_summary['retention_overdue_count']} overdue. "
        f"C360: {len(c360_results)} domains. SAR overdue: {sar_overdue}. "
        f"Zone: {current_zone.value}."
    )
    _log_step(state, 4, "DataGovernanceAgent", reason,
              "UK GDPR retention + Customer 360 + SAR governance synthesis", outcome)
    return state


# ---------------------------------------------------------------------------
# Agent 5 — DataInfraReportAgent (Claude Sonnet 4.6)
# ---------------------------------------------------------------------------

def data_infra_report_agent(state: DataInfraState) -> DataInfraState:
    """
    Agent 5: DataInfraReportAgent
    LLM: Claude Sonnet 4.6

    Generates the formal Data Infrastructure Health Report for the AWB CRO
    and Board Risk Committee. Produces executive_summary (300 words),
    risk_narrative (BCBS 239 and GDPR regulatory mapping), and action_items.
    Satisfies BCBS 239 P1 (Governance), PRA SS1/23 para 7, and
    BAP-2026-DI-001 §7 reporting obligations.
    """
    reason = (
        "ReAct reasoning: Claude Sonnet 4.6 generates the formal CRO report. "
        "BCBS 239 P1 (Governance) requires documented evidence that senior management "
        "oversees data quality and aggregation capabilities. PRA SS1/23 para 7 requires "
        "human-readable audit documentation for all model data decisions. "
        "BAP-2026-DI-001 §7 mandates Board Risk Committee reporting within 5 business "
        "days of any BCBS 239 principle scoring below 90%."
    )

    current_zone = RiskZone(state.get("overall_risk_zone", RiskZone.GREEN.value))
    dq_results = state.get("dq_results", [])
    bcbs239_results = state.get("bcbs239_results", [])
    governance_summary = state.get("data_governance_summary", {})

    # BCBS 239 summary stats
    non_compliant_principles = [
        r["principle"] for r in bcbs239_results
        if r.get("score_pct", 100.0) < BCBS239_MIN_SCORE_PCT
    ]
    cro_escalations = [
        r["principle"] for r in bcbs239_results
        if r.get("cro_escalation_required", False)
    ]
    weighted_avg = (
        sum(r.get("score_pct", 0.0) for r in bcbs239_results) / len(bcbs239_results)
        if bcbs239_results else 0.0
    )

    report_prompt = f"""
Generate an AWB Data Infrastructure Health Report for the CRO and Board Risk Committee.

Platform status: {'REQUIRES ATTENTION' if current_zone != RiskZone.GREEN else 'HEALTHY'}
Overall risk zone: {current_zone.value}
Run date: {state.get('run_date', date.today().isoformat())}
Trigger: {state.get('trigger_event', 'scheduled assessment')}

BCBS 239 Summary:
- 11 principles assessed, weighted average: {weighted_avg:.1f}%
- Non-compliant (< 90%): {non_compliant_principles}
- CRO escalations (< 80%): {cro_escalations}
- Pre-ERDW baseline: 52% (PRA Section 166 cost: £680,000)

Data Quality: {len(dq_results)} datasets assessed
- FAIL: {sum(1 for r in dq_results if r.get('overall_severity') == 'FAIL')}
- WARN: {sum(1 for r in dq_results if r.get('overall_severity') == 'WARN')}

Governance:
- Retention overdue: {governance_summary.get('retention_overdue_count', 0)}
- SAR overdue: {governance_summary.get('sar_overdue', 0)}
- Customer 360 freshness breaches: {governance_summary.get('c360_freshness_breaches', 0)}

AWB data estate: {AWB_TOTAL_RECORDS_BILLIONS}B records, {AWB_ERDW_TABLES} ERDW tables, {C360_DOMAINS_COUNT} C360 domains.

Regulatory context: BCBS 239 (Basel 2013/2022), PRA SS1/23 §4-7,
UK GDPR Art.5/15/17, DPA 2018, FCA COBS 9.1.3R, MLR 2017 Reg 40,
DORA Art.17, BAP-2026-DI-001.

Produce:
1. executive_summary: 300-word CRO/Board-ready executive summary
2. risk_narrative: Detailed BCBS 239 principle findings with PRA regulatory mapping
3. action_items: Numbered prioritised remediation steps with owners and deadlines
"""
    llm_response = _call_claude_sonnet(report_prompt)

    sections = llm_response.split("\n\n") if llm_response else []
    exec_summary = sections[0] if sections else (
        f"BCBS 239 weighted average {weighted_avg:.1f}%. Zone: {current_zone.value}. "
        f"Non-compliant principles: {non_compliant_principles}."
    )
    risk_narrative = sections[1] if len(sections) > 1 else (
        f"Zone {current_zone.value}. BCBS 239 P11 at 74% — below 80% escalation threshold."
    )
    action_items_raw = sections[2] if len(sections) > 2 else (
        "1. P11 Distribution: Deploy Airflow distribution DAG by Q3 2026.\n"
        "2. P6 Adaptability: Build ad hoc query layer on analytics zone.\n"
        "3. P2 Architecture: Complete T24-to-COREP data lineage documentation."
    )

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
        f"Data infra report generated. BCBS 239 avg: {weighted_avg:.1f}%. "
        f"Action items: {len(action_items)}. Zone: {current_zone.value}."
    )
    _log_step(state, 5, "DataInfraReportAgent", reason,
              "Generate CRO Data Infrastructure report via Claude Sonnet 4.6", outcome)
    return state


# ---------------------------------------------------------------------------
# HITL gate
# ---------------------------------------------------------------------------

def hitl_gate(state: DataInfraState) -> DataInfraState:
    """
    HITL Gate: Human-in-the-Loop data governance approval.

    BCBS 239 P1 requires senior management oversight of data risk.
    PRA SS1/23 para 7 and EU AI Act Art.14 require human sign-off
    for HIGH-RISK AI system data decisions.

    APPROVE only when:
    - All BCBS 239 principles score >= 90%
    - No data quality FAIL findings
    - No retention overdue categories
    - No SAR SLA breaches
    - Customer 360 identity match rate >= 98% across all domains
    - Overall zone is GREEN
    """
    current_zone = RiskZone(state.get("overall_risk_zone", RiskZone.GREEN.value))
    bcbs239_results = state.get("bcbs239_results", [])
    dq_results = state.get("dq_results", [])
    governance_summary = state.get("data_governance_summary", {})

    has_bcbs_gap = any(
        r.get("score_pct", 100.0) < BCBS239_MIN_SCORE_PCT for r in bcbs239_results
    )
    has_dq_fail = any(
        r.get("overall_severity") == DataQualitySeverity.FAIL.value for r in dq_results
    )
    has_retention_overdue = governance_summary.get("retention_overdue_count", 0) > 0
    has_sar_overdue = governance_summary.get("sar_overdue", 0) > 0
    has_identity_failure = governance_summary.get("c360_identity_failures", 0) > 0

    if (current_zone == RiskZone.GREEN
            and not has_bcbs_gap
            and not has_dq_fail
            and not has_retention_overdue
            and not has_sar_overdue
            and not has_identity_failure):
        decision = HITLDecision.APPROVE
    else:
        decision = HITLDecision.ESCALATE

    state["hitl_decision"] = decision.value

    breaches = []
    if has_bcbs_gap:
        non_compliant = [r["principle"] for r in bcbs239_results if r.get("score_pct", 100.0) < BCBS239_MIN_SCORE_PCT]
        breaches.append(f"BCBS 239 gaps: {non_compliant}")
    if has_dq_fail:
        breaches.append("Data quality FAIL findings")
    if has_retention_overdue:
        breaches.append(f"Retention overdue: {governance_summary.get('retention_overdue_count')} categories")
    if has_sar_overdue:
        breaches.append(f"SAR SLA breached: {governance_summary.get('sar_overdue')} cases")
    if has_identity_failure:
        breaches.append("Customer 360 identity match below 98%")

    _log_step(
        state, 6, "HITLGate",
        "BCBS 239 P1 governance + PRA SS1/23 §7 + EU AI Act Art.14",
        "Evaluate HITL data governance approval",
        f"Decision: {decision.value}. Zone: {current_zone.value}. "
        + (f"Breaches: {breaches}" if breaches else "All clear."),
    )
    return state


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _build_graph() -> Any:
    graph = StateGraph(DataInfraState)

    graph.add_node("data_quality", data_quality_agent)
    graph.add_node("bcbs239_compliance", bcbs239_compliance_agent)
    graph.add_node("feature_store_health", feature_store_health_agent)
    graph.add_node("data_governance", data_governance_agent)
    graph.add_node("data_infra_report", data_infra_report_agent)
    graph.add_node("hitl", hitl_gate)

    graph.add_edge(START, "data_quality")
    graph.add_edge("data_quality", "bcbs239_compliance")
    graph.add_edge("bcbs239_compliance", "feature_store_health")
    graph.add_edge("feature_store_health", "data_governance")
    graph.add_edge("data_governance", "data_infra_report")
    graph.add_edge("data_infra_report", "hitl")
    graph.add_edge("hitl", END)

    graph.set_entry_point("data_quality")
    return graph.compile()


_AWB_DATA_INFRA_GRAPH = None


def _get_graph():
    global _AWB_DATA_INFRA_GRAPH
    if _AWB_DATA_INFRA_GRAPH is None:
        _AWB_DATA_INFRA_GRAPH = _build_graph()
    return _AWB_DATA_INFRA_GRAPH


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_agentic_data_infrastructure(
    run_date: str,
    trigger_event: str,
    dq_inputs: Dict[str, Any],
    bcbs239_inputs: Dict[str, Any],
    feature_store_inputs: Dict[str, Any],
    governance_inputs: Dict[str, Any],
    c360_inputs: Optional[Dict[str, Any]] = None,
) -> DataInfraState:
    """
    Run the AWB Agentic Data Infrastructure Pipeline (MR-2026-063-DI).

    Args:
        run_date:             ISO date string (e.g. "2026-05-24")
        trigger_event:        What triggered this run
        dq_inputs:            Dict with key "datasets" — list of dicts with:
                              dataset_id, dataset_name, completeness_pct,
                              accuracy_pct, timeliness_hours, uniqueness_pct
        bcbs239_inputs:       Dict with key "scores" — {principle: score_pct}
                              Defaults to AWB Q1 2026 scores if empty
        feature_store_inputs: Dict with key "feature_sets" — list of dicts with:
                              feature_set_id, feature_version, skew_pct,
                              redis_hit_rate_pct, redis_p50_ms, version_drift_days
        governance_inputs:    Dict with keys:
                              "retention_categories" (list of retention dicts),
                              "sar_backlog_count" (int),
                              "sar_overdue_count" (int)
        c360_inputs:          Optional dict with key "domains" — list of dicts with:
                              domain, freshness_minutes, records_count,
                              identity_match_rate_pct

    Returns:
        DataInfraState with all fields populated including:
        - dq_results, bcbs239_results, feature_store_results
        - retention_results, c360_results, data_governance_summary
        - hitl_decision, executive_summary, risk_narrative, action_items
        - hop_chain (6 steps: 5 agents + HITL)
    """
    run_id = str(uuid.uuid4())
    log.info("Starting MR-2026-063-DI run_id=%s date=%s trigger=%s",
             run_id, run_date, trigger_event)

    initial_state = DataInfraState(
        run_id=run_id,
        run_date=run_date,
        trigger_event=trigger_event,
        dq_inputs=dq_inputs,
        bcbs239_inputs=bcbs239_inputs,
        feature_store_inputs=feature_store_inputs,
        governance_inputs=governance_inputs,
        c360_inputs=c360_inputs or {"domains": []},
        dq_results=[],
        bcbs239_results=[],
        feature_store_results=[],
        retention_results=[],
        c360_results=[],
        overall_risk_zone=RiskZone.GREEN.value,
        data_governance_summary={},
        hitl_decision=HITLDecision.PENDING.value,
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
    except Exception as exc:
        log.exception("Data infra pipeline error: %s", exc)
        initial_state.setdefault("errors", []).append(str(exc))
        initial_state["hitl_decision"] = HITLDecision.ESCALATE.value
        return initial_state

    log.info(
        "MR-2026-063-DI complete: zone=%s hitl=%s hop_chain_steps=%d",
        final_state.get("overall_risk_zone"),
        final_state.get("hitl_decision"),
        len(final_state.get("hop_chain", [])),
    )
    return DataInfraState(final_state)


# ---------------------------------------------------------------------------
# AWB Q1 2026 demonstration run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    demo_dq_inputs = {
        "datasets": [
            {
                "dataset_id": "erdw.credit_decisions",
                "dataset_name": "ERDW Credit Decision Records",
                "completeness_pct": 99.97,
                "accuracy_pct": 99.94,
                "timeliness_hours": 1.2,
                "uniqueness_pct": 99.999,
            },
            {
                "dataset_id": "erdw.risk_positions_intraday",
                "dataset_name": "ERDW Intraday Risk Positions",
                "completeness_pct": 99.91,
                "accuracy_pct": 99.95,
                "timeliness_hours": 3.8,    # within 4-hour SLA
                "uniqueness_pct": 100.0,
            },
            {
                "dataset_id": "erdw.kyc_records",
                "dataset_name": "ERDW KYC Records",
                "completeness_pct": 99.88,  # AMBER — below 99.5% threshold?
                "accuracy_pct": 99.92,
                "timeliness_hours": 2.1,
                "uniqueness_pct": 99.998,
            },
        ]
    }

    # Use AWB Q1 2026 production scores
    demo_bcbs239_inputs = {
        "scores": AWB_BCBS239_SCORES
    }

    demo_feature_store_inputs = {
        "feature_sets": [
            {
                "feature_set_id": "fs.churn_features_v2",
                "feature_version": "v2.1.0",
                "skew_pct": 0.008,          # 0.8% — below 2% warn threshold
                "redis_hit_rate_pct": 97.3,
                "redis_p50_ms": 2.1,
                "version_drift_days": 2,
            },
            {
                "feature_set_id": "fs.credit_features_v3",
                "feature_version": "v3.4.0",
                "skew_pct": 0.013,          # 1.3% — below warn threshold
                "redis_hit_rate_pct": 98.1,
                "redis_p50_ms": 1.8,
                "version_drift_days": 1,
            },
        ]
    }

    demo_governance_inputs = {
        "retention_categories": [
            {
                "data_category": "credit_decisions",
                "retention_years_policy": RETENTION_CREDIT_DECISIONS_YEARS,
                "oldest_record_years": 3.2,
                "s3_object_lock_active": True,
                "next_deletion_run": "2030-01-15",
                "regulatory_basis": "FCA COBS 9.1.3R",
            },
            {
                "data_category": "sar_records",
                "retention_years_policy": RETENTION_SAR_RECORDS_YEARS,
                "oldest_record_years": 2.1,
                "s3_object_lock_active": True,
                "next_deletion_run": "2028-06-30",
                "regulatory_basis": "MLR 2017 Regulation 40",
            },
            {
                "data_category": "model_outputs",
                "retention_years_policy": RETENTION_MODEL_OUTPUTS_YEARS,
                "oldest_record_years": 1.8,
                "s3_object_lock_active": True,
                "next_deletion_run": "2031-01-01",
                "regulatory_basis": "PRA SS1/23 Section 4",
            },
            {
                "data_category": "audit_logs",
                "retention_years_policy": RETENTION_AUDIT_LOGS_YEARS,
                "oldest_record_years": 2.4,
                "s3_object_lock_active": True,
                "next_deletion_run": "2030-12-31",
                "regulatory_basis": "FCA COBS 9 / DORA Art.17",
            },
        ],
        "sar_backlog_count": 12,
        "sar_overdue_count": 0,    # all within 30-day UK GDPR Art.15 SLA
    }

    demo_c360_inputs = {
        "domains": [
            {"domain": "identity_kyc",      "freshness_minutes": 8.0,  "records_count": 287_000, "identity_match_rate_pct": 99.3},
            {"domain": "credit_history",    "freshness_minutes": 22.0, "records_count": 201_000, "identity_match_rate_pct": 99.1},
            {"domain": "transactions",      "freshness_minutes": 4.0,  "records_count": 287_000, "identity_match_rate_pct": 99.7},
            {"domain": "product_holdings",  "freshness_minutes": 18.0, "records_count": 287_000, "identity_match_rate_pct": 99.5},
            {"domain": "risk_scores",       "freshness_minutes": 12.0, "records_count": 271_000, "identity_match_rate_pct": 98.9},
            {"domain": "complaints",        "freshness_minutes": 25.0, "records_count": 4_200,   "identity_match_rate_pct": 98.6},
            {"domain": "channel_behaviour", "freshness_minutes": 6.0,  "records_count": 265_000, "identity_match_rate_pct": 99.2},
            {"domain": "vulnerability",     "freshness_minutes": 28.0, "records_count": 14_300,  "identity_match_rate_pct": 98.4},
        ]
    }

    result = asyncio.run(run_agentic_data_infrastructure(
        run_date="2026-05-24",
        trigger_event="scheduled_daily_q1_2026_review",
        dq_inputs=demo_dq_inputs,
        bcbs239_inputs=demo_bcbs239_inputs,
        feature_store_inputs=demo_feature_store_inputs,
        governance_inputs=demo_governance_inputs,
        c360_inputs=demo_c360_inputs,
    ))

    print("\n" + "=" * 70)
    print("AWB Data Infrastructure Pipeline (MR-2026-063-DI) — Q1 2026 Review")
    print("=" * 70)
    print(f"Run ID:         {result.get('run_id')}")
    print(f"Overall Zone:   {result.get('overall_risk_zone')}")
    print(f"HITL Decision:  {result.get('hitl_decision')}")
    print(f"Hop-chain:      {len(result.get('hop_chain', []))} steps")
    print("\nBCBS 239 Summary:")
    for r in result.get("bcbs239_results", []):
        flag = "✓" if r["severity"] == "COMPLIANT" else ("⚠" if r["severity"] == "AMBER" else "✗")
        print(f"  {flag} {r['principle']:30s} {r['score_pct']:5.1f}%  {r['severity']}")
    print("\nData Quality:")
    for r in result.get("dq_results", []):
        print(f"  {r['dataset_id']:40s} {r['overall_severity']}")
    print("\nHop-chain:")
    for step in result.get("hop_chain", []):
        print(f"  Step {step['seq']}: [{step['agent']}] {step['outcome']}")
    print("\nExecutive Summary (first 300 chars):")
    print(result.get("executive_summary", "")[:300])
    print("=" * 70)
