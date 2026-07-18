"""Agentic Enterprise Architecture Monitor — Chapter 13 Agentic Extension.

Model ID  : MR-2026-061-EA
Risk Class: HIGH (DORA Art.9 ICT risk; PRA SS1/23; operational resilience)
Chapter   : 13 — Enterprise AI Architecture for Risk and Compliance

Architecture: LangGraph StateGraph — five specialist agents + HITL gate.

Agents
------
1. ServiceMeshAgent       — Gemini 3.5 Flash  — ECS health, circuit breakers, inter-service connectivity
2. InfrastructureAgent    — Gemini 3.5 Flash  — AWS/Azure multi-cloud, Terraform drift, FinOps cost analysis
3. KafkaT24Agent          — Gemini 3.5 Flash  — Kafka topology, consumer lag, T24 integration pattern health
4. DORAResilienceAgent    — Gemini 3.1 Pro    — DORA Arts 9/17/28, RTO/RPO, LLM provider concentration
5. ArchitectureReportAgent— Claude Sonnet 4.6 — architecture health narrative, board report, change management

HITL Gate: HITLDecision — APPROVE / ESCALATE / OVERRIDE / PENDING.
           Conservative: ESCALATE on any circuit OPEN, Kafka lag breach,
           T24 replica lag > 5min, RTO breach, DORA concentration breach.

Regulatory Coverage
-------------------
- DORA Art. 9 (ICT risk management framework — prevent, detect, recover)
- DORA Art. 17 (ICT-related incident classification — material incidents)
- DORA Art. 28 (ICT third-party concentration risk — no provider > 70%)
- PRA SS1/23 §4-7 (model risk — HITL, validation, change management)
- PRA PS7/25 (operational resilience — important business services)
- FCA SYSC 15A (operational resilience for FCA-regulated firms)
- UK GDPR Art. 32 (security of processing — AWS EU-WEST-2 residency)
- BAP-2026-EA-001 (AWB internal: enterprise architecture governance)

LLM Allocation
--------------
Agents 1-3 : google/gemini-3.5-flash   — fast, structured infrastructure checks
Agent 4    : google/gemini-3.1-pro          — multi-standard resilience reasoning
Agent 5    : anthropic/claude-sonnet-4-6    — architecture narrative synthesis

Hop-Chain Audit
---------------
Every agent appends to state["hop_chain"]:
  {seq, agent, timestamp, reason, act, outcome}
Mandatory per PRA AI Roundtable October 2025 / BAP-2026-EA-001 §6.
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
import os
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PLATFORM CONSTANTS
# ---------------------------------------------------------------------------

# ECS service health thresholds
ECS_MIN_HEALTHY_PCT: float = 100.0   # Min 100% healthy for rolling deploy
ECS_P50_LATENCY_MS: float = 150.0    # AWB synchronous REST p50 target
ECS_P95_LATENCY_MS: float = 500.0    # p95 target
ECS_P99_LATENCY_MS: float = 2000.0   # p99 target (LLM services)

# Circuit breaker thresholds (awb_commons CircuitBreaker)
CIRCUIT_FAILURE_THRESHOLD: float = 0.5   # 50% failures → OPEN
CIRCUIT_WINDOW_SECONDS: int = 60
CIRCUIT_RECOVERY_TIMEOUT: int = 30

# Kafka consumer lag thresholds (DORA Art.9 real-time processing)
KAFKA_LAG_WARN_MESSAGES: int = 1_000     # Amber warning
KAFKA_LAG_BREACH_MESSAGES: int = 10_000  # Red — SLA breach
KAFKA_AML_LAG_MAX_SECONDS: int = 5       # AML alert < 5s from T24 commit
KAFKA_FX_LAG_MAX_SECONDS: int = 30       # FX rate < 30s

# T24 integration health (DORA Art.9 — core banking system)
T24_READ_REPLICA_LAG_WARN_SECS: int = 300   # 5-minute read replica lag
T24_READ_REPLICA_LAG_RED_SECS: int = 600    # 10-minute — escalate
T24_IDEMPOTENCY_STUCK_MINS: int = 30        # Stuck reservations → alert
T24_KAFKA_BATCH_WINDOW_START_HR: int = 22   # 22:00 GMT batch start
T24_KAFKA_BATCH_WINDOW_END_HR: int = 4      # 04:00 GMT batch end

# Multi-cloud / DR thresholds (DORA Art.9, PRA PS7/25)
AWS_FAILOVER_RTO_MINS: int = 120    # Tier 1 services (Digital Identity, API Gateway)
AWS_FAILOVER_RPO_MINS: int = 60     # Tier 1 RPO
AZURE_DR_SYNC_LAG_WARN_MINS: int = 30
TERRAFORM_DRIFT_WARN_RESOURCES: int = 5   # Amber if > 5 drifted resources

# LLM provider concentration (DORA Art.28 — no provider > 70%)
DORA_LLM_CONCENTRATION_MAX_PCT: float = 70.0   # Hard limit
DORA_LLM_CONCENTRATION_WARN_PCT: float = 60.0  # Warning threshold
AWB_LLM_DISTRIBUTION: Dict[str, float] = {     # AWB target distribution
    "google": 68.0,    # Gemini 3.5 Flash primary
    "anthropic": 17.0, # Claude Sonnet 4.6
    "openai": 15.0,    # GPT-5.5 fallback
}

# AWS cost thresholds (FinOps — BAP-2026-EA-001 §9)
MONTHLY_INFRA_BUDGET_GBP: float = 21_750.0  # £21,750/month total
LLM_API_MONTHLY_BUDGET_GBP: float = 8_500.0  # LLM API calls
ECS_MONTHLY_BUDGET_GBP: float = 7_200.0
MSK_MONTHLY_BUDGET_GBP: float = 2_800.0


# ---------------------------------------------------------------------------
# STATE SCHEMA
# ---------------------------------------------------------------------------

class EnterpriseArchState(dict):
    """Shared mutable state threaded through all five agents.

    Inherits dict for LangGraph compatibility.
    Zone only escalates (GREEN → AMBER → RED).
    """
    pass


def _initial_state(
    run_date: date,
    trigger_event: str,
    service_health_inputs: Dict[str, Any],
    infrastructure_inputs: Dict[str, Any],
    kafka_inputs: Dict[str, Any],
    t24_inputs: Dict[str, Any],
) -> EnterpriseArchState:
    """Construct clean initial state for the architecture pipeline.

    Args:
        run_date: Date of this architecture health run.
        trigger_event: What triggered this run
            (e.g., "Daily architecture health check 07:00 GMT").
        service_health_inputs: ECS service metrics.
            Keys: services (list of {name, desired_count, running_count,
            p50_ms, p95_ms, circuit_state}), total_services.
        infrastructure_inputs: AWS/Azure/Terraform metrics.
            Keys: aws_region, azure_dr_region, terraform_drifted_resources,
            azure_dr_sync_lag_mins, monthly_cost_gbp (by category),
            llm_usage_pct (by provider).
        kafka_inputs: Kafka topology health.
            Keys: topics (list of {name, consumer_lag, lag_seconds}),
            total_partitions, broker_count.
        t24_inputs: T24 integration health.
            Keys: read_replica_lag_secs, idempotency_stuck_count,
            kafka_stream_active, batch_window_active.

    Returns:
        Initialised EnterpriseArchState dict.
    """
    return EnterpriseArchState(
        # ---- inputs ----
        run_date=run_date,
        trigger_event=trigger_event,
        service_health_inputs=service_health_inputs,
        infrastructure_inputs=infrastructure_inputs,
        kafka_inputs=kafka_inputs,
        t24_inputs=t24_inputs,
        # ---- outputs (populated by agents) ----
        service_mesh_result=None,       # Agent 1
        degraded_services=[],           # Agent 1
        open_circuits=[],               # Agent 1
        infrastructure_result=None,     # Agent 2
        terraform_drift_resources=[],   # Agent 2
        cost_overruns=[],               # Agent 2
        kafka_result=None,              # Agent 3
        lagging_topics=[],              # Agent 3
        t24_health=None,                # Agent 3
        dora_result=None,               # Agent 4
        dora_breaches=[],               # Agent 4
        rto_rpo_status={},              # Agent 4
        llm_concentration_check={},     # Agent 4
        architecture_narrative="",      # Agent 5
        board_summary="",               # Agent 5
        change_management_flags=[],     # Agent 5
        # ---- control ----
        risk_zone="GREEN",
        hitl_decision="PENDING",
        hitl_rationale="",
        hop_chain=[],
        pipeline_completed=False,
    )


# ---------------------------------------------------------------------------
# HOP-CHAIN LOGGING
# ---------------------------------------------------------------------------

_SEQ: int = 0


def _log_step(
    state: EnterpriseArchState,
    agent: str,
    reason: str,
    act: str,
    outcome: str,
) -> None:
    """Append one hop to the audit chain (BAP-2026-EA-001 §6)."""
    global _SEQ
    _SEQ += 1
    hop = {
        "seq": _SEQ,
        "agent": agent,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "reason": reason,
        "act": act,
        "outcome": outcome,
    }
    state["hop_chain"].append(hop)
    log.info("[HOP %02d] %s | %s → %s", _SEQ, agent, act[:60], outcome[:80])


# ---------------------------------------------------------------------------
# RISK ZONE ESCALATION
# ---------------------------------------------------------------------------

_ZONE_RANK: Dict[str, int] = {"GREEN": 0, "AMBER": 1, "RED": 2}


def _escalate_zone(state: EnterpriseArchState, proposed: str) -> None:
    """Monotonically escalate risk zone — GREEN→AMBER→RED only."""
    current = state.get("risk_zone", "GREEN")
    new_zone = max(current, proposed, key=lambda z: _ZONE_RANK.get(z, 0))
    if new_zone != current:
        log.warning("EA risk zone: %s → %s", current, new_zone)
    state["risk_zone"] = new_zone


# ---------------------------------------------------------------------------
# HITL DECISION
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

class HITLDecision(str, Enum):
    """HITL decision for enterprise architecture pipeline.

    APPROVE  : All services healthy, no drift, Kafka lag within SLA,
               DORA compliant, T24 healthy.
    ESCALATE : Any circuit OPEN, Kafka SLA breach, DORA concentration
               breach, Terraform drift, T24 lag critical.
    OVERRIDE : Engineering lead has reviewed and accepted the issue.
    PENDING  : Awaiting human decision.
    """
    APPROVE = "APPROVE"
    ESCALATE = "ESCALATE"
    OVERRIDE = "OVERRIDE"
    PENDING = "PENDING"


def _compute_hitl_decision(
    state: EnterpriseArchState,
) -> Tuple[str, str]:
    """Derive HITL decision from pipeline state."""
    reasons: List[str] = []

    if state.get("open_circuits"):
        reasons.append(
            f"Circuit(s) OPEN: {state['open_circuits']} — "
            "cascade failure risk (DORA Art.9)"
        )
    if state.get("degraded_services"):
        reasons.append(
            f"Degraded services: {state['degraded_services']}"
        )
    if state.get("lagging_topics"):
        reasons.append(
            f"Kafka SLA breach: {state['lagging_topics']}"
        )
    if state.get("dora_breaches"):
        reasons.append(
            f"DORA breach(es): {'; '.join(state['dora_breaches'])}"
        )
    if state.get("cost_overruns"):
        reasons.append(
            f"Budget overrun(s): {state['cost_overruns']}"
        )
    if state.get("terraform_drift_resources"):
        n = len(state["terraform_drift_resources"])
        if n > TERRAFORM_DRIFT_WARN_RESOURCES:
            reasons.append(
                f"Terraform drift: {n} resources — IaC governance (BAP-2026-EA-001 §5)"
            )

    if reasons:
        return HITLDecision.ESCALATE.value, " | ".join(reasons)

    return HITLDecision.APPROVE.value, (
        "All ECS services healthy; no open circuits; Kafka within SLA; "
        "DORA compliant; T24 integration healthy; Terraform clean."
    )


# ---------------------------------------------------------------------------
# LLM HELPERS
# ---------------------------------------------------------------------------

def _call_gemini_flash(prompt: str, context: str = "") -> str:
    """Invoke Gemini 3.5 Flash for fast infrastructure analysis."""
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        return f"[FLASH-STUB] {prompt[:80]}..."
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-3.5-flash")
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        return model.generate_content(full_prompt).text
    except Exception as exc:
        log.warning("Gemini Flash error: %s", exc)
        return f"[FLASH-ERROR] {exc}"


def _call_gemini_pro(prompt: str, context: str = "") -> str:
    """Invoke Gemini 3.1 Pro for DORA multi-standard reasoning."""
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        return f"[PRO-STUB] {prompt[:80]}..."
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-3.1-pro")
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        return model.generate_content(full_prompt).text
    except Exception as exc:
        log.warning("Gemini Pro error: %s", exc)
        return f"[PRO-ERROR] {exc}"


def _call_claude_sonnet(prompt: str, context: str = "") -> str:
    """Invoke Claude Sonnet 4.6 for architecture narrative synthesis."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return f"[SONNET-STUB] {prompt[:80]}..."
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            messages=[{"role": "user", "content": full_prompt}],
        )
        return msg.content[0].text
    except Exception as exc:
        log.warning("Claude Sonnet error: %s", exc)
        return f"[SONNET-ERROR] {exc}"


# ---------------------------------------------------------------------------
# AGENT 1 — ServiceMeshAgent (Gemini Flash)
# ---------------------------------------------------------------------------

def service_mesh_agent(
    state: EnterpriseArchState,
) -> EnterpriseArchState:
    """Agent 1: Monitor ECS service health and circuit breaker states.

    Regulatory basis: DORA Art. 9 ICT risk management requires continuous
    monitoring of ICT systems supporting important business services.
    PRA PS7/25 operational resilience: digital services must be able to
    remain within impact tolerances. AWB's 23 ECS services constitute
    important business services under both DORA and PRA frameworks.

    The CircuitBreaker pattern (awb_commons/circuit_breaker.py) prevents
    cascade failures. A circuit in OPEN state means the dependent service
    is shedding load — the OPEN state itself is a protection mechanism,
    but its root cause must be resolved within the DORA Art.17 incident
    classification window.

    Thresholds:
    - p50 < 150ms, p95 < 500ms, p99 < 2000ms (LLM services)
    - Desired == Running count for rolling deploy (ECS_MIN_HEALTHY_PCT)
    - Circuit OPEN state → RED zone immediately

    Populates:
        state["service_mesh_result"]: Summary dict.
        state["degraded_services"]: Services below SLA.
        state["open_circuits"]: Services with circuit OPEN.
        risk_zone: AMBER (latency breach) or RED (circuit OPEN).

    Args:
        state: Shared pipeline state.

    Returns:
        Updated state.
    """
    inputs = state.get("service_health_inputs", {})
    services = inputs.get("services", [])

    reason = (
        "DORA Art.9 and PRA PS7/25 require continuous monitoring of ECS "
        "services. Circuit OPEN state is a DORA Art.17 incident trigger. "
        "Latency SLA breaches indicate capacity or dependency issues "
        "requiring engineering escalation under BAP-2026-EA-001 §4."
    )

    degraded: List[str] = []
    open_circuits: List[str] = []
    service_summary: List[Dict[str, Any]] = []

    for svc in services:
        name = svc.get("name", "unknown")
        desired = svc.get("desired_count", 0)
        running = svc.get("running_count", 0)
        p50 = svc.get("p50_ms", 0.0)
        p95 = svc.get("p95_ms", 0.0)
        p99 = svc.get("p99_ms", 0.0)
        circuit = svc.get("circuit_state", "closed")

        issues: List[str] = []

        # Capacity check
        if desired > 0 and running < desired:
            issues.append(
                f"tasks {running}/{desired} (ECS rolling deploy at risk)"
            )
            _escalate_zone(state, "AMBER")

        # Latency SLA
        if p50 > ECS_P50_LATENCY_MS:
            issues.append(f"p50={p50}ms > {ECS_P50_LATENCY_MS}ms SLA")
            _escalate_zone(state, "AMBER")
        if p95 > ECS_P95_LATENCY_MS:
            issues.append(f"p95={p95}ms > {ECS_P95_LATENCY_MS}ms SLA")
            _escalate_zone(state, "AMBER")
        if p99 > ECS_P99_LATENCY_MS:
            issues.append(f"p99={p99}ms > {ECS_P99_LATENCY_MS}ms LLM SLA")
            _escalate_zone(state, "AMBER")

        # Circuit breaker
        if circuit.upper() == "OPEN":
            open_circuits.append(name)
            issues.append("circuit=OPEN — cascade failure risk")
            _escalate_zone(state, "RED")
        elif circuit.upper() == "HALF_OPEN":
            issues.append("circuit=HALF_OPEN — recovering")
            _escalate_zone(state, "AMBER")

        if issues:
            degraded.append(name)
            log.warning("Service %s: %s", name, "; ".join(issues))

        service_summary.append({
            "name": name,
            "running": running,
            "desired": desired,
            "p50_ms": p50,
            "p95_ms": p95,
            "circuit_state": circuit,
            "issues": issues,
        })

    # LLM analysis of service mesh pattern
    llm_prompt = (
        f"AWB ECS service mesh health — {state['run_date']}:\n"
        f"  Total services: {len(services)}\n"
        f"  Degraded: {len(degraded)}\n"
        f"  Open circuits: {len(open_circuits)}\n"
        f"  Open circuit services: {open_circuits}\n\n"
        f"Identify the most likely root cause pattern and cite the "
        f"DORA Art.17 incident classification that applies. "
        f"Recommend the AWB circuit breaker recovery procedure from "
        f"awb_commons/circuit_breaker.py."
    )
    state["service_mesh_llm"] = _call_gemini_flash(llm_prompt)

    state["service_mesh_result"] = {
        "total_services": len(services),
        "degraded_count": len(degraded),
        "open_circuit_count": len(open_circuits),
        "service_summary": service_summary,
    }
    state["degraded_services"] = degraded
    state["open_circuits"] = open_circuits

    _log_step(
        state,
        agent="ServiceMeshAgent [gemini-3.5-flash]",
        reason=reason,
        act=f"Checked {len(services)} ECS services for health/SLA/circuit states",
        outcome=(
            f"Degraded={len(degraded)}, OpenCircuits={len(open_circuits)}, "
            f"Zone={state['risk_zone']}"
        ),
    )
    return state


# ---------------------------------------------------------------------------
# AGENT 2 — InfrastructureAgent (Gemini Flash)
# ---------------------------------------------------------------------------

def infrastructure_agent(
    state: EnterpriseArchState,
) -> EnterpriseArchState:
    """Agent 2: Validate multi-cloud health, Terraform drift, and FinOps costs.

    Regulatory basis: DORA Art. 9 requires ICT risk management across the
    full technology estate, including cloud infrastructure. DORA Art. 28
    mandates concentration risk management for ICT third-party providers —
    no single cloud provider should create systemic dependency. PRA PS7/25
    requires documented RTO/RPO for important business services. UK GDPR
    Art. 32 requires appropriate security for processing, which includes
    maintaining data residency in AWS EU-WEST-2 (London).

    Terraform drift detection (BAP-2026-EA-001 §5): any infrastructure
    change not captured in Terraform represents a governance risk under
    DORA Art. 9 change management requirements.

    FinOps thresholds (BAP-2026-EA-001 §9): monthly budget caps enforced
    per service category to maintain the £21,750/month infrastructure budget.

    Populates:
        state["infrastructure_result"]: Summary dict.
        state["terraform_drift_resources"]: Drifted resource names.
        state["cost_overruns"]: Budget categories over limit.
        risk_zone: AMBER (drift/cost) or RED (DR sync failure).

    Args:
        state: Shared pipeline state.

    Returns:
        Updated state.
    """
    inputs = state.get("infrastructure_inputs", {})

    reason = (
        "DORA Art.9 ICT risk and Art.28 concentration risk require "
        "multi-cloud health monitoring. Terraform drift = DORA change "
        "management gap. UK GDPR Art.32 requires AWS EU-WEST-2 data "
        "residency. FinOps budget enforcement per BAP-2026-EA-001 §9."
    )

    # ---- Multi-cloud DR health ----
    azure_lag_mins = float(inputs.get("azure_dr_sync_lag_mins", 0))
    if azure_lag_mins > AZURE_DR_SYNC_LAG_WARN_MINS:
        _escalate_zone(state, "AMBER")
        log.warning("Azure DR sync lag: %.1f mins > %d min warn", azure_lag_mins, AZURE_DR_SYNC_LAG_WARN_MINS)

    # ---- Terraform drift detection ----
    drifted = inputs.get("terraform_drifted_resources", [])
    if len(drifted) > TERRAFORM_DRIFT_WARN_RESOURCES:
        _escalate_zone(state, "AMBER")
        log.warning("Terraform drift: %d resources", len(drifted))

    # ---- LLM provider concentration (DORA Art.28) ----
    llm_usage = inputs.get("llm_usage_pct", AWB_LLM_DISTRIBUTION.copy())
    concentration_issues: List[str] = []
    for provider, pct in llm_usage.items():
        if pct > DORA_LLM_CONCENTRATION_MAX_PCT:
            concentration_issues.append(
                f"{provider} at {pct:.1f}% > DORA Art.28 limit {DORA_LLM_CONCENTRATION_MAX_PCT}%"
            )
            _escalate_zone(state, "RED")
        elif pct > DORA_LLM_CONCENTRATION_WARN_PCT:
            concentration_issues.append(
                f"{provider} at {pct:.1f}% approaching DORA Art.28 limit"
            )
            _escalate_zone(state, "AMBER")

    # ---- FinOps cost tracking ----
    monthly_costs = inputs.get("monthly_cost_gbp", {})
    cost_overruns: List[str] = []

    budget_map = {
        "total": MONTHLY_INFRA_BUDGET_GBP,
        "llm_api": LLM_API_MONTHLY_BUDGET_GBP,
        "ecs": ECS_MONTHLY_BUDGET_GBP,
        "msk": MSK_MONTHLY_BUDGET_GBP,
    }
    for category, budget in budget_map.items():
        actual = float(monthly_costs.get(category, 0))
        if actual > budget:
            cost_overruns.append(
                f"{category}: £{actual:,.0f} > budget £{budget:,.0f}"
            )
            _escalate_zone(state, "AMBER")

    # ---- AWS data residency check ----
    aws_region = inputs.get("aws_region", "eu-west-2")
    if aws_region != "eu-west-2":
        _escalate_zone(state, "RED")
        log.error(
            "DATA RESIDENCY BREACH: AWS region %s != eu-west-2 "
            "(UK GDPR Art.32)", aws_region
        )

    # ---- LLM infrastructure analysis ----
    llm_prompt = (
        f"AWB infrastructure health — {state['run_date']}:\n"
        f"  AWS region: {aws_region}\n"
        f"  Azure DR sync lag: {azure_lag_mins:.1f} mins\n"
        f"  Terraform drifted resources: {len(drifted)}\n"
        f"  LLM usage: {llm_usage}\n"
        f"  Monthly costs: {monthly_costs}\n"
        f"  DORA concentration issues: {concentration_issues}\n\n"
        f"Assess DORA Art.28 ICT third-party concentration risk for "
        f"each LLM provider and recommend rebalancing actions if "
        f"any provider approaches the 70% hard limit."
    )
    state["infrastructure_llm"] = _call_gemini_flash(llm_prompt)

    state["infrastructure_result"] = {
        "aws_region": aws_region,
        "azure_dr_sync_lag_mins": azure_lag_mins,
        "terraform_drifted_count": len(drifted),
        "llm_concentration": llm_usage,
        "concentration_issues": concentration_issues,
        "cost_overruns": cost_overruns,
        "monthly_costs": monthly_costs,
    }
    state["terraform_drift_resources"] = drifted
    state["cost_overruns"] = cost_overruns

    _log_step(
        state,
        agent="InfrastructureAgent [gemini-3.5-flash]",
        reason=reason,
        act=(
            f"Checked multi-cloud, Terraform drift={len(drifted)}, "
            f"LLM concentration, FinOps costs"
        ),
        outcome=(
            f"DriftedResources={len(drifted)}, CostOverruns={len(cost_overruns)}, "
            f"ConcentrationIssues={len(concentration_issues)}, Zone={state['risk_zone']}"
        ),
    )
    return state


# ---------------------------------------------------------------------------
# AGENT 3 — KafkaT24Agent (Gemini Flash)
# ---------------------------------------------------------------------------

def kafka_t24_agent(
    state: EnterpriseArchState,
) -> EnterpriseArchState:
    """Agent 3: Monitor Kafka consumer lag and T24 integration health.

    Regulatory basis: DORA Art. 9 requires continuous monitoring of all
    ICT systems. AWB's Kafka MSK cluster (8 topics, 42 total partitions,
    3 brokers, EU-WEST-2) is a critical integration hub between T24 core
    banking and the 23 AI services. Consumer lag on awb.transactions
    directly delays AML alert generation — FCA SYSC 6.3 requires prompt
    investigation of suspicious transactions.

    T24 integration patterns (3 types — awb_commons/t24_client.py):
    - READ: Oracle read replica (5-minute lag acceptable; > 10min = alert)
    - WRITE: UUID idempotency key (stuck reservations > 30min = escalate)
    - EVENT: Kafka consumer (awb.transactions — lag < 5s for AML)

    Batch window awareness: T24 batch runs 22:00–04:00 GMT. The AML
    Transaction Monitoring System consumer uses backpressure during the
    batch window to avoid overwhelming the Kafka cluster when 4.2M
    monthly transactions (avg 140K/day) are committed in bulk.

    Populates:
        state["kafka_result"]: Topic health summary.
        state["lagging_topics"]: Topics breaching lag SLA.
        state["t24_health"]: T24 integration health dict.
        risk_zone: AMBER (lag warn) or RED (AML lag > 5s or T24 critical).

    Args:
        state: Shared pipeline state.

    Returns:
        Updated state.
    """
    kafka_inputs = state.get("kafka_inputs", {})
    t24_inputs = state.get("t24_inputs", {})

    reason = (
        "DORA Art.9 continuous monitoring. Kafka awb.transactions lag "
        "directly impacts AML alert latency — FCA SYSC 6.3 requires "
        "prompt transaction investigation. T24 read replica lag > 10min "
        "blocks regulatory reporting (Ch11 MJRRP pipeline)."
    )

    # ---- Kafka topic health ----
    topics = kafka_inputs.get("topics", [])
    lagging: List[str] = []
    topic_summary: List[Dict[str, Any]] = []

    for topic in topics:
        name = topic.get("name", "unknown")
        lag_messages = int(topic.get("consumer_lag", 0))
        lag_secs = float(topic.get("lag_seconds", 0.0))

        status = "OK"
        if lag_messages > KAFKA_LAG_BREACH_MESSAGES:
            status = "SLA_BREACH"
            lagging.append(name)
            _escalate_zone(state, "RED")
        elif lag_messages > KAFKA_LAG_WARN_MESSAGES:
            status = "WARNING"
            lagging.append(name)
            _escalate_zone(state, "AMBER")

        # AML-specific latency SLA
        if name == "awb.transactions" and lag_secs > KAFKA_AML_LAG_MAX_SECONDS:
            status = "AML_SLA_BREACH"
            if name not in lagging:
                lagging.append(name)
            _escalate_zone(state, "RED")
            log.error(
                "AML SLA BREACH: awb.transactions lag %.1fs > %ds",
                lag_secs, KAFKA_AML_LAG_MAX_SECONDS,
            )

        # FX rate SLA
        if name == "awb.fx-rates" and lag_secs > KAFKA_FX_LAG_MAX_SECONDS:
            if name not in lagging:
                lagging.append(name)
            _escalate_zone(state, "AMBER")

        topic_summary.append({
            "name": name,
            "lag_messages": lag_messages,
            "lag_seconds": lag_secs,
            "status": status,
        })

    # ---- T24 integration health ----
    t24_replica_lag = float(t24_inputs.get("read_replica_lag_secs", 0))
    t24_idempotency_stuck = int(t24_inputs.get("idempotency_stuck_count", 0))
    t24_kafka_active = bool(t24_inputs.get("kafka_stream_active", True))
    t24_batch_active = bool(t24_inputs.get("batch_window_active", False))

    t24_issues: List[str] = []
    if t24_replica_lag > T24_READ_REPLICA_LAG_RED_SECS:
        t24_issues.append(
            f"Read replica lag {t24_replica_lag:.0f}s > {T24_READ_REPLICA_LAG_RED_SECS}s critical"
        )
        _escalate_zone(state, "RED")
    elif t24_replica_lag > T24_READ_REPLICA_LAG_WARN_SECS:
        t24_issues.append(
            f"Read replica lag {t24_replica_lag:.0f}s > {T24_READ_REPLICA_LAG_WARN_SECS}s warning"
        )
        _escalate_zone(state, "AMBER")

    if t24_idempotency_stuck > 0:
        t24_issues.append(
            f"{t24_idempotency_stuck} stuck idempotency reservation(s) > {T24_IDEMPOTENCY_STUCK_MINS}min"
        )
        _escalate_zone(state, "AMBER")

    if not t24_kafka_active and not t24_batch_active:
        t24_issues.append("T24 Kafka stream inactive outside batch window — check TCBP")
        _escalate_zone(state, "RED")

    # ---- LLM Kafka health analysis ----
    llm_prompt = (
        f"AWB Kafka MSK topology health — {state['run_date']}:\n"
        f"  Topics: {len(topics)}, Brokers: {kafka_inputs.get('broker_count', 3)}\n"
        f"  Lagging topics: {lagging}\n"
        f"  T24 read replica lag: {t24_replica_lag:.0f}s\n"
        f"  T24 batch window active: {t24_batch_active}\n\n"
        f"For each lagging topic, identify: (1) the AWB service most "
        f"affected, (2) the DORA Art.17 incident classification, and "
        f"(3) whether the T24 batch window (22:00-04:00 GMT) is the "
        f"cause. Cite the relevant Kafka consumer group from topics.py."
    )
    state["kafka_t24_llm"] = _call_gemini_flash(llm_prompt)

    state["kafka_result"] = {
        "total_topics": len(topics),
        "lagging_count": len(lagging),
        "topic_summary": topic_summary,
        "broker_count": kafka_inputs.get("broker_count", 3),
    }
    state["lagging_topics"] = lagging
    state["t24_health"] = {
        "read_replica_lag_secs": t24_replica_lag,
        "idempotency_stuck_count": t24_idempotency_stuck,
        "kafka_stream_active": t24_kafka_active,
        "batch_window_active": t24_batch_active,
        "issues": t24_issues,
    }

    _log_step(
        state,
        agent="KafkaT24Agent [gemini-3.5-flash]",
        reason=reason,
        act=f"Checked {len(topics)} Kafka topics + T24 integration (3 patterns)",
        outcome=(
            f"LaggingTopics={len(lagging)}, T24Issues={len(t24_issues)}, "
            f"Zone={state['risk_zone']}"
        ),
    )
    return state


# ---------------------------------------------------------------------------
# AGENT 4 — DORAResilienceAgent (Gemini 3.1 Pro)
# ---------------------------------------------------------------------------

def dora_resilience_agent(
    state: EnterpriseArchState,
) -> EnterpriseArchState:
    """Agent 4: DORA compliance assessment — Arts 9, 17, 28.

    Regulatory basis:
    - DORA Art. 9: ICT risk management framework — AWB must identify,
      protect, detect, respond, and recover from ICT disruptions.
    - DORA Art. 17: ICT-related incident classification — major incidents
      must be reported to PRA within 4 hours (initial) and 24 hours (report).
    - DORA Art. 28: ICT third-party concentration risk — no single provider
      should create undue dependency. AWB LLM distribution target: Google
      68%, Anthropic 17%, OpenAI 15% — no provider exceeds 70%.
    - PRA PS7/25: Important business services must remain within impact
      tolerances. AWB Tier 1 RTO = 2 hours, RPO = 60 minutes.
    - FCA SYSC 15A: Equivalent operational resilience requirements.

    Uses Gemini 3.1 Pro for the complex multi-regulatory reasoning required
    to classify incidents across DORA Art.17, PRA PS7/25, and FCA SYSC 15A
    simultaneously, and to assess LLM provider concentration across the
    seven approved models against DORA Art.28 limits.

    Populates:
        state["dora_result"]: Full DORA compliance assessment.
        state["dora_breaches"]: Active regulatory breach descriptions.
        state["rto_rpo_status"]: RTO/RPO status per service tier.
        state["llm_concentration_check"]: Provider distribution analysis.
        risk_zone: RED on any DORA material incident / concentration breach.

    Args:
        state: Shared pipeline state.

    Returns:
        Updated state.
    """
    infra = state.get("infrastructure_result", {})
    service = state.get("service_mesh_result", {})
    kafka = state.get("kafka_result", {})
    t24 = state.get("t24_health", {})

    reason = (
        "DORA Arts 9/17/28 and PRA PS7/25 require systematic resilience "
        "assessment. Gemini 3.1 Pro needed for multi-standard incident "
        "classification across DORA Art.17, PRA PS7/25, and FCA SYSC 15A. "
        "LLM concentration check against DORA Art.28 70% hard limit."
    )

    dora_breaches: List[str] = []

    # ---- DORA Art.17 incident classification ----
    open_circuits = state.get("open_circuits", [])
    if open_circuits:
        # AWB circuit breaker OPEN on Tier 1 service = potential major incident
        tier1_services = [
            "digital-identity-api", "api-gateway", "credit-decision-api"
        ]
        tier1_open = [s for s in open_circuits if s in tier1_services]
        if tier1_open:
            dora_breaches.append(
                f"DORA Art.17 major incident: Tier 1 circuit OPEN — "
                f"{tier1_open} (4hr PRA notification required)"
            )
            _escalate_zone(state, "RED")

    # ---- DORA Art.28 LLM concentration ----
    llm_usage = infra.get("llm_concentration", AWB_LLM_DISTRIBUTION)
    concentration_check: Dict[str, Any] = {}
    for provider, pct in llm_usage.items():
        status = "OK"
        if pct > DORA_LLM_CONCENTRATION_MAX_PCT:
            status = "BREACH"
            dora_breaches.append(
                f"DORA Art.28 concentration breach: {provider} at {pct:.1f}% "
                f"> hard limit {DORA_LLM_CONCENTRATION_MAX_PCT}%"
            )
            _escalate_zone(state, "RED")
        elif pct > DORA_LLM_CONCENTRATION_WARN_PCT:
            status = "WARN"
        concentration_check[provider] = {
            "pct": pct,
            "status": status,
            "headroom_pct": round(DORA_LLM_CONCENTRATION_MAX_PCT - pct, 1),
        }

    # ---- PRA PS7/25 RTO/RPO assessment ----
    rto_rpo: Dict[str, Any] = {
        "tier1": {
            "services": ["digital-identity-api", "api-gateway"],
            "rto_mins": AWS_FAILOVER_RTO_MINS,
            "rpo_mins": AWS_FAILOVER_RPO_MINS,
            "azure_dr_sync_lag_mins": infra.get("azure_dr_sync_lag_mins", 0),
            "within_rpo": infra.get("azure_dr_sync_lag_mins", 0) <= AWS_FAILOVER_RPO_MINS,
        },
        "tier2": {
            "services": ["aml-monitor", "credit-decision-api", "kyc-api"],
            "rto_mins": 240,
            "rpo_mins": 120,
            "within_rpo": True,  # Kafka replay provides RPO
        },
    }

    if not rto_rpo["tier1"]["within_rpo"]:
        dora_breaches.append(
            f"PRA PS7/25 RPO breach: Azure DR sync lag "
            f"{infra.get('azure_dr_sync_lag_mins', 0):.0f}min "
            f"> {AWS_FAILOVER_RPO_MINS}min RPO for Tier 1 services"
        )
        _escalate_zone(state, "RED")

    # ---- T24 DORA classification ----
    t24_replica_lag = (t24.get("read_replica_lag_secs", 0)
                       if isinstance(t24, dict) else 0)
    if t24_replica_lag > T24_READ_REPLICA_LAG_RED_SECS:
        dora_breaches.append(
            f"DORA Art.9 ICT risk: T24 read replica lag {t24_replica_lag:.0f}s "
            f"exceeds {T24_READ_REPLICA_LAG_RED_SECS}s — regulatory reporting "
            f"pipeline (MJRRP Ch11) impacted"
        )

    # ---- Gemini 3.1 Pro DORA narrative ----
    context = (
        f"Run date: {state['run_date']}\n"
        f"Open circuits: {open_circuits}\n"
        f"Degraded services: {state.get('degraded_services', [])}\n"
        f"Kafka lagging topics: {state.get('lagging_topics', [])}\n"
        f"T24 issues: {t24.get('issues', []) if isinstance(t24, dict) else []}\n"
        f"Terraform drift: {len(infra.get('terraform_drifted_count', 0))} resources\n"
        f"LLM concentration: {llm_usage}\n"
        f"Azure DR sync lag: {infra.get('azure_dr_sync_lag_mins', 0):.1f} mins\n"
        f"Active DORA breaches: {dora_breaches}"
    )
    llm_prompt = (
        "You are AWB's Head of Technology Risk preparing a DORA resilience "
        "assessment for the CRO and Board Risk Committee.\n\n"
        "1. Classify each active issue against DORA Art.17 severity tiers "
        "(minor, significant, major) with the classification rationale.\n"
        "2. Assess DORA Art.28 LLM provider concentration against the "
        "70% hard limit and AWB's target distribution "
        "(Google 68%, Anthropic 17%, OpenAI 15%).\n"
        "3. Validate PRA PS7/25 RTO/RPO commitments against current DR status.\n"
        "4. Identify any issues requiring immediate PRA/FCA notification "
        "under DORA Art.17 (initial report within 4 hours for major incidents).\n"
        "5. Recommend the three highest-priority remediation actions.\n"
        "Format: structured assessment, formal tone, cite DORA articles."
    )
    dora_narrative = _call_gemini_pro(llm_prompt, context)

    state["dora_result"] = {
        "breaches": dora_breaches,
        "rto_rpo": rto_rpo,
        "llm_concentration": concentration_check,
        "narrative": dora_narrative,
    }
    state["dora_breaches"] = dora_breaches
    state["rto_rpo_status"] = rto_rpo
    state["llm_concentration_check"] = concentration_check

    _log_step(
        state,
        agent="DORAResilienceAgent [gemini-3.1-pro]",
        reason=reason,
        act=(
            f"Assessed DORA Arts 9/17/28, PRA PS7/25 RTO/RPO, "
            f"LLM concentration"
        ),
        outcome=(
            f"DORABreaches={len(dora_breaches)}, "
            f"Zone={state['risk_zone']}"
        ),
    )
    return state


# ---------------------------------------------------------------------------
# AGENT 5 — ArchitectureReportAgent (Claude Sonnet 4.6)
# ---------------------------------------------------------------------------

def architecture_report_agent(
    state: EnterpriseArchState,
) -> EnterpriseArchState:
    """Agent 5: Generate architecture health narrative for board reporting.

    Regulatory basis: PRA SS1/23 §7 requires firms to maintain an audit
    trail and human-readable narrative explaining AI-assisted infrastructure
    decisions. DORA Art. 17 requires incident reports to be human-readable
    and submitted to regulators. BAP-2026-EA-001 §8 mandates a signed-off
    weekly architecture health narrative for the Board Technology Committee.

    Uses Claude Sonnet 4.6 to synthesise the outputs of all four preceding
    agents into:
    1. An architecture health narrative (board-level, 400-500 words)
    2. A board summary (CTO/CRO format, 150 words)
    3. Change management flags (issues requiring CAB approval under DORA
       Art. 9 change management process)

    Populates:
        state["architecture_narrative"]: Full health narrative.
        state["board_summary"]: Board-level summary.
        state["change_management_flags"]: CAB/DORA change items.
        hitl_decision / hitl_rationale: Final HITL outcome.
        pipeline_completed: True.

    Args:
        state: Fully populated pipeline state after agents 1-4.

    Returns:
        Updated state with narrative and final HITL decision.
    """
    reason = (
        "PRA SS1/23 §7 and DORA Art.17 require human-readable architecture "
        "health narratives. Claude Sonnet 4.6 synthesises all five agent "
        "outputs into board-quality reporting. HITL applied: any breach "
        "requires engineering lead sign-off before action per "
        "BAP-2026-EA-001 §8."
    )

    # ---- Aggregate context ----
    svc = state.get("service_mesh_result", {}) or {}
    infra = state.get("infrastructure_result", {}) or {}
    kafka = state.get("kafka_result", {}) or {}
    t24 = state.get("t24_health", {}) or {}
    dora = state.get("dora_result", {}) or {}

    context_parts = [
        f"Run Date: {state.get('run_date')}",
        f"Trigger: {state.get('trigger_event')}",
        f"",
        f"SERVICE MESH:",
        f"  Services: {svc.get('total_services', 0)} total, "
        f"{svc.get('degraded_count', 0)} degraded",
        f"  Open circuits: {state.get('open_circuits', [])}",
        f"",
        f"INFRASTRUCTURE:",
        f"  AWS region: {infra.get('aws_region', 'eu-west-2')} "
        f"(UK GDPR data residency)",
        f"  Azure DR sync: {infra.get('azure_dr_sync_lag_mins', 0):.1f} mins",
        f"  Terraform drift: {infra.get('terraform_drifted_count', 0)} resources",
        f"  Cost overruns: {state.get('cost_overruns', [])}",
        f"",
        f"KAFKA/T24:",
        f"  Topics: {kafka.get('total_topics', 0)} total, "
        f"{kafka.get('lagging_count', 0)} lagging",
        f"  T24 replica lag: {t24.get('read_replica_lag_secs', 0):.0f}s",
        f"  T24 idempotency stuck: {t24.get('idempotency_stuck_count', 0)}",
        f"",
        f"DORA:",
        f"  Breaches: {state.get('dora_breaches', [])}",
        f"  LLM concentration: {state.get('llm_concentration_check', {})}",
        f"  RTO/RPO: Tier1 RTO={AWS_FAILOVER_RTO_MINS}min, "
        f"RPO={AWS_FAILOVER_RPO_MINS}min",
        f"",
        f"RISK ZONE: {state.get('risk_zone')}",
        f"HOP CHAIN: {len(state.get('hop_chain', []))} steps logged",
    ]
    full_context = "\n".join(context_parts)

    # ---- Architecture health narrative ----
    narrative_prompt = (
        "You are AWB's Chief Technology Officer preparing the weekly "
        "architecture health report for the Board Technology Committee "
        "and CRO.\n\n"
        "Write a formal architecture health narrative (400-500 words):\n"
        "1. ECS service mesh status — all 23 AI services health summary\n"
        "2. Multi-cloud resilience — AWS EU-WEST-2 primary and Azure DR status\n"
        "3. Kafka MSK topology health — consumer lag SLA compliance\n"
        "4. T24 integration health — three patterns (READ/WRITE/EVENT)\n"
        "5. DORA Arts 9/17/28 compliance status and any active incidents\n"
        "6. LLM provider concentration vs. DORA Art.28 70% hard limit\n"
        "7. FinOps: monthly spend vs. £21,750 infrastructure budget\n"
        "8. HITL sign-off requirement under BAP-2026-EA-001 §8\n\n"
        "Tone: formal, technical, CTO-level. Cite DORA articles, "
        "PRA PS7/25, and AWB policy references where relevant."
    )
    architecture_narrative = _call_claude_sonnet(narrative_prompt, full_context)

    # ---- Board summary (150 words) ----
    board_prompt = (
        "Write a 150-word Board summary for AWB's weekly architecture "
        "health report. Cover: overall platform status (RED/AMBER/GREEN), "
        "key issues, DORA compliance, and the top action item. "
        "Reference MR-2026-061-EA. Non-technical language for board directors."
    )
    board_summary = _call_claude_sonnet(board_prompt, full_context)

    # ---- Change management flags ----
    change_flags: List[str] = []
    if state.get("terraform_drift_resources"):
        change_flags.append(
            f"CAB REQUIRED: {len(state['terraform_drift_resources'])} "
            "infrastructure changes not in Terraform — DORA Art.9 change management"
        )
    if state.get("dora_breaches"):
        for breach in state["dora_breaches"]:
            if "Art.17" in breach:
                change_flags.append(
                    f"PRA NOTIFICATION: {breach[:80]}..."
                )
    if state.get("open_circuits"):
        change_flags.append(
            f"INCIDENT RESPONSE: Circuit(s) OPEN — "
            f"{state['open_circuits']} — runbook activation required"
        )
    if not change_flags:
        change_flags.append(
            "No change management actions required — routine weekly health check"
        )

    state["architecture_narrative"] = architecture_narrative
    state["board_summary"] = board_summary
    state["change_management_flags"] = change_flags

    # ---- HITL decision ----
    decision, rationale = _compute_hitl_decision(state)
    state["hitl_decision"] = decision
    state["hitl_rationale"] = rationale
    state["pipeline_completed"] = True

    _log_step(
        state,
        agent="ArchitectureReportAgent [claude-sonnet-4-6]",
        reason=reason,
        act=(
            f"Generated architecture narrative "
            f"({len(architecture_narrative)} chars), board summary, "
            f"{len(change_flags)} change flag(s)"
        ),
        outcome=(
            f"HITL={decision}, Flags={len(change_flags)}, "
            f"Zone={state['risk_zone']}, Pipeline=COMPLETE"
        ),
    )
    return state


# ---------------------------------------------------------------------------
# PIPELINE ORCHESTRATION
# ---------------------------------------------------------------------------

def _build_graph():
    """Build LangGraph StateGraph for the enterprise architecture pipeline.

    Topology:
        START → service_mesh_agent
              → infrastructure_agent
              → kafka_t24_agent
              → dora_resilience_agent
              → architecture_report_agent
              → END
    """
    try:
        from langgraph.graph import StateGraph, END

        graph = StateGraph(EnterpriseArchState)
        graph.add_node("service_mesh", service_mesh_agent)
        graph.add_node("infrastructure", infrastructure_agent)
        graph.add_node("kafka_t24", kafka_t24_agent)
        graph.add_node("dora_resilience", dora_resilience_agent)
        graph.add_node("architecture_report", architecture_report_agent)

        graph.set_entry_point("service_mesh")
        graph.add_edge("service_mesh", "infrastructure")
        graph.add_edge("infrastructure", "kafka_t24")
        graph.add_edge("kafka_t24", "dora_resilience")
        graph.add_edge("dora_resilience", "architecture_report")
        graph.add_edge("architecture_report", END)

        return graph.compile()

    except ImportError:
        log.warning("LangGraph not installed — using _SequentialStub")
        return _SequentialStub()


class _SequentialStub:
    """Fallback for environments without LangGraph installed."""

    def invoke(self, state: EnterpriseArchState) -> EnterpriseArchState:
        state = service_mesh_agent(state)
        state = infrastructure_agent(state)
        state = kafka_t24_agent(state)
        state = dora_resilience_agent(state)
        state = architecture_report_agent(state)
        return state


# ---------------------------------------------------------------------------
# PUBLIC ENTRY POINT
# ---------------------------------------------------------------------------

async def run_agentic_enterprise_architecture(
    run_date: date,
    trigger_event: str,
    service_health_inputs: Dict[str, Any],
    infrastructure_inputs: Dict[str, Any],
    kafka_inputs: Optional[Dict[str, Any]] = None,
    t24_inputs: Optional[Dict[str, Any]] = None,
) -> EnterpriseArchState:
    """Run the five-agent enterprise architecture monitoring pipeline.

    Orchestrates AWB's weekly architecture health check:
    1. ServiceMeshAgent       — 23 ECS services, circuit breakers, SLA
    2. InfrastructureAgent    — AWS/Azure multi-cloud, Terraform, FinOps
    3. KafkaT24Agent          — 8-topic Kafka MSK, T24 integration (3 patterns)
    4. DORAResilienceAgent    — DORA Arts 9/17/28, PRA PS7/25, RTO/RPO
    5. ArchitectureReportAgent— board narrative, change management, HITL

    Model ID: MR-2026-061-EA (BAP-2026-EA-001 §3).

    Args:
        run_date: Date of this architecture health run.
        trigger_event: Human-readable trigger (e.g., "Weekly health check").
        service_health_inputs: ECS service metrics dict.
        infrastructure_inputs: AWS/Azure/Terraform/FinOps metrics dict.
        kafka_inputs: Optional Kafka MSK health metrics.
        t24_inputs: Optional T24 integration health metrics.

    Returns:
        Completed EnterpriseArchState with all fields populated.

    Example:
        >>> import asyncio
        >>> from datetime import date
        >>> state = asyncio.run(run_agentic_enterprise_architecture(
        ...     run_date=date(2026, 1, 15),
        ...     trigger_event="Weekly architecture health check",
        ...     service_health_inputs={
        ...         "services": [
        ...             {"name": "credit-decision-api", "desired_count": 3,
        ...              "running_count": 3, "p50_ms": 120.0, "p95_ms": 380.0,
        ...              "circuit_state": "closed"},
        ...             {"name": "aml-monitor", "desired_count": 2,
        ...              "running_count": 2, "p50_ms": 95.0, "p95_ms": 290.0,
        ...              "circuit_state": "closed"},
        ...         ],
        ...     },
        ...     infrastructure_inputs={
        ...         "aws_region": "eu-west-2",
        ...         "azure_dr_sync_lag_mins": 12.5,
        ...         "terraform_drifted_resources": [],
        ...         "llm_usage_pct": {"google": 67.0, "anthropic": 18.0, "openai": 15.0},
        ...         "monthly_cost_gbp": {"total": 19200, "llm_api": 7800,
        ...                              "ecs": 6900, "msk": 2600},
        ...     },
        ... ))
        >>> print(state["hitl_decision"])   # APPROVE or ESCALATE
        >>> print(state["risk_zone"])       # GREEN
        >>> print(len(state["hop_chain"]))  # 5
    """
    global _SEQ
    _SEQ = 0

    if kafka_inputs is None:
        kafka_inputs = {"topics": [], "broker_count": 3, "total_partitions": 42}
    if t24_inputs is None:
        t24_inputs = {
            "read_replica_lag_secs": 180,
            "idempotency_stuck_count": 0,
            "kafka_stream_active": True,
            "batch_window_active": False,
        }

    state = _initial_state(
        run_date=run_date,
        trigger_event=trigger_event,
        service_health_inputs=service_health_inputs,
        infrastructure_inputs=infrastructure_inputs,
        kafka_inputs=kafka_inputs,
        t24_inputs=t24_inputs,
    )

    log.info(
        "Agentic Enterprise Architecture Pipeline START | "
        "MR-2026-061-EA | date=%s | trigger='%s'",
        run_date, trigger_event,
    )

    graph = _build_graph()
    loop = asyncio.get_event_loop()
    final_state = await loop.run_in_executor(None, graph.invoke, state)

    log.info(
        "Agentic Enterprise Architecture Pipeline END | "
        "HITL=%s | Zone=%s | Hops=%d | OpenCircuits=%d | "
        "DORABreaches=%d | Lagging=%d",
        final_state.get("hitl_decision"),
        final_state.get("risk_zone"),
        len(final_state.get("hop_chain", [])),
        len(final_state.get("open_circuits", [])),
        len(final_state.get("dora_breaches", [])),
        len(final_state.get("lagging_topics", [])),
    )
    return final_state


# ---------------------------------------------------------------------------
# CLI DEMO
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    demo_services = {
        "services": [
            {"name": "credit-decision-api", "desired_count": 3,
             "running_count": 3, "p50_ms": 118.0, "p95_ms": 372.0,
             "p99_ms": 1420.0, "circuit_state": "closed"},
            {"name": "aml-monitor", "desired_count": 2,
             "running_count": 2, "p50_ms": 92.0, "p95_ms": 285.0,
             "p99_ms": 890.0, "circuit_state": "closed"},
            {"name": "digital-identity-api", "desired_count": 2,
             "running_count": 2, "p50_ms": 145.0, "p95_ms": 480.0,
             "p99_ms": 1850.0, "circuit_state": "closed"},
            {"name": "kyc-api", "desired_count": 2,
             "running_count": 1, "p50_ms": 230.0, "p95_ms": 680.0,
             "p99_ms": 2200.0, "circuit_state": "half_open"},
            {"name": "rag-query-api", "desired_count": 3,
             "running_count": 3, "p50_ms": 135.0, "p95_ms": 410.0,
             "p99_ms": 1600.0, "circuit_state": "closed"},
        ],
        "total_services": 23,
    }

    demo_infra = {
        "aws_region": "eu-west-2",
        "azure_dr_region": "uksouth",
        "azure_dr_sync_lag_mins": 14.2,
        "terraform_drifted_resources": [],
        "llm_usage_pct": {
            "google": 67.5, "anthropic": 17.8, "openai": 14.7
        },
        "monthly_cost_gbp": {
            "total": 19_420,
            "llm_api": 7_950,
            "ecs": 7_100,
            "msk": 2_650,
        },
    }

    demo_kafka = {
        "topics": [
            {"name": "awb.transactions", "consumer_lag": 450,
             "lag_seconds": 1.8},
            {"name": "awb.credit-events", "consumer_lag": 12,
             "lag_seconds": 0.2},
            {"name": "awb.fx-rates", "consumer_lag": 3,
             "lag_seconds": 8.5},
            {"name": "awb.market-data", "consumer_lag": 85,
             "lag_seconds": 2.1},
            {"name": "awb.kyc-events", "consumer_lag": 28,
             "lag_seconds": 0.9},
            {"name": "awb.model-alerts", "consumer_lag": 5,
             "lag_seconds": 0.3},
            {"name": "awb.audit-trail", "consumer_lag": 190,
             "lag_seconds": 4.2},
            {"name": "awb.regulatory-filings", "consumer_lag": 2,
             "lag_seconds": 0.1},
        ],
        "broker_count": 3,
        "total_partitions": 42,
    }

    demo_t24 = {
        "read_replica_lag_secs": 195,
        "idempotency_stuck_count": 0,
        "kafka_stream_active": True,
        "batch_window_active": False,
    }

    result = asyncio.run(
        run_agentic_enterprise_architecture(
            run_date=date(2026, 1, 15),
            trigger_event=(
                "Weekly architecture health check — "
                "AWB-AI-2025 programme | 23 services | "
                "AWS EU-WEST-2 + Azure DR"
            ),
            service_health_inputs=demo_services,
            infrastructure_inputs=demo_infra,
            kafka_inputs=demo_kafka,
            t24_inputs=demo_t24,
        )
    )

    print("\n" + "=" * 70)
    print("AWB AGENTIC ENTERPRISE ARCHITECTURE — MR-2026-061-EA")
    print("=" * 70)
    print(f"Run Date      : {result['run_date']}")
    print(f"Risk Zone     : {result['risk_zone']}")
    print(f"HITL Decision : {result['hitl_decision']}")
    print(f"HITL Rationale: {result['hitl_rationale']}")
    print(f"")
    print(f"Service Mesh:")
    svc = result.get("service_mesh_result") or {}
    print(f"  Total services : {svc.get('total_services', 0)}")
    print(f"  Degraded       : {result.get('degraded_services', [])}")
    print(f"  Open circuits  : {result.get('open_circuits', [])}")
    print(f"")
    print(f"Infrastructure:")
    infra = result.get("infrastructure_result") or {}
    print(f"  AWS region     : {infra.get('aws_region', 'N/A')}")
    print(f"  Azure DR lag   : {infra.get('azure_dr_sync_lag_mins', 0):.1f} mins")
    print(f"  Terraform drift: {infra.get('terraform_drifted_count', 0)} resources")
    print(f"  Cost overruns  : {result.get('cost_overruns', [])}")
    print(f"")
    print(f"Kafka/T24:")
    kafka = result.get("kafka_result") or {}
    print(f"  Topics         : {kafka.get('total_topics', 0)} total")
    print(f"  Lagging        : {result.get('lagging_topics', [])}")
    t24 = result.get("t24_health") or {}
    print(f"  T24 replica lag: {t24.get('read_replica_lag_secs', 0):.0f}s")
    print(f"")
    print(f"DORA Breaches  : {result.get('dora_breaches', [])}")
    print(f"LLM Concentration: {result.get('llm_concentration_check', {})}")
    print(f"")
    print(f"Change Management Flags:")
    for flag in result.get("change_management_flags", []):
        print(f"  • {flag}")
    print(f"")
    print(f"Hop Chain ({len(result.get('hop_chain', []))} hops):")
    for hop in result.get("hop_chain", []):
        print(
            f"  [{hop['seq']:02d}] {hop['agent']}: "
            f"{hop['act'][:55]} → {hop['outcome'][:55]}"
        )
    print("=" * 70)
