"""
agentic_integrated_platform.py
================================
Chapter 16 — Building an Integrated AI Risk Platform: Complete Implementation
Section 16.3A — Agentic Platform Orchestrator (MR-2026-074-IP)

AWB Avon & Wessex Bank plc
Programme: AWB-AI-2025 | Model Reference: MR-2026-074-IP
PRA SS1/23 Risk Rating: CRITICAL
Regulatory Basis: PRA SS1/23, FCA SYSC 6.3.3R, DORA Art.28, CRR3 Art.316

Agentic Pipeline: Five specialist agents + deterministic HITL gate
  Agent 1 — PlatformHealthAgent       (Gemini 3.5 Flash) — 23-system health rollup
  Agent 2 — CrossDomainRiskAgent      (Gemini 3.5 Flash) — credit/market/liquidity correlation
  Agent 3 — RegulatoryBreachAgent     (Gemini 3.5 Flash) — FCA/PRA/DORA breach detection
  Agent 4 — CapitalAdequacyAgent      (Gemini 3.1 Pro)   — CET1 and RWA scenario modelling
  Agent 5 — PlatformSummaryAgent      (Claude Sonnet 4.6) — CRO/CFO executive narrative

LangGraph topology: START → health → cross_risk → reg_breach → capital → summary → hitl → END
HITL trigger: any RED zone, RegulatoryBreach ≥ MATERIAL, CET1 < 14.5%, or DORA P1 incident

All agents follow ReAct: reason() before act() — enforced via _log_step().
Hop-chain appended to IntegratedPlatformState["hop_chain"] for FCA audit trail.

Constants (live AWB June 2026 values):
  CET1_REGULATORY_MIN_PCT    = 10.5   # CRR3 Art. 92 Pillar 1
  CET1_BUFFER_MIN_PCT        = 13.0   # AWB Board buffer trigger
  CET1_HITL_THRESHOLD_PCT    = 14.5   # HITL escalation
  RWA_WARN_PCT               = 5.0    # 5% RWA increase triggers AMBER
  RWA_BREACH_PCT             = 10.0   # 10% RWA increase triggers RED
  DORA_P1_RTO_MINUTES        = 120    # DORA Art.12 Recovery Time Objective
  DORA_P1_PRA_NOTIFY_HOURS   = 4      # PRA notification SLA
  BCBS239_CONSOLIDATED_MIN   = 90.0   # Platform-wide BCBS 239 consolidated score
  MODEL_HEALTH_WARN_PCT      = 85.0   # <85% systems GREEN → AMBER
  MODEL_HEALTH_RED_PCT       = 70.0   # <70% systems GREEN → RED
  CROSS_CORR_WARN_THRESHOLD  = 0.65   # Cross-domain risk correlation AMBER
  CROSS_CORR_RED_THRESHOLD   = 0.80   # Cross-domain risk correlation RED
  VaR_BREACH_MULTIPLIER      = 1.10   # VaR 10% above limit → AMBER
  LIQUIDITY_LCR_MIN_PCT      = 100.0  # LCR minimum (CRR3)
  LIQUIDITY_LCR_BUFFER_PCT   = 120.0  # AWB internal buffer
  PSI_PLATFORM_WARN          = 0.10   # Platform-wide model drift AMBER
  DORA_CONCENTRATION_MAX_PCT = 70.0   # No single cloud provider > 70%
  TOKEN_BUDGET_PER_RUN       = 50_000
  COST_BUDGET_GBP_PER_RUN    = 2.50
"""

# ── MCP Runtime Data Access (Section 3.9B) ──────────────────────────────────
# The PlatformHealthAgent reads live model status via MCPModelInventoryServer:
#   from credit_agent.mcp_servers import AWBMCPServerRegistry
#   registry = AWBMCPServerRegistry.default()
#   status = registry.call_tool("model_lookup", {"model_ref": "MR-2026-041"}, "MR-2026-074-IP")
# MCPBloombergServer supplies VaR inputs; MCPFCAHandbookServer checks breach rules.
# AWBMCPServerRegistry.default() creates: MCPFCAHandbookServer, MCPBloombergServer, MCPModelInventoryServer
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("awb.agentic_integrated_platform")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CET1_REGULATORY_MIN_PCT = 10.5
CET1_BUFFER_MIN_PCT = 13.0
CET1_HITL_THRESHOLD_PCT = 14.5
RWA_WARN_PCT = 5.0
RWA_BREACH_PCT = 10.0
DORA_P1_RTO_MINUTES = 120
DORA_P1_PRA_NOTIFY_HOURS = 4
BCBS239_CONSOLIDATED_MIN = 90.0
MODEL_HEALTH_WARN_PCT = 85.0
MODEL_HEALTH_RED_PCT = 70.0
CROSS_CORR_WARN_THRESHOLD = 0.65
CROSS_CORR_RED_THRESHOLD = 0.80
VaR_BREACH_MULTIPLIER = 1.10
LIQUIDITY_LCR_MIN_PCT = 100.0
LIQUIDITY_LCR_BUFFER_PCT = 120.0
PSI_PLATFORM_WARN = 0.10
DORA_CONCENTRATION_MAX_PCT = 70.0
TOKEN_BUDGET_PER_RUN = 50_000
COST_BUDGET_GBP_PER_RUN = 2.50

# AWB 23-system registry snapshot (MR references → current health status)
AWB_PLATFORM_SYSTEMS: Dict[str, Dict[str, Any]] = {
    "MR-2026-035": {"name": "Credit RAG Knowledge Base",        "chapter": 4,  "status": "GREEN",  "auc_roc": 0.847, "psi": 0.031},
    "MR-2026-036": {"name": "Credit Decision Agent",            "chapter": 3,  "status": "GREEN",  "auc_roc": 0.891, "psi": 0.018},
    "MR-2026-037": {"name": "IFRS 9 PD Model (LGBM)",           "chapter": 6,  "status": "GREEN",  "auc_roc": 0.923, "psi": 0.042},
    "MR-2026-038": {"name": "IFRS 9 LGD Model",                 "chapter": 6,  "status": "GREEN",  "auc_roc": 0.887, "psi": 0.029},
    "MR-2026-039": {"name": "IFRS 9 EAD Model",                 "chapter": 6,  "status": "GREEN",  "auc_roc": 0.871, "psi": 0.033},
    "MR-2026-040": {"name": "IFRS 9 Staging Engine",            "chapter": 6,  "status": "GREEN",  "auc_roc": 0.912, "psi": 0.021},
    "MR-2026-041": {"name": "Real-Time VaR Engine",             "chapter": 7,  "status": "AMBER",  "auc_roc": None,  "psi": 0.048},
    "MR-2026-042": {"name": "CVA Computation Engine",           "chapter": 7,  "status": "GREEN",  "auc_roc": None,  "psi": 0.027},
    "MR-2026-043": {"name": "Algo Trading Backtester",          "chapter": 7,  "status": "GREEN",  "auc_roc": 0.834, "psi": 0.019},
    "MR-2026-044": {"name": "Operational Risk NLP Classifier",  "chapter": 8,  "status": "GREEN",  "auc_roc": 0.879, "psi": 0.037},
    "MR-2026-045": {"name": "Fraud Detection Model",            "chapter": 8,  "status": "GREEN",  "auc_roc": 0.944, "psi": 0.014},
    "MR-2026-046": {"name": "Basel III SMA Capital Calculator", "chapter": 8,  "status": "GREEN",  "auc_roc": None,  "psi": 0.008},
    "MR-2026-047": {"name": "LCR Forecasting Model",            "chapter": 9,  "status": "GREEN",  "auc_roc": 0.856, "psi": 0.031},
    "MR-2026-048": {"name": "NSFR Optimiser",                   "chapter": 9,  "status": "GREEN",  "auc_roc": None,  "psi": 0.022},
    "MR-2026-049": {"name": "Intraday Liquidity Agent",         "chapter": 9,  "status": "GREEN",  "auc_roc": None,  "psi": 0.016},
    "MR-2026-050": {"name": "Model Validation Orchestrator",    "chapter": 10, "status": "GREEN",  "auc_roc": None,  "psi": 0.009},
    "MR-2026-051": {"name": "COREP/FINREP Automation",          "chapter": 11, "status": "GREEN",  "auc_roc": None,  "psi": 0.005},
    "MR-2026-052": {"name": "Regulatory Change Tracker",        "chapter": 11, "status": "GREEN",  "auc_roc": None,  "psi": 0.011},
    "MR-2026-053": {"name": "Consumer Duty Classifier",         "chapter": 5,  "status": "GREEN",  "auc_roc": 0.883, "psi": 0.028},
    "MR-2026-060-AML": {"name": "AML/KYC Transaction Monitor", "chapter": 12, "status": "GREEN",  "auc_roc": 0.916, "psi": 0.019},
    "MR-2026-061-EA": {"name": "Enterprise Architecture Agent", "chapter": 13, "status": "GREEN",  "auc_roc": None,  "psi": 0.007},
    "MR-2026-062-MLO": {"name": "MLOps/LLMOps Orchestrator",   "chapter": 14, "status": "GREEN",  "auc_roc": None,  "psi": 0.006},
    "MR-2026-063-DI":  {"name": "Data Infrastructure Agent",   "chapter": 15, "status": "GREEN",  "auc_roc": None,  "psi": 0.004},
}

# LLM provider concentration (DORA Art.28 — no single provider > 70%)
AWB_LLM_CONCENTRATION: Dict[str, float] = {
    "google_gemini": 68.0,   # Agents 1-3 across all 16 pipelines
    "anthropic_claude": 17.0, # Agent 5 (summary) across all 16 pipelines
    "openai_gpt4o": 15.0,     # Fallback / specific use cases
}

# AWB June 2026 capital and liquidity snapshot
AWB_CAPITAL_SNAPSHOT = {
    "cet1_pct": 15.2,           # CET1 ratio — above HITL threshold
    "tier1_pct": 16.8,
    "total_capital_pct": 19.1,
    "rwa_gbp_bn": 8.4,
    "rwa_change_pct": 2.3,      # Within WARN threshold (< 5%)
    "lcr_pct": 138.0,           # Above AWB buffer (120%)
    "nsfr_pct": 112.0,
    "leverage_ratio_pct": 5.8,
}


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class RiskZone(str, Enum):
    GREEN = "GREEN"
    AMBER = "AMBER"
    RED = "RED"
    CRITICAL = "CRITICAL"


class HITLDecision(str, Enum):
    APPROVE = "APPROVE"
    ESCALATE = "ESCALATE"
    OVERRIDE = "OVERRIDE"
    PENDING = "PENDING"


class BreachSeverity(str, Enum):
    NONE = "NONE"
    MINOR = "MINOR"
    MATERIAL = "MATERIAL"
    SERIOUS = "SERIOUS"


class DORAIncidentClass(str, Enum):
    P1_MAJOR = "P1_MAJOR"       # PRA notification within 4 hours
    P2_SIGNIFICANT = "P2_SIGNIFICANT"
    P3_MINOR = "P3_MINOR"
    NO_INCIDENT = "NO_INCIDENT"


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class IntegratedPlatformState(dict):
    """
    LangGraph state for the Integrated Platform Orchestrator (MR-2026-074-IP).
    Inherits dict so LangGraph's StateGraph reducer can merge updates.
    """
    pass


def _initial_state(
    run_date: str,
    trigger_event: str,
    platform_inputs: Dict[str, Any],
    capital_inputs: Dict[str, Any],
    risk_inputs: Dict[str, Any],
    regulatory_inputs: Dict[str, Any],
    dora_inputs: Dict[str, Any],
) -> IntegratedPlatformState:
    return IntegratedPlatformState(
        run_id=str(uuid.uuid4()),
        model_ref="MR-2026-074-IP",
        run_date=run_date,
        trigger_event=trigger_event,
        platform_inputs=platform_inputs,
        capital_inputs=capital_inputs,
        risk_inputs=risk_inputs,
        regulatory_inputs=regulatory_inputs,
        dora_inputs=dora_inputs,
        # Platform health
        platform_system_count=0,
        platform_green_count=0,
        platform_amber_count=0,
        platform_red_count=0,
        platform_health_pct=0.0,
        platform_zone=RiskZone.GREEN,
        platform_alerts=[],
        # Cross-domain risk
        credit_market_correlation=0.0,
        market_liquidity_correlation=0.0,
        cross_domain_zone=RiskZone.GREEN,
        cross_domain_findings=[],
        # Regulatory breach
        breach_severity=BreachSeverity.NONE,
        breach_details=[],
        regulatory_zone=RiskZone.GREEN,
        dora_incident_class=DORAIncidentClass.NO_INCIDENT,
        # Capital adequacy
        cet1_pct=0.0,
        cet1_zone=RiskZone.GREEN,
        rwa_change_pct=0.0,
        capital_scenario_findings=[],
        # Aggregate
        overall_zone=RiskZone.GREEN,
        hitl_decision=HITLDecision.PENDING,
        hitl_rationale="",
        executive_summary="",
        cro_briefing="",
        hop_chain=[],
        errors=[],
        tokens_used=0,
        cost_gbp=0.0,
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

_ZONE_RANK = {RiskZone.GREEN: 0, RiskZone.AMBER: 1, RiskZone.RED: 2, RiskZone.CRITICAL: 3}


def _escalate_zone(current: RiskZone, proposed: RiskZone) -> RiskZone:
    """Monotonically upward zone escalation only."""
    return proposed if _ZONE_RANK.get(proposed, 0) > _ZONE_RANK.get(current, 0) else current


def _log_step(
    state: IntegratedPlatformState,
    agent: str,
    reason: str,
    act: str,
    outcome: str,
) -> None:
    """Append a ReAct hop to the state hop_chain for FCA audit trail."""
    state["hop_chain"].append({
        "seq": len(state["hop_chain"]) + 1,
        "agent": agent,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "act": act,
        "outcome": outcome,
    })


def _sha256(data: Any) -> str:
    return hashlib.sha256(json.dumps(data, default=str).encode()).hexdigest()


def _charge_tokens(state: IntegratedPlatformState, tokens: int, cost_gbp: float) -> None:
    state["tokens_used"] = state.get("tokens_used", 0) + tokens
    state["cost_gbp"] = state.get("cost_gbp", 0.0) + cost_gbp
    if state["tokens_used"] > TOKEN_BUDGET_PER_RUN:
        raise RuntimeError(
            f"Token budget exceeded: {state['tokens_used']:,} > {TOKEN_BUDGET_PER_RUN:,}"
        )
    if state["cost_gbp"] > COST_BUDGET_GBP_PER_RUN:
        raise RuntimeError(
            f"Cost budget exceeded: £{state['cost_gbp']:.2f} > £{COST_BUDGET_GBP_PER_RUN:.2f}"
        )


# ---------------------------------------------------------------------------
# LLM Stub (Gemini Flash / Gemini Pro / Claude Sonnet 4.6)
# ---------------------------------------------------------------------------

async def _call_llm(
    model: str,
    system_prompt: str,
    user_prompt: str,
    state: IntegratedPlatformState,
) -> str:
    """
    Production implementation routes to:
      - "gemini-3.5-flash"  → google.generativeai (agents 1-3)
      - "gemini-3.1-pro"    → google.generativeai (agent 4)
      - "claude-sonnet-4-6" → anthropic.Anthropic  (agent 5)

    Stub returns deterministic response for unit-test / offline use.
    """
    token_est = (len(system_prompt) + len(user_prompt)) // 4
    cost_per_1k = {"gemini-3.5-flash": 0.000075, "gemini-3.1-pro": 0.00125, "claude-sonnet-4-6": 0.003}
    cost = (token_est / 1000) * cost_per_1k.get(model, 0.001)
    _charge_tokens(state, token_est, cost)

    try:
        if model.startswith("gemini"):
            import google.generativeai as genai
            client = genai.GenerativeModel(model)
            resp = client.generate_content(f"{system_prompt}\n\n{user_prompt}")
            return resp.text
        elif model == "claude-sonnet-4-6":
            import anthropic
            client = anthropic.Anthropic()
            msg = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return msg.content[0].text
    except Exception as exc:
        logger.warning("LLM call failed (%s): %s — using stub", model, exc)

    # Deterministic stub
    return (
        f"[STUB:{model}] Platform analysis complete. "
        f"Run: {state.get('run_id','?')[:8]}. "
        f"Regulatory basis: PRA SS1/23, FCA SYSC 6.3.3R, DORA Art.28."
    )


# ---------------------------------------------------------------------------
# Agent 1 — PlatformHealthAgent  (Gemini 3.5 Flash)
# ---------------------------------------------------------------------------

class PlatformHealthAgent:
    """
    Rolls up health status across all 23 AWB production AI systems.
    Detects PSI drift, DORA concentration breach, and system-level RED zones.

    ReAct pattern:
      Reason: "PSI > 0.10 on MR-2026-041 VaR — AMBER drift. DORA concentration check needed."
      Act:    roll_up_system_health(systems=AWB_PLATFORM_SYSTEMS, dora=AWB_LLM_CONCENTRATION)
      Outcome: platform_zone, alert list, health_pct
    """
    NAME = "PlatformHealthAgent"
    MODEL = "gemini-3.5-flash"

    async def run(self, state: IntegratedPlatformState) -> IntegratedPlatformState:
        reason = (
            "Assess real-time health of all 23 AWB production AI systems. "
            "Flag any PSI drift ≥ 0.10 (AMBER) per SS1/23 §4.3. "
            "Check DORA Art.28 LLM concentration: no provider > 70%. "
            "Regulatory basis: PRA SS1/23 §3.1, FCA SYSC 6.3.3R, DORA Art.28."
        )
        _log_step(state, self.NAME, reason, "Initialising platform health roll-up", "START")

        # Merge live inputs over defaults
        systems = {**AWB_PLATFORM_SYSTEMS}
        if state.get("platform_inputs", {}).get("system_overrides"):
            systems.update(state["platform_inputs"]["system_overrides"])

        green = amber = red = 0
        alerts: List[str] = []

        for mr_ref, sys_info in systems.items():
            status = sys_info.get("status", "GREEN")
            psi = sys_info.get("psi", 0.0)

            # PSI drift check
            if psi >= PSI_PLATFORM_WARN and status == "GREEN":
                status = "AMBER"
                alerts.append(f"{mr_ref} ({sys_info['name']}): PSI={psi:.3f} ≥ {PSI_PLATFORM_WARN} → AMBER drift")

            if status == "GREEN":
                green += 1
            elif status == "AMBER":
                amber += 1
                alerts.append(f"{mr_ref}: AMBER — {sys_info['name']}")
            else:
                red += 1
                alerts.append(f"{mr_ref}: RED — {sys_info['name']} requires immediate review")

        total = green + amber + red
        health_pct = (green / total * 100) if total > 0 else 0.0

        # Determine platform zone
        if health_pct < MODEL_HEALTH_RED_PCT or red > 0:
            platform_zone = RiskZone.RED
        elif health_pct < MODEL_HEALTH_WARN_PCT or amber > 2:
            platform_zone = RiskZone.AMBER
        else:
            platform_zone = RiskZone.GREEN

        # DORA concentration check
        concentration = {**AWB_LLM_CONCENTRATION}
        if state.get("dora_inputs", {}).get("llm_concentration"):
            concentration.update(state["dora_inputs"]["llm_concentration"])
        for provider, pct in concentration.items():
            if pct > DORA_CONCENTRATION_MAX_PCT:
                alerts.append(
                    f"DORA Art.28 BREACH: {provider} concentration={pct:.1f}% > {DORA_CONCENTRATION_MAX_PCT:.0f}% limit"
                )
                platform_zone = _escalate_zone(platform_zone, RiskZone.RED)

        # LLM analysis
        act_desc = f"roll_up_system_health(total={total}, green={green}, amber={amber}, red={red})"
        _log_step(state, self.NAME, reason, act_desc, f"health_pct={health_pct:.1f}%, zone={platform_zone}")

        prompt = (
            f"AWB Platform Health Summary — {state['run_date']}\n"
            f"Total systems: {total} | GREEN: {green} | AMBER: {amber} | RED: {red}\n"
            f"Health %: {health_pct:.1f}% | Zone: {platform_zone}\n"
            f"Alerts: {'; '.join(alerts[:5]) if alerts else 'None'}\n"
            f"Summarise in 2 sentences for a PRA supervisor. Reference SS1/23 §3.1."
        )
        llm_summary = await _call_llm(
            self.MODEL,
            "You are an AWB platform health specialist. Apply PRA SS1/23 standards.",
            prompt,
            state,
        )

        state["platform_system_count"] = total
        state["platform_green_count"] = green
        state["platform_amber_count"] = amber
        state["platform_red_count"] = red
        state["platform_health_pct"] = round(health_pct, 2)
        state["platform_zone"] = platform_zone
        state["platform_alerts"] = alerts
        state["overall_zone"] = _escalate_zone(state["overall_zone"], platform_zone)

        _log_step(
            state, self.NAME, reason,
            f"LLM narrative: {llm_summary[:80]}…",
            f"Platform zone finalised: {platform_zone}",
        )
        return state


# ---------------------------------------------------------------------------
# Agent 2 — CrossDomainRiskAgent  (Gemini 3.5 Flash)
# ---------------------------------------------------------------------------

class CrossDomainRiskAgent:
    """
    Detects cross-domain risk correlations: credit ↔ market, market ↔ liquidity.
    AWB June 2026: credit stress scenarios from VaR engine feed IFRS 9 staging;
    liquidity buffer must cover peak VaR drawdown.

    ReAct pattern:
      Reason: "VaR spike correlates with LCR drawdown — systemic risk indicator."
      Act:    compute_cross_domain_correlation(credit_var, market_var, lcr_series)
      Outcome: cross_domain_zone, correlation coefficients, findings
    """
    NAME = "CrossDomainRiskAgent"
    MODEL = "gemini-3.5-flash"

    async def run(self, state: IntegratedPlatformState) -> IntegratedPlatformState:
        reason = (
            "Assess cross-domain risk correlations across credit, market, and liquidity domains. "
            "High credit-market correlation (> 0.65) signals systemic stress — AMBER. "
            "High market-liquidity correlation (> 0.80) signals funding cliff risk — RED. "
            "Regulatory basis: CRR3 Art.325, ILAAP guidelines, FCA SYSC 19D."
        )
        _log_step(state, self.NAME, reason, "Loading cross-domain risk inputs", "START")

        risk_in = state.get("risk_inputs", {})
        credit_market_corr = risk_in.get("credit_market_correlation", 0.42)
        market_liquidity_corr = risk_in.get("market_liquidity_correlation", 0.38)
        var_limit_breach = risk_in.get("var_limit_breach", False)
        lcr_pct = state.get("capital_inputs", {}).get("lcr_pct", AWB_CAPITAL_SNAPSHOT["lcr_pct"])

        findings: List[str] = []
        cross_zone = RiskZone.GREEN

        # Credit-market correlation
        if credit_market_corr >= CROSS_CORR_RED_THRESHOLD:
            cross_zone = _escalate_zone(cross_zone, RiskZone.RED)
            findings.append(
                f"CRITICAL: credit-market correlation={credit_market_corr:.2f} ≥ {CROSS_CORR_RED_THRESHOLD} "
                f"— systemic stress indicator; CRR3 Art.325 stress test required"
            )
        elif credit_market_corr >= CROSS_CORR_WARN_THRESHOLD:
            cross_zone = _escalate_zone(cross_zone, RiskZone.AMBER)
            findings.append(
                f"AMBER: credit-market correlation={credit_market_corr:.2f} — elevated systemic exposure"
            )

        # Market-liquidity correlation
        if market_liquidity_corr >= CROSS_CORR_RED_THRESHOLD:
            cross_zone = _escalate_zone(cross_zone, RiskZone.RED)
            findings.append(
                f"RED: market-liquidity correlation={market_liquidity_corr:.2f} — funding cliff risk"
            )
        elif market_liquidity_corr >= CROSS_CORR_WARN_THRESHOLD:
            cross_zone = _escalate_zone(cross_zone, RiskZone.AMBER)
            findings.append(
                f"AMBER: market-liquidity correlation={market_liquidity_corr:.2f} — monitor closely"
            )

        # VaR limit breach
        if var_limit_breach:
            cross_zone = _escalate_zone(cross_zone, RiskZone.RED)
            findings.append(
                "RED: VaR limit breach detected — MR-2026-041 VaR Engine; "
                "MAR Art.325 capital add-on may apply"
            )

        # LCR below AWB buffer
        if lcr_pct < LIQUIDITY_LCR_BUFFER_PCT:
            cross_zone = _escalate_zone(cross_zone, RiskZone.AMBER)
            findings.append(
                f"AMBER: LCR={lcr_pct:.1f}% < AWB buffer {LIQUIDITY_LCR_BUFFER_PCT:.0f}% "
                f"(regulatory min {LIQUIDITY_LCR_MIN_PCT:.0f}% maintained)"
            )

        act_desc = (
            f"compute_cross_domain_correlation("
            f"credit_market={credit_market_corr:.2f}, "
            f"market_liquidity={market_liquidity_corr:.2f}, "
            f"lcr={lcr_pct:.1f}%)"
        )
        _log_step(state, self.NAME, reason, act_desc, f"cross_zone={cross_zone}, findings={len(findings)}")

        prompt = (
            f"AWB Cross-Domain Risk Assessment — {state['run_date']}\n"
            f"Credit-Market Correlation: {credit_market_corr:.2f}\n"
            f"Market-Liquidity Correlation: {market_liquidity_corr:.2f}\n"
            f"LCR: {lcr_pct:.1f}%\n"
            f"Zone: {cross_zone}\n"
            f"Findings: {'; '.join(findings) if findings else 'No material cross-domain stress'}\n"
            f"Provide 2-sentence ICAAP-grade assessment. Reference CRR3 Art.325."
        )
        llm_analysis = await _call_llm(
            self.MODEL,
            "You are an AWB risk manager specialising in cross-domain systemic risk.",
            prompt,
            state,
        )

        state["credit_market_correlation"] = credit_market_corr
        state["market_liquidity_correlation"] = market_liquidity_corr
        state["cross_domain_zone"] = cross_zone
        state["cross_domain_findings"] = findings
        state["overall_zone"] = _escalate_zone(state["overall_zone"], cross_zone)

        _log_step(
            state, self.NAME, reason,
            f"LLM: {llm_analysis[:80]}…",
            f"Cross-domain zone: {cross_zone}",
        )
        return state


# ---------------------------------------------------------------------------
# Agent 3 — RegulatoryBreachAgent  (Gemini 3.5 Flash)
# ---------------------------------------------------------------------------

class RegulatoryBreachAgent:
    """
    Detects regulatory breaches across FCA, PRA, DORA, and EU AI Act obligations.
    Classifies severity: MINOR, MATERIAL, SERIOUS.
    MATERIAL or SERIOUS → HITL escalation mandatory.

    ReAct pattern:
      Reason: "DORA Art.28 concentration threshold approaching — pre-notification check."
      Act:    assess_regulatory_obligations(fca_rules, pra_rules, dora_rules, eu_ai_act)
      Outcome: breach_severity, breach_details, dora_incident_class
    """
    NAME = "RegulatoryBreachAgent"
    MODEL = "gemini-3.5-flash"

    async def run(self, state: IntegratedPlatformState) -> IntegratedPlatformState:
        reason = (
            "Check all AWB regulatory obligations: FCA SYSC 6.3.3R (model governance), "
            "PRA SS1/23 (AI/ML model risk), DORA Art.10/12/28 (ICT resilience/concentration), "
            "EU AI Act Art.9/13 (high-risk system requirements). "
            "MATERIAL breach → mandatory CRO escalation within 24h. "
            "SERIOUS breach → PRA/FCA notification SLA triggered."
        )
        _log_step(state, self.NAME, reason, "Loading regulatory obligation inputs", "START")

        reg_in = state.get("regulatory_inputs", {})
        dora_in = state.get("dora_inputs", {})

        breach_details: List[str] = []
        breach_severity = BreachSeverity.NONE
        dora_class = DORAIncidentClass.NO_INCIDENT
        reg_zone = RiskZone.GREEN

        # --- FCA checks ---
        consumer_duty_score = reg_in.get("consumer_duty_score_pct", 94.0)
        if consumer_duty_score < 85.0:
            breach_severity = BreachSeverity.MATERIAL
            breach_details.append(
                f"FCA CONSUMER DUTY: score={consumer_duty_score:.1f}% < 85% threshold — "
                f"CONC 5.2.1R breach; MRC escalation required"
            )
            reg_zone = _escalate_zone(reg_zone, RiskZone.RED)

        corep_error = reg_in.get("corep_manual_error_detected", False)
        if corep_error:
            breach_severity = BreachSeverity.SERIOUS
            breach_details.append(
                "PRA: COREP manual error detected — s166 FSMA risk; "
                "SUP 15.3.1R notification SLA = 24h"
            )
            reg_zone = _escalate_zone(reg_zone, RiskZone.CRITICAL)

        # --- PRA SS1/23 checks ---
        ss123_overdue = reg_in.get("ss123_validation_overdue_count", 0)
        if ss123_overdue > 0:
            sev = BreachSeverity.MATERIAL if ss123_overdue >= 2 else BreachSeverity.MINOR
            if _ZONE_RANK.get(sev, 0) > _ZONE_RANK.get(breach_severity, 0):
                breach_severity = sev
            breach_details.append(
                f"SS1/23: {ss123_overdue} model(s) overdue validation — "
                f"PRA §4 annual validation requirement; escalate to MRC"
            )
            reg_zone = _escalate_zone(reg_zone, RiskZone.AMBER if ss123_overdue < 2 else RiskZone.RED)

        # --- EU AI Act ---
        high_risk_unregistered = reg_in.get("eu_ai_act_unregistered_high_risk", 0)
        if high_risk_unregistered > 0:
            breach_severity = BreachSeverity.MATERIAL
            breach_details.append(
                f"EU AI ACT Art.9/13: {high_risk_unregistered} high-risk system(s) not registered "
                f"in EU AI Act database — fines up to 3% global turnover"
            )
            reg_zone = _escalate_zone(reg_zone, RiskZone.RED)

        # --- DORA checks ---
        dora_rto_breached = dora_in.get("rto_breached", False)
        dora_incident_duration_min = dora_in.get("incident_duration_minutes", 0)
        dora_systems_affected = dora_in.get("systems_affected_count", 0)

        if dora_rto_breached or dora_incident_duration_min > DORA_P1_RTO_MINUTES:
            dora_class = DORAIncidentClass.P1_MAJOR
            breach_severity = BreachSeverity.SERIOUS
            breach_details.append(
                f"DORA Art.12 P1 INCIDENT: RTO={dora_incident_duration_min}min > "
                f"{DORA_P1_RTO_MINUTES}min limit; PRA notification within {DORA_P1_PRA_NOTIFY_HOURS}h required"
            )
            reg_zone = _escalate_zone(reg_zone, RiskZone.CRITICAL)
        elif dora_systems_affected > 3:
            dora_class = DORAIncidentClass.P2_SIGNIFICANT
            breach_details.append(
                f"DORA Art.10: {dora_systems_affected} systems affected — P2 significant incident; "
                f"root-cause analysis within 72h"
            )
            reg_zone = _escalate_zone(reg_zone, RiskZone.AMBER)

        act_desc = (
            f"assess_regulatory_obligations("
            f"consumer_duty={consumer_duty_score:.1f}%, "
            f"corep_error={corep_error}, "
            f"ss123_overdue={ss123_overdue}, "
            f"dora_class={dora_class})"
        )
        _log_step(state, self.NAME, reason, act_desc, f"severity={breach_severity}, zone={reg_zone}")

        prompt = (
            f"AWB Regulatory Breach Assessment — {state['run_date']}\n"
            f"Severity: {breach_severity} | Zone: {reg_zone}\n"
            f"DORA Incident: {dora_class}\n"
            f"Breaches: {'; '.join(breach_details) if breach_details else 'No material breaches'}\n"
            f"Provide 2-sentence assessment suitable for Chief Compliance Officer. "
            f"Reference specific FCA/PRA rule numbers."
        )
        llm_assessment = await _call_llm(
            self.MODEL,
            "You are an AWB regulatory compliance specialist. Reference FCA/PRA rules precisely.",
            prompt,
            state,
        )

        state["breach_severity"] = breach_severity
        state["breach_details"] = breach_details
        state["regulatory_zone"] = reg_zone
        state["dora_incident_class"] = dora_class
        state["overall_zone"] = _escalate_zone(state["overall_zone"], reg_zone)

        _log_step(
            state, self.NAME, reason,
            f"LLM: {llm_assessment[:80]}…",
            f"Regulatory zone: {reg_zone}, severity: {breach_severity}",
        )
        return state


# ---------------------------------------------------------------------------
# Agent 4 — CapitalAdequacyAgent  (Gemini 3.1 Pro)
# ---------------------------------------------------------------------------

class CapitalAdequacyAgent:
    """
    Assesses CET1 ratio, RWA movements, and capital stress scenarios.
    Gemini 3.1 Pro used for complex multi-scenario regulatory capital modelling.

    ReAct pattern:
      Reason: "RWA +8% shock would breach AWB Board buffer at 13%. Assess P2A impact."
      Act:    model_capital_scenarios(cet1_base, rwa_stress, p2a_add_on)
      Outcome: cet1_pct, cet1_zone, scenario_findings
    """
    NAME = "CapitalAdequacyAgent"
    MODEL = "gemini-3.1-pro"

    async def run(self, state: IntegratedPlatformState) -> IntegratedPlatformState:
        reason = (
            "Model CET1 adequacy under base, stress (RWA +8%), and severe stress (RWA +15%) scenarios. "
            "CET1 < 14.5% → HITL mandatory per AWB Board resolution. "
            "CET1 < 13.0% → Board buffer breach — emergency capital plan required. "
            "CET1 < 10.5% → PRA Pillar 1 breach — immediate notification. "
            "Regulatory basis: CRR3 Art.92, PRA Pillar 2A, ICG."
        )
        _log_step(state, self.NAME, reason, "Loading capital adequacy inputs", "START")

        cap_in = state.get("capital_inputs", {})
        cet1_base = cap_in.get("cet1_pct", AWB_CAPITAL_SNAPSHOT["cet1_pct"])
        rwa_change = cap_in.get("rwa_change_pct", AWB_CAPITAL_SNAPSHOT["rwa_change_pct"])
        rwa_gbp_bn = cap_in.get("rwa_gbp_bn", AWB_CAPITAL_SNAPSHOT["rwa_gbp_bn"])
        p2a_add_on = cap_in.get("p2a_add_on_pct", 1.8)  # AWB PRA Pillar 2A

        scenario_findings: List[str] = []
        cet1_zone = RiskZone.GREEN

        # RWA movement check
        if rwa_change >= RWA_BREACH_PCT:
            cet1_zone = _escalate_zone(cet1_zone, RiskZone.RED)
            scenario_findings.append(
                f"RED: RWA change={rwa_change:.1f}% ≥ {RWA_BREACH_PCT:.0f}% breach threshold — "
                f"CRR3 Art.92 significant movement; model governance review required"
            )
        elif rwa_change >= RWA_WARN_PCT:
            cet1_zone = _escalate_zone(cet1_zone, RiskZone.AMBER)
            scenario_findings.append(
                f"AMBER: RWA change={rwa_change:.1f}% ≥ {RWA_WARN_PCT:.0f}% warn threshold — monitor"
            )

        # CET1 base scenario
        if cet1_base < CET1_REGULATORY_MIN_PCT:
            cet1_zone = _escalate_zone(cet1_zone, RiskZone.CRITICAL)
            scenario_findings.append(
                f"CRITICAL: CET1={cet1_base:.1f}% < {CET1_REGULATORY_MIN_PCT:.1f}% Pillar 1 minimum — "
                f"PRA immediate notification required; CRR3 Art.92"
            )
        elif cet1_base < CET1_BUFFER_MIN_PCT:
            cet1_zone = _escalate_zone(cet1_zone, RiskZone.RED)
            scenario_findings.append(
                f"RED: CET1={cet1_base:.1f}% < {CET1_BUFFER_MIN_PCT:.1f}% Board buffer — "
                f"emergency capital plan required"
            )
        elif cet1_base < CET1_HITL_THRESHOLD_PCT:
            cet1_zone = _escalate_zone(cet1_zone, RiskZone.AMBER)
            scenario_findings.append(
                f"AMBER: CET1={cet1_base:.1f}% < {CET1_HITL_THRESHOLD_PCT:.1f}% HITL threshold — "
                f"escalation triggered"
            )

        # Stress scenario: RWA +8%
        rwa_stress_factor = 1.08
        cet1_stress = cet1_base * (rwa_gbp_bn / (rwa_gbp_bn * rwa_stress_factor))
        if cet1_stress < CET1_BUFFER_MIN_PCT:
            cet1_zone = _escalate_zone(cet1_zone, RiskZone.AMBER)
            scenario_findings.append(
                f"STRESS (+8% RWA): CET1 would fall to {cet1_stress:.1f}% — "
                f"below AWB Board buffer {CET1_BUFFER_MIN_PCT:.1f}%"
            )

        # Severe stress scenario: RWA +15%
        rwa_severe_factor = 1.15
        cet1_severe = cet1_base * (rwa_gbp_bn / (rwa_gbp_bn * rwa_severe_factor))
        if cet1_severe < CET1_REGULATORY_MIN_PCT + p2a_add_on:
            cet1_zone = _escalate_zone(cet1_zone, RiskZone.RED)
            scenario_findings.append(
                f"SEVERE STRESS (+15% RWA): CET1 would fall to {cet1_severe:.1f}% — "
                f"below Pillar 1+P2A requirement of {CET1_REGULATORY_MIN_PCT + p2a_add_on:.1f}%"
            )

        act_desc = (
            f"model_capital_scenarios("
            f"cet1_base={cet1_base:.1f}%, "
            f"stress_cet1={cet1_stress:.1f}%, "
            f"severe_cet1={cet1_severe:.1f}%, "
            f"rwa_change={rwa_change:.1f}%)"
        )
        _log_step(state, self.NAME, reason, act_desc, f"cet1_zone={cet1_zone}")

        prompt = (
            f"AWB Capital Adequacy Assessment — {state['run_date']}\n"
            f"CET1 Base: {cet1_base:.1f}% | RWA Change: {rwa_change:.1f}%\n"
            f"Stress CET1 (+8% RWA): {cet1_stress:.1f}% | Severe CET1 (+15% RWA): {cet1_severe:.1f}%\n"
            f"Pillar 1 Minimum: {CET1_REGULATORY_MIN_PCT:.1f}% | P2A: {p2a_add_on:.1f}% | "
            f"Board Buffer: {CET1_BUFFER_MIN_PCT:.1f}%\n"
            f"Zone: {cet1_zone}\n"
            f"Findings: {'; '.join(scenario_findings) if scenario_findings else 'CET1 adequate across all scenarios'}\n"
            f"Provide 3-sentence ICAAP-grade capital adequacy assessment. "
            f"Reference CRR3 Art.92, PRA Pillar 2A framework."
        )
        llm_capital = await _call_llm(
            self.MODEL,
            "You are AWB's Head of Capital Management. Provide precise CRR3/PRA Pillar 2A analysis.",
            prompt,
            state,
        )

        state["cet1_pct"] = cet1_base
        state["cet1_zone"] = cet1_zone
        state["rwa_change_pct"] = rwa_change
        state["capital_scenario_findings"] = scenario_findings
        state["overall_zone"] = _escalate_zone(state["overall_zone"], cet1_zone)

        _log_step(
            state, self.NAME, reason,
            f"LLM capital: {llm_capital[:80]}…",
            f"Capital zone: {cet1_zone}",
        )
        return state


# ---------------------------------------------------------------------------
# Agent 5 — PlatformSummaryAgent  (Claude Sonnet 4.6)
# ---------------------------------------------------------------------------

class PlatformSummaryAgent:
    """
    Synthesises all agent findings into a CRO/CFO executive summary and
    CRO daily briefing. Uses Claude Sonnet 4.6 for nuanced narrative synthesis.

    ReAct pattern:
      Reason: "Overall zone RED requires concise CRO escalation narrative with regulatory references."
      Act:    synthesise_executive_summary(platform, cross_risk, regulatory, capital findings)
      Outcome: executive_summary, cro_briefing, hitl_rationale
    """
    NAME = "PlatformSummaryAgent"
    MODEL = "claude-sonnet-4-6"

    async def run(self, state: IntegratedPlatformState) -> IntegratedPlatformState:
        reason = (
            "Synthesise findings from all four specialist agents into CRO/CFO executive narrative. "
            "Overall zone determines escalation tone and regulatory citation priority. "
            "Claude Sonnet 4.6 selected for nuanced risk narrative synthesis per AWB LLM policy. "
            "Regulatory basis: PRA SS1/23 §7 (board reporting), FCA SYSC 4.3.1R (senior management)."
        )
        _log_step(state, self.NAME, reason, "Aggregating all agent findings for synthesis", "START")

        overall_zone = state.get("overall_zone", RiskZone.GREEN)
        platform_alerts = state.get("platform_alerts", [])
        cross_findings = state.get("cross_domain_findings", [])
        breach_details = state.get("breach_details", [])
        capital_findings = state.get("capital_scenario_findings", [])

        all_findings = platform_alerts[:3] + cross_findings[:3] + breach_details[:3] + capital_findings[:3]
        breach_severity = state.get("breach_severity", BreachSeverity.NONE)
        dora_class = state.get("dora_incident_class", DORAIncidentClass.NO_INCIDENT)

        summary_prompt = (
            f"AWB Integrated AI Risk Platform — Executive Summary\n"
            f"Date: {state['run_date']} | Run ID: {state['run_id'][:8]}\n"
            f"Overall Zone: {overall_zone}\n\n"
            f"Platform Health: {state.get('platform_green_count',0)}/{state.get('platform_system_count',23)} systems GREEN "
            f"({state.get('platform_health_pct',0):.1f}%)\n"
            f"Cross-Domain: credit-market ρ={state.get('credit_market_correlation',0):.2f}, "
            f"market-liquidity ρ={state.get('market_liquidity_correlation',0):.2f}\n"
            f"Regulatory Breach: {breach_severity} | DORA: {dora_class}\n"
            f"CET1: {state.get('cet1_pct',15.2):.1f}% | RWA Δ: {state.get('rwa_change_pct',0):.1f}%\n\n"
            f"Key Findings:\n"
            + ("\n".join(f"• {f}" for f in all_findings) if all_findings else "• No material issues identified")
            + f"\n\nWrite a 150-word CRO executive summary suitable for board risk committee. "
            f"Cite specific regulations (SS1/23, CRR3 Art.92, DORA Art.28, FCA SYSC). "
            f"End with clear recommended action: APPROVE, ESCALATE, or EMERGENCY ESCALATE."
        )

        executive_summary = await _call_llm(
            self.MODEL,
            (
                "You are the Chief Risk Officer of Avon & Wessex Bank plc. "
                "Write precise, board-grade risk summaries citing specific UK/EU regulatory references. "
                "Use PRA SS1/23, CRR3, DORA, and FCA Handbook references accurately."
            ),
            summary_prompt,
            state,
        )

        # CRO daily briefing (shorter)
        briefing_prompt = (
            f"AWB CRO Daily Risk Briefing — {state['run_date']}\n"
            f"Zone: {overall_zone} | Systems: {state.get('platform_health_pct',0):.0f}% healthy\n"
            f"Top issue: {all_findings[0] if all_findings else 'No material issues'}\n"
            f"Write a 3-bullet point CRO morning briefing. Each bullet ≤ 20 words."
        )
        cro_briefing = await _call_llm(
            self.MODEL,
            "You are the AWB CRO. Write concise risk briefings for senior management.",
            briefing_prompt,
            state,
        )

        # Determine HITL rationale
        hitl_rationale_parts: List[str] = []
        if overall_zone in (RiskZone.RED, RiskZone.CRITICAL):
            hitl_rationale_parts.append(f"Overall zone {overall_zone} triggers mandatory HITL")
        if breach_severity in (BreachSeverity.MATERIAL, BreachSeverity.SERIOUS):
            hitl_rationale_parts.append(f"Regulatory breach severity {breach_severity} requires CRO sign-off")
        if state.get("cet1_pct", 15.2) < CET1_HITL_THRESHOLD_PCT:
            hitl_rationale_parts.append(
                f"CET1={state.get('cet1_pct',15.2):.1f}% < {CET1_HITL_THRESHOLD_PCT:.1f}% HITL threshold"
            )
        if dora_class == DORAIncidentClass.P1_MAJOR:
            hitl_rationale_parts.append("DORA P1 incident requires PRA notification within 4h")

        hitl_rationale = "; ".join(hitl_rationale_parts) if hitl_rationale_parts else "No mandatory escalation triggers"

        state["executive_summary"] = executive_summary
        state["cro_briefing"] = cro_briefing
        state["hitl_rationale"] = hitl_rationale

        act_desc = f"synthesise_executive_summary(zone={overall_zone}, findings_count={len(all_findings)})"
        _log_step(
            state, self.NAME, reason,
            act_desc,
            f"Summary generated ({len(executive_summary)} chars), HITL rationale: {hitl_rationale[:60]}…",
        )
        return state


# ---------------------------------------------------------------------------
# HITL Gate — deterministic, no LLM
# ---------------------------------------------------------------------------

class HITLGate:
    """
    Deterministic Human-in-the-Loop gate. No LLM involved.
    Evaluates state against hard thresholds. Default: ESCALATE if any breach.

    Escalation triggers:
      • overall_zone in (RED, CRITICAL)
      • breach_severity in (MATERIAL, SERIOUS)
      • cet1_pct < CET1_HITL_THRESHOLD_PCT (14.5%)
      • dora_incident_class == P1_MAJOR
      • platform_red_count > 0
    """
    NAME = "HITLGate"

    def evaluate(self, state: IntegratedPlatformState) -> IntegratedPlatformState:
        reason = (
            "Deterministic HITL evaluation — no LLM. "
            "Assess mandatory escalation triggers per AWB Board Risk Policy v3.2 and PRA SS1/23 §7. "
            "Default action: ESCALATE if any breach detected."
        )

        triggers: List[str] = []

        if state.get("overall_zone") in (RiskZone.RED, RiskZone.CRITICAL):
            triggers.append(f"zone={state['overall_zone']}")

        if state.get("breach_severity") in (BreachSeverity.MATERIAL, BreachSeverity.SERIOUS):
            triggers.append(f"breach={state['breach_severity']}")

        cet1 = state.get("cet1_pct", CET1_HITL_THRESHOLD_PCT + 1)
        if cet1 < CET1_HITL_THRESHOLD_PCT:
            triggers.append(f"CET1={cet1:.1f}%<{CET1_HITL_THRESHOLD_PCT:.1f}%")

        if state.get("dora_incident_class") == DORAIncidentClass.P1_MAJOR:
            triggers.append("DORA P1 incident")

        if state.get("platform_red_count", 0) > 0:
            triggers.append(f"platform_red={state['platform_red_count']}")

        decision = HITLDecision.ESCALATE if triggers else HITLDecision.APPROVE
        state["hitl_decision"] = decision

        _log_step(
            state, self.NAME, reason,
            f"evaluate_hitl_triggers(triggers={triggers})",
            f"decision={decision} | triggers={triggers if triggers else 'none'}",
        )
        return state


# ---------------------------------------------------------------------------
# Sequential Stub (fallback when LangGraph not installed)
# ---------------------------------------------------------------------------

class _SequentialStub:
    """
    Runs the five agents + HITL gate sequentially without LangGraph.
    Provides identical state output to the StateGraph version.
    """
    def __init__(self) -> None:
        self._health = PlatformHealthAgent()
        self._cross = CrossDomainRiskAgent()
        self._reg = RegulatoryBreachAgent()
        self._capital = CapitalAdequacyAgent()
        self._summary = PlatformSummaryAgent()
        self._hitl = HITLGate()

    async def ainvoke(self, state: IntegratedPlatformState) -> IntegratedPlatformState:
        state = await self._health.run(state)
        state = await self._cross.run(state)
        state = await self._reg.run(state)
        state = await self._capital.run(state)
        state = await self._summary.run(state)
        state = self._hitl.evaluate(state)
        return state


# ---------------------------------------------------------------------------
# Pipeline Builder
# ---------------------------------------------------------------------------

def _build_pipeline() -> Any:
    try:
        from langgraph.graph import StateGraph, END, START
        agents = {
            "health":    PlatformHealthAgent(),
            "cross_risk": CrossDomainRiskAgent(),
            "reg_breach": RegulatoryBreachAgent(),
            "capital":   CapitalAdequacyAgent(),
            "summary":   PlatformSummaryAgent(),
        }
        hitl = HITLGate()

        async def node_health(state):    return await agents["health"].run(state)
        async def node_cross(state):     return await agents["cross_risk"].run(state)
        async def node_reg(state):       return await agents["reg_breach"].run(state)
        async def node_capital(state):   return await agents["capital"].run(state)
        async def node_summary(state):   return await agents["summary"].run(state)
        def node_hitl(state):            return hitl.evaluate(state)

        g = StateGraph(IntegratedPlatformState)
        for name, fn in [
            ("health", node_health), ("cross_risk", node_cross),
            ("reg_breach", node_reg), ("capital", node_capital),
            ("summary", node_summary), ("hitl_gate", node_hitl),
        ]:
            g.add_node(name, fn)

        g.add_edge(START, "health")
        g.add_edge("health", "cross_risk")
        g.add_edge("cross_risk", "reg_breach")
        g.add_edge("reg_breach", "capital")
        g.add_edge("capital", "summary")
        g.add_edge("summary", "hitl_gate")
        g.add_edge("hitl_gate", END)

        return g.compile()

    except ImportError:
        logger.info("LangGraph not installed — using SequentialStub for MR-2026-074-IP")
        return _SequentialStub()


# ---------------------------------------------------------------------------
# Public Entrypoint
# ---------------------------------------------------------------------------

async def run_agentic_integrated_platform(
    run_date: str,
    trigger_event: str,
    platform_inputs: Optional[Dict[str, Any]] = None,
    capital_inputs: Optional[Dict[str, Any]] = None,
    risk_inputs: Optional[Dict[str, Any]] = None,
    regulatory_inputs: Optional[Dict[str, Any]] = None,
    dora_inputs: Optional[Dict[str, Any]] = None,
) -> IntegratedPlatformState:
    """
    Main entry point for the Integrated Platform Orchestrator.

    Args:
        run_date:           ISO date string, e.g. "2026-03-31"
        trigger_event:      What triggered this run, e.g. "SCHEDULED_DAILY", "DORA_INCIDENT", "CRO_REQUEST"
        platform_inputs:    Dict with optional system_overrides, dora.llm_concentration overrides
        capital_inputs:     Dict with cet1_pct, rwa_change_pct, rwa_gbp_bn, p2a_add_on_pct, lcr_pct
        risk_inputs:        Dict with credit_market_correlation, market_liquidity_correlation, var_limit_breach
        regulatory_inputs:  Dict with consumer_duty_score_pct, corep_manual_error_detected,
                            ss123_validation_overdue_count, eu_ai_act_unregistered_high_risk
        dora_inputs:        Dict with rto_breached, incident_duration_minutes, systems_affected_count,
                            llm_concentration

    Returns:
        IntegratedPlatformState with all agent findings, HITL decision, and hop-chain audit trail.

    Example:
        state = asyncio.run(run_agentic_integrated_platform(
            run_date="2026-03-31",
            trigger_event="SCHEDULED_DAILY",
            capital_inputs={"cet1_pct": 15.2, "rwa_change_pct": 2.3},
            regulatory_inputs={"consumer_duty_score_pct": 94.0},
        ))
        print(state["hitl_decision"])     # HITLDecision.APPROVE
        print(state["overall_zone"])      # RiskZone.GREEN
        print(len(state["hop_chain"]))    # 12 (2 per agent × 5 agents + 2 HITL)
    """
    state = _initial_state(
        run_date=run_date,
        trigger_event=trigger_event,
        platform_inputs=platform_inputs or {},
        capital_inputs=capital_inputs or {},
        risk_inputs=risk_inputs or {},
        regulatory_inputs=regulatory_inputs or {},
        dora_inputs=dora_inputs or {},
    )

    logger.info(
        "AWB Integrated Platform Orchestrator — run_id=%s, date=%s, trigger=%s",
        state["run_id"][:8], run_date, trigger_event,
    )

    pipeline = _build_pipeline()
    final_state = await pipeline.ainvoke(state)

    logger.info(
        "MR-2026-074-IP complete — zone=%s, hitl=%s, tokens=%d, cost=£%.4f, hops=%d",
        final_state.get("overall_zone"),
        final_state.get("hitl_decision"),
        final_state.get("tokens_used", 0),
        final_state.get("cost_gbp", 0.0),
        len(final_state.get("hop_chain", [])),
    )
    return final_state


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def validate_platform_state(state: IntegratedPlatformState) -> Tuple[bool, List[str]]:
    """
    Post-run validation for CI/CD pipeline and unit tests.
    Returns (is_valid, list_of_violations).
    """
    violations: List[str] = []
    required_fields = [
        "run_id", "model_ref", "run_date", "trigger_event",
        "platform_health_pct", "platform_zone",
        "credit_market_correlation", "cross_domain_zone",
        "breach_severity", "regulatory_zone",
        "cet1_pct", "cet1_zone",
        "overall_zone", "hitl_decision",
        "executive_summary", "hop_chain",
    ]
    for f in required_fields:
        if f not in state:
            violations.append(f"Missing required field: {f}")

    if state.get("model_ref") != "MR-2026-074-IP":
        violations.append(f"Unexpected model_ref: {state.get('model_ref')}")

    if len(state.get("hop_chain", [])) < 10:
        violations.append(f"Insufficient hop_chain: {len(state.get('hop_chain', []))}")

    if state.get("tokens_used", 0) > TOKEN_BUDGET_PER_RUN:
        violations.append(f"Token budget breach: {state['tokens_used']} > {TOKEN_BUDGET_PER_RUN}")

    if state.get("cost_gbp", 0) > COST_BUDGET_GBP_PER_RUN:
        violations.append(f"Cost budget breach: £{state['cost_gbp']:.4f} > £{COST_BUDGET_GBP_PER_RUN}")

    hitl = state.get("hitl_decision")
    if hitl not in (HITLDecision.APPROVE, HITLDecision.ESCALATE, HITLDecision.OVERRIDE):
        violations.append(f"Invalid HITL decision: {hitl}")

    return len(violations) == 0, violations


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    async def _demo() -> None:
        print("=" * 72)
        print("AWB Integrated AI Risk Platform — MR-2026-074-IP")
        print("Chapter 16 Agentic Orchestrator Demo")
        print("=" * 72)

        # Scenario 1: Normal daily run (all GREEN)
        print("\n[SCENARIO 1] Scheduled daily run — March 31, 2026")
        state1 = await run_agentic_integrated_platform(
            run_date="2026-03-31",
            trigger_event="SCHEDULED_DAILY",
            capital_inputs={"cet1_pct": 15.2, "rwa_change_pct": 2.3, "rwa_gbp_bn": 8.4, "lcr_pct": 138.0},
            risk_inputs={"credit_market_correlation": 0.42, "market_liquidity_correlation": 0.38},
            regulatory_inputs={"consumer_duty_score_pct": 94.0, "corep_manual_error_detected": False,
                               "ss123_validation_overdue_count": 0, "eu_ai_act_unregistered_high_risk": 0},
            dora_inputs={"rto_breached": False, "incident_duration_minutes": 0, "systems_affected_count": 0},
        )
        is_valid, violations = validate_platform_state(state1)
        print(f"  Overall Zone:  {state1['overall_zone']}")
        print(f"  HITL Decision: {state1['hitl_decision']}")
        print(f"  Platform:      {state1['platform_health_pct']:.1f}% healthy ({state1['platform_green_count']}/{state1['platform_system_count']} GREEN)")
        print(f"  CET1:          {state1['cet1_pct']:.1f}%")
        print(f"  Hop chain:     {len(state1['hop_chain'])} steps")
        print(f"  Tokens:        {state1['tokens_used']:,} | Cost: £{state1['cost_gbp']:.4f}")
        print(f"  Valid:         {is_valid} | Violations: {violations}")

        print()

        # Scenario 2: DORA incident + regulatory breach
        print("\n[SCENARIO 2] DORA P1 incident — emergency escalation")
        state2 = await run_agentic_integrated_platform(
            run_date="2026-03-31",
            trigger_event="DORA_INCIDENT_ALERT",
            capital_inputs={"cet1_pct": 14.1, "rwa_change_pct": 7.8, "rwa_gbp_bn": 8.4, "lcr_pct": 118.0},
            risk_inputs={"credit_market_correlation": 0.71, "market_liquidity_correlation": 0.45, "var_limit_breach": True},
            regulatory_inputs={"consumer_duty_score_pct": 88.0, "corep_manual_error_detected": False,
                               "ss123_validation_overdue_count": 2, "eu_ai_act_unregistered_high_risk": 0},
            dora_inputs={"rto_breached": True, "incident_duration_minutes": 150, "systems_affected_count": 5},
        )
        is_valid2, violations2 = validate_platform_state(state2)
        print(f"  Overall Zone:  {state2['overall_zone']}")
        print(f"  HITL Decision: {state2['hitl_decision']}")
        print(f"  DORA Class:    {state2['dora_incident_class']}")
        print(f"  Breach:        {state2['breach_severity']}")
        print(f"  CET1:          {state2['cet1_pct']:.1f}%")
        print(f"  HITL Rationale:{state2['hitl_rationale'][:100]}…")
        print(f"  Hop chain:     {len(state2['hop_chain'])} steps")
        print(f"  Valid:         {is_valid2} | Violations: {violations2}")

        print()
        print("=" * 72)
        print("Hop chain (Scenario 1):")
        for hop in state1["hop_chain"]:
            print(f"  [{hop['seq']:02d}] {hop['agent']}: {hop['act'][:70]}")
        print("=" * 72)

    asyncio.run(_demo())
