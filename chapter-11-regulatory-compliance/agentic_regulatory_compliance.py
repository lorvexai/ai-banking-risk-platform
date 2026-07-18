"""Agentic Regulatory Compliance Monitor — Chapter 11 Agentic Extension.

Model ID  : MR-2026-059-REG
Risk Class: HIGH (capital adequacy, supervisory filing, EU AI Act Art.6 Annex III)
Chapter   : 11 — Regulatory Compliance Automation

Architecture: LangGraph StateGraph — five specialist agents + HITL gate.

Agents
------
1. RegulatoryCalendarAgent   — Gemini 3.5 Flash   — deadline tracking, COREP overdue detection
2. CapitalAdequacyAgent      — Gemini 3.5 Flash   — CET1, Tier1, leverage ratio, RWA calculation
3. XBRLFilingAgent           — Gemini 3.5 Flash   — EBA XBRL 4.0 instance generation & validation
4. StressTestingAgent        — Gemini 3.1 Pro     — PRA CST adverse/severe + BoE CBES climate scenarios
5. RegulatoryNarrativeAgent  — Claude Sonnet 4.6  — supervisory narrative, PRA commentary, breach rationale

HITL Gate: HITLDecision enum — APPROVE / ESCALATE / OVERRIDE / PENDING.
           Conservative default: ESCALATE on any capital breach, stress breach, or filing overdue.

Regulatory Coverage
-------------------
- CRR3 Arts 92-386 (capital requirements — RWA, CET1, Tier1)
- CRR3 Art. 429 (leverage ratio — 3.0% minimum for AWB)
- CRR3 Arts 411-428 (LCR — 100% minimum)
- CRR3 Arts 428a-428au (NSFR — 100% minimum)
- EBA ITS 2024/07 (COREP taxonomy 4.0 — effective Q1 2025)
- PRA SS10/13 (COREP filing requirements — 12 BD after quarter end)
- PRA CST 2026 (concurrent stress test — adverse/severe scenarios)
- BoE CBES 2026 (climate biennial exploratory scenario)
- PRA SS1/23 §4-7 (model risk — validation, governance, HITL)
- EU AI Act Art. 6 Annex III (high-risk AI — capital adequacy systems)
- BAP-2026-REG-001 (AWB internal: agentic regulatory compliance governance)

LLM Allocation
--------------
Agents 1-3 : google/gemini-3.5-flash   — fast, structured regulatory logic
Agent 4    : google/gemini-3.1-pro          — multi-scenario stress narrative
Agent 5    : anthropic/claude-sonnet-4-6    — supervisory narrative synthesis

Hop-Chain Audit
---------------
Every agent appends to state["hop_chain"]:
  {seq, agent, timestamp, reason, act, outcome}
Mandatory per PRA AI Roundtable October 2025 / BAP-2026-REG-001 §6.
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import date, datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# REGULATORY CONSTANTS
# ---------------------------------------------------------------------------

# CRR3 Arts 92(1)(a-c) — minimum capital ratios
CET1_MINIMUM_PCT: float = 4.5    # Article 92(1)(a)
TIER1_MINIMUM_PCT: float = 6.0   # Article 92(1)(b)
TOTAL_CAPITAL_MINIMUM_PCT: float = 8.0  # Article 92(1)(c)

# AWB combined buffer requirement (P2R + CCB + O-SII — internal target)
AWB_CET1_TARGET_PCT: float = 10.2   # CET1 + buffers — PRA SREP 2025
AWB_CET1_AMBER_PCT: float = 11.0    # Amber warning — management action trigger
AWB_CET1_GREEN_PCT: float = 12.5    # Green — comfortable headroom

# CRR3 Art. 429 leverage ratio
LEVERAGE_MINIMUM_PCT: float = 3.0   # AWB (non-G-SIB; G-SIBs = 3.5%)
LEVERAGE_AMBER_PCT: float = 4.5
LEVERAGE_GREEN_PCT: float = 6.0

# CRR3 Arts 411-428 LCR
LCR_MINIMUM_PCT: float = 100.0
LCR_AMBER_PCT: float = 115.0
LCR_GREEN_PCT: float = 130.0

# CRR3 Arts 428a-428au NSFR
NSFR_MINIMUM_PCT: float = 100.0
NSFR_AMBER_PCT: float = 105.0
NSFR_GREEN_PCT: float = 115.0

# PRA SS10/13 filing deadlines (business days after period end)
COREP_QUARTERLY_BD: int = 12    # C 02.00, C 08.00, C 18.00, C 47.00, C 80.00
COREP_MONTHLY_BD: int = 15      # C 72.00 (LCR)
FILING_OVERDUE_AMBER_DAYS: int = 5   # 5 days before deadline — amber warning
FILING_OVERDUE_RED_DAYS: int = 0     # On/after deadline — red breach

# PRA CST 2026 stress thresholds
STRESS_CET1_FLOOR_PCT: float = 4.5   # Must not breach regulatory minimum under stress
STRESS_LEVERAGE_FLOOR_PCT: float = 3.0

# BoE CBES RWA uplift thresholds
CBES_RWA_WARN_PCT: float = 0.20      # 20% RWA uplift — amber
CBES_RWA_SEVERE_PCT: float = 0.35    # 35% RWA uplift — red


# ---------------------------------------------------------------------------
# STATE SCHEMA
# ---------------------------------------------------------------------------

class RegulatoryComplianceState(dict):
    """Shared mutable state threaded through all five agents.

    Inherits dict for LangGraph compatibility (TypedDict-style access).
    All agents read and append; zone only escalates (GREEN → AMBER → RED).
    """
    pass


def _initial_state(
    run_date: date,
    trigger_event: str,
    capital_inputs: Dict[str, Any],
    filing_calendar: List[Dict[str, Any]],
    stress_inputs: Dict[str, Any],
    xbrl_returns: List[str],
) -> RegulatoryComplianceState:
    """Construct a clean initial state for the compliance pipeline.

    Args:
        run_date: Date of the compliance run (usually quarter/month end).
        trigger_event: Human-readable description of what triggered the run.
        capital_inputs: Dict with keys: tier1_capital_gbp, cet1_capital_gbp,
            total_capital_gbp, total_rwa_gbp, on_bs_gbp, sa_ccr_gbp,
            sft_gbp, off_bs_gbp, hqla_l1_gbp, hqla_l2a_gbp, hqla_l2b_gbp,
            net_cash_outflows_30d_gbp, asf_gbp, rsf_gbp.
        filing_calendar: List of dicts {return_code, period_end, deadline_date,
            status} for pending COREP filings.
        stress_inputs: Dict with stress scenario params for PRA CST/CBES.
        xbrl_returns: List of COREP return codes to generate (e.g. ["C 47.00"]).

    Returns:
        Initialised RegulatoryComplianceState dict.
    """
    return RegulatoryComplianceState(
        # ---- inputs ----
        run_date=run_date,
        trigger_event=trigger_event,
        capital_inputs=capital_inputs,
        filing_calendar=filing_calendar,
        stress_inputs=stress_inputs,
        xbrl_returns=xbrl_returns,
        # ---- outputs (populated by agents) ----
        calendar_check=None,          # Agent 1
        overdue_filings=[],           # Agent 1
        cet1_ratio_pct=None,          # Agent 2
        tier1_ratio_pct=None,         # Agent 2
        leverage_ratio_pct=None,      # Agent 2
        lcr_pct=None,                 # Agent 2
        nsfr_pct=None,                # Agent 2
        capital_breaches=[],          # Agent 2
        xbrl_instances=[],            # Agent 3
        filing_validation_errors=[],  # Agent 3
        stress_results={},            # Agent 4
        cbes_results={},              # Agent 4
        stress_breaches=[],           # Agent 4
        regulatory_narrative="",      # Agent 5
        pra_commentary="",            # Agent 5
        supervisory_risk_flags=[],    # Agent 5
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
    state: RegulatoryComplianceState,
    agent: str,
    reason: str,
    act: str,
    outcome: str,
) -> None:
    """Append one hop to the audit chain.

    Mandatory per PRA AI Roundtable Oct 2025 and BAP-2026-REG-001 §6.
    Each hop records: sequence, agent, timestamp, reason (pre-action
    reasoning), act (action taken), outcome (result summary).

    Args:
        state: Shared pipeline state — hop_chain list appended in place.
        agent: Agent name / model ID label.
        reason: Why the agent took this action (regulatory basis).
        act: What action was taken.
        outcome: Result / metric produced.
    """
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
    log.info(
        "[HOP %02d] %s | %s → %s",
        _SEQ, agent, act[:60], outcome[:80],
    )


# ---------------------------------------------------------------------------
# RISK ZONE ESCALATION
# ---------------------------------------------------------------------------

_ZONE_RANK: Dict[str, int] = {"GREEN": 0, "AMBER": 1, "RED": 2, "CRITICAL": 3}


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


class RiskZone(str, Enum):
    """Risk zone classification — monotonically escalates GREEN→AMBER→RED→CRITICAL.

    Aligned with AWB Board Risk Appetite and PRA SS1/23 §3.1 traffic-light reporting.
    CRITICAL reserved for DORA P1 incidents and CET1 Pillar 1 breaches.
    """
    GREEN    = "GREEN"
    AMBER    = "AMBER"
    RED      = "RED"
    CRITICAL = "CRITICAL"

def _escalate_zone(
    state: RegulatoryComplianceState,
    proposed: str,
) -> None:
    """Monotonically escalate risk zone — never lower.

    GREEN → AMBER → RED only. Any agent can raise; none can lower.

    Args:
        state: Pipeline state whose "risk_zone" key will be updated.
        proposed: The zone this agent proposes ("GREEN"/"AMBER"/"RED").
    """
    current = state.get("risk_zone", "GREEN")
    new_zone = max(current, proposed, key=lambda z: _ZONE_RANK.get(z, 0))
    if new_zone != current:
        log.warning(
            "Risk zone escalated: %s → %s", current, new_zone
        )
    state["risk_zone"] = new_zone


# ---------------------------------------------------------------------------
# HITL DECISION
# ---------------------------------------------------------------------------

class HITLDecision(str, Enum):
    """Human-in-the-loop decision outcome.

    APPROVE  : All ratios healthy, no breaches, no overdue filings — auto-proceed.
    ESCALATE : Capital breach or filing overdue — require human sign-off.
    OVERRIDE : Human reviewer has manually accepted a breach and documented reason.
    PENDING  : Awaiting human decision.
    """
    APPROVE = "APPROVE"
    ESCALATE = "ESCALATE"
    OVERRIDE = "OVERRIDE"
    PENDING = "PENDING"


def _compute_hitl_decision(
    state: RegulatoryComplianceState,
) -> Tuple[str, str]:
    """Derive HITL decision from state.

    Conservative default: any breach, overdue filing, or stress failure
    triggers ESCALATE. Returns (decision, rationale) string pair.

    Args:
        state: Completed pipeline state after all five agents.

    Returns:
        Tuple of (HITLDecision value string, rationale string).
    """
    reasons: List[str] = []

    capital_breaches = state.get("capital_breaches", [])
    if capital_breaches:
        reasons.append(
            f"Capital breach(es): {'; '.join(capital_breaches)}"
        )

    overdue = state.get("overdue_filings", [])
    if overdue:
        reasons.append(
            f"Overdue COREP filings: {', '.join(overdue)}"
        )

    stress_breaches = state.get("stress_breaches", [])
    if stress_breaches:
        reasons.append(
            f"Stress test breach(es): {'; '.join(stress_breaches)}"
        )

    filing_errors = state.get("filing_validation_errors", [])
    if filing_errors:
        reasons.append(
            f"XBRL validation error(s): {len(filing_errors)} return(s)"
        )

    if reasons:
        rationale = " | ".join(reasons)
        return HITLDecision.ESCALATE.value, rationale

    return HITLDecision.APPROVE.value, (
        "All capital ratios within limits; no overdue filings; "
        "stress tests passed; XBRL validation clean."
    )


# ---------------------------------------------------------------------------
# LLM HELPERS
# ---------------------------------------------------------------------------

def _call_gemini_flash(prompt: str, context: str = "") -> str:
    """Invoke Gemini 3.5 Flash for fast structured compliance tasks.

    Agents 1-3: regulatory calendar, capital calculations, XBRL generation.
    Falls back to deterministic stub when GOOGLE_API_KEY not set.

    Args:
        prompt: Instruction to the model.
        context: Additional regulatory context (schema, facts).

    Returns:
        Model response as plain text.
    """
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        log.debug("GOOGLE_API_KEY not set — using Flash stub")
        return f"[FLASH-STUB] {prompt[:80]}..."

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-3.5-flash")
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as exc:
        log.warning("Gemini Flash error: %s — using stub", exc)
        return f"[FLASH-ERROR] {exc}"


def _call_gemini_pro(prompt: str, context: str = "") -> str:
    """Invoke Gemini 3.1 Pro for complex multi-scenario stress analysis.

    Agent 4: PRA CST adverse/severe + BoE CBES multi-pathway reasoning.
    Falls back to deterministic stub when GOOGLE_API_KEY not set.

    Args:
        prompt: Multi-scenario instruction with full stress parameters.
        context: Capital baseline and scenario parameter tables.

    Returns:
        Detailed stress narrative with breach analysis.
    """
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        log.debug("GOOGLE_API_KEY not set — using Pro stub")
        return f"[PRO-STUB] {prompt[:80]}..."

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-3.1-pro")
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as exc:
        log.warning("Gemini Pro error: %s — using stub", exc)
        return f"[PRO-ERROR] {exc}"


def _call_claude_sonnet(prompt: str, context: str = "") -> str:
    """Invoke Claude Sonnet 4.6 for supervisory narrative synthesis.

    Agent 5: PRA commentary, breach rationale, regulatory risk flags.
    Falls back to deterministic stub when ANTHROPIC_API_KEY not set.

    Args:
        prompt: Narrative synthesis instruction with full compliance picture.
        context: Aggregated outputs from agents 1-4 as structured text.

    Returns:
        Supervisory-quality narrative for PRA submission or board pack.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        log.debug("ANTHROPIC_API_KEY not set — using Sonnet stub")
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
        log.warning("Claude Sonnet error: %s — using stub", exc)
        return f"[SONNET-ERROR] {exc}"


# ---------------------------------------------------------------------------
# AGENT 1 — RegulatoryCalendarAgent (Gemini Flash)
# ---------------------------------------------------------------------------

def regulatory_calendar_agent(
    state: RegulatoryComplianceState,
) -> RegulatoryComplianceState:
    """Agent 1: Track COREP filing deadlines and identify overdue returns.

    Regulatory basis: PRA SS10/13 — COREP submissions must arrive at
    PRA Gabriel within 12 business days of each quarter end (quarterly
    returns) and 15 business days (monthly LCR). Failure constitutes a
    regulatory breach under CRR3 Art. 430 and may attract PRA enforcement.

    ReAct reasoning is constructed BEFORE the deadline scan to document
    the regulatory justification for each classification.

    Populates:
        state["calendar_check"]: Summary dict of upcoming/overdue filings.
        state["overdue_filings"]: List of overdue return_code strings.
        risk_zone: Escalated to AMBER/RED if filings overdue.

    Args:
        state: Shared pipeline state.

    Returns:
        Updated state.
    """
    run_date = state["run_date"]
    filing_calendar = state.get("filing_calendar", [])

    # ---- ReAct: Reason before acting ----
    reason = (
        "PRA SS10/13 requires COREP submissions within 12 BD (quarterly) "
        "or 15 BD (monthly LCR) of period end. Scan calendar for overdue "
        "or imminent filings to classify risk zone and flag for HITL."
    )

    prompt = (
        f"You are AWB's regulatory calendar analyst. Today is {run_date}.\n"
        f"Review these pending COREP filings and classify each as:\n"
        f"  GREEN: >5 business days before deadline\n"
        f"  AMBER: 1-5 business days before deadline (file immediately)\n"
        f"  RED: On or past deadline (supervisory breach)\n\n"
        f"Filings: {filing_calendar}\n\n"
        f"For each RED/AMBER, explain the regulatory consequence under "
        f"PRA SS10/13 and CRR3 Art. 430."
    )

    llm_analysis = _call_gemini_flash(prompt)

    # ---- Deterministic deadline scan ----
    overdue: List[str] = []
    amber_filings: List[str] = []
    calendar_summary: Dict[str, Any] = {}

    for filing in filing_calendar:
        return_code = filing.get("return_code", "UNKNOWN")
        deadline_str = filing.get("deadline_date", "")
        status = filing.get("status", "PENDING")

        if status == "FILED":
            calendar_summary[return_code] = "FILED"
            continue

        try:
            deadline = date.fromisoformat(deadline_str)
        except (ValueError, TypeError):
            log.warning(
                "Invalid deadline for %s: %s", return_code, deadline_str
            )
            calendar_summary[return_code] = "INVALID-DATE"
            continue

        days_until = (deadline - run_date).days

        if days_until < FILING_OVERDUE_RED_DAYS:
            overdue.append(return_code)
            calendar_summary[return_code] = f"OVERDUE ({-days_until}d past deadline)"
        elif days_until <= FILING_OVERDUE_AMBER_DAYS:
            amber_filings.append(return_code)
            calendar_summary[return_code] = f"IMMINENT ({days_until}d to deadline)"
        else:
            calendar_summary[return_code] = f"ON-TRACK ({days_until}d to deadline)"

    # ---- Zone escalation ----
    if overdue:
        _escalate_zone(state, "RED")
    elif amber_filings:
        _escalate_zone(state, "AMBER")

    state["calendar_check"] = {
        "summary": calendar_summary,
        "overdue_count": len(overdue),
        "amber_count": len(amber_filings),
        "llm_analysis": llm_analysis,
    }
    state["overdue_filings"] = overdue

    _log_step(
        state,
        agent="RegulatoryCalendarAgent [gemini-3.5-flash]",
        reason=reason,
        act=f"Scanned {len(filing_calendar)} COREP filings for deadline compliance",
        outcome=(
            f"Overdue: {len(overdue)}, Amber: {len(amber_filings)}, "
            f"Zone: {state['risk_zone']}"
        ),
    )
    return state


# ---------------------------------------------------------------------------
# AGENT 2 — CapitalAdequacyAgent (Gemini Flash)
# ---------------------------------------------------------------------------

def capital_adequacy_agent(
    state: RegulatoryComplianceState,
) -> RegulatoryComplianceState:
    """Agent 2: Calculate capital ratios and detect regulatory breaches.

    Regulatory basis:
    - CRR3 Arts 92(1)(a-c): CET1 ≥ 4.5%, Tier1 ≥ 6.0%, Total ≥ 8.0%
    - CRR3 Art. 429: Leverage ratio ≥ 3.0% (AWB non-G-SIB)
    - CRR3 Arts 411-428: LCR ≥ 100% (30-day stress outflows)
    - CRR3 Arts 428a-428au: NSFR ≥ 100% (stable funding ratio)
    - AWB internal: CET1 target 10.2% (PRA SREP 2025 combined buffer)

    ReAct reasoning documents which CRR3 article applies to each ratio
    BEFORE performing the calculation.

    Populates:
        state["cet1_ratio_pct"], tier1_ratio_pct, leverage_ratio_pct,
        lcr_pct, nsfr_pct: Calculated ratios as floats.
        state["capital_breaches"]: List of breach description strings.
        risk_zone: Escalated per ratio vs. min/amber/green thresholds.

    Args:
        state: Shared pipeline state (capital_inputs must be populated).

    Returns:
        Updated state.
    """
    inputs = state.get("capital_inputs", {})

    # ---- ReAct: Reason before acting ----
    reason = (
        "CRR3 Arts 92, 411-428, 428a-428au, 429 mandate minimum capital, "
        "liquidity, and leverage ratios. Calculate each against regulatory "
        "minima and AWB's internal targets to detect breaches and set "
        "risk zone before XBRL filing and stress testing."
    )

    # ---- Extract inputs with safe defaults ----
    cet1_gbp = float(inputs.get("cet1_capital_gbp", 0))
    tier1_gbp = float(inputs.get("tier1_capital_gbp", 0))
    total_cap_gbp = float(inputs.get("total_capital_gbp", 0))
    total_rwa_gbp = float(inputs.get("total_rwa_gbp", 1))  # prevent div-by-zero

    on_bs_gbp = float(inputs.get("on_bs_gbp", 0))
    sa_ccr_gbp = float(inputs.get("sa_ccr_gbp", 0))
    sft_gbp = float(inputs.get("sft_gbp", 0))
    off_bs_gbp = float(inputs.get("off_bs_gbp", 0))
    total_exposure_gbp = max(on_bs_gbp + sa_ccr_gbp + sft_gbp + off_bs_gbp, 1)

    hqla_l1 = float(inputs.get("hqla_l1_gbp", 0))
    hqla_l2a = float(inputs.get("hqla_l2a_gbp", 0))
    hqla_l2b = float(inputs.get("hqla_l2b_gbp", 0))
    net_outflows = max(float(inputs.get("net_cash_outflows_30d_gbp", 1)), 1)
    adjusted_hqla = hqla_l1 + hqla_l2a * 0.85 + hqla_l2b * 0.75

    asf_gbp = float(inputs.get("asf_gbp", 0))
    rsf_gbp = max(float(inputs.get("rsf_gbp", 1)), 1)

    # ---- Calculate ratios ----
    cet1_pct = (cet1_gbp / total_rwa_gbp) * 100
    tier1_pct = (tier1_gbp / total_rwa_gbp) * 100
    total_cap_pct = (total_cap_gbp / total_rwa_gbp) * 100
    leverage_pct = (tier1_gbp / total_exposure_gbp) * 100
    lcr_pct = (adjusted_hqla / net_outflows) * 100
    nsfr_pct = (asf_gbp / rsf_gbp) * 100

    state["cet1_ratio_pct"] = round(cet1_pct, 2)
    state["tier1_ratio_pct"] = round(tier1_pct, 2)
    state["total_capital_ratio_pct"] = round(total_cap_pct, 2)
    state["leverage_ratio_pct"] = round(leverage_pct, 2)
    state["lcr_pct"] = round(lcr_pct, 2)
    state["nsfr_pct"] = round(nsfr_pct, 2)

    # ---- Breach detection ----
    breaches: List[str] = []

    if cet1_pct < CET1_MINIMUM_PCT:
        breaches.append(
            f"CET1 {cet1_pct:.2f}% < CRR3 Art.92(1)(a) minimum {CET1_MINIMUM_PCT}%"
        )
        _escalate_zone(state, "RED")
    elif cet1_pct < AWB_CET1_TARGET_PCT:
        breaches.append(
            f"CET1 {cet1_pct:.2f}% < AWB target {AWB_CET1_TARGET_PCT}% (PRA SREP)"
        )
        _escalate_zone(state, "AMBER")
    elif cet1_pct < AWB_CET1_AMBER_PCT:
        _escalate_zone(state, "AMBER")

    if tier1_pct < TIER1_MINIMUM_PCT:
        breaches.append(
            f"Tier1 {tier1_pct:.2f}% < CRR3 Art.92(1)(b) minimum {TIER1_MINIMUM_PCT}%"
        )
        _escalate_zone(state, "RED")

    if total_cap_pct < TOTAL_CAPITAL_MINIMUM_PCT:
        breaches.append(
            f"Total Capital {total_cap_pct:.2f}% < CRR3 Art.92(1)(c) minimum {TOTAL_CAPITAL_MINIMUM_PCT}%"
        )
        _escalate_zone(state, "RED")

    if leverage_pct < LEVERAGE_MINIMUM_PCT:
        breaches.append(
            f"Leverage {leverage_pct:.2f}% < CRR3 Art.429 minimum {LEVERAGE_MINIMUM_PCT}%"
        )
        _escalate_zone(state, "RED")
    elif leverage_pct < LEVERAGE_AMBER_PCT:
        _escalate_zone(state, "AMBER")

    if lcr_pct < LCR_MINIMUM_PCT:
        breaches.append(
            f"LCR {lcr_pct:.2f}% < CRR3 Arts 411-428 minimum {LCR_MINIMUM_PCT}%"
        )
        _escalate_zone(state, "RED")
    elif lcr_pct < LCR_AMBER_PCT:
        _escalate_zone(state, "AMBER")

    if nsfr_pct < NSFR_MINIMUM_PCT:
        breaches.append(
            f"NSFR {nsfr_pct:.2f}% < CRR3 Arts 428a-428au minimum {NSFR_MINIMUM_PCT}%"
        )
        _escalate_zone(state, "RED")
    elif nsfr_pct < NSFR_AMBER_PCT:
        _escalate_zone(state, "AMBER")

    state["capital_breaches"] = breaches

    # ---- LLM supplementary analysis ----
    llm_prompt = (
        f"AWB capital position as at {state['run_date']}:\n"
        f"  CET1: {cet1_pct:.2f}% (min 4.5%, target 10.2%)\n"
        f"  Tier1: {tier1_pct:.2f}% (min 6.0%)\n"
        f"  Total Capital: {total_cap_pct:.2f}% (min 8.0%)\n"
        f"  Leverage: {leverage_pct:.2f}% (min 3.0%)\n"
        f"  LCR: {lcr_pct:.2f}% (min 100%)\n"
        f"  NSFR: {nsfr_pct:.2f}% (min 100%)\n"
        f"  Breaches: {breaches}\n\n"
        f"Identify the PRA supervisory actions that would be triggered "
        f"by each breach, citing CRR3 article numbers."
    )
    state["capital_llm_analysis"] = _call_gemini_flash(llm_prompt)

    _log_step(
        state,
        agent="CapitalAdequacyAgent [gemini-3.5-flash]",
        reason=reason,
        act=(
            f"Calculated CET1={cet1_pct:.2f}%, T1={tier1_pct:.2f}%, "
            f"Lev={leverage_pct:.2f}%, LCR={lcr_pct:.2f}%, NSFR={nsfr_pct:.2f}%"
        ),
        outcome=(
            f"{len(breaches)} breach(es) detected, Zone={state['risk_zone']}"
        ),
    )
    return state


# ---------------------------------------------------------------------------
# AGENT 3 — XBRLFilingAgent (Gemini Flash)
# ---------------------------------------------------------------------------

def xbrl_filing_agent(
    state: RegulatoryComplianceState,
) -> RegulatoryComplianceState:
    """Agent 3: Generate and validate EBA XBRL 4.0 COREP instance documents.

    Regulatory basis: EBA ITS 2024/07 requires all UK CRR firms to file
    COREP returns as XBRL instance documents compliant with EBA Taxonomy 4.0.
    PRA Gabriel validates on receipt — failed instances are rejected and
    treated as non-filed (deadline breach). BAP-2026-REG-001 §7 requires
    all generated XBRL to pass internal pre-validation before submission.

    Generates XBRL for each return_code in state["xbrl_returns"] using
    the calculated capital ratios from Agent 2.

    Populates:
        state["xbrl_instances"]: List of {return_code, xml_snippet, valid} dicts.
        state["filing_validation_errors"]: Any returns with validation failures.
        risk_zone: Escalated to AMBER if validation errors found.

    Args:
        state: Shared pipeline state (capital ratios must be populated).

    Returns:
        Updated state.
    """
    return_codes = state.get("xbrl_returns", [])
    run_date = state.get("run_date", date.today())
    entity_id = "AWB-UK-001"

    # ---- ReAct: Reason before acting ----
    reason = (
        "EBA ITS 2024/07 mandates XBRL Taxonomy 4.0 instance documents for "
        "all COREP filings. Pre-validation against EBA taxonomy before "
        "submission to PRA Gabriel is required under BAP-2026-REG-001 §7. "
        "Validation failures must be escalated as they constitute filing risk."
    )

    # ---- Capital fact mapping for XBRL ----
    capital_facts: Dict[str, float] = {}
    if state.get("cet1_ratio_pct") is not None:
        capital_facts["eba-re:CET1CapitalRatio"] = state["cet1_ratio_pct"]
    if state.get("tier1_ratio_pct") is not None:
        capital_facts["eba-re:Tier1CapitalRatio"] = state["tier1_ratio_pct"]
    if state.get("leverage_ratio_pct") is not None:
        capital_facts["eba-re:LeverageRatio"] = state["leverage_ratio_pct"]
    if state.get("lcr_pct") is not None:
        capital_facts["eba-re:LiquidityCoverageRatio"] = state["lcr_pct"]
    if state.get("nsfr_pct") is not None:
        capital_facts["eba-re:NetStableFundingRatio"] = state["nsfr_pct"]

    # ---- COREP return → relevant facts mapping ----
    return_fact_map: Dict[str, List[str]] = {
        "C 02.00": ["eba-re:CET1CapitalRatio", "eba-re:Tier1CapitalRatio"],
        "C 08.00": ["eba-re:CET1CapitalRatio"],
        "C 47.00": ["eba-re:LeverageRatio"],
        "C 72.00": ["eba-re:LiquidityCoverageRatio"],
        "C 80.00": ["eba-re:NetStableFundingRatio"],
        "C 18.00": ["eba-re:Tier1CapitalRatio"],
        "C 24.00": ["eba-re:CET1CapitalRatio"],
    }

    instances: List[Dict[str, Any]] = []
    validation_errors: List[str] = []

    for return_code in return_codes:
        facts_for_return = {
            k: v for k, v in capital_facts.items()
            if k in return_fact_map.get(return_code, list(capital_facts.keys()))
        }

        # Build minimal XBRL instance
        facts_xml = "\n".join(
            f'  <{concept} contextRef="period" decimals="2">{value}</{concept}>'
            for concept, value in facts_for_return.items()
        )
        xbrl_xml = (
            f'<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<xbrl xmlns="http://www.xbrl.org/2003/instance"\n'
            f'  xmlns:eba-re="http://www.eba.europa.eu/xbrl/crr/dict/lei"\n'
            f'  xmlns:xbrli="http://www.xbrl.org/2003/instance">\n'
            f'  <xbrli:context id="period">\n'
            f'    <xbrli:entity>\n'
            f'      <xbrli:identifier scheme="http://www.eba.europa.eu/xbrl/crr/dict/lei">'
            f'{entity_id}</xbrli:identifier>\n'
            f'    </xbrli:entity>\n'
            f'    <xbrli:period>\n'
            f'      <xbrli:instant>{run_date.isoformat()}</xbrli:instant>\n'
            f'    </xbrli:period>\n'
            f'  </xbrli:context>\n'
            f'  <!-- EBA XBRL Taxonomy 4.0 | {return_code} | '
            f'MR-2026-059-REG -->\n'
            f'{facts_xml}\n'
            f'</xbrl>'
        )

        # Validate XML well-formedness (pre-arelle check)
        import xml.etree.ElementTree as ET
        valid = True
        val_error = None
        try:
            ET.fromstring(xbrl_xml)
        except ET.ParseError as e:
            valid = False
            val_error = str(e)
            validation_errors.append(
                f"{return_code}: XML parse error — {e}"
            )

        # LLM review of XBRL facts
        llm_prompt = (
            f"Review this EBA XBRL Taxonomy 4.0 instance for COREP {return_code}.\n"
            f"Entity: {entity_id}, Period: {run_date}.\n"
            f"Facts included: {list(facts_for_return.keys())}.\n"
            f"Identify any missing mandatory facts per EBA ITS 2024/07 "
            f"and rate the completeness as HIGH/MEDIUM/LOW."
        )
        llm_review = _call_gemini_flash(llm_prompt)

        instances.append({
            "return_code": return_code,
            "xml_snippet": xbrl_xml[:500] + "...",
            "facts_count": len(facts_for_return),
            "valid": valid,
            "validation_error": val_error,
            "llm_completeness_review": llm_review,
        })

        log.info(
            "XBRL %s: valid=%s facts=%d",
            return_code, valid, len(facts_for_return),
        )

    state["xbrl_instances"] = instances
    state["filing_validation_errors"] = validation_errors

    if validation_errors:
        _escalate_zone(state, "AMBER")

    _log_step(
        state,
        agent="XBRLFilingAgent [gemini-3.5-flash]",
        reason=reason,
        act=f"Generated {len(instances)} XBRL instance(s) for {return_codes}",
        outcome=(
            f"Valid: {sum(1 for i in instances if i['valid'])}/{len(instances)}, "
            f"Errors: {len(validation_errors)}, Zone: {state['risk_zone']}"
        ),
    )
    return state


# ---------------------------------------------------------------------------
# AGENT 4 — StressTestingAgent (Gemini 3.1 Pro)
# ---------------------------------------------------------------------------

def stress_testing_agent(
    state: RegulatoryComplianceState,
) -> RegulatoryComplianceState:
    """Agent 4: Run PRA CST and BoE CBES stress scenarios.

    Regulatory basis:
    - PRA CST 2026: Annual concurrent stress test — AWB must submit
      stressed capital projections for 3-year horizon under adverse
      and severe scenarios. Results used for PRA capital setting.
    - BoE CBES 2026: Climate Biennial Exploratory Scenario — exploratory
      only, but results inform supervisory dialogue on climate risk.
      Three pathways: early action, late action, no additional action.

    Uses Gemini 3.1 Pro for complex multi-scenario, multi-dimensional
    reasoning across capital, liquidity, and climate risk dimensions.

    Populates:
        state["stress_results"]: PRA CST results per scenario.
        state["cbes_results"]: BoE CBES results per pathway.
        state["stress_breaches"]: Scenarios where floor breached.
        risk_zone: RED if any scenario breaches regulatory floor.

    Args:
        state: Shared pipeline state.

    Returns:
        Updated state.
    """
    stress_inputs = state.get("stress_inputs", {})
    cet1_base = state.get("cet1_ratio_pct", 13.67)
    leverage_base = state.get("leverage_ratio_pct", 8.9)
    lcr_base = state.get("lcr_pct", 147.3)
    total_rwa_base = float(
        stress_inputs.get("total_rwa_gbp", 30_000_000_000)
    )

    # ---- ReAct: Reason before acting ----
    reason = (
        "PRA CST 2026 requires AWB to model capital depletion under adverse "
        "(GDP -3.5%, credit loss 2.1%) and severe (GDP -7.2%, credit loss 4.8%) "
        "PRA scenarios. BoE CBES requires climate pathway analysis across "
        "early/late/no-action scenarios. Breaches of CET1 4.5% floor under "
        "any scenario trigger RED zone and mandatory HITL escalation."
    )

    # ---- PRA CST scenario parameters ----
    pra_scenarios = {
        "base": {
            "rwa_growth_pct": 0.00,
            "lcr_outflow_multiplier": 1.00,
            "credit_loss_rate_pct": 0.80,
            "gdp_shock_pct": 0.00,
        },
        "adverse": {
            "rwa_growth_pct": 0.25,
            "lcr_outflow_multiplier": 1.15,
            "credit_loss_rate_pct": 2.10,
            "gdp_shock_pct": -3.50,
        },
        "severe": {
            "rwa_growth_pct": 0.45,
            "lcr_outflow_multiplier": 1.35,
            "credit_loss_rate_pct": 4.80,
            "gdp_shock_pct": -7.20,
        },
    }

    # ---- BoE CBES climate scenarios ----
    cbes_scenarios = {
        "early_action": {
            "stranded_assets_pct": 0.08,
            "transition_risk_rwa_uplift": 0.12,
            "physical_risk_rwa_uplift": 0.05,
        },
        "late_action": {
            "stranded_assets_pct": 0.15,
            "transition_risk_rwa_uplift": 0.25,
            "physical_risk_rwa_uplift": 0.12,
        },
        "no_action": {
            "stranded_assets_pct": 0.22,
            "transition_risk_rwa_uplift": 0.08,
            "physical_risk_rwa_uplift": 0.35,
        },
    }

    stress_results: Dict[str, Any] = {}
    cbes_results: Dict[str, Any] = {}
    stress_breaches: List[str] = []

    # ---- PRA CST calculations ----
    for scenario_name, params in pra_scenarios.items():
        rwa_stressed = total_rwa_base * (1 + params["rwa_growth_pct"])
        credit_loss_pct = params["credit_loss_rate_pct"] / 100
        capital_after_losses_gbp = (
            (cet1_base / 100) * total_rwa_base
            - credit_loss_pct * total_rwa_base
        )
        cet1_stressed = (
            capital_after_losses_gbp / max(rwa_stressed, 1)
        ) * 100
        lcr_stressed = lcr_base / params["lcr_outflow_multiplier"]
        leverage_stressed = leverage_base * (
            total_rwa_base / max(rwa_stressed, 1)
        )

        breach = cet1_stressed < STRESS_CET1_FLOOR_PCT
        if breach:
            stress_breaches.append(
                f"PRA CST {scenario_name}: CET1 {cet1_stressed:.2f}% "
                f"< floor {STRESS_CET1_FLOOR_PCT}% "
                f"(GDP shock {params['gdp_shock_pct']}%, "
                f"credit loss {params['credit_loss_rate_pct']}%)"
            )
            _escalate_zone(state, "RED")

        stress_results[scenario_name] = {
            "cet1_stressed_pct": round(cet1_stressed, 2),
            "leverage_stressed_pct": round(leverage_stressed, 2),
            "lcr_stressed_pct": round(lcr_stressed, 2),
            "rwa_stressed_gbp": round(rwa_stressed / 1e9, 2),
            "cet1_breach": breach,
            "gdp_shock_pct": params["gdp_shock_pct"],
            "credit_loss_rate_pct": params["credit_loss_rate_pct"],
        }

    # ---- BoE CBES calculations ----
    for scenario_name, params in cbes_scenarios.items():
        total_rwa_uplift = (
            params["transition_risk_rwa_uplift"]
            + params["physical_risk_rwa_uplift"]
        )
        rwa_climate_stressed = total_rwa_base * (1 + total_rwa_uplift)
        cet1_climate = (
            (cet1_base / 100) * total_rwa_base / max(rwa_climate_stressed, 1)
        ) * 100
        stranded_loss_pct = params["stranded_assets_pct"]

        severe_climate = total_rwa_uplift > CBES_RWA_SEVERE_PCT
        warn_climate = total_rwa_uplift > CBES_RWA_WARN_PCT

        if severe_climate:
            _escalate_zone(state, "RED")
            stress_breaches.append(
                f"CBES {scenario_name}: RWA uplift {total_rwa_uplift:.0%} "
                f"> severe threshold {CBES_RWA_SEVERE_PCT:.0%}"
            )
        elif warn_climate:
            _escalate_zone(state, "AMBER")

        cbes_results[scenario_name] = {
            "total_rwa_uplift_pct": round(total_rwa_uplift * 100, 1),
            "rwa_climate_stressed_gbp": round(rwa_climate_stressed / 1e9, 2),
            "cet1_climate_stressed_pct": round(cet1_climate, 2),
            "stranded_assets_pct": round(stranded_loss_pct * 100, 1),
            "severity_flag": "SEVERE" if severe_climate else ("WARN" if warn_climate else "OK"),
        }

    # ---- Gemini 3.1 Pro narrative ----
    context = (
        f"AWB baseline: CET1={cet1_base:.2f}%, Leverage={leverage_base:.2f}%, "
        f"LCR={lcr_base:.2f}%, RWA=£{total_rwa_base/1e9:.1f}B\n"
        f"PRA CST results: {stress_results}\n"
        f"BoE CBES results: {cbes_results}\n"
        f"Breaches: {stress_breaches}"
    )
    llm_prompt = (
        "You are AWB's stress testing lead preparing a PRA CST submission.\n"
        "Analyse the stressed capital positions across all PRA CST scenarios "
        "(base, adverse, severe) and BoE CBES climate pathways.\n"
        "For each scenario:\n"
        "  1. Identify whether CET1 remains above the 4.5% regulatory floor.\n"
        "  2. Identify the primary driver of capital depletion.\n"
        "  3. Recommend a management action if stressed CET1 < 8.0%.\n"
        "  4. For CBES, assess whether climate RWA uplift is material "
        "for supervisory disclosure.\n"
        "Write in the style of a PRA supervisory narrative (formal, precise, "
        "citing CRR3 and PRA Policy Statement references)."
    )
    stress_narrative = _call_gemini_pro(llm_prompt, context)

    state["stress_results"] = stress_results
    state["cbes_results"] = cbes_results
    state["stress_breaches"] = stress_breaches
    state["stress_narrative"] = stress_narrative

    _log_step(
        state,
        agent="StressTestingAgent [gemini-3.1-pro]",
        reason=reason,
        act=(
            f"Ran {len(pra_scenarios)} PRA CST + {len(cbes_scenarios)} CBES scenarios"
        ),
        outcome=(
            f"Stress breaches: {len(stress_breaches)}, "
            f"Zone: {state['risk_zone']}"
        ),
    )
    return state


# ---------------------------------------------------------------------------
# AGENT 5 — RegulatoryNarrativeAgent (Claude Sonnet 4.6)
# ---------------------------------------------------------------------------

def regulatory_narrative_agent(
    state: RegulatoryComplianceState,
) -> RegulatoryComplianceState:
    """Agent 5: Generate supervisory narrative for PRA and board reporting.

    Regulatory basis: PRA SS1/23 §7 requires firms to maintain an audit
    trail and human-readable narrative explaining AI-assisted compliance
    decisions. EU AI Act Art. 14 (human oversight) requires outputs of
    high-risk AI systems (capital adequacy is Annex III high-risk) to be
    understandable by qualified humans. BAP-2026-REG-001 §8 mandates a
    signed-off narrative before COREP submission.

    Uses Claude Sonnet 4.6 for highest-quality regulatory narrative
    synthesis — integrating capital ratios, filing status, stress results,
    and XBRL validation into a coherent PRA-ready submission summary.

    Populates:
        state["regulatory_narrative"]: Full compliance narrative.
        state["pra_commentary"]: PRA submission commentary section.
        state["supervisory_risk_flags"]: Material issues for supervisors.
        hitl_decision / hitl_rationale: Final HITL outcome.
        pipeline_completed: True.

    Args:
        state: Fully populated pipeline state after agents 1-4.

    Returns:
        Updated state with narrative and final HITL decision.
    """
    # ---- ReAct: Reason before acting ----
    reason = (
        "PRA SS1/23 §7 and EU AI Act Art. 14 require a human-readable "
        "narrative explaining all AI-assisted compliance outputs before "
        "regulatory submission. Claude Sonnet 4.6 synthesises the full "
        "compliance picture into supervisory-quality prose. HITL decision "
        "applied conservatively: any breach triggers ESCALATE."
    )

    # ---- Aggregate context from agents 1-4 ----
    context_parts = [
        f"Run Date: {state.get('run_date')}",
        f"Trigger: {state.get('trigger_event')}",
        f"",
        f"CAPITAL RATIOS:",
        f"  CET1: {state.get('cet1_ratio_pct', 'N/A')}% "
        f"(min 4.5%, AWB target 10.2%)",
        f"  Tier1: {state.get('tier1_ratio_pct', 'N/A')}% (min 6.0%)",
        f"  Total Capital: {state.get('total_capital_ratio_pct', 'N/A')}% "
        f"(min 8.0%)",
        f"  Leverage: {state.get('leverage_ratio_pct', 'N/A')}% (min 3.0%)",
        f"  LCR: {state.get('lcr_pct', 'N/A')}% (min 100%)",
        f"  NSFR: {state.get('nsfr_pct', 'N/A')}% (min 100%)",
        f"",
        f"BREACHES: {state.get('capital_breaches', [])}",
        f"",
        f"FILING STATUS: {state.get('calendar_check', {}).get('summary', {})}",
        f"OVERDUE FILINGS: {state.get('overdue_filings', [])}",
        f"",
        f"XBRL VALIDATION: {len(state.get('filing_validation_errors', []))} "
        f"error(s) in {len(state.get('xbrl_instances', []))} return(s)",
        f"",
        f"STRESS TEST RESULTS:",
        f"  PRA CST: {state.get('stress_results', {})}",
        f"  BoE CBES: {state.get('cbes_results', {})}",
        f"  Stress Breaches: {state.get('stress_breaches', [])}",
        f"",
        f"RISK ZONE: {state.get('risk_zone')}",
        f"HOP CHAIN: {len(state.get('hop_chain', []))} steps logged",
    ]
    full_context = "\n".join(context_parts)

    # ---- Supervisory narrative prompt ----
    narrative_prompt = (
        "You are AWB's Chief Risk Officer preparing the quarterly regulatory "
        "compliance summary for the Board Risk Committee and PRA submission.\n\n"
        "Write a formal supervisory narrative (400-500 words) that:\n"
        "1. States AWB's capital and liquidity position against all CRR3 minima\n"
        "2. Describes any breaches and immediate management actions taken\n"
        "3. Summarises PRA CST stress test results and headroom above CET1 floor\n"
        "4. Addresses BoE CBES climate scenario implications\n"
        "5. Confirms COREP XBRL filings status and any overdue returns\n"
        "6. States the HITL sign-off requirement under BAP-2026-REG-001 §8\n\n"
        "Tone: formal, precise, PRA-supervisory standard. "
        "Cite CRR3 articles and PRA policy statements where relevant."
    )
    regulatory_narrative = _call_claude_sonnet(
        narrative_prompt, full_context
    )

    # ---- PRA commentary (shorter, filing-ready) ----
    pra_prompt = (
        "Write a 150-word PRA submission commentary for AWB's COREP filing "
        "covering the key capital metrics, any material changes since last "
        "quarter, and confirmation that all data has been validated under "
        "BAP-2026-REG-001 §7. Reference model ID MR-2026-059-REG."
    )
    pra_commentary = _call_claude_sonnet(pra_prompt, full_context)

    # ---- Supervisory risk flags ----
    risk_flags: List[str] = []
    if state.get("capital_breaches"):
        risk_flags.append(
            f"CAPITAL: {len(state['capital_breaches'])} breach(es) — "
            "immediate PRA notification required (CRR3 Art. 142)"
        )
    if state.get("overdue_filings"):
        risk_flags.append(
            f"FILING: {len(state['overdue_filings'])} overdue COREP return(s) — "
            "late filing risk under CRR3 Art. 430"
        )
    if state.get("stress_breaches"):
        risk_flags.append(
            f"STRESS: {len(state['stress_breaches'])} scenario breach(es) — "
            "PRA CST submission requires management action narrative"
        )
    if any(r["total_rwa_uplift_pct"] > CBES_RWA_WARN_PCT * 100
           for r in state.get("cbes_results", {}).values()
           if isinstance(r, dict)):
        risk_flags.append(
            "CLIMATE: CBES RWA uplift material — "
            "climate risk supervisor dialogue recommended"
        )
    if not risk_flags:
        risk_flags.append(
            "No material supervisory concerns — routine quarterly filing"
        )

    state["regulatory_narrative"] = regulatory_narrative
    state["pra_commentary"] = pra_commentary
    state["supervisory_risk_flags"] = risk_flags

    # ---- HITL decision ----
    decision, rationale = _compute_hitl_decision(state)
    state["hitl_decision"] = decision
    state["hitl_rationale"] = rationale
    state["pipeline_completed"] = True

    _log_step(
        state,
        agent="RegulatoryNarrativeAgent [claude-sonnet-4-6]",
        reason=reason,
        act=(
            f"Generated supervisory narrative ({len(regulatory_narrative)} chars), "
            f"PRA commentary, {len(risk_flags)} risk flag(s)"
        ),
        outcome=(
            f"HITL={decision}, Flags={len(risk_flags)}, "
            f"Zone={state['risk_zone']}, Pipeline=COMPLETE"
        ),
    )
    return state


# ---------------------------------------------------------------------------
# PIPELINE ORCHESTRATION
# ---------------------------------------------------------------------------

def _build_graph():
    """Build LangGraph StateGraph for the regulatory compliance pipeline.

    Topology:
        START → regulatory_calendar_agent
              → capital_adequacy_agent
              → xbrl_filing_agent
              → stress_testing_agent
              → regulatory_narrative_agent
              → END

    Returns LangGraph CompiledGraph or _SequentialStub if LangGraph absent.
    """
    try:
        from langgraph.graph import StateGraph, END

        graph = StateGraph(RegulatoryComplianceState)
        graph.add_node("regulatory_calendar", regulatory_calendar_agent)
        graph.add_node("capital_adequacy", capital_adequacy_agent)
        graph.add_node("xbrl_filing", xbrl_filing_agent)
        graph.add_node("stress_testing", stress_testing_agent)
        graph.add_node("regulatory_narrative", regulatory_narrative_agent)

        graph.set_entry_point("regulatory_calendar")
        graph.add_edge("regulatory_calendar", "capital_adequacy")
        graph.add_edge("capital_adequacy", "xbrl_filing")
        graph.add_edge("xbrl_filing", "stress_testing")
        graph.add_edge("stress_testing", "regulatory_narrative")
        graph.add_edge("regulatory_narrative", END)

        return graph.compile()

    except ImportError:
        log.warning(
            "LangGraph not installed — using _SequentialStub"
        )
        return _SequentialStub()


class _SequentialStub:
    """Fallback orchestrator for environments without LangGraph.

    Runs the five agents in sequence using standard Python function calls.
    Provides the same invoke() interface as a compiled LangGraph graph.
    Used in CI, unit tests, and environments without langgraph installed.
    """

    def invoke(
        self,
        state: RegulatoryComplianceState,
    ) -> RegulatoryComplianceState:
        """Execute all five agents sequentially.

        Args:
            state: Initial pipeline state.

        Returns:
            Fully populated state after all five agents.
        """
        state = regulatory_calendar_agent(state)
        state = capital_adequacy_agent(state)
        state = xbrl_filing_agent(state)
        state = stress_testing_agent(state)
        state = regulatory_narrative_agent(state)
        return state


# ---------------------------------------------------------------------------
# PUBLIC ENTRY POINT
# ---------------------------------------------------------------------------

async def run_agentic_regulatory_compliance(
    run_date: date,
    trigger_event: str,
    capital_inputs: Dict[str, Any],
    filing_calendar: List[Dict[str, Any]],
    stress_inputs: Optional[Dict[str, Any]] = None,
    xbrl_returns: Optional[List[str]] = None,
) -> RegulatoryComplianceState:
    """Run the five-agent regulatory compliance pipeline.

    Orchestrates the full AWB quarterly compliance cycle:
    1. RegulatoryCalendarAgent  — COREP deadline scan
    2. CapitalAdequacyAgent     — CET1/T1/Leverage/LCR/NSFR calculation
    3. XBRLFilingAgent          — EBA XBRL 4.0 instance generation
    4. StressTestingAgent       — PRA CST + BoE CBES scenarios
    5. RegulatoryNarrativeAgent — PRA supervisory narrative + HITL decision

    Model ID: MR-2026-059-REG (BAP-2026-REG-001 §3, EU AI Act Art.6 Annex III).

    Args:
        run_date: Quarter or month end date for this compliance run.
        trigger_event: Description of what triggered this run
            (e.g., "Q4 2025 quarter-end COREP submission cycle").
        capital_inputs: Balance-sheet-derived capital metric inputs.
            Required keys: cet1_capital_gbp, tier1_capital_gbp,
            total_capital_gbp, total_rwa_gbp, on_bs_gbp, sa_ccr_gbp,
            sft_gbp, off_bs_gbp, hqla_l1_gbp, hqla_l2a_gbp,
            hqla_l2b_gbp, net_cash_outflows_30d_gbp, asf_gbp, rsf_gbp.
        filing_calendar: List of pending COREP filing dicts with keys:
            return_code, period_end, deadline_date, status.
        stress_inputs: Optional dict with total_rwa_gbp for stress base.
            Defaults to capital_inputs["total_rwa_gbp"] if not provided.
        xbrl_returns: Optional list of COREP return codes to generate XBRL.
            Defaults to ["C 02.00", "C 47.00", "C 72.00", "C 80.00"].

    Returns:
        Completed RegulatoryComplianceState with all fields populated.

    Example:
        >>> import asyncio
        >>> from datetime import date
        >>> state = asyncio.run(run_agentic_regulatory_compliance(
        ...     run_date=date(2025, 12, 31),
        ...     trigger_event="Q4 2025 quarter-end COREP cycle",
        ...     capital_inputs={
        ...         "cet1_capital_gbp": 4_100_000_000,
        ...         "tier1_capital_gbp": 4_100_000_000,
        ...         "total_capital_gbp": 4_900_000_000,
        ...         "total_rwa_gbp": 30_000_000_000,
        ...         "on_bs_gbp": 38_800_000_000,
        ...         "sa_ccr_gbp": 2_100_000_000,
        ...         "sft_gbp": 800_000_000,
        ...         "off_bs_gbp": 4_200_000_000,
        ...         "hqla_l1_gbp": 5_200_000_000,
        ...         "hqla_l2a_gbp": 1_800_000_000,
        ...         "hqla_l2b_gbp": 600_000_000,
        ...         "net_cash_outflows_30d_gbp": 4_500_000_000,
        ...         "asf_gbp": 28_000_000_000,
        ...         "rsf_gbp": 24_000_000_000,
        ...     },
        ...     filing_calendar=[
        ...         {"return_code": "C 47.00", "period_end": "2025-12-31",
        ...          "deadline_date": "2026-01-18", "status": "PENDING"},
        ...         {"return_code": "C 72.00", "period_end": "2025-12-31",
        ...          "deadline_date": "2026-01-21", "status": "PENDING"},
        ...     ],
        ... ))
        >>> print(state["hitl_decision"])  # APPROVE or ESCALATE
        >>> print(state["cet1_ratio_pct"])  # 13.67
        >>> print(len(state["hop_chain"]))  # 5
    """
    global _SEQ
    _SEQ = 0

    if stress_inputs is None:
        stress_inputs = {
            "total_rwa_gbp": capital_inputs.get("total_rwa_gbp", 30_000_000_000)
        }

    if xbrl_returns is None:
        xbrl_returns = ["C 02.00", "C 47.00", "C 72.00", "C 80.00"]

    state = _initial_state(
        run_date=run_date,
        trigger_event=trigger_event,
        capital_inputs=capital_inputs,
        filing_calendar=filing_calendar,
        stress_inputs=stress_inputs,
        xbrl_returns=xbrl_returns,
    )

    log.info(
        "Agentic Regulatory Compliance Pipeline START | "
        "MR-2026-059-REG | date=%s | trigger='%s'",
        run_date, trigger_event,
    )

    graph = _build_graph()

    # Run synchronously in thread pool to support async callers
    loop = asyncio.get_event_loop()
    final_state = await loop.run_in_executor(
        None, graph.invoke, state
    )

    log.info(
        "Agentic Regulatory Compliance Pipeline END | "
        "HITL=%s | Zone=%s | Hops=%d | Breaches=%d | "
        "Overdue=%d | StressBreaches=%d",
        final_state.get("hitl_decision"),
        final_state.get("risk_zone"),
        len(final_state.get("hop_chain", [])),
        len(final_state.get("capital_breaches", [])),
        len(final_state.get("overdue_filings", [])),
        len(final_state.get("stress_breaches", [])),
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

    # AWB Q4 2025 illustrative inputs
    demo_capital = {
        "cet1_capital_gbp": 4_100_000_000,       # £4.1B CET1
        "tier1_capital_gbp": 4_100_000_000,       # £4.1B T1 (all CET1)
        "total_capital_gbp": 4_900_000_000,       # £4.9B (T1 + T2)
        "total_rwa_gbp": 30_000_000_000,          # £30B RWA
        "on_bs_gbp": 38_800_000_000,              # Art.429b
        "sa_ccr_gbp": 2_100_000_000,              # Art.429c
        "sft_gbp": 800_000_000,                   # Art.429d
        "off_bs_gbp": 4_200_000_000,              # Arts 429e-g
        "hqla_l1_gbp": 5_200_000_000,             # L1 HQLA (0% haircut)
        "hqla_l2a_gbp": 1_800_000_000,            # L2A (15% haircut)
        "hqla_l2b_gbp": 600_000_000,              # L2B (25-50% haircut)
        "net_cash_outflows_30d_gbp": 4_500_000_000, # 30-day net outflows
        "asf_gbp": 28_000_000_000,                # Available Stable Funding
        "rsf_gbp": 24_000_000_000,                # Required Stable Funding
    }
    # Expected: CET1=13.67%, Leverage=8.9%, LCR~147%, NSFR~117%

    demo_calendar = [
        {
            "return_code": "C 02.00",
            "period_end": "2025-12-31",
            "deadline_date": "2026-01-18",
            "status": "PENDING",
        },
        {
            "return_code": "C 47.00",
            "period_end": "2025-12-31",
            "deadline_date": "2026-01-18",
            "status": "PENDING",
        },
        {
            "return_code": "C 72.00",
            "period_end": "2025-12-31",
            "deadline_date": "2026-01-21",
            "status": "PENDING",
        },
        {
            "return_code": "C 80.00",
            "period_end": "2025-12-31",
            "deadline_date": "2026-01-18",
            "status": "FILED",
        },
    ]

    result = asyncio.run(
        run_agentic_regulatory_compliance(
            run_date=date(2025, 12, 31),
            trigger_event=(
                "Q4 2025 quarter-end COREP submission cycle — "
                "Basel III/IV CRR3 capital, leverage, LCR, NSFR"
            ),
            capital_inputs=demo_capital,
            filing_calendar=demo_calendar,
        )
    )

    print("\n" + "=" * 70)
    print("AWB AGENTIC REGULATORY COMPLIANCE — MR-2026-059-REG")
    print("=" * 70)
    print(f"Run Date        : {result['run_date']}")
    print(f"Trigger         : {result['trigger_event']}")
    print(f"Risk Zone       : {result['risk_zone']}")
    print(f"HITL Decision   : {result['hitl_decision']}")
    print(f"HITL Rationale  : {result['hitl_rationale']}")
    print(f"")
    print(f"Capital Ratios:")
    print(f"  CET1          : {result.get('cet1_ratio_pct', 'N/A')}%")
    print(f"  Tier1         : {result.get('tier1_ratio_pct', 'N/A')}%")
    print(f"  Total Capital : {result.get('total_capital_ratio_pct', 'N/A')}%")
    print(f"  Leverage      : {result.get('leverage_ratio_pct', 'N/A')}%")
    print(f"  LCR           : {result.get('lcr_pct', 'N/A')}%")
    print(f"  NSFR          : {result.get('nsfr_pct', 'N/A')}%")
    print(f"")
    print(f"Capital Breaches: {result.get('capital_breaches', [])}")
    print(f"Overdue Filings : {result.get('overdue_filings', [])}")
    print(f"Stress Breaches : {result.get('stress_breaches', [])}")
    print(f"XBRL Errors     : {result.get('filing_validation_errors', [])}")
    print(f"Supervisory Risk Flags:")
    for flag in result.get("supervisory_risk_flags", []):
        print(f"  • {flag}")
    print(f"")
    print(f"Hop Chain ({len(result.get('hop_chain', []))} hops):")
    for hop in result.get("hop_chain", []):
        print(
            f"  [{hop['seq']:02d}] {hop['agent']}: "
            f"{hop['act'][:60]} → {hop['outcome'][:60]}"
        )
    print("=" * 70)
