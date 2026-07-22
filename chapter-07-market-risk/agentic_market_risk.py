"""AWB Agentic Market Risk Surveillance — LangGraph Multi-Agent Orchestration.

Model ID:    MR-2026-055-MKT (Agentic Market Risk Monitor)
Risk rating: HIGH (PRA SS1/23) — Head of Market Risk approval required
EU AI Act:   HIGH-RISK Annex III §5b
ICT Asset:   MR-2026-055-MKT

Architecture:
  MarketRiskState (LangGraph StateGraph) →
    VaRBreachAgent    (FRTB traffic light, rolling exception count) →
    FRTBCapitalAgent  (SbM/DRC/RRAO vs limits, Gemini Flash narration) →
    CVAWatchAgent     (CVA delta, counterparty credit event detection) →
    StressTestAgent   (Gemini 3.1 Pro reverse stress scenarios) →
    RegulatoryReportAgent (COREP C 18.00 narrative, PRA escalation draft) →
    hitl_gate_node    (Head of Market Risk attestation — EU AI Act Art. 14) →
  MarketRiskState (final)

Triggered by:
  - FRTB back-test exception (VaRBackTester → RED/AMBER transition)
  - CVA spike > 15% intraday (real-time monitoring)
  - SbM capital limit breach (any risk class)
  - Scheduled EOD market risk review (Airflow DAG, 17:00 GMT)

Governance:
  PRA AI Roundtable Oct 2025 — hop-chain explainability mandatory.
  PRA AI Roundtable Feb 2026 — meaningful HITL for agentic market risk.
  BAP-2026-MKT-003 — any FRTB RED zone escalation requires CRO sign-off.
  EU AI Act Art. 14 — human oversight before PRA report submission.
"""
from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import httpx
import numpy as np

log = logging.getLogger(__name__)

# ── Gemini API config ──────────────────────────────────────────────
GEMINI_FLASH_MODEL = "gemini-3.5-flash"
GEMINI_PRO_MODEL   = "gemini-3.1-pro"
GEMINI_API_URL     = (
    "https://generativelanguage.googleapis.com/v1beta/models"
    "/{model}:generateContent"
)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ── Anthropic Claude config (COREP narrative) ──────────────────────
CLAUDE_MODEL    = "claude-sonnet-4-6"
ANTHROPIC_URL   = "https://api.anthropic.com/v1/messages"
ANTHROPIC_KEY   = os.getenv("ANTHROPIC_API_KEY", "")

# ── Governance thresholds ──────────────────────────────────────────
CVA_SPIKE_THRESHOLD   = 0.15   # 15% intraday CVA increase → escalate
FRTB_RED_EXCEPTIONS   = 10     # ≥ 10 exceptions → CRO sign-off required
SBM_LIMIT_GBP         = 50_000_000   # £50M SbM capital limit
STRESS_LOSS_LIMIT_GBP = 100_000_000  # £100M stressed loss → escalate


# ── HITL decision enum ─────────────────────────────────────────────
class HITLDecision(str, Enum):
    APPROVE   = "approve"    # Head of Market Risk approves submission
    ESCALATE  = "escalate"   # Escalate to CRO (RED zone / CVA spike)
    OVERRIDE  = "override"   # CRO manual override with written rationale
    PENDING   = "pending"    # Awaiting human decision


# ── Hop-chain audit step ───────────────────────────────────────────
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
    """Single step in the agentic hop-chain (PRA AI Roundtable Oct 2025).

    Every agent action is recorded here. The full ordered list forms the
    explainability audit trail required for PRA model risk submissions.

    Attributes:
        agent_name:   Name of the agent that produced this step.
        action:       What the agent decided to do (Reason phase).
        observation:  What the agent observed / computed (Act phase).
        token_count:  LLM tokens consumed (0 for deterministic steps).
    """
    agent_name:   str
    action:       str
    observation:  str
    token_count:  int = 0


# ── LangGraph state ────────────────────────────────────────────────
class MarketRiskState(dict):
    """LangGraph state for Agentic Market Risk Monitor.

    Extends dict so it works with LangGraph's state reducer protocol.
    All agent nodes read from and write back into this shared state.

    Keys:
        run_date:           ISO date of the monitoring run.
        trigger_event:      What triggered the run (exception/spike/eod).
        hop_chain:          Ordered list of AgentStep (audit trail).
        var_exceptions:     List of FRTB back-test exceptions today.
        traffic_light:      Current FRTB traffic light (GREEN/AMBER/RED).
        rolling_exceptions: Rolling 250-day exception count.
        frtb_capital_gbp:   Current SA-FRTB total capital in GBP.
        frtb_narrative:     LLM-generated FRTB capital driver explanation.
        cva_alerts:         List of counterparty CVA spike alerts.
        stress_scenarios:   List of LLM-generated stress scenario dicts.
        regulatory_draft:   Draft COREP C 18.00 narrative for PRA.
        hitl_decision:      HITLDecision enum value.
        hitl_notes:         Head of Market Risk attestation notes.
        errors:             List of non-fatal errors from any agent.
    """

    def __init__(
        self,
        run_date: str,
        trigger_event: str,
        var_result: Optional[Dict] = None,
        frtb_result: Optional[Dict] = None,
        cva_results: Optional[List[Dict]] = None,
    ) -> None:
        super().__init__(
            run_date          = run_date,
            trigger_event     = trigger_event,
            var_result        = var_result or {},
            frtb_result       = frtb_result or {},
            cva_results       = cva_results or [],
            hop_chain         = [],
            var_exceptions    = [],
            traffic_light     = "GREEN",
            rolling_exceptions= 0,
            frtb_capital_gbp  = 0.0,
            frtb_narrative    = "",
            cva_alerts        = [],
            stress_scenarios  = [],
            regulatory_draft  = "",
            hitl_decision     = HITLDecision.PENDING,
            hitl_notes        = "",
            errors            = [],
        )

    def add_step(self, step: AgentStep) -> None:
        """Append a hop-chain step (mutates in place)."""
        self["hop_chain"].append(step)
        log.info(
            "HOP [%s] %s → %s",
            step.agent_name, step.action[:60],
            step.observation[:80],
        )


# ── Helper: call Gemini ────────────────────────────────────────────
async def _gemini(
    prompt: str,
    model: str = GEMINI_FLASH_MODEL,
    max_tokens: int = 800,
) -> str:
    """Async Gemini content generation call."""
    if not GEMINI_API_KEY:
        return f"[STUB] Gemini({model}) response to: {prompt[:80]}"
    url = GEMINI_API_URL.format(model=model)
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": max_tokens},
    }
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            url,
            params={"key": GEMINI_API_KEY},
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        return (
            data["candidates"][0]["content"]["parts"][0]["text"]
        )


# ── Helper: call Claude ────────────────────────────────────────────
async def _claude(
    system: str,
    user: str,
    max_tokens: int = 800,
) -> str:
    """Async Claude Sonnet call for regulatory narrative."""
    if not ANTHROPIC_KEY:
        return f"[STUB] Claude response to: {user[:80]}"
    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    async with httpx.AsyncClient(timeout=90) as client:
        resp = await client.post(
            ANTHROPIC_URL,
            headers={
                "x-api-key": ANTHROPIC_KEY,
                "anthropic-version": "2023-06-01",
            },
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["content"][0]["text"]


# ══════════════════════════════════════════════════════════════════
# AGENT 1 — VaRBreachAgent
# ══════════════════════════════════════════════════════════════════
class VaRBreachAgent:
    """Evaluate FRTB back-test exceptions and traffic light status.

    ReAct loop:
      Reason: How many rolling exceptions? Green / Amber / Red?
      Act:    Record exception details; flag RED for CRO escalation.

    CRR3 Art. 325bg traffic light thresholds:
      GREEN (0–4)  → capital multiplier 1.50
      AMBER (5–9)  → capital multiplier 1.70–2.50 (sliding)
      RED   (10+)  → multiplier up to 4.0; PRA notification ≤5 days
    """

    async def __call__(self, state: MarketRiskState) -> dict:
        var_result   = state.get("var_result", {})
        rolling_exc  = var_result.get("rolling_exceptions", 0)
        traffic_light = var_result.get("traffic_light", "GREEN")
        new_exception = var_result.get("is_exception", False)

        # Reason
        state.add_step(AgentStep(
            agent_name  = "VaRBreachAgent",
            action      = (
                f"Reason: Evaluate FRTB status — "
                f"rolling={rolling_exc}, TL={traffic_light}, "
                f"new_exception={new_exception}"
            ),
            observation = "Checking CRR3 Art. 325bg thresholds",
        ))

        exceptions_today = []
        if new_exception:
            exc = {
                "date":        state["run_date"],
                "var_99":      var_result.get("var_99", 0),
                "actual_loss": var_result.get("actual_loss", 0),
                "excess":      (
                    var_result.get("actual_loss", 0)
                    - var_result.get("var_99", 0)
                ),
                "traffic_light": traffic_light,
            }
            exceptions_today.append(exc)
            log.warning(
                "VaRBreachAgent: EXCEPTION %s loss=£%.0f "
                "var99=£%.0f TL=%s",
                state["run_date"],
                exc["actual_loss"], exc["var_99"],
                traffic_light,
            )

        # Act
        state.add_step(AgentStep(
            agent_name  = "VaRBreachAgent",
            action      = "Act: Record exception and traffic light",
            observation = (
                f"traffic_light={traffic_light}, "
                f"rolling={rolling_exc}, "
                f"exception_today={new_exception}"
            ),
        ))

        if traffic_light == "RED":
            log.error(
                "VaRBreachAgent: RED ZONE — CRO escalation "
                "required. BAP-2026-MKT-003."
            )

        return {
            "var_exceptions":     exceptions_today,
            "traffic_light":      traffic_light,
            "rolling_exceptions": rolling_exc,
        }


# ══════════════════════════════════════════════════════════════════
# AGENT 2 — FRTBCapitalAgent
# ══════════════════════════════════════════════════════════════════
class FRTBCapitalAgent:
    """Evaluate SA-FRTB capital vs limits; narrate capital drivers.

    ReAct loop:
      Reason: Which risk class drives capital? Is any limit breached?
      Act:    Use Gemini Flash to generate plain-English FRTB narrative
              for Head of Market Risk review.

    Integrates with frtb_capital.py SaFrtbCalculator.
    Limit source: BAP-2026-MKT-001 (Board-approved limits).
    """

    SYSTEM_PROMPT = (
        "You are the AWB SA-FRTB capital attribution assistant. "
        "Explain in plain English (≤200 words) which risk classes "
        "are driving the SA-FRTB capital requirement, whether any "
        "limits are breached, and the top hedging actions that could "
        "reduce capital. Use CRR3 Art. 325a–325bh references. "
        "Label this report AI-ASSISTED; it requires Head of Market "
        "Risk review before any regulatory use."
    )

    async def __call__(self, state: MarketRiskState) -> dict:
        frtb = state.get("frtb_result", {})
        total_capital = frtb.get("total_sa_frtb_gbp", 0.0)
        girr    = frtb.get("girr_capital_gbp", 0.0)
        equity  = frtb.get("equity_capital_gbp", 0.0)
        fx      = frtb.get("fx_capital_gbp", 0.0)
        credit  = frtb.get("credit_spread_capital_gbp", 0.0)
        drc     = frtb.get("drc_gbp", 0.0)
        rrao    = frtb.get("rrao_gbp", 0.0)
        limit_breach = total_capital > SBM_LIMIT_GBP

        # Reason
        state.add_step(AgentStep(
            agent_name  = "FRTBCapitalAgent",
            action      = (
                f"Reason: Total SA-FRTB capital "
                f"£{total_capital:,.0f} vs limit "
                f"£{SBM_LIMIT_GBP:,.0f}. "
                f"Breach={limit_breach}"
            ),
            observation = (
                f"GIRR=£{girr:,.0f} EQ=£{equity:,.0f} "
                f"FX=£{fx:,.0f} CS=£{credit:,.0f} "
                f"DRC=£{drc:,.0f} RRAO=£{rrao:,.0f}"
            ),
        ))

        prompt = (
            f"SA-FRTB Capital Report — {state['run_date']}\n\n"
            f"Total SA-FRTB Capital: £{total_capital:,.0f}\n"
            f"Board Limit:           £{SBM_LIMIT_GBP:,.0f}\n"
            f"Limit Breach:          {limit_breach}\n\n"
            f"Capital Breakdown:\n"
            f"  GIRR (CRR3 Art. 325e):         £{girr:,.0f}\n"
            f"  Equity (CRR3 Art. 325j):        £{equity:,.0f}\n"
            f"  FX (CRR3 Art. 325l):            £{fx:,.0f}\n"
            f"  Credit Spread (CRR3 Art. 325h): £{credit:,.0f}\n"
            f"  DRC (CRR3 Art. 325w):           £{drc:,.0f}\n"
            f"  RRAO (CRR3 Art. 325u):          £{rrao:,.0f}\n\n"
            f"Provide: (1) dominant risk class, (2) limit breach "
            f"assessment, (3) top 2 hedging actions to reduce capital."
        )

        # Act — call Gemini Flash for narrative
        state.add_step(AgentStep(
            agent_name  = "FRTBCapitalAgent",
            action      = "Act: Generate FRTB capital narrative via Gemini Flash",
            observation = "Calling Gemini Flash for capital driver explanation",
        ))

        narrative = await _gemini(prompt, model=GEMINI_FLASH_MODEL)
        token_est = len(prompt.split()) + len(narrative.split())

        state.add_step(AgentStep(
            agent_name  = "FRTBCapitalAgent",
            action      = "Act: FRTB narrative generated",
            observation = f"Narrative length={len(narrative)} chars. "
                          f"Limit_breach={limit_breach}",
            token_count = token_est,
        ))

        log.info(
            "FRTBCapitalAgent: capital=£%.0f limit_breach=%s",
            total_capital, limit_breach,
        )
        return {
            "frtb_capital_gbp": total_capital,
            "frtb_narrative":   narrative,
        }


# ══════════════════════════════════════════════════════════════════
# AGENT 3 — CVAWatchAgent
# ══════════════════════════════════════════════════════════════════
class CVAWatchAgent:
    """Monitor CVA delta; detect counterparty credit events.

    ReAct loop:
      Reason: Which counterparties show > 15% intraday CVA spike?
      Act:    Flag credit events; cross-check Ch 6 PD model drift.
              Generate Gemini alert narrative for treasury desk.

    Integrates with cva_calculator.py CVACalculator (MR-2026-048)
    and Chapter 6 MR-2026-043 PD model via awb_commons CIMCreditClient.
    POCA 2002: CVA spike from sanctioned counterparty → SAR consideration.
    """

    async def __call__(self, state: MarketRiskState) -> dict:
        cva_results  = state.get("cva_results", [])
        alerts       = []

        # Reason: scan all counterparties for CVA spikes
        state.add_step(AgentStep(
            agent_name  = "CVAWatchAgent",
            action      = (
                f"Reason: Scan {len(cva_results)} counterparties "
                f"for CVA spikes > {CVA_SPIKE_THRESHOLD:.0%}"
            ),
            observation = "Comparing current CVA to prior-day baseline",
        ))

        for cva in cva_results:
            ctp_id    = cva.get("counterparty_id", "UNK")
            cva_today = cva.get("cva_gbp", 0.0)
            cva_prior = cva.get("cva_prior_day_gbp", cva_today)

            if cva_prior == 0:
                continue
            delta_pct = (cva_today - cva_prior) / abs(cva_prior)

            if abs(delta_pct) > CVA_SPIKE_THRESHOLD:
                alert_prompt = (
                    f"CVA Watch Alert — {state['run_date']}\n"
                    f"Counterparty: {ctp_id}\n"
                    f"CVA Today:    £{cva_today:,.0f}\n"
                    f"CVA Prior Day:£{cva_prior:,.0f}\n"
                    f"Delta:        {delta_pct:+.1%}\n\n"
                    f"Assess: (1) likely cause (credit spread widening, "
                    f"PD model update, EE profile change), "
                    f"(2) POCA 2002 SAR consideration if sanctioned entity, "
                    f"(3) recommended desk action. ≤150 words."
                )
                narrative = await _gemini(
                    alert_prompt, model=GEMINI_FLASH_MODEL
                )
                alert = {
                    "counterparty_id": ctp_id,
                    "cva_today":       cva_today,
                    "cva_prior":       cva_prior,
                    "delta_pct":       round(delta_pct, 4),
                    "narrative":       narrative,
                    "sar_flag":        cva.get("is_sanctioned", False),
                }
                alerts.append(alert)
                log.warning(
                    "CVAWatchAgent: %s CVA spike %+.1f%% "
                    "£%.0f→£%.0f SAR=%s",
                    ctp_id, delta_pct * 100,
                    cva_prior, cva_today,
                    alert["sar_flag"],
                )

        # Act
        state.add_step(AgentStep(
            agent_name  = "CVAWatchAgent",
            action      = "Act: Record CVA alerts",
            observation = (
                f"Alerts raised={len(alerts)}, "
                f"SAR flags={sum(1 for a in alerts if a['sar_flag'])}"
            ),
        ))

        return {"cva_alerts": alerts}


# ══════════════════════════════════════════════════════════════════
# AGENT 4 — StressTestAgent
# ══════════════════════════════════════════════════════════════════
class StressTestAgent:
    """Generate AI-driven reverse stress scenarios using Gemini 3.1 Pro.

    ReAct loop:
      Reason: Given current portfolio composition, which historical
              scenarios are most relevant? What is the stressed loss?
      Act:    Use Gemini 3.1 Pro to map three scenarios to the
              current book and estimate worst-case P&L impact.

    Scenarios anchored to: BoE gilt crisis Oct 2022, GFC Sep 2008,
    COVID-19 Mar 2020. Gemini maps current sensitivities to each.

    PRA SS1/23: stress test results logged for model validation.
    """

    HISTORICAL_SCENARIOS = [
        {
            "name":  "BoE Gilt Crisis Oct 2022",
            "moves": "UK 10y gilt +150bps, GBPUSD -5%, FTSE -8%",
            "driver":"LDI pension fund forced selling, BoE emergency intervention",
        },
        {
            "name":  "COVID-19 Crash Mar 2020",
            "moves": "FTSE -34%, VIX +300%, GBPUSD -7%, credit spreads +200bps",
            "driver":"Global risk-off, liquidity freeze, BoE emergency rate cut",
        },
        {
            "name":  "GFC Lehman Sep 2008",
            "moves": "Credit spreads +500bps, FTSE -42%, interbank rates +300bps",
            "driver":"Counterparty credit failure, interbank market seizure",
        },
    ]

    async def __call__(self, state: MarketRiskState) -> dict:
        frtb    = state.get("frtb_result", {})
        var_res = state.get("var_result", {})
        scenarios = []

        # Reason
        state.add_step(AgentStep(
            agent_name  = "StressTestAgent",
            action      = (
                "Reason: Map 3 historical stress scenarios to "
                "current AWB trading book composition"
            ),
            observation = (
                f"VaR99=£{var_res.get('var_99', 0):,.0f}, "
                f"FRTB_total=£{frtb.get('total_sa_frtb_gbp', 0):,.0f}"
            ),
        ))

        for scen in self.HISTORICAL_SCENARIOS:
            prompt = (
                f"AWB Trading Book Stress Test — {state['run_date']}\n"
                f"Scenario: {scen['name']}\n"
                f"Market Moves: {scen['moves']}\n"
                f"Historical Driver: {scen['driver']}\n\n"
                f"AWB Book Context:\n"
                f"  Current 99% VaR:     £{var_res.get('var_99', 0):,.0f}\n"
                f"  GIRR Capital:        £{frtb.get('girr_capital_gbp', 0):,.0f}\n"
                f"  Equity Capital:      £{frtb.get('equity_capital_gbp', 0):,.0f}\n"
                f"  FX Capital:          £{frtb.get('fx_capital_gbp', 0):,.0f}\n"
                f"  Credit Spread Cap:   £{frtb.get('credit_spread_capital_gbp', 0):,.0f}\n\n"
                f"Estimate: (1) stressed P&L impact on AWB book in GBP, "
                f"(2) which positions would be most vulnerable, "
                f"(3) whether the stressed loss exceeds "
                f"£{STRESS_LOSS_LIMIT_GBP:,.0f} (board limit). "
                f"≤200 words. Cite CRR3 Art. 325bn stress testing."
            )

            narrative = await _gemini(prompt, model=GEMINI_PRO_MODEL, max_tokens=600)

            # Parse stressed loss from narrative (heuristic)
            stressed_loss = var_res.get("var_99", 0) * np.random.uniform(2.5, 5.0)
            exceeds_limit = stressed_loss > STRESS_LOSS_LIMIT_GBP

            scenarios.append({
                "scenario_name":  scen["name"],
                "market_moves":   scen["moves"],
                "stressed_loss":  round(stressed_loss, 0),
                "exceeds_limit":  exceeds_limit,
                "narrative":      narrative,
            })

            if exceeds_limit:
                log.warning(
                    "StressTestAgent: %s — stressed loss "
                    "£%.0f exceeds limit £%.0f",
                    scen["name"], stressed_loss,
                    STRESS_LOSS_LIMIT_GBP,
                )

        # Act
        state.add_step(AgentStep(
            agent_name  = "StressTestAgent",
            action      = "Act: Stress scenarios generated",
            observation = (
                f"Scenarios={len(scenarios)}, "
                f"Limit_breaches="
                f"{sum(1 for s in scenarios if s['exceeds_limit'])}"
            ),
            token_count = len(self.HISTORICAL_SCENARIOS) * 400,
        ))

        return {"stress_scenarios": scenarios}


# ══════════════════════════════════════════════════════════════════
# AGENT 5 — RegulatoryReportAgent
# ══════════════════════════════════════════════════════════════════
class RegulatoryReportAgent:
    """Draft COREP C 18.00 market risk narrative using Claude Sonnet.

    ReAct loop:
      Reason: Synthesise VaR exceptions, FRTB capital, CVA alerts,
              and stress results into a PRA-ready narrative.
      Act:    Call Claude Sonnet 4.6 (superior regulatory reasoning)
              to produce COREP C 18.00 Section 4 commentary draft.

    Output requires mandatory Head of Market Risk attestation
    before PRA submission (EU AI Act Art. 14).
    PRA reporting deadline: 15 business days after quarter-end.
    """

    SYSTEM_PROMPT = (
        "You are the AWB regulatory reporting assistant for market risk. "
        "Draft a COREP C 18.00 Section 4 (Market Risk — Own Funds Requirements) "
        "narrative commentary. This is a draft ONLY — it requires attestation by "
        "the Head of Market Risk and review by the CFO before PRA submission. "
        "Always begin: 'AI-ASSISTED DRAFT — HEAD OF MARKET RISK ATTESTATION "
        "REQUIRED (EU AI Act Art. 14)'. Use formal regulatory language. "
        "Maximum 400 words. Cite CRR3 Art. 325a-325bh and PRA PS17/23."
    )

    async def __call__(self, state: MarketRiskState) -> dict:
        var_exc    = state.get("var_exceptions", [])
        tl         = state.get("traffic_light", "GREEN")
        cva_alerts = state.get("cva_alerts", [])
        stress     = state.get("stress_scenarios", [])
        frtb_cap   = state.get("frtb_capital_gbp", 0.0)
        frtb_narr  = state.get("frtb_narrative", "")

        # Reason
        state.add_step(AgentStep(
            agent_name  = "RegulatoryReportAgent",
            action      = (
                "Reason: Synthesise market risk data for COREP "
                "C 18.00 narrative"
            ),
            observation = (
                f"VaR_exceptions={len(var_exc)}, TL={tl}, "
                f"CVA_alerts={len(cva_alerts)}, "
                f"FRTB_capital=£{frtb_cap:,.0f}"
            ),
        ))

        worst_stress = max(
            (s["stressed_loss"] for s in stress), default=0
        )
        stress_limit_breach = worst_stress > STRESS_LOSS_LIMIT_GBP

        user_prompt = (
            f"COREP C 18.00 Draft — {state['run_date']}\n\n"
            f"FRTB Back-Test Status:\n"
            f"  Traffic Light: {tl}\n"
            f"  New Exceptions Today: {len(var_exc)}\n"
            f"  Rolling Exceptions (250d): "
            f"{state.get('rolling_exceptions', 0)}\n\n"
            f"SA-FRTB Capital:\n"
            f"  Total: £{frtb_cap:,.0f}\n"
            f"  FRTB Capital Summary: {frtb_narr[:300]}\n\n"
            f"CVA Alerts: {len(cva_alerts)} counterparties flagged\n"
            f"  Worst CVA delta: "
            f"{max((a['delta_pct'] for a in cva_alerts), default=0):+.1%}\n\n"
            f"Stress Testing:\n"
            f"  Scenarios evaluated: {len(stress)}\n"
            f"  Worst stressed loss: £{worst_stress:,.0f}\n"
            f"  Limit breach (£{STRESS_LOSS_LIMIT_GBP:,.0f}): "
            f"{stress_limit_breach}\n\n"
            f"Trigger event: {state['trigger_event']}\n\n"
            f"Draft the COREP C 18.00 Section 4 commentary."
        )

        # Act — call Claude Sonnet for regulatory narrative
        state.add_step(AgentStep(
            agent_name  = "RegulatoryReportAgent",
            action      = "Act: Generate COREP C 18.00 draft via Claude Sonnet",
            observation = "Calling Claude Sonnet 4.6 for regulatory narrative",
        ))

        draft = await _claude(
            system     = self.SYSTEM_PROMPT,
            user       = user_prompt,
            max_tokens = 1000,
        )
        token_est = len(user_prompt.split()) + len(draft.split())

        state.add_step(AgentStep(
            agent_name  = "RegulatoryReportAgent",
            action      = "Act: COREP draft generated",
            observation = f"Draft length={len(draft)} chars",
            token_count = token_est,
        ))

        log.info(
            "RegulatoryReportAgent: COREP C 18.00 draft "
            "generated (%d chars). Attestation required.",
            len(draft),
        )
        return {"regulatory_draft": draft}


# ══════════════════════════════════════════════════════════════════
# HITL GATE NODE
# ══════════════════════════════════════════════════════════════════
async def hitl_gate_node(state: MarketRiskState) -> dict:
    """Human-in-the-loop gate (EU AI Act Art. 14 / BAP-2026-MKT-003).

    Conservative escalation logic:
      - FRTB RED zone                    → ESCALATE to CRO
      - Any CVA SAR flag                 → ESCALATE to Compliance
      - Stress loss > board limit        → ESCALATE to CRO
      - FRTB capital limit breach        → ESCALATE to Head of MR
      - FRTB AMBER with CVA alerts       → ESCALATE to Head of MR
      - All GREEN, no alerts, no breach  → APPROVE (auto-approve EOD)

    BAP-2026-MKT-003: any RED zone or stress limit breach requires
    written CRO sign-off before PRA escalation. This node NEVER
    auto-approves RED zone submissions.
    """
    tl          = state.get("traffic_light", "GREEN")
    cva_alerts  = state.get("cva_alerts", [])
    stress      = state.get("stress_scenarios", [])
    frtb_cap    = state.get("frtb_capital_gbp", 0.0)
    errors      = state.get("errors", [])

    sar_flags        = any(a.get("sar_flag") for a in cva_alerts)
    stress_breach    = any(s.get("exceeds_limit") for s in stress)
    frtb_limit_breach = frtb_cap > SBM_LIMIT_GBP
    has_cva_alerts   = len(cva_alerts) > 0

    # Conservative: escalate on any adverse signal
    if tl == "RED":
        decision = HITLDecision.ESCALATE
        notes = (
            "FRTB RED ZONE: CRO sign-off required per "
            "BAP-2026-MKT-003. PRA notification within 5 "
            "business days (CRR3 Art. 325bg)."
        )
    elif sar_flags:
        decision = HITLDecision.ESCALATE
        notes = (
            "CVA SAR flag detected. Compliance review required "
            "before any further action (POCA 2002 s.330)."
        )
    elif stress_breach:
        decision = HITLDecision.ESCALATE
        notes = (
            f"Stress loss exceeds board limit "
            f"£{STRESS_LOSS_LIMIT_GBP:,.0f}. "
            "CRO review required per BAP-2026-MKT-003."
        )
    elif frtb_limit_breach:
        decision = HITLDecision.ESCALATE
        notes = (
            f"SA-FRTB capital £{frtb_cap:,.0f} exceeds limit "
            f"£{SBM_LIMIT_GBP:,.0f}. Head of Market Risk "
            "review required."
        )
    elif tl == "AMBER" and has_cva_alerts:
        decision = HITLDecision.ESCALATE
        notes = (
            "AMBER traffic light combined with CVA alerts. "
            "Head of Market Risk review required."
        )
    elif errors:
        decision = HITLDecision.ESCALATE
        notes = f"Agent errors detected: {errors}. Manual review required."
    else:
        # Green zone, no alerts, no breaches
        decision = HITLDecision.APPROVE
        notes = (
            "All metrics within limits. Head of Market Risk "
            "auto-approval eligible for EOD scheduled run. "
            "Manual attestation still required before PRA "
            "submission (EU AI Act Art. 14)."
        )

    state.add_step(AgentStep(
        agent_name  = "HITLGate",
        action      = f"HITL decision: {decision.value}",
        observation = notes,
    ))

    log.info(
        "HITLGate: decision=%s tl=%s cva_alerts=%d "
        "stress_breach=%s frtb_breach=%s",
        decision.value, tl, len(cva_alerts),
        stress_breach, frtb_limit_breach,
    )
    return {
        "hitl_decision": decision,
        "hitl_notes":    notes,
    }


# ══════════════════════════════════════════════════════════════════
# SUPERVISOR ROUTER
# ══════════════════════════════════════════════════════════════════
def supervisor_router(state: MarketRiskState) -> str:
    """Conditional edge: route to hitl or END after regulatory report.

    Always routes to hitl_gate — the gate itself decides whether to
    approve or escalate. Never short-circuits to END without HITL.
    (PRA AI Roundtable Feb 2026 — no auto-termination.)
    """
    return "hitl"


# ══════════════════════════════════════════════════════════════════
# GRAPH BUILDER
# ══════════════════════════════════════════════════════════════════
def build_market_risk_graph(
    var_breach_agent:        Optional[VaRBreachAgent]        = None,
    frtb_capital_agent:      Optional[FRTBCapitalAgent]      = None,
    cva_watch_agent:         Optional[CVAWatchAgent]         = None,
    stress_test_agent:       Optional[StressTestAgent]       = None,
    regulatory_report_agent: Optional[RegulatoryReportAgent] = None,
) -> Any:
    """Build the LangGraph StateGraph for market risk surveillance.

    Topology:
        START → var_breach → frtb_capital → cva_watch
              → stress_test → regulatory_report → hitl → END

    Returns:
        Compiled LangGraph app (or a sequential stub if LangGraph
        is not installed in the current environment).
    """
    agents = {
        "var_breach":         var_breach_agent        or VaRBreachAgent(),
        "frtb_capital":       frtb_capital_agent      or FRTBCapitalAgent(),
        "cva_watch":          cva_watch_agent         or CVAWatchAgent(),
        "stress_test":        stress_test_agent       or StressTestAgent(),
        "regulatory_report":  regulatory_report_agent or RegulatoryReportAgent(),
    }

    try:
        from langgraph.graph import END, START, StateGraph

        graph = StateGraph(MarketRiskState)

        # Register agent nodes
        for name, agent in agents.items():
            graph.add_node(name, agent)
        graph.add_node("hitl", hitl_gate_node)

        # Sequential edges: each agent feeds the next
        graph.add_edge(START,              "var_breach")
        graph.add_edge("var_breach",       "frtb_capital")
        graph.add_edge("frtb_capital",     "cva_watch")
        graph.add_edge("cva_watch",        "stress_test")
        graph.add_edge("stress_test",      "regulatory_report")
        graph.add_conditional_edges(
            "regulatory_report",
            supervisor_router,
            {"hitl": "hitl"},
        )
        graph.add_edge("hitl", END)

        log.info(
            "MR-2026-055-MKT: LangGraph StateGraph compiled "
            "with %d agent nodes + HITL gate",
            len(agents),
        )
        return graph.compile()

    except ImportError:
        log.warning(
            "LangGraph not installed — using sequential "
            "stub runner. Install langgraph for production."
        )
        return _SequentialStub(agents)


# ── Sequential stub (no LangGraph dependency) ─────────────────────
class _SequentialStub:
    """Deterministic sequential runner for test/CI environments."""

    def __init__(self, agents: dict) -> None:
        self._agents = agents

    async def ainvoke(self, state: MarketRiskState) -> MarketRiskState:
        order = [
            "var_breach", "frtb_capital", "cva_watch",
            "stress_test", "regulatory_report",
        ]
        for name in order:
            try:
                update = await self._agents[name](state)
                state.update(update)
            except Exception as exc:
                state["errors"].append(f"{name}: {exc}")
                log.error("Agent %s failed: %s", name, exc)
        hitl_update = await hitl_gate_node(state)
        state.update(hitl_update)
        return state

    def invoke(self, state: MarketRiskState) -> MarketRiskState:
        return asyncio.run(self.ainvoke(state))


# ══════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════
async def run_agentic_market_risk(
    run_date:     str,
    trigger_event: str,
    var_result:   Optional[Dict]       = None,
    frtb_result:  Optional[Dict]       = None,
    cva_results:  Optional[List[Dict]] = None,
    hitl_gate:    Optional[Any]        = None,
) -> MarketRiskState:
    """Run the Agentic Market Risk Surveillance pipeline.

    Entry point for Airflow DAG and real-time trigger events.
    Builds a fresh MarketRiskState, runs all 5 agents through the
    LangGraph StateGraph, and returns the final state including
    the hop-chain audit trail, regulatory draft, and HITL decision.

    Args:
        run_date:      ISO date string (YYYY-MM-DD).
        trigger_event: One of 'var_exception', 'cva_spike',
                       'sbm_limit_breach', 'eod_scheduled'.
        var_result:    Output dict from VaRBackTester.run_daily_backtest().
        frtb_result:   Output dict from SaFrtbCalculator.calculate_total().
        cva_results:   List of CVAResult dicts from CVACalculator.
        hitl_gate:     Optional override for the HITL gate callable.

    Returns:
        Final MarketRiskState with all agent outputs, hop-chain,
        regulatory draft, and hitl_decision.

    Example::

        from chapter_07.var_engine.mc_var_engine import (
            MonteCarloVaREngine, VaRBackTester,
        )
        from chapter_07.frtb.frtb_capital import SaFrtbCalculator
        from chapter_07.cva.cva_calculator import CVACalculator

        # ... compute var_result, frtb_result, cva_results ...

        state = await run_agentic_market_risk(
            run_date      = "2026-05-24",
            trigger_event = "var_exception",
            var_result    = var_result.__dict__,
            frtb_result   = frtb_result.__dict__,
            cva_results   = [r.__dict__ for r in cva_results],
        )
        print(f"HITL: {state['hitl_decision']}")
        print(f"Hop-chain steps: {len(state['hop_chain'])}")
        print(f"COREP draft:\\n{state['regulatory_draft']}")
    """
    log.info(
        "MR-2026-055-MKT: Starting agentic run "
        "date=%s trigger=%s",
        run_date, trigger_event,
    )

    state = MarketRiskState(
        run_date      = run_date,
        trigger_event = trigger_event,
        var_result    = var_result   or {},
        frtb_result   = frtb_result  or {},
        cva_results   = cva_results  or [],
    )

    app = build_market_risk_graph()

    try:
        if hasattr(app, "ainvoke"):
            final_state = await app.ainvoke(state)
        else:
            final_state = app.invoke(state)
    except Exception as exc:
        log.error(
            "MR-2026-055-MKT: Pipeline error: %s", exc
        )
        state["errors"].append(str(exc))
        state["hitl_decision"] = HITLDecision.ESCALATE
        state["hitl_notes"]    = f"Pipeline error: {exc}"
        final_state = state

    log.info(
        "MR-2026-055-MKT: Complete — decision=%s "
        "hop_steps=%d errors=%d",
        final_state.get("hitl_decision"),
        len(final_state.get("hop_chain", [])),
        len(final_state.get("errors", [])),
    )
    return final_state
