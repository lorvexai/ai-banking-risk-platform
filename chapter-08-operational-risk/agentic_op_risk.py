"""AWB Agentic Operational Risk Monitor — LangGraph Multi-Agent Orchestration.

Model ID:    MR-2026-056-OPS (Agentic Op Risk Monitor)
Risk rating: HIGH (PRA SS1/23) — COO approval required for major incidents
EU AI Act:   HIGH-RISK Annex III §5b
ICT Asset:   OPS-2026-056

Architecture:
  OpRiskState (LangGraph StateGraph) →
    FraudTriageAgent      (payment + credit fraud severity; Gemini Flash) →
    OpLossClassifierAgent (Basel III SMA category; Gemini Pro reasoning) →
    SMAImpactAgent        (ILM delta on SMA capital; deterministic + narrative) →
    DORAIncidentAgent     (DORA Art. 17 classification; reporting obligation) →
    ConsumerDutyAgent     (FCA PS22/9 Consumer Duty notification check) →
    hitl_gate_node        (COO/CISO escalation — EU AI Act Art. 14) →
  OpRiskState (final)

Triggered by:
  - Payment fraud alert CRITICAL or HIGH severity (MR-2026-049)
  - New op loss event above £50K (MR-2026-050 NLP extractor output)
  - DORA ICT incident ticket opened (ServiceNow webhook)
  - EOD batch: reconcile all intraday op risk events (Airflow 18:00 GMT)

Governance:
  PRA AI Roundtable Oct 2025 — hop-chain explainability mandatory.
  BAP-2026-OPS-001 — op loss events > £100K require COO notification.
  DORA Art. 17 — major ICT incidents reported to PRA within 4 hours.
  FCA PS22/9 — Consumer Duty customer harm notification within 72 hours.
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import httpx

log = logging.getLogger(__name__)

# ── LLM config ────────────────────────────────────────────────────
GEMINI_FLASH_MODEL = "gemini-3.5-flash"
GEMINI_PRO_MODEL   = "gemini-3.1-pro"
GEMINI_API_URL     = (
    "https://generativelanguage.googleapis.com/v1beta/models"
    "/{model}:generateContent"
)
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
ANTHROPIC_URL   = "https://api.anthropic.com/v1/messages"
ANTHROPIC_KEY   = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL    = "claude-sonnet-4-6"

# ── Governance thresholds ──────────────────────────────────────────
COO_NOTIFICATION_GBP   = 100_000     # BAP-2026-OPS-001
DORA_MAJOR_THRESHOLD   = 50_000      # DORA Art. 17 — £50K+ or 100K+ customers
CONSUMER_DUTY_THRESHOLD = 1_000      # PS22/9 — affects > 1,000 customers
ILM_CHANGE_ALERT        = 0.05       # Flag if ILM changes by > 5%

# ── Basel III SMA constants ────────────────────────────────────────
BIC_GBP            = 52_000_000      # AWB Business Indicator Component
ILM_PRE_AI         = 2.94            # Pre-programme ILM baseline
ILM_DENOMINATOR    = 0.035 * BIC_GBP


# ── HITL decision ──────────────────────────────────────────────────
class HITLDecision(str, Enum):
    APPROVE   = "approve"
    ESCALATE  = "escalate"
    OVERRIDE  = "override"
    PENDING   = "pending"


# ── Basel III SMA event categories ────────────────────────────────
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


class SMACategory(str, Enum):
    INTERNAL_FRAUD        = "ET1_INTERNAL_FRAUD"
    EXTERNAL_FRAUD        = "ET2_EXTERNAL_FRAUD"
    EMPLOYMENT_PRACTICES  = "ET3_EMPLOYMENT_PRACTICES"
    CLIENTS_PRODUCTS      = "ET4_CLIENTS_PRODUCTS"
    PHYSICAL_ASSETS       = "ET5_PHYSICAL_ASSETS"
    BUSINESS_DISRUPTION   = "ET6_BUSINESS_DISRUPTION"
    EXECUTION_DELIVERY    = "ET7_EXECUTION_DELIVERY"
    UNKNOWN               = "UNKNOWN"


# ── DORA classification ────────────────────────────────────────────
class DORAClassification(str, Enum):
    MAJOR          = "MAJOR"          # PRA notification ≤ 4 hours
    SIGNIFICANT    = "SIGNIFICANT"    # Internal escalation ≤ 24 hours
    STANDARD       = "STANDARD"       # Normal incident management
    NOT_ICT        = "NOT_ICT"        # Not an ICT incident


# ── Hop-chain audit step ───────────────────────────────────────────
@dataclass
class AgentStep:
    """Single step in the agentic hop-chain.

    Mandatory per PRA AI Roundtable Oct 2025. Every Reason and Act
    step is recorded for model risk review and PRA audit.

    Attributes:
        agent_name:   Agent that produced this step.
        action:       What the agent decided / observed.
        observation:  Result of the action.
        token_count:  LLM tokens consumed (0 for deterministic steps).
    """
    agent_name:   str
    action:       str
    observation:  str
    token_count:  int = 0


# ── LangGraph state ────────────────────────────────────────────────
class OpRiskState(dict):
    """LangGraph state for the Agentic Op Risk Monitor.

    Extends dict for LangGraph state reducer protocol.
    All five agents read from and write into this shared state.

    Keys:
        run_date:             ISO date of the monitoring run.
        trigger_event:        What triggered the run.
        fraud_alerts:         List of fraud alert dicts (MR-2026-049).
        op_loss_events:       List of op loss event dicts (MR-2026-050).
        ict_incidents:        List of ServiceNow ICT incident dicts.
        hop_chain:            Ordered AgentStep audit trail.
        fraud_triage:         FraudTriageAgent output dict.
        sma_classification:   OpLossClassifierAgent output dict.
        sma_impact:           SMAImpactAgent output dict.
        dora_classification:  DORAIncidentAgent output dict.
        consumer_duty:        ConsumerDutyAgent output dict.
        hitl_decision:        HITLDecision enum value.
        hitl_notes:           Escalation rationale for COO/CISO.
        errors:               Non-fatal errors from any agent.
    """

    def __init__(
        self,
        run_date:      str,
        trigger_event: str,
        fraud_alerts:  Optional[List[Dict]] = None,
        op_loss_events: Optional[List[Dict]] = None,
        ict_incidents: Optional[List[Dict]] = None,
    ) -> None:
        super().__init__(
            run_date            = run_date,
            trigger_event       = trigger_event,
            fraud_alerts        = fraud_alerts    or [],
            op_loss_events      = op_loss_events  or [],
            ict_incidents       = ict_incidents   or [],
            hop_chain           = [],
            fraud_triage        = {},
            sma_classification  = {},
            sma_impact          = {},
            dora_classification = {},
            consumer_duty       = {},
            hitl_decision       = HITLDecision.PENDING,
            hitl_notes          = "",
            errors              = [],
        )

    def add_step(self, step: AgentStep) -> None:
        self["hop_chain"].append(step)
        log.info(
            "HOP [%s] %s → %s",
            step.agent_name, step.action[:60],
            step.observation[:80],
        )


# ── Helper: Gemini ─────────────────────────────────────────────────
async def _gemini(
    prompt: str,
    model: str = GEMINI_FLASH_MODEL,
    max_tokens: int = 600,
) -> str:
    if not GEMINI_API_KEY:
        return f"[STUB] Gemini({model}): {prompt[:80]}"
    url = GEMINI_API_URL.format(model=model)
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": max_tokens},
    }
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            url, params={"key": GEMINI_API_KEY}, json=payload
        )
        resp.raise_for_status()
        return (
            resp.json()["candidates"][0]
            ["content"]["parts"][0]["text"]
        )


# ── Helper: Claude ─────────────────────────────────────────────────
async def _claude(
    system: str, user: str, max_tokens: int = 800
) -> str:
    if not ANTHROPIC_KEY:
        return f"[STUB] Claude: {user[:80]}"
    payload = {
        "model": CLAUDE_MODEL, "max_tokens": max_tokens,
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
        return resp.json()["content"][0]["text"]


# ══════════════════════════════════════════════════════════════════
# AGENT 1 — FraudTriageAgent
# ══════════════════════════════════════════════════════════════════
class FraudTriageAgent:
    """Triage payment and credit fraud alerts; generate narrative.

    ReAct loop:
      Reason: What is the combined fraud severity across all alerts?
              Is total exposure above the COO notification threshold?
      Act:    Gemini Flash generates fraud triage narrative for the
              Fraud Operations team; flags CRITICAL for auto-block.

    Reads: MR-2026-049 PaymentFraudDetector alerts
           MR-2026-051 CreditFraudScorer results
    """

    async def __call__(self, state: OpRiskState) -> dict:
        alerts    = state.get("fraud_alerts", [])
        critical  = [a for a in alerts if a.get("severity") == "CRITICAL"]
        high      = [a for a in alerts if a.get("severity") == "HIGH"]
        total_exp = sum(a.get("amount_gbp", 0) for a in alerts)
        coo_flag  = total_exp > COO_NOTIFICATION_GBP

        state.add_step(AgentStep(
            agent_name  = "FraudTriageAgent",
            action      = (
                f"Reason: {len(alerts)} alerts — "
                f"CRITICAL={len(critical)}, HIGH={len(high)}, "
                f"total_exposure=£{total_exp:,.0f}"
            ),
            observation = f"COO_notification={coo_flag}",
        ))

        prompt = (
            f"AWB Fraud Triage — {state['run_date']}\n\n"
            f"Total Alerts:          {len(alerts)}\n"
            f"CRITICAL (auto-block): {len(critical)}\n"
            f"HIGH (human review):   {len(high)}\n"
            f"Total Exposure:        £{total_exp:,.0f}\n"
            f"COO Notification:      {coo_flag} "
            f"(threshold £{COO_NOTIFICATION_GBP:,})\n\n"
            f"CRITICAL Alerts:\n"
            + "\n".join(
                f"  - tx={a.get('transaction_id','?')} "
                f"£{a.get('amount_gbp',0):,.0f} "
                f"rules={a.get('triggered_rules',[])}"
                for a in critical[:5]
            )
            + "\n\nGenerate: (1) fraud triage summary for ops team, "
            f"(2) immediate actions, (3) whether POCA 2002 s.330 "
            f"SAR obligation is triggered. ≤200 words."
        )
        narrative = await _gemini(prompt)

        state.add_step(AgentStep(
            agent_name  = "FraudTriageAgent",
            action      = "Act: Fraud triage narrative generated",
            observation = (
                f"CRITICAL={len(critical)}, exposure=£{total_exp:,.0f}, "
                f"COO_flag={coo_flag}"
            ),
            token_count = len(prompt.split()) + len(narrative.split()),
        ))

        if critical:
            log.warning(
                "FraudTriageAgent: %d CRITICAL alerts £%.0f total. "
                "Auto-block triggered. COO_flag=%s",
                len(critical), total_exp, coo_flag,
            )

        return {
            "fraud_triage": {
                "total_alerts":   len(alerts),
                "critical_count": len(critical),
                "high_count":     len(high),
                "total_exposure_gbp": total_exp,
                "coo_notification":   coo_flag,
                "narrative":          narrative,
            }
        }


# ══════════════════════════════════════════════════════════════════
# AGENT 2 — OpLossClassifierAgent
# ══════════════════════════════════════════════════════════════════
class OpLossClassifierAgent:
    """Improve Basel III SMA category classification via LLM reasoning.

    ReAct loop:
      Reason: The NLP keyword pre-filter (MR-2026-050) assigns a
              provisional SMA category. LLM review catches misclassification
              that keyword matching misses (e.g. a payment fraud loss
              miscategorised as EXECUTION_DELIVERY).
      Act:    Gemini Pro reviews each event and confirms or corrects
              the category; provides regulatory rationale.

    CRR3 Art. 316: SMA category determines capital bucket.
    Misclassification risk: under-stated capital or incorrect ILM.
    """

    SYSTEM_PROMPT = (
        "You are the AWB operational risk categorisation specialist. "
        "For each loss event, confirm or correct the provisional Basel III "
        "SMA category (ET1–ET7 per CRR3 Art. 316), provide a one-sentence "
        "regulatory rationale, and assign a confidence score 0–1. "
        "Return JSON: {event_id, confirmed_category, rationale, confidence}."
    )

    async def __call__(self, state: OpRiskState) -> dict:
        events = state.get("op_loss_events", [])

        state.add_step(AgentStep(
            agent_name  = "OpLossClassifierAgent",
            action      = f"Reason: Review {len(events)} op loss events for SMA category accuracy",
            observation = "Using Gemini Pro for LLM-assisted category review",
        ))

        classifications = []
        total_tokens    = 0

        for ev in events:
            event_id    = ev.get("event_id", "UNK")
            description = ev.get("description", "")
            prov_cat    = ev.get("category", "UNKNOWN")
            amount      = ev.get("loss_amount_gbp", 0)

            prompt = (
                f"Basel III SMA Category Review\n"
                f"Event ID:              {event_id}\n"
                f"Provisional Category:  {prov_cat}\n"
                f"Loss Amount:           £{amount:,.0f}\n"
                f"Description:           {description[:300]}\n\n"
                f"Confirm or correct the SMA category (ET1–ET7). "
                f"Return JSON only: "
                f"{{\"event_id\":\"{event_id}\","
                f"\"confirmed_category\":\"...\","
                f"\"rationale\":\"...\","
                f"\"confidence\":0.0}}"
            )
            response = await _gemini(prompt, model=GEMINI_PRO_MODEL)
            total_tokens += len(prompt.split()) + len(response.split())

            # Parse or default
            import json as _json
            try:
                result = _json.loads(response)
            except Exception:
                result = {
                    "event_id":          event_id,
                    "confirmed_category": prov_cat,
                    "rationale":         "LLM parse error — provisional category retained",
                    "confidence":        0.6,
                }
            classifications.append(result)

            if result["confirmed_category"] != prov_cat:
                log.warning(
                    "OpLossClassifier: event=%s category changed "
                    "%s→%s (confidence=%.2f)",
                    event_id, prov_cat,
                    result["confirmed_category"],
                    result["confidence"],
                )

        state.add_step(AgentStep(
            agent_name  = "OpLossClassifierAgent",
            action      = "Act: SMA category classifications complete",
            observation = (
                f"events={len(classifications)}, "
                f"reclassified="
                f"{sum(1 for c in classifications if c.get('confirmed_category') != events[i].get('category','') for i, c in enumerate(classifications) if i < len(events))}"
            ),
            token_count = total_tokens,
        ))

        return {"sma_classification": {"classifications": classifications}}


# ══════════════════════════════════════════════════════════════════
# AGENT 3 — SMAImpactAgent
# ══════════════════════════════════════════════════════════════════
class SMAImpactAgent:
    """Calculate ILM and SMA capital impact of new loss events.

    ReAct loop:
      Reason: How do new op loss events change the 10-year average?
              What is the delta to the ILM and hence SMA capital?
      Act:    Compute updated ILM; use Claude Sonnet to narrate
              capital change for CFO and Board Risk Committee.

    CRR3 Art. 323: ILM = 1 + ln(1 + avg_annual_loss / (0.035 * BIC))
    AWB pre-AI ILM: 2.94 → post-AI target: 2.72
    Capital saving per 0.01 ILM improvement ≈ £520K at AWB BIC.
    """

    SYSTEM_PROMPT = (
        "You are the AWB operational risk capital specialist. "
        "Explain in plain English (≤200 words) how new operational loss events "
        "affect the Basel III SMA Internal Loss Multiplier and SMA capital requirement. "
        "Use CRR3 Art. 316–323 references. Label the report AI-ASSISTED; it requires "
        "CFO review before Board Risk Committee submission."
    )

    async def __call__(self, state: OpRiskState) -> dict:
        events      = state.get("op_loss_events", [])
        new_loss    = sum(e.get("loss_amount_gbp", 0) for e in events)

        # Deterministic ILM calculation (CRR3 Art. 323)
        # Simulate: add new losses to 10-year window
        # AWB baseline: avg annual loss = £12.8M pre-AI, £8.4M post-AI
        baseline_avg = 8_400_000   # post-AI programme baseline
        incremental  = new_loss / 10.0  # spread over 10-year window

        new_avg_annual = baseline_avg + incremental
        new_ilm = 1.0 + math.log(
            1.0 + new_avg_annual / ILM_DENOMINATOR
        )
        new_ilm = max(1.0, min(10.0, new_ilm))
        ilm_delta = new_ilm - ILM_PRE_AI

        sma_capital_pre  = BIC_GBP * ILM_PRE_AI
        sma_capital_post = BIC_GBP * new_ilm
        capital_delta    = sma_capital_post - sma_capital_pre

        state.add_step(AgentStep(
            agent_name  = "SMAImpactAgent",
            action      = (
                f"Reason: new_loss=£{new_loss:,.0f}, "
                f"baseline_avg_annual=£{baseline_avg:,.0f}"
            ),
            observation = (
                f"new_ILM={new_ilm:.4f}, delta={ilm_delta:+.4f}, "
                f"capital_delta=£{capital_delta:+,.0f}"
            ),
        ))

        prompt = (
            f"SMA Capital Impact Report — {state['run_date']}\n\n"
            f"New Op Loss Events:     £{new_loss:,.0f} (today)\n"
            f"Baseline Avg Annual:    £{baseline_avg:,.0f}\n"
            f"Updated Avg Annual:     £{new_avg_annual:,.0f}\n"
            f"Pre-event ILM:          {ILM_PRE_AI:.4f}\n"
            f"Updated ILM:            {new_ilm:.4f} ({ilm_delta:+.4f})\n"
            f"SMA Capital Change:     £{capital_delta:+,.0f}\n"
            f"CRR3 Art. 317 BIC:      £{BIC_GBP/1e6:.0f}M\n\n"
            f"Explain the capital impact and whether the board "
            f"AI programme target (ILM 2.72) remains achievable."
        )
        narrative = await _claude(
            system     = self.SYSTEM_PROMPT,
            user       = prompt,
            max_tokens = 600,
        )

        state.add_step(AgentStep(
            agent_name  = "SMAImpactAgent",
            action      = "Act: SMA capital narrative generated via Claude Sonnet",
            observation = f"ILM={new_ilm:.4f}, capital_delta=£{capital_delta:+,.0f}",
            token_count = len(prompt.split()) + len(narrative.split()),
        ))

        alert_ilm = abs(ilm_delta) > ILM_CHANGE_ALERT
        if alert_ilm:
            log.warning(
                "SMAImpactAgent: ILM delta %+.4f exceeds %.2f threshold. "
                "CFO notification required.",
                ilm_delta, ILM_CHANGE_ALERT,
            )

        return {
            "sma_impact": {
                "new_ilm":          round(new_ilm, 4),
                "ilm_delta":        round(ilm_delta, 4),
                "sma_capital_delta": round(capital_delta, 0),
                "alert_triggered":   alert_ilm,
                "narrative":         narrative,
            }
        }


# ══════════════════════════════════════════════════════════════════
# AGENT 4 — DORAIncidentAgent
# ══════════════════════════════════════════════════════════════════
class DORAIncidentAgent:
    """Classify ICT incidents under DORA Art. 17; set reporting obligation.

    ReAct loop:
      Reason: Does this incident meet DORA major incident criteria?
              Customer impact, operational disruption, financial loss.
      Act:    Gemini Pro classifies severity; sets PRA notification
              timer (4 hours for MAJOR, 24 hours for SIGNIFICANT).

    DORA Art. 17 major incident criteria (EBA RTS 2024):
      - Transactions affected > 25% of normal daily volume, OR
      - Service unavailable > 2 hours, OR
      - Financial impact > £50K, OR
      - Reputational risk assessment: HIGH.
    """

    MAJOR_CRITERIA = {
        "tx_pct_threshold":    0.25,
        "downtime_hours":      2.0,
        "financial_gbp":       50_000,
    }

    async def __call__(self, state: OpRiskState) -> dict:
        incidents = state.get("ict_incidents", [])
        results   = []

        state.add_step(AgentStep(
            agent_name  = "DORAIncidentAgent",
            action      = f"Reason: Classify {len(incidents)} ICT incidents under DORA Art. 17",
            observation = "Applying EBA RTS 2024 DORA major incident criteria",
        ))

        for inc in incidents:
            inc_id      = inc.get("incident_id", "INC-?")
            description = inc.get("description", "")
            tx_pct      = inc.get("transactions_affected_pct", 0)
            downtime    = inc.get("downtime_hours", 0)
            financial   = inc.get("financial_impact_gbp", 0)
            customers   = inc.get("customers_affected", 0)

            # Deterministic pre-classification
            is_major = (
                tx_pct >= self.MAJOR_CRITERIA["tx_pct_threshold"]
                or downtime >= self.MAJOR_CRITERIA["downtime_hours"]
                or financial >= self.MAJOR_CRITERIA["financial_gbp"]
            )

            prompt = (
                f"DORA Art. 17 Incident Classification\n"
                f"Incident ID:           {inc_id}\n"
                f"Description:           {description[:300]}\n"
                f"Transactions affected: {tx_pct:.1%}\n"
                f"Service downtime:      {downtime:.1f} hours\n"
                f"Financial impact:      £{financial:,.0f}\n"
                f"Customers affected:    {customers:,}\n"
                f"Pre-classification:    {'MAJOR' if is_major else 'SIGNIFICANT/STANDARD'}\n\n"
                f"Classify as MAJOR/SIGNIFICANT/STANDARD/NOT_ICT. "
                f"State: (1) which DORA Art. 17 criterion is met, "
                f"(2) PRA notification deadline, "
                f"(3) required initial report content. ≤150 words."
            )
            narrative = await _gemini(prompt, model=GEMINI_PRO_MODEL)

            classification = DORAClassification.MAJOR if is_major else DORAClassification.SIGNIFICANT
            notification_hours = 4 if classification == DORAClassification.MAJOR else 24

            results.append({
                "incident_id":         inc_id,
                "classification":      classification.value,
                "notification_hours":  notification_hours,
                "customers_affected":  customers,
                "narrative":           narrative,
            })

            if classification == DORAClassification.MAJOR:
                log.error(
                    "DORAIncidentAgent: MAJOR incident %s — "
                    "PRA notification within %d hours. "
                    "CISO escalation required.",
                    inc_id, notification_hours,
                )

        state.add_step(AgentStep(
            agent_name  = "DORAIncidentAgent",
            action      = "Act: DORA classifications complete",
            observation = (
                f"incidents={len(results)}, "
                f"major={sum(1 for r in results if r['classification']=='MAJOR')}"
            ),
        ))

        return {"dora_classification": {"incidents": results}}


# ══════════════════════════════════════════════════════════════════
# AGENT 5 — ConsumerDutyAgent
# ══════════════════════════════════════════════════════════════════
class ConsumerDutyAgent:
    """Check FCA Consumer Duty PS22/9 customer notification obligations.

    ReAct loop:
      Reason: Do any fraud events or op loss incidents constitute
              foreseeable harm to retail customers under PS22/9?
      Act:    Gemini Flash determines notification obligation;
              drafts customer communication if required.

    FCA PS22/9 Consumer Duty (effective 31 July 2023):
      - Firms must act to avoid foreseeable harm to retail customers.
      - Material harm events triggering customer harm assessment:
        payment fraud, data breach, service outage > 2h.
      - Notification required if > 1,000 retail customers affected.
    """

    async def __call__(self, state: OpRiskState) -> dict:
        fraud_triage = state.get("fraud_triage", {})
        dora_class   = state.get("dora_classification", {})
        incidents    = dora_class.get("incidents", [])
        alerts       = state.get("fraud_alerts", [])

        total_customers = sum(
            inc.get("customers_affected", 0) for inc in incidents
        )
        major_incidents = [
            i for i in incidents if i["classification"] == "MAJOR"
        ]
        fraud_customer_count = len(alerts)  # proxy: 1 customer per alert
        notification_required = (
            total_customers > CONSUMER_DUTY_THRESHOLD
            or len(major_incidents) > 0
        )

        state.add_step(AgentStep(
            agent_name  = "ConsumerDutyAgent",
            action      = (
                f"Reason: customers_affected={total_customers}, "
                f"major_incidents={len(major_incidents)}, "
                f"fraud_alerts={len(alerts)}"
            ),
            observation = f"notification_required={notification_required}",
        ))

        prompt = (
            f"FCA Consumer Duty PS22/9 Assessment — {state['run_date']}\n\n"
            f"Customers affected by ICT incidents: {total_customers:,}\n"
            f"DORA MAJOR incidents:               {len(major_incidents)}\n"
            f"Payment fraud alerts:               {len(alerts)}\n"
            f"Total fraud exposure:               "
            f"£{fraud_triage.get('total_exposure_gbp', 0):,.0f}\n"
            f"Notification threshold:             {CONSUMER_DUTY_THRESHOLD:,} customers\n\n"
            f"Assess: (1) Is PS22/9 foreseeable harm test met? "
            f"(2) Notification obligation and 72-hour deadline, "
            f"(3) Required communication to affected customers. ≤200 words."
        )
        narrative = await _gemini(prompt)

        state.add_step(AgentStep(
            agent_name  = "ConsumerDutyAgent",
            action      = "Act: Consumer Duty assessment complete",
            observation = (
                f"notification_required={notification_required}, "
                f"customers={total_customers}"
            ),
            token_count = len(prompt.split()) + len(narrative.split()),
        ))

        if notification_required:
            log.warning(
                "ConsumerDutyAgent: PS22/9 notification required — "
                "%d customers affected. 72-hour deadline.",
                total_customers,
            )

        return {
            "consumer_duty": {
                "notification_required": notification_required,
                "customers_affected":    total_customers,
                "major_incidents":       len(major_incidents),
                "narrative":             narrative,
            }
        }


# ══════════════════════════════════════════════════════════════════
# HITL GATE
# ══════════════════════════════════════════════════════════════════
async def hitl_gate_node(state: OpRiskState) -> dict:
    """Human-in-the-loop gate (EU AI Act Art. 14 / BAP-2026-OPS-001).

    Conservative escalation:
      - DORA MAJOR incident           → ESCALATE to CISO + COO
      - Fraud exposure > £100K        → ESCALATE to COO
      - ILM alert triggered           → ESCALATE to CFO
      - Consumer Duty notification    → ESCALATE to Compliance
      - Any agent error               → ESCALATE (safety-first)
      - All GREEN                     → APPROVE (EOD scheduled run)
    """
    fraud_triage = state.get("fraud_triage", {})
    sma_impact   = state.get("sma_impact", {})
    dora         = state.get("dora_classification", {})
    duty         = state.get("consumer_duty", {})
    errors       = state.get("errors", [])

    dora_major       = any(
        i["classification"] == "MAJOR"
        for i in dora.get("incidents", [])
    )
    coo_required     = fraud_triage.get("coo_notification", False)
    ilm_alert        = sma_impact.get("alert_triggered", False)
    duty_notify      = duty.get("notification_required", False)

    if dora_major:
        decision = HITLDecision.ESCALATE
        notes = (
            "DORA MAJOR incident: PRA notification required within "
            "4 hours. CISO and COO sign-off required per "
            "BAP-2026-OPS-001 and DORA Art. 17."
        )
    elif coo_required:
        decision = HITLDecision.ESCALATE
        notes = (
            f"Fraud exposure £{fraud_triage.get('total_exposure_gbp',0):,.0f} "
            f"exceeds £{COO_NOTIFICATION_GBP:,} threshold. "
            "COO notification required per BAP-2026-OPS-001."
        )
    elif ilm_alert:
        decision = HITLDecision.ESCALATE
        notes = (
            f"ILM delta {sma_impact.get('ilm_delta',0):+.4f} exceeds "
            f"{ILM_CHANGE_ALERT:.0%} alert threshold. "
            "CFO review required before Board Risk Committee."
        )
    elif duty_notify:
        decision = HITLDecision.ESCALATE
        notes = (
            "FCA Consumer Duty PS22/9 notification required within "
            "72 hours. Compliance Director sign-off needed."
        )
    elif errors:
        decision = HITLDecision.ESCALATE
        notes = f"Agent errors: {errors}. Manual review required."
    else:
        decision = HITLDecision.APPROVE
        notes = (
            "All metrics within normal parameters. EOD run complete. "
            "Head of Operational Risk sign-off required before "
            "any PRA submission (EU AI Act Art. 14)."
        )

    state.add_step(AgentStep(
        agent_name  = "HITLGate",
        action      = f"HITL decision: {decision.value}",
        observation = notes,
    ))

    log.info(
        "HITLGate: decision=%s dora_major=%s coo=%s ilm_alert=%s duty=%s",
        decision.value, dora_major, coo_required, ilm_alert, duty_notify,
    )
    return {"hitl_decision": decision, "hitl_notes": notes}


# ══════════════════════════════════════════════════════════════════
# SUPERVISOR ROUTER
# ══════════════════════════════════════════════════════════════════
def supervisor_router(state: OpRiskState) -> str:
    return "hitl"


# ══════════════════════════════════════════════════════════════════
# GRAPH BUILDER
# ══════════════════════════════════════════════════════════════════
def build_op_risk_graph(
    fraud_triage_agent:      Optional[FraudTriageAgent]      = None,
    op_loss_classifier:      Optional[OpLossClassifierAgent] = None,
    sma_impact_agent:        Optional[SMAImpactAgent]        = None,
    dora_incident_agent:     Optional[DORAIncidentAgent]     = None,
    consumer_duty_agent:     Optional[ConsumerDutyAgent]     = None,
) -> Any:
    """Build the LangGraph StateGraph for operational risk monitoring.

    Topology:
        START → fraud_triage → op_loss_classifier → sma_impact
              → dora_incident → consumer_duty → hitl → END
    """
    agents = {
        "fraud_triage":       fraud_triage_agent   or FraudTriageAgent(),
        "op_loss_classifier": op_loss_classifier   or OpLossClassifierAgent(),
        "sma_impact":         sma_impact_agent     or SMAImpactAgent(),
        "dora_incident":      dora_incident_agent  or DORAIncidentAgent(),
        "consumer_duty":      consumer_duty_agent  or ConsumerDutyAgent(),
    }

    try:
        from langgraph.graph import END, START, StateGraph
        graph = StateGraph(OpRiskState)
        for name, agent in agents.items():
            graph.add_node(name, agent)
        graph.add_node("hitl", hitl_gate_node)

        graph.add_edge(START,                "fraud_triage")
        graph.add_edge("fraud_triage",       "op_loss_classifier")
        graph.add_edge("op_loss_classifier", "sma_impact")
        graph.add_edge("sma_impact",         "dora_incident")
        graph.add_edge("dora_incident",      "consumer_duty")
        graph.add_conditional_edges(
            "consumer_duty", supervisor_router, {"hitl": "hitl"}
        )
        graph.add_edge("hitl", END)

        log.info(
            "MR-2026-056-OPS: LangGraph StateGraph compiled "
            "with %d agents + HITL gate", len(agents)
        )
        return graph.compile()

    except ImportError:
        log.warning("LangGraph not installed — using sequential stub.")
        return _SequentialStub(agents)


class _SequentialStub:
    def __init__(self, agents: dict) -> None:
        self._agents = agents

    async def ainvoke(self, state: OpRiskState) -> OpRiskState:
        order = [
            "fraud_triage", "op_loss_classifier",
            "sma_impact", "dora_incident", "consumer_duty",
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

    def invoke(self, state: OpRiskState) -> OpRiskState:
        return asyncio.run(self.ainvoke(state))


# ══════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════
async def run_agentic_op_risk(
    run_date:       str,
    trigger_event:  str,
    fraud_alerts:   Optional[List[Dict]] = None,
    op_loss_events: Optional[List[Dict]] = None,
    ict_incidents:  Optional[List[Dict]] = None,
) -> OpRiskState:
    """Run the Agentic Operational Risk Monitor pipeline.

    Entry point for Airflow DAG and real-time event triggers.
    Builds OpRiskState, runs 5 agents through LangGraph StateGraph,
    returns final state with hop-chain audit trail and HITL decision.

    Args:
        run_date:       ISO date (YYYY-MM-DD).
        trigger_event:  One of 'fraud_alert', 'op_loss_event',
                        'dora_incident', 'eod_scheduled'.
        fraud_alerts:   List of FraudAlert dicts (MR-2026-049).
        op_loss_events: List of OpLossEvent dicts (MR-2026-050).
        ict_incidents:  List of ServiceNow ICT incident dicts.

    Returns:
        Final OpRiskState with all agent outputs, hop-chain,
        and hitl_decision.

    Example::

        state = await run_agentic_op_risk(
            run_date      = "2026-05-24",
            trigger_event = "fraud_alert",
            fraud_alerts  = [alert.__dict__ for alert in alerts],
            op_loss_events= [ev.__dict__ for ev in loss_events],
            ict_incidents = servicenow_incidents,
        )
        print(f"HITL: {state['hitl_decision']}")
        print(f"SMA ILM delta: {state['sma_impact']['ilm_delta']:+.4f}")
        print(f"DORA major: {state['dora_classification']}")
        print(f"Hop-chain: {len(state['hop_chain'])} steps")
    """
    log.info(
        "MR-2026-056-OPS: Starting run date=%s trigger=%s "
        "fraud=%d op_loss=%d ict=%d",
        run_date, trigger_event,
        len(fraud_alerts or []),
        len(op_loss_events or []),
        len(ict_incidents or []),
    )

    state = OpRiskState(
        run_date       = run_date,
        trigger_event  = trigger_event,
        fraud_alerts   = fraud_alerts   or [],
        op_loss_events = op_loss_events or [],
        ict_incidents  = ict_incidents  or [],
    )

    app = build_op_risk_graph()
    try:
        final = await app.ainvoke(state) if hasattr(app, "ainvoke") else app.invoke(state)
    except Exception as exc:
        log.error("MR-2026-056-OPS pipeline error: %s", exc)
        state["errors"].append(str(exc))
        state["hitl_decision"] = HITLDecision.ESCALATE
        state["hitl_notes"]    = f"Pipeline error: {exc}"
        final = state

    log.info(
        "MR-2026-056-OPS: Complete decision=%s hop=%d errors=%d",
        final.get("hitl_decision"),
        len(final.get("hop_chain", [])),
        len(final.get("errors", [])),
    )
    return final
