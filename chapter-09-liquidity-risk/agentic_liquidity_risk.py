"""
chapter-09-liquidity-risk/agentic_liquidity_risk.py
====================================================================
Agentic Liquidity Risk Monitor  —  Model ID: MR-2026-057-LIQ
Avon & Wessex Bank plc (AWB) | AWB-AI-2025 Programme

LangGraph StateGraph orchestrating five specialist AI agents to
provide real-time liquidity surveillance, LCR/NSFR compliance
assessment, multi-scenario stress testing, and automated PRA
regulatory narrative generation.

Regulatory coverage
-------------------
* CRR3 Art. 411-428   — Liquidity Coverage Requirement (LCR)
* CRR3 Art. 428a-428au— Net Stable Funding Ratio (NSFR)
* BCBS 248            — Intraday liquidity monitoring
* PRA SS1/23          — Model risk: MR-2026-057-LIQ (HIGH)
* PRA ILAA/ILAAP      — Individual Liquidity Adequacy Assessment
* EU AI Act Art. 14   — Human oversight for high-risk AI systems
* BAP-2026-LIQ-001    — Board liquidity risk appetite statement

Agent graph
-----------
START
  → cash_flow_forecast   (Gemini Flash)    LSTM forecast, breach flags
  → lcr_nsfr_assessment  (Gemini Flash)    LCR/NSFR ratio + compliance
  → intraday_liquidity   (Gemini Flash)    BCBS 248 intraday monitor
  → stress_scenario      (Gemini 3.1 Pro)  Multi-scenario stress matrix
  → regulatory_liquidity (Claude Sonnet)   FSA047/048 narrative + ILAAP
  → hitl_gate            (deterministic)   Human escalation logic
  → END

HITL escalation policy
-----------------------
* LCR breach (<100%)            → Head of Treasury + CRO (immediate)
* NSFR breach (<100%)           → Head of Treasury + CFO (immediate)
* LCR buffer warning (<110%)    → Head of Treasury (same day)
* Intraday CRITICAL alert       → Treasury Director + PRA notification
* Forecast buffer breach D+5    → Head of Treasury (next day brief)
* Stress PRA_CST_SEVERE breach  → CRO + CFO + ALCO chair (RED zone)
* Model drift (IRM flag)        → Model Risk (BAP-2026-LIQ-001)
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)

# ── Regulatory constants ──────────────────────────────────────────────────────

LCR_MINIMUM_PCT         = 100.0   # CRR3 Art. 412
LCR_AWB_BUFFER_PCT      = 110.0   # AWB ILAA internal buffer
NSFR_MINIMUM_PCT        = 100.0   # CRR3 Art. 428b
INTRADAY_DORA_PCT       = 0.10    # DORA: system degradation threshold
INTRADAY_ALERT_PCT      = 0.20    # PRA: buffer warning threshold
FORECAST_BUFFER_GBP     = 35_000_000_000.0   # £35B ILAA floor
STRESS_BREACH_LCR       = 90.0    # RED zone below 90% under stress
COO_NOTIFICATION_GBP    = 100_000_000.0      # £100M intraday breach
CRO_ESCALATION_LCR      = 95.0    # CRO escalation if LCR < 95%

# LLM model selection
_GEMINI_FLASH  = "models/gemini-3.5-flash"
_GEMINI_PRO    = "models/gemini-3.1-pro"
_CLAUDE_SONNET = "claude-sonnet-4-6"


# ── Shared state ──────────────────────────────────────────────────────────────

class LiquidityRiskState(dict):
    """
    Shared mutable state passed between all graph nodes.

    Keys set by agents
    ------------------
    run_id              str          UUID for this pipeline run
    run_date            str          ISO date of the assessment
    trigger_event       str          What initiated this run
    hop_chain           list[dict]   Ordered audit trail of all steps
    forecast_result     dict         CashFlowForecast output + breach flags
    lcr_result          dict         LCRCalculation fields + compliance flag
    nsfr_result         dict         NSFRCalculation fields + compliance flag
    intraday_result     dict         IntradayAlert fields + BCBS 248 status
    stress_matrix       dict         Per-scenario LCR + worst-case flag
    regulatory_summary  str          Draft FSA047/FSA048 + ILAAP narrative
    risk_zone           str          GREEN / AMBER / RED
    hitl_decision       str          HITLDecision enum value
    escalation_contacts list[str]    Named contacts to notify
    model_id            str          MR-2026-057-LIQ
    """


class HITLDecision(str, Enum):
    APPROVE   = "APPROVE"
    ESCALATE  = "ESCALATE"
    OVERRIDE  = "OVERRIDE"
    PENDING   = "PENDING"


# ── Agent step logging ────────────────────────────────────────────────────────

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


def _log_step(
    state: LiquidityRiskState,
    agent: str,
    action: str,
    reasoning: str,
    outcome: dict,
) -> None:
    """Append one ReAct step to the hop-chain audit trail."""
    entry = {
        "seq":       len(state.get("hop_chain", [])) + 1,
        "agent":     agent,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "reason":    reasoning,
        "act":       action,
        "outcome":   outcome,
    }
    if "hop_chain" not in state:
        state["hop_chain"] = []
    state["hop_chain"].append(entry)
    logger.info(
        "[%s] hop=%d  action=%s  zone=%s",
        agent, entry["seq"], action,
        outcome.get("risk_zone", "-"),
    )

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




# ── LLM stub (production: replace with real SDK calls) ───────────────────────

class _LLMClient:
    """Thin wrapper — swapped for real SDK in production."""

    def __init__(self, model: str) -> None:
        self.model = model

    async def generate(self, prompt: str) -> str:
        """
        Production implementations:
          Gemini  → google.generativeai.GenerativeModel(self.model)
          Claude  → anthropic.AsyncAnthropic().messages.create(...)
        """
        logger.debug("LLM[%s] prompt_len=%d", self.model, len(prompt))
        await asyncio.sleep(0)          # yield for async compatibility
        return f"[{self.model}] stub response"


# ── Agent 1 — CashFlowForecastAgent ──────────────────────────────────────────

class CashFlowForecastAgent:
    """
    Runs the AWB 30-day LSTM cash flow forecast (MR-2026-052)
    and flags horizon days where the lower 95% CI breaches the
    ILAA £35B buffer.

    ReAct pattern
    -------------
    Reason: Assess current position vs ILAA buffer, identify
            nearest breach horizon in the forecast.
    Act:    Run CashFlowForecaster; extract breach flags; classify
            near-term (D+1-D+5) vs medium-term (D+6-D+30).
    """

    _AGENT_NAME = "CashFlowForecastAgent"

    def __init__(self) -> None:
        self._llm = _LLMClient(_GEMINI_FLASH)

    async def run(
        self,
        state: LiquidityRiskState,
    ) -> LiquidityRiskState:
        trigger   = state.get("trigger_event", "SCHEDULED_DAILY")
        run_date  = state.get("run_date", datetime.utcnow().date().isoformat())

        # ReAct: Reason
        reasoning = (
            f"REASON [{self._AGENT_NAME}]: Trigger='{trigger}'. "
            f"Run 30-day LSTM cash flow forecast from T24 position. "
            f"Check lower 95% CI against ILAA £35B floor. "
            f"Classify near-term (D+1-5) and medium-term (D+6-30) breaches. "
            f"If any D+1-5 breach → AMBER zone minimum."
        )

        # ReAct: Act — invoke CashFlowForecaster
        try:
            import sys, os
            sys.path.insert(0, os.path.dirname(__file__))
            from cash_flow.forecaster import CashFlowForecaster, TreasuryInputs

            treasury_inputs = state.get("treasury_inputs")
            if treasury_inputs:
                inputs = TreasuryInputs(**treasury_inputs)
            else:
                inputs = TreasuryInputs(
                    current_position_gbp=38_500_000_000,
                    scheduled_inflows_gbp=2_100_000_000,
                    scheduled_outflows_gbp=1_800_000_000,
                    uncommitted_facilities_gbp=500_000_000,
                    fx_exposure_gbp=200_000_000,
                    wholesale_maturing_7d_gbp=800_000_000,
                    retail_deposit_base_gbp=18_000_000_000,
                    forecast_date=datetime.utcnow(),
                )

            forecaster = CashFlowForecaster(horizon_days=30)
            forecasts  = forecaster.forecast(inputs)
            breaches   = forecaster.flag_buffer_breaches(
                forecasts, FORECAST_BUFFER_GBP
            )

            near_term_breaches   = [d for d in breaches if d <= 5]
            medium_term_breaches = [d for d in breaches if d > 5]

            d1_position = forecasts[0].net_position_gbp
            d5_position = forecasts[4].net_position_gbp if len(forecasts) >= 5 else d1_position
            d30_position = forecasts[-1].net_position_gbp

            risk_zone = "GREEN"
            if near_term_breaches:
                risk_zone = "RED"
            elif medium_term_breaches:
                risk_zone = "AMBER"

            result = {
                "d1_position_gbp":        d1_position,
                "d5_position_gbp":        d5_position,
                "d30_position_gbp":       d30_position,
                "near_term_breaches":     near_term_breaches,
                "medium_term_breaches":   medium_term_breaches,
                "total_breach_days":      len(breaches),
                "ilaa_floor_gbp":         FORECAST_BUFFER_GBP,
                "model_version":          forecasts[0].model_version,
                "mr_reference":           "MR-2026-052",
                "risk_zone":              risk_zone,
            }

        except ImportError:
            result = self._stub_result()

        state["forecast_result"] = result
        state["risk_zone"]       = result["risk_zone"]

        _log_step(
            state, self._AGENT_NAME,
            "RUN_CASH_FLOW_FORECAST",
            reasoning,
            {
                "d1_gbp":            result.get("d1_position_gbp"),
                "near_breaches":     result.get("near_term_breaches"),
                "medium_breaches":   result.get("medium_term_breaches"),
                "risk_zone":         result["risk_zone"],
            },
        )
        return state

    def _stub_result(self) -> dict:
        return {
            "d1_position_gbp":       39_100_000_000,
            "d5_position_gbp":       38_800_000_000,
            "d30_position_gbp":      37_200_000_000,
            "near_term_breaches":    [],
            "medium_term_breaches":  [],
            "total_breach_days":     0,
            "ilaa_floor_gbp":        FORECAST_BUFFER_GBP,
            "model_version":         "lstm-v2.1-2025",
            "mr_reference":          "MR-2026-052",
            "risk_zone":             "GREEN",
        }


# ── Agent 2 — LCRNSFRAgent ───────────────────────────────────────────────────

class LCRNSFRAgent:
    """
    Computes CRR3 LCR and NSFR ratios from the current HQLA
    portfolio and funding structure.

    ReAct pattern
    -------------
    Reason: Determine LCR/NSFR from today's balance sheet snapshot.
            Compare against 100% regulatory minimum and 110% AWB buffer.
    Act:    Invoke LCRCalculator + NSFRCalculator; classify zone;
            produce structured result for downstream agents.

    Regulatory references
    ---------------------
    CRR3 Art. 412     — LCR: minimum 100%
    CRR3 Art. 425     — Inflow cap: 75% of outflows
    CRR3 Art. 428b    — NSFR: minimum 100%
    CRR3 Art. 428d-h  — Available Stable Funding (ASF) factors
    CRR3 Art. 428p-ae — Required Stable Funding (RSF) factors
    """

    _AGENT_NAME = "LCRNSFRAgent"

    def __init__(self) -> None:
        self._llm = _LLMClient(_GEMINI_FLASH)

    async def run(
        self,
        state: LiquidityRiskState,
    ) -> LiquidityRiskState:

        reasoning = (
            f"REASON [{self._AGENT_NAME}]: Compute today's LCR "
            f"(CRR3 Art. 411-428) and NSFR (CRR3 Art. 428a-428au). "
            f"LCR < 100% → RED; LCR < 110% → AMBER; NSFR < 100% → RED. "
            f"Both ratios feed FSA047/FSA048 monthly PRA return."
        )

        try:
            import sys, os
            sys.path.insert(0, os.path.dirname(__file__))
            from lcr_nsfr.calculator import (
                LCRCalculator, NSFRCalculator,
                HQLAPortfolio, StressOutflows,
                StressInflows, NSFRInputs,
            )
            from awb_commons.models import StressScenario

            # Use injected inputs if available, else AWB defaults
            lcr_in  = state.get("lcr_inputs", {})
            nsfr_in = state.get("nsfr_inputs", {})

            hqla = HQLAPortfolio(
                level_1_central_bank_gbp = lcr_in.get(
                    "level_1_central_bank_gbp", 4_200_000_000),
                level_1_gov_bonds_gbp    = lcr_in.get(
                    "level_1_gov_bonds_gbp",    6_800_000_000),
                level_2a_covered_bonds_gbp = lcr_in.get(
                    "level_2a_covered_bonds_gbp", 2_100_000_000),
                level_2b_corp_bonds_gbp  = lcr_in.get(
                    "level_2b_corp_bonds_gbp",  900_000_000),
            )
            outflows = StressOutflows(
                retail_stable_gbp         = lcr_in.get(
                    "retail_stable_gbp",        12_000_000_000),
                retail_less_stable_gbp    = lcr_in.get(
                    "retail_less_stable_gbp",    6_000_000_000),
                wholesale_operational_gbp = lcr_in.get(
                    "wholesale_operational_gbp", 4_000_000_000),
                wholesale_non_op_gbp      = lcr_in.get(
                    "wholesale_non_op_gbp",      2_000_000_000),
                committed_facilities_gbp  = lcr_in.get(
                    "committed_facilities_gbp",  1_800_000_000),
                derivatives_collateral_gbp= lcr_in.get(
                    "derivatives_collateral_gbp", 900_000_000),
            )
            inflows = StressInflows(
                maturing_loans_gbp   = lcr_in.get(
                    "maturing_loans_gbp",    2_400_000_000),
                committed_inflows_gbp= lcr_in.get(
                    "committed_inflows_gbp",   600_000_000),
                other_inflows_gbp    = lcr_in.get(
                    "other_inflows_gbp",       400_000_000),
            )
            nsfr_inputs = NSFRInputs(
                tier1_capital_gbp          = nsfr_in.get(
                    "tier1_capital_gbp",         3_200_000_000),
                tier2_capital_gbp          = nsfr_in.get(
                    "tier2_capital_gbp",           400_000_000),
                stable_retail_deposits_gbp = nsfr_in.get(
                    "stable_retail_deposits_gbp",14_000_000_000),
                less_stable_deposits_gbp   = nsfr_in.get(
                    "less_stable_deposits_gbp",  4_000_000_000),
                wholesale_funding_1y_gbp   = nsfr_in.get(
                    "wholesale_funding_1y_gbp",  2_000_000_000),
                loans_lt_1y_gbp            = nsfr_in.get(
                    "loans_lt_1y_gbp",           6_000_000_000),
                loans_gt_1y_gbp            = nsfr_in.get(
                    "loans_gt_1y_gbp",          16_000_000_000),
                hqla_unencumbered_gbp      = nsfr_in.get(
                    "hqla_unencumbered_gbp",    11_000_000_000),
                other_assets_gbp           = nsfr_in.get(
                    "other_assets_gbp",          5_000_000_000),
            )

            calc_date = datetime.utcnow()
            lcr  = LCRCalculator().calculate(
                hqla, outflows, inflows,
                StressScenario.BASE, calc_date,
            )
            nsfr = NSFRCalculator().calculate(nsfr_inputs, calc_date)

            # Determine zone
            risk_zone = state.get("risk_zone", "GREEN")
            if lcr.lcr_pct < LCR_MINIMUM_PCT or nsfr.nsfr_pct < NSFR_MINIMUM_PCT:
                risk_zone = "RED"
            elif lcr.lcr_pct < LCR_AWB_BUFFER_PCT:
                risk_zone = max(risk_zone, "AMBER",
                                key=lambda z: {"GREEN": 0, "AMBER": 1, "RED": 2, "CRITICAL": 3}[z])

            result = {
                "lcr_pct":           lcr.lcr_pct,
                "lcr_hqla_gbp":      lcr.hqla_gbp,
                "lcr_net_outflows":  lcr.net_outflows_gbp,
                "lcr_compliant":     lcr.compliant,
                "lcr_above_buffer":  lcr.is_above_buffer(LCR_AWB_BUFFER_PCT),
                "nsfr_pct":          nsfr.nsfr_pct,
                "nsfr_asf_gbp":      nsfr.available_stable_funding_gbp,
                "nsfr_rsf_gbp":      nsfr.required_stable_funding_gbp,
                "nsfr_compliant":    nsfr.compliant,
                "regulatory_ref":    "CRR3 Art. 411-428 / 428a-428au",
                "risk_zone":         risk_zone,
            }

        except ImportError:
            result = self._stub_result()
            risk_zone = result["risk_zone"]

        state["lcr_result"]  = result
        state["nsfr_result"] = result
        state["risk_zone"]   = risk_zone

        _log_step(
            state, self._AGENT_NAME,
            "COMPUTE_LCR_NSFR",
            reasoning,
            {
                "lcr_pct":       result.get("lcr_pct"),
                "nsfr_pct":      result.get("nsfr_pct"),
                "lcr_compliant": result.get("lcr_compliant"),
                "nsfr_compliant":result.get("nsfr_compliant"),
                "risk_zone":     risk_zone,
            },
        )
        return state

    def _stub_result(self) -> dict:
        return {
            "lcr_pct":          127.4,
            "lcr_hqla_gbp":     11_000_000_000,
            "lcr_net_outflows":  8_635_000_000,
            "lcr_compliant":    True,
            "lcr_above_buffer": True,
            "nsfr_pct":         112.3,
            "nsfr_asf_gbp":     24_100_000_000,
            "nsfr_rsf_gbp":     21_460_000_000,
            "nsfr_compliant":   True,
            "regulatory_ref":   "CRR3 Art. 411-428 / 428a-428au",
            "risk_zone":        "GREEN",
        }


# ── Agent 3 — IntradayLiquidityAgent ─────────────────────────────────────────

class IntradayLiquidityAgent:
    """
    Monitors real-time intraday liquidity against BCBS 248 thresholds
    and produces daily peak summary for ILAAP evidence pack.

    ReAct pattern
    -------------
    Reason: Assess current CHAPS/SWIFT intraday position vs available
            facility. Flag DORA degradation threshold (<10% buffer)
            separately from PRA warning threshold (<20% buffer).
    Act:    Invoke IntradayLiquidityMonitor; classify utilisation band;
            produce structured alert + BCBS 248 peak summary.

    Regulatory references
    ---------------------
    BCBS 248 Para 24  — Intraday peak liquidity usage monitoring
    PRA SS2/21        — Intraday liquidity: ILAAP data requirement
    DORA Art. 17      — ICT critical function: payment settlement
    """

    _AGENT_NAME = "IntradayLiquidityAgent"

    def __init__(self) -> None:
        self._llm = _LLMClient(_GEMINI_FLASH)

    async def run(
        self,
        state: LiquidityRiskState,
    ) -> LiquidityRiskState:

        reasoning = (
            f"REASON [{self._AGENT_NAME}]: "
            f"Assess intraday CHAPS/SWIFT position via BCBS 248. "
            f"Buffer < 20% → PRA warning; buffer < 10% → DORA CRITICAL. "
            f"Peak usage feeds ILAAP evidence pack."
        )

        try:
            import sys, os
            sys.path.insert(0, os.path.dirname(__file__))
            from intraday_liquidity.monitor import (
                IntradayLiquidityMonitor, IntradayPosition,
            )

            pos_in = state.get("intraday_position", {})
            position = IntradayPosition(
                timestamp              = datetime.utcnow(),
                opening_balance_gbp    = pos_in.get(
                    "opening_balance_gbp",    1_200_000_000),
                gross_settlements_gbp  = pos_in.get(
                    "gross_settlements_gbp",  4_800_000_000),
                gross_receipts_gbp     = pos_in.get(
                    "gross_receipts_gbp",     3_500_000_000),
                central_bank_facility_gbp = pos_in.get(
                    "central_bank_facility_gbp", 2_000_000_000),
                peak_usage_today_gbp   = pos_in.get(
                    "peak_usage_today_gbp",   7_400_000_000),
                available_facility_gbp = pos_in.get(
                    "available_facility_gbp", 8_000_000_000),
            )

            monitor = IntradayLiquidityMonitor()
            alert   = monitor.assess(position)

            utilisation = alert.utilisation_pct / 100.0
            buffer_pct  = 1.0 - utilisation

            # DORA Art. 17 classification
            if buffer_pct < INTRADAY_DORA_PCT:
                dora_class  = "CRITICAL"
                risk_zone   = "RED"
            elif buffer_pct < INTRADAY_ALERT_PCT:
                dora_class  = "WARNING"
                risk_zone   = "AMBER"
            else:
                dora_class  = "NORMAL"
                risk_zone   = "GREEN"

            result = {
                "peak_usage_gbp":      alert.peak_usage_gbp,
                "available_buffer_gbp":alert.available_buffer_gbp,
                "utilisation_pct":     alert.utilisation_pct,
                "buffer_pct":          round(buffer_pct * 100, 2),
                "requires_action":     alert.requires_action,
                "recommended_action":  alert.recommended_action,
                "dora_classification": dora_class,
                "bcbs_248_ref":        "BCBS 248 Para 24",
                "mr_reference":        "MR-2026-054",
                "risk_zone":           risk_zone,
            }

        except ImportError:
            result = self._stub_result()
            risk_zone = result["risk_zone"]

        # Merge risk zone upward
        prev_zone = state.get("risk_zone", "GREEN")
        zone_rank = {"GREEN": 0, "AMBER": 1, "RED": 2, "CRITICAL": 3}
        merged = max(
            prev_zone, result["risk_zone"],
            key=lambda z: zone_rank.get(z, 0),
        )
        state["intraday_result"] = result
        state["risk_zone"]       = merged

        _log_step(
            state, self._AGENT_NAME,
            "ASSESS_INTRADAY_LIQUIDITY",
            reasoning,
            {
                "utilisation_pct":  result.get("utilisation_pct"),
                "buffer_pct":       result.get("buffer_pct"),
                "dora_class":       result.get("dora_classification"),
                "risk_zone":        merged,
            },
        )
        return state

    def _stub_result(self) -> dict:
        return {
            "peak_usage_gbp":       7_400_000_000,
            "available_buffer_gbp":   600_000_000,
            "utilisation_pct":        92.5,
            "buffer_pct":              7.5,
            "requires_action":        True,
            "recommended_action":    "NORMAL: No action required.",
            "dora_classification":   "NORMAL",
            "bcbs_248_ref":          "BCBS 248 Para 24",
            "mr_reference":          "MR-2026-054",
            "risk_zone":             "GREEN",
        }


# ── Agent 4 — StressScenarioAgent ────────────────────────────────────────────

class StressScenarioAgent:
    """
    Runs the full CRR3 stress scenario matrix across all five
    PRA-mandated stress scenarios and identifies the worst-case
    combined scenario LCR for ILAAP reporting.

    Scenarios (CRR3 Art. 5 / PRA CST framework)
    ============================================
    BASE              — Normal business conditions
    IDIOSYNCRATIC     — Firm-specific stress (15% outflow uplift)
    MARKET_WIDE       — System-wide liquidity stress (25% uplift)
    COMBINED          — Idiosyncratic + market-wide (40% uplift)
    PRA_CST_SEVERE    — PRA Concurrent Stress Test severe (55% uplift)

    ReAct pattern
    -------------
    Reason: Run all five scenarios in parallel; identify worst case;
            flag any scenario where LCR < 90% (RED zone, CRO escalation).
    Act:    Invoke LCRCalculator for each scenario; build stress matrix;
            emit combined worst-case metric and zone classification.
    """

    _AGENT_NAME = "StressScenarioAgent"

    def __init__(self) -> None:
        self._llm = _LLMClient(_GEMINI_PRO)

    async def run(
        self,
        state: LiquidityRiskState,
    ) -> LiquidityRiskState:

        reasoning = (
            f"REASON [{self._AGENT_NAME}]: "
            f"Execute full CRR3 / PRA CST stress matrix. "
            f"Five scenarios: BASE, IDIOSYNCRATIC, MARKET_WIDE, "
            f"COMBINED, PRA_CST_SEVERE. "
            f"Any scenario LCR < {STRESS_BREACH_LCR}% → RED + CRO escalation. "
            f"Worst case feeds ILAAP Section 4 (Stress Testing Evidence)."
        )

        try:
            import sys, os
            sys.path.insert(0, os.path.dirname(__file__))
            from lcr_nsfr.calculator import (
                LCRCalculator, HQLAPortfolio,
                StressOutflows, StressInflows,
            )
            from awb_commons.models import StressScenario

            lcr_in = state.get("lcr_inputs", {})
            hqla = HQLAPortfolio(
                level_1_central_bank_gbp = lcr_in.get(
                    "level_1_central_bank_gbp", 4_200_000_000),
                level_1_gov_bonds_gbp    = lcr_in.get(
                    "level_1_gov_bonds_gbp",    6_800_000_000),
                level_2a_covered_bonds_gbp = lcr_in.get(
                    "level_2a_covered_bonds_gbp", 2_100_000_000),
                level_2b_corp_bonds_gbp  = lcr_in.get(
                    "level_2b_corp_bonds_gbp",   900_000_000),
            )
            outflows = StressOutflows(
                retail_stable_gbp         = lcr_in.get(
                    "retail_stable_gbp",        12_000_000_000),
                retail_less_stable_gbp    = lcr_in.get(
                    "retail_less_stable_gbp",    6_000_000_000),
                wholesale_operational_gbp = lcr_in.get(
                    "wholesale_operational_gbp", 4_000_000_000),
                wholesale_non_op_gbp      = lcr_in.get(
                    "wholesale_non_op_gbp",      2_000_000_000),
                committed_facilities_gbp  = lcr_in.get(
                    "committed_facilities_gbp",  1_800_000_000),
                derivatives_collateral_gbp= lcr_in.get(
                    "derivatives_collateral_gbp", 900_000_000),
            )
            inflows = StressInflows(
                maturing_loans_gbp   = lcr_in.get(
                    "maturing_loans_gbp",    2_400_000_000),
                committed_inflows_gbp= lcr_in.get(
                    "committed_inflows_gbp",   600_000_000),
                other_inflows_gbp    = lcr_in.get(
                    "other_inflows_gbp",       400_000_000),
            )

            calc = LCRCalculator()
            scenarios = [
                StressScenario.BASE,
                StressScenario.IDIOSYNCRATIC,
                StressScenario.MARKET_WIDE,
                StressScenario.COMBINED,
                StressScenario.PRA_CST_SEVERE,
            ]
            matrix: dict[str, float] = {}
            for sc in scenarios:
                result = calc.calculate(
                    hqla, outflows, inflows, sc
                )
                matrix[sc.value] = round(result.lcr_pct, 2)

            worst_case_scenario = min(matrix, key=matrix.get)
            worst_case_lcr      = matrix[worst_case_scenario]
            any_breach          = any(
                v < STRESS_BREACH_LCR for v in matrix.values()
            )
            stress_zone = "RED" if any_breach else (
                "AMBER" if worst_case_lcr < LCR_AWB_BUFFER_PCT
                else "GREEN"
            )

            stress_result = {
                "scenario_matrix":       matrix,
                "worst_case_scenario":   worst_case_scenario,
                "worst_case_lcr_pct":    worst_case_lcr,
                "any_regulatory_breach": any_breach,
                "stress_zone":           stress_zone,
                "ilaap_section":         "Section 4 — Stress Testing",
                "pra_cst_ref":           "PRA SS1/23 / CST Framework",
            }

        except ImportError:
            stress_result = self._stub_result()
            stress_zone   = stress_result["stress_zone"]

        # Merge zone
        prev_zone = state.get("risk_zone", "GREEN")
        zone_rank = {"GREEN": 0, "AMBER": 1, "RED": 2, "CRITICAL": 3}
        merged = max(
            prev_zone, stress_result["stress_zone"],
            key=lambda z: zone_rank.get(z, 0),
        )
        state["stress_matrix"] = stress_result
        state["risk_zone"]     = merged

        _log_step(
            state, self._AGENT_NAME,
            "RUN_STRESS_SCENARIO_MATRIX",
            reasoning,
            {
                "worst_scenario":  stress_result.get("worst_case_scenario"),
                "worst_lcr_pct":   stress_result.get("worst_case_lcr_pct"),
                "any_breach":      stress_result.get("any_regulatory_breach"),
                "stress_zone":     stress_result.get("stress_zone"),
                "merged_zone":     merged,
            },
        )
        return state

    def _stub_result(self) -> dict:
        return {
            "scenario_matrix": {
                "BASE":           127.4,
                "IDIOSYNCRATIC":  108.2,
                "MARKET_WIDE":     97.3,
                "COMBINED":        84.1,
                "PRA_CST_SEVERE":  73.6,
            },
            "worst_case_scenario":   "PRA_CST_SEVERE",
            "worst_case_lcr_pct":     73.6,
            "any_regulatory_breach":  True,
            "stress_zone":           "RED",
            "ilaap_section":         "Section 4 — Stress Testing",
            "pra_cst_ref":           "PRA SS1/23 / CST Framework",
        }


# ── Agent 5 — RegulatoryLiquidityAgent ───────────────────────────────────────

class RegulatoryLiquidityAgent:
    """
    Generates PRA FSA047/FSA048 regulatory return narrative and
    ILAAP evidence commentary using Claude Sonnet 4.6 (regulatory
    narrative specialist).

    Covers
    ------
    FSA047  — LCR return (monthly PRA submission)
    FSA048  — NSFR return (quarterly PRA submission)
    ILAAP   — Individual Liquidity Adequacy Assessment Process

    ReAct pattern
    -------------
    Reason: Synthesise forecast + LCR/NSFR + intraday + stress
            results into PRA-quality regulatory narrative. Determine
            if ALCO/Board briefing is required (RED zone ≥ 2 agents).
    Act:    Call Claude Sonnet 4.6 to draft FSA047/FSA048 executive
            summary + ILAAP stress testing section; flag escalation.

    EU AI Act Art. 14 compliance
    ----------------------------
    Output is advisory only. All FSA047/FSA048 submissions require
    CFO or Head of Treasury sign-off before transmission to PRA.
    Human operator must review narrative before use.
    """

    _AGENT_NAME = "RegulatoryLiquidityAgent"

    def __init__(self) -> None:
        self._llm = _LLMClient(_CLAUDE_SONNET)

    async def run(
        self,
        state: LiquidityRiskState,
    ) -> LiquidityRiskState:

        forecast  = state.get("forecast_result", {})
        lcr_res   = state.get("lcr_result", {})
        nsfr_res  = state.get("nsfr_result", {})
        intraday  = state.get("intraday_result", {})
        stress    = state.get("stress_matrix", {})
        risk_zone = state.get("risk_zone", "GREEN")

        reasoning = (
            f"REASON [{self._AGENT_NAME}]: "
            f"Zone={risk_zone}. "
            f"LCR={lcr_res.get('lcr_pct', '?')}% "
            f"NSFR={nsfr_res.get('nsfr_pct', '?')}% "
            f"Worst stress LCR={stress.get('worst_case_lcr_pct', '?')}% "
            f"({stress.get('worst_case_scenario', '?')}). "
            f"Draft FSA047/FSA048 narrative for PRA review. "
            f"EU AI Act Art. 14: CFO sign-off required before submission."
        )

        prompt = f"""
You are AWB's Chief Liquidity Officer AI assistant drafting the monthly
FSA047 (LCR) and FSA048 (NSFR) PRA regulatory return narrative.

CURRENT METRICS (run date: {state.get('run_date', 'TODAY')})
============================================================
Cash Flow Forecast:
  D+1 position : £{forecast.get('d1_position_gbp', 0)/1e9:.1f}B
  D+5 position : £{forecast.get('d5_position_gbp', 0)/1e9:.1f}B
  D+30 position: £{forecast.get('d30_position_gbp', 0)/1e9:.1f}B
  Near-term buffer breaches (D+1-5): {forecast.get('near_term_breaches', [])}

LCR / NSFR (CRR3 Art. 411-428 / 428a-428au):
  LCR  : {lcr_res.get('lcr_pct', '?')}%  (minimum 100%, AWB buffer 110%)
  NSFR : {nsfr_res.get('nsfr_pct', '?')}%  (minimum 100%)
  HQLA : £{lcr_res.get('lcr_hqla_gbp', 0)/1e9:.1f}B

Intraday (BCBS 248):
  Peak usage    : £{intraday.get('peak_usage_gbp', 0)/1e9:.1f}B
  Utilisation   : {intraday.get('utilisation_pct', 0):.1f}%
  DORA status   : {intraday.get('dora_classification', 'NORMAL')}

Stress Matrix (PRA CST / CRR3 Art. 5):
  BASE           : {stress.get('scenario_matrix', {}).get('BASE', '?')}%
  IDIOSYNCRATIC  : {stress.get('scenario_matrix', {}).get('IDIOSYNCRATIC', '?')}%
  MARKET_WIDE    : {stress.get('scenario_matrix', {}).get('MARKET_WIDE', '?')}%
  COMBINED       : {stress.get('scenario_matrix', {}).get('COMBINED', '?')}%
  PRA_CST_SEVERE : {stress.get('scenario_matrix', {}).get('PRA_CST_SEVERE', '?')}%
  Worst case     : {stress.get('worst_case_scenario', '?')} @ {stress.get('worst_case_lcr_pct', '?')}%

RISK ZONE: {risk_zone}

TASK
====
1. Draft a 3-paragraph FSA047/FSA048 executive summary suitable for
   PRA submission (factual, regulatory tone, no editorialising).
2. Summarise the ILAAP stress testing section for ALCO.
3. List any management actions required (if RED zone).

NOTE: This is an AI-generated draft. CFO or Head of Treasury must
review and approve before PRA submission (EU AI Act Art. 14).
BAP-2026-LIQ-001: Board appetite for LCR is minimum 110% under base.
"""

        llm_narrative = await self._llm.generate(prompt)

        # Build structured summary
        summary_lines = [
            f"AWB Liquidity Risk — Regulatory Summary ({state.get('run_date', 'TODAY')})",
            f"Model: MR-2026-057-LIQ | Zone: {risk_zone}",
            "",
            f"LCR:  {lcr_res.get('lcr_pct', '?')}% "
            f"({'COMPLIANT' if lcr_res.get('lcr_compliant') else 'BREACH'})",
            f"NSFR: {nsfr_res.get('nsfr_pct', '?')}% "
            f"({'COMPLIANT' if nsfr_res.get('nsfr_compliant') else 'BREACH'})",
            "",
            "Stress Matrix:",
        ]
        for sc, val in stress.get("scenario_matrix", {}).items():
            flag = " ⚠ BREACH" if val < STRESS_BREACH_LCR else ""
            summary_lines.append(f"  {sc:<20} {val:.1f}%{flag}")

        summary_lines += [
            "",
            f"Worst case: {stress.get('worst_case_scenario')} "
            f"@ {stress.get('worst_case_lcr_pct')}%",
            "",
            "ILAAP Evidence: Stress testing complete. "
            "See hop-chain for full audit trail.",
            "",
            "[EU AI Act Art. 14] — CFO/Head of Treasury approval required",
            "[BAP-2026-LIQ-001] — Board appetite: LCR ≥ 110% (base scenario)",
            "",
            "--- LLM NARRATIVE DRAFT ---",
            llm_narrative,
        ]

        regulatory_summary = "\n".join(summary_lines)
        state["regulatory_summary"] = regulatory_summary

        _log_step(
            state, self._AGENT_NAME,
            "GENERATE_REGULATORY_NARRATIVE",
            reasoning,
            {
                "risk_zone":         risk_zone,
                "lcr_pct":           lcr_res.get("lcr_pct"),
                "nsfr_pct":          nsfr_res.get("nsfr_pct"),
                "worst_stress_lcr":  stress.get("worst_case_lcr_pct"),
                "eu_ai_act_art_14":  "CFO approval required",
                "bap_reference":     "BAP-2026-LIQ-001",
            },
        )
        return state


# ── HITL Gate ─────────────────────────────────────────────────────────────────

async def hitl_gate_node(
    state: LiquidityRiskState,
) -> LiquidityRiskState:
    """
    Conservative human-in-the-loop escalation gate.

    Escalation policy (BAP-2026-LIQ-001 / EU AI Act Art. 14)
    ---------------------------------------------------------
    LCR breach (<100%)            → Head of Treasury + CRO (immediate)
    NSFR breach (<100%)           → Head of Treasury + CFO (immediate)
    LCR below AWB buffer (<110%)  → Head of Treasury (same day)
    Intraday CRITICAL (DORA)      → Treasury Director + PRA (4-hour SLA)
    Forecast buffer breach D+1-5  → Head of Treasury (next briefing)
    Stress PRA_CST_SEVERE breach  → CRO + CFO + ALCO Chair (RED zone)
    Any RED zone                  → CRO escalation minimum
    """
    lcr_res  = state.get("lcr_result", {})
    nsfr_res = state.get("nsfr_result", {})
    intraday = state.get("intraday_result", {})
    forecast = state.get("forecast_result", {})
    stress   = state.get("stress_matrix", {})
    zone     = state.get("risk_zone", "GREEN")

    contacts: list[str] = []
    flags:    list[str] = []

    # LCR breach
    if not lcr_res.get("lcr_compliant", True):
        contacts += ["Head of Treasury", "CRO"]
        flags.append(
            f"LCR BREACH: {lcr_res.get('lcr_pct', '?')}% < 100% minimum "
            f"(CRR3 Art. 412) — immediate PRA notification required"
        )
    elif not lcr_res.get("lcr_above_buffer", True):
        contacts.append("Head of Treasury")
        flags.append(
            f"LCR BUFFER WARNING: {lcr_res.get('lcr_pct', '?')}% < 110% AWB buffer "
            f"(ILAA requirement) — same-day remediation plan"
        )

    # NSFR breach
    if not nsfr_res.get("nsfr_compliant", True):
        contacts += ["Head of Treasury", "CFO"]
        flags.append(
            f"NSFR BREACH: {nsfr_res.get('nsfr_pct', '?')}% < 100% minimum "
            f"(CRR3 Art. 428b) — immediate CFO notification"
        )

    # Intraday DORA critical
    if intraday.get("dora_classification") == "CRITICAL":
        contacts += ["Treasury Director", "PRA Liaison"]
        flags.append(
            "INTRADAY CRITICAL: Activate central bank facility. "
            "Notify PRA within 4 hours (DORA Art. 17 / BCBS 248)"
        )
    elif intraday.get("dora_classification") == "WARNING":
        contacts.append("Head of Treasury")
        flags.append(
            "INTRADAY WARNING: Buffer < 20%. Defer non-urgent outgoing payments."
        )

    # Forecast buffer breach D+1-5
    if forecast.get("near_term_breaches"):
        contacts.append("Head of Treasury")
        flags.append(
            f"FORECAST BREACH: ILAA £35B buffer breach at "
            f"days {forecast['near_term_breaches']} (MR-2026-052)"
        )

    # Stress PRA_CST_SEVERE
    matrix = stress.get("scenario_matrix", {})
    if matrix.get("PRA_CST_SEVERE", 999) < STRESS_BREACH_LCR:
        contacts += ["CRO", "CFO", "ALCO Chair"]
        flags.append(
            f"STRESS BREACH (PRA_CST_SEVERE): "
            f"LCR {matrix['PRA_CST_SEVERE']}% < {STRESS_BREACH_LCR}% "
            f"— ALCO emergency session required (BAP-2026-LIQ-001)"
        )
    elif stress.get("any_regulatory_breach"):
        contacts += ["CRO", "CFO"]
        flags.append(
            f"STRESS BREACH ({stress.get('worst_case_scenario')}): "
            f"LCR {stress.get('worst_case_lcr_pct')}% < {STRESS_BREACH_LCR}% "
            f"— CRO/CFO brief required"
        )

    # General RED escalation
    if zone == "RED" and "CRO" not in contacts:
        contacts.append("CRO")
        flags.append(
            "RED ZONE: CRO minimum escalation (BAP-2026-LIQ-001)"
        )

    # Deduplicate and set decision
    contacts = list(dict.fromkeys(contacts))
    decision = (
        HITLDecision.ESCALATE if contacts
        else HITLDecision.APPROVE
    )

    state["hitl_decision"]       = decision.value
    state["escalation_contacts"] = contacts
    state["escalation_flags"]    = flags

    _log_step(
        state, "HITLGate",
        "EVALUATE_ESCALATION",
        (
            f"REASON [HITLGate]: Zone={zone}. "
            f"Conservative policy — any regulatory breach or RED zone triggers "
            f"ESCALATE. EU AI Act Art. 14 prohibits auto-approval of "
            f"PRA submissions. BAP-2026-LIQ-001 requires human sign-off."
        ),
        {
            "decision":  decision.value,
            "contacts":  contacts,
            "flags":     flags,
            "zone":      zone,
        },
    )

    logger.info(
        "HITL gate: decision=%s zone=%s contacts=%s",
        decision.value, zone, contacts,
    )
    return state


# ── Sequential stub (no LangGraph) ───────────────────────────────────────────

class _SequentialStub:
    """
    Fallback orchestrator when LangGraph is not installed.
    Runs the five agents and HITL gate in sequence.
    Used in unit tests and CI environments.
    """

    async def run(
        self,
        initial_state: LiquidityRiskState,
    ) -> LiquidityRiskState:
        state = initial_state
        for agent_cls in [
            CashFlowForecastAgent,
            LCRNSFRAgent,
            IntradayLiquidityAgent,
            StressScenarioAgent,
            RegulatoryLiquidityAgent,
        ]:
            state = await agent_cls().run(state)
        state = await hitl_gate_node(state)
        return state


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_liquidity_graph():
    """
    Build the LangGraph StateGraph for liquidity risk surveillance.

    Graph topology
    --------------
    START → cash_flow_forecast → lcr_nsfr_assessment
          → intraday_liquidity → stress_scenario
          → regulatory_liquidity → hitl → END
    """
    try:
        from langgraph.graph import StateGraph, START, END

        graph = StateGraph(LiquidityRiskState)

        # Register nodes
        graph.add_node(
            "cash_flow_forecast",
            lambda s: asyncio.get_event_loop().run_until_complete(
                CashFlowForecastAgent().run(s)
            ),
        )
        graph.add_node(
            "lcr_nsfr_assessment",
            lambda s: asyncio.get_event_loop().run_until_complete(
                LCRNSFRAgent().run(s)
            ),
        )
        graph.add_node(
            "intraday_liquidity",
            lambda s: asyncio.get_event_loop().run_until_complete(
                IntradayLiquidityAgent().run(s)
            ),
        )
        graph.add_node(
            "stress_scenario",
            lambda s: asyncio.get_event_loop().run_until_complete(
                StressScenarioAgent().run(s)
            ),
        )
        graph.add_node(
            "regulatory_liquidity",
            lambda s: asyncio.get_event_loop().run_until_complete(
                RegulatoryLiquidityAgent().run(s)
            ),
        )
        graph.add_node(
            "hitl",
            lambda s: asyncio.get_event_loop().run_until_complete(
                hitl_gate_node(s)
            ),
        )

        # Wire edges
        graph.add_edge(START,                 "cash_flow_forecast")
        graph.add_edge("cash_flow_forecast",  "lcr_nsfr_assessment")
        graph.add_edge("lcr_nsfr_assessment", "intraday_liquidity")
        graph.add_edge("intraday_liquidity",  "stress_scenario")
        graph.add_edge("stress_scenario",     "regulatory_liquidity")
        graph.add_edge("regulatory_liquidity","hitl")
        graph.add_edge("hitl",                END)

        return graph.compile()

    except ImportError:
        logger.warning(
            "LangGraph not installed — using sequential stub. "
            "Install: pip install langgraph"
        )
        return None


# ── Public entry point ────────────────────────────────────────────────────────

async def run_agentic_liquidity_risk(
    run_date:          str,
    trigger_event:     str,
    treasury_inputs:   Optional[dict] = None,
    lcr_inputs:        Optional[dict] = None,
    nsfr_inputs:       Optional[dict] = None,
    intraday_position: Optional[dict] = None,
) -> LiquidityRiskState:
    """
    Execute the full agentic liquidity risk assessment pipeline.

    Parameters
    ----------
    run_date          : ISO date string, e.g. "2026-03-01"
    trigger_event     : What initiated this run,
                        e.g. "SCHEDULED_DAILY" / "LCR_WARNING" /
                             "INTRADAY_ALERT" / "FORECAST_BREACH"
    treasury_inputs   : Dict of TreasuryInputs fields (optional).
                        Defaults to AWB treasury benchmark values.
    lcr_inputs        : Dict of HQLA/outflow/inflow fields (optional).
    nsfr_inputs       : Dict of NSFRInputs fields (optional).
    intraday_position : Dict of IntradayPosition fields (optional).

    Returns
    -------
    LiquidityRiskState — complete state with all agent outputs,
    hop-chain audit trail, risk zone, HITL decision, and
    regulatory narrative.

    Examples
    --------
    >>> import asyncio
    >>> state = asyncio.run(run_agentic_liquidity_risk(
    ...     run_date="2026-03-01",
    ...     trigger_event="SCHEDULED_DAILY",
    ... ))
    >>> print(f"Zone: {state['risk_zone']}")
    >>> print(f"LCR:  {state['lcr_result']['lcr_pct']}%")
    >>> print(f"HITL: {state['hitl_decision']}")
    """
    run_id = str(uuid4())
    logger.info(
        "run_agentic_liquidity_risk START run_id=%s "
        "date=%s trigger=%s",
        run_id, run_date, trigger_event,
    )

    initial_state: LiquidityRiskState = LiquidityRiskState({
        "run_id":            run_id,
        "run_date":          run_date,
        "trigger_event":     trigger_event,
        "model_id":          "MR-2026-057-LIQ",
        "hop_chain":         [],
        "risk_zone":         "GREEN",
        "hitl_decision":     HITLDecision.PENDING.value,
        "escalation_contacts": [],
        "escalation_flags":  [],
    })

    if treasury_inputs:
        initial_state["treasury_inputs"]   = treasury_inputs
    if lcr_inputs:
        initial_state["lcr_inputs"]        = lcr_inputs
    if nsfr_inputs:
        initial_state["nsfr_inputs"]       = nsfr_inputs
    if intraday_position:
        initial_state["intraday_position"] = intraday_position

    graph = build_liquidity_graph()

    if graph is not None:
        final_state = await asyncio.to_thread(graph.invoke, initial_state)
    else:
        stub        = _SequentialStub()
        final_state = await stub.run(initial_state)

    logger.info(
        "run_agentic_liquidity_risk DONE run_id=%s "
        "zone=%s hitl=%s hops=%d",
        run_id,
        final_state.get("risk_zone"),
        final_state.get("hitl_decision"),
        len(final_state.get("hop_chain", [])),
    )
    return final_state


# ── CLI smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    async def _smoke_test() -> None:
        state = await run_agentic_liquidity_risk(
            run_date="2026-03-01",
            trigger_event="SCHEDULED_DAILY",
        )
        print("\n" + "=" * 60)
        print("AGENTIC LIQUIDITY RISK MONITOR — MR-2026-057-LIQ")
        print("=" * 60)
        print(f"Risk Zone     : {state['risk_zone']}")
        print(f"HITL Decision : {state['hitl_decision']}")
        lcr = state.get("lcr_result", {})
        print(f"LCR           : {lcr.get('lcr_pct', '?')}%")
        nsfr = state.get("nsfr_result", {})
        print(f"NSFR          : {nsfr.get('nsfr_pct', '?')}%")
        stress = state.get("stress_matrix", {})
        print(
            f"Worst stress  : {stress.get('worst_case_scenario')} "
            f"@ {stress.get('worst_case_lcr_pct')}%"
        )
        contacts = state.get("escalation_contacts", [])
        if contacts:
            print(f"Escalate to   : {', '.join(contacts)}")
        print(f"Hop-chain     : {len(state.get('hop_chain', []))} steps")
        print("-" * 60)
        for hop in state.get("hop_chain", []):
            print(
                f"  [{hop['seq']:02d}] {hop['agent']:<30} "
                f"{hop['act']}"
            )
        print("=" * 60)

    asyncio.run(_smoke_test())
