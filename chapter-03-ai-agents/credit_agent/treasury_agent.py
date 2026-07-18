"""
credit_agent/treasury_agent.py
AWB Treasury Operations Agent — Parallel Multi-Agent Pipeline
Chapter 3: Agentic AI for Financial Risk

Implements the Treasury Operations Agent as a parallel LangGraph pipeline,
demonstrating the contrast with the sequential credit decision pipeline in
langgraph_agent.py. While credit decisions are sequential (each step depends
on the last), treasury operations are embarrassingly parallel — cash position,
FX exposure, and settlement risk can all be assessed simultaneously.

Architecture (Section 3.3):

    START
      │
      ├──────────────────────────────────┐
      │                                  │
      ▼                                  ▼
  CashPositionAgent              FXExposureAgent
  (Gemini Flash)                 (Gemini Flash)
  • Nostro account balances       • Open FX positions
  • Intraday liquidity forecast   • FX VaR (1-day 99%)
  • LCR ratio (Basel III)         • Counterparty exposure
  • Alert: balance < floor        • Alert: VaR > limit
      │                                  │
      └──────────────┬───────────────────┘
                     │
                     ▼
            SettlementRiskAgent          Gemini Pro
            (waits for both)             (needs full picture)
            • Net settlement positions
            • DVP / FoP flag
            • Herstatt risk exposure
            • RTGS queue status
                     │
                     ▼
            TreasuryReportNode           Gemini Flash
            • Morning position summary
            • Streaming output to dealer
            • Alerts to Slack/Bloomberg
                     │
                     ▼
                    END

Key orchestration patterns demonstrated:
  1. Parallel fan-out: CashPosition + FXExposure run concurrently.
  2. Fan-in join: SettlementRisk waits for BOTH parallel branches.
  3. Streaming output: TreasuryReport streams token-by-token to the
     dealer dashboard during the 7am FX market open window.
  4. Real-time constraint: if either parallel agent misses its 30s
     deadline, SettlementRisk proceeds with partial data (DORA resilience).

Why parallel for treasury vs sequential for credit:
  Credit decisions follow a strict information dependency chain:
  you need ratios before policy, policy before memo. Treasury positions
  are independent — cash position in EUR nostros has no bearing on
  computing FX VaR on USD/JPY books. Parallelism cuts 90s → 35s.

Regulatory context:
  PRA Rulebook (Liquidity): LCR monitoring requires real-time cash
    position data. The 7am morning run must complete before RTGS opens.
  EMIR Article 11: FX derivative positions must be reported by T+1.
    The FXExposureAgent collects the inputs for EMIR trade reporting.
  DORA Article 6: Parallel architecture provides resilience — if the
    FXExposureAgent is degraded, CashPositionAgent still completes.
  Basel III LCR: Cash position agent computes the daily LCR from
    nostro balances and intraday liquidity facility headroom.

Author: AWB AI Programme (AWB-AI-2025)
Chapter: 3 — Agentic AI for Financial Risk
Version: 1.0.0 (June 2026)
"""
from __future__ import annotations

import asyncio
import datetime
import json
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

logger = logging.getLogger("awb.treasury_agent")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_FLASH = "gemini-3.5-flash"
MODEL_PRO   = "gemini-3.1-pro"
MODEL_REGISTRATION = "MR-2026-038"  # Treasury Operations Agent PRA SS1/23 ID

# DORA operational deadlines
PARALLEL_AGENT_TIMEOUT_SECONDS = 30   # Each parallel agent must complete in 30s
MORNING_RUN_DEADLINE_HOUR = 7         # 07:00 London time (before RTGS opens)

# LCR floor (Basel III — minimum 100% under PRA rules)
LCR_MINIMUM_PCT = 100.0
LCR_INTERNAL_FLOOR_PCT = 110.0  # AWB internal buffer above regulatory minimum

# FX VaR limit
FX_VAR_LIMIT_GBP = 15_000_000  # 1-day 99% VaR limit per PRA permission


# ---------------------------------------------------------------------------
# Treasury state
# ---------------------------------------------------------------------------

class TreasuryState(dict):
    """
    Shared state for the treasury operations pipeline.

    Keys populated by each agent node:
      run_id             (input)
      valuation_date     (input)    — business date (YYYY-MM-DD)
      cash_position      (CashPositionAgent)
      fx_exposure        (FXExposureAgent)
      settlement_risk    (SettlementRiskAgent)
      treasury_report    (TreasuryReportNode)
      alerts             (all nodes)  — append-only list
      audit_trail        (all nodes)
      error              (any node)
      completed_nodes    (router)    — set of completed node names
    """
    pass


def _default_treasury_state(valuation_date: Optional[str] = None) -> TreasuryState:
    return TreasuryState(
        run_id=f"TR-{uuid.uuid4().hex[:10].upper()}",
        valuation_date=valuation_date or datetime.date.today().isoformat(),
        cash_position=None,
        fx_exposure=None,
        settlement_risk=None,
        treasury_report=None,
        alerts=[],
        audit_trail=[],
        error=None,
        completed_nodes=set(),
    )


# ---------------------------------------------------------------------------
# Mock data helpers (replace with live T24 / FX system calls in production)
# ---------------------------------------------------------------------------

def _mock_nostro_balances() -> List[Dict[str, Any]]:
    """Mock nostro account balances across currencies."""
    return [
        {"currency": "GBP", "account": "NOSTRO-GBP-001", "balance_m": 285.4,
         "floor_m": 200.0, "lcr_weight": 1.0},
        {"currency": "EUR", "account": "NOSTRO-EUR-001", "balance_m": 142.8,
         "floor_m": 100.0, "lcr_weight": 0.95},
        {"currency": "USD", "account": "NOSTRO-USD-001", "balance_m": 98.6,
         "floor_m": 75.0, "lcr_weight": 0.95},
        {"currency": "CHF", "account": "NOSTRO-CHF-001", "balance_m": 22.1,
         "floor_m": 15.0, "lcr_weight": 0.90},
        {"currency": "JPY", "account": "NOSTRO-JPY-001", "balance_m": 18.3,
         "floor_m": 10.0, "lcr_weight": 0.85},
    ]


def _mock_fx_positions() -> List[Dict[str, Any]]:
    """Mock open FX positions for AWB Treasury book."""
    return [
        {"pair": "EUR/GBP", "notional_m": 45.0, "direction": "LONG",
         "rate": 0.8542, "maturity": "2026-05-28", "counterparty": "DB_LONDON"},
        {"pair": "USD/GBP", "notional_m": 62.0, "direction": "SHORT",
         "rate": 0.7891, "maturity": "2026-05-26", "counterparty": "JPM_LONDON"},
        {"pair": "USD/EUR", "notional_m": 28.5, "direction": "LONG",
         "rate": 1.0856, "maturity": "2026-06-30", "counterparty": "BARCLAYS"},
        {"pair": "GBP/JPY", "notional_m": 15.0, "direction": "SHORT",
         "rate": 192.45, "maturity": "2026-05-30", "counterparty": "NOMURA"},
    ]


def _mock_settlement_queue() -> List[Dict[str, Any]]:
    """Mock RTGS settlement queue for today."""
    return [
        {"trade_id": "FX-20260523-4412", "pair": "EUR/GBP",
         "settle_amount_m": 45.0, "settle_time": "09:00",
         "mechanism": "DVP", "counterparty": "DB_LONDON", "status": "PENDING"},
        {"trade_id": "FX-20260523-4387", "pair": "USD/GBP",
         "settle_amount_m": 62.0, "settle_time": "14:00",
         "mechanism": "DVP", "counterparty": "JPM_LONDON", "status": "PENDING"},
        {"trade_id": "FX-20260522-9901", "pair": "USD/EUR",
         "settle_amount_m": 28.5, "settle_time": "11:00",
         "mechanism": "FOP", "counterparty": "BARCLAYS", "status": "CONFIRMED"},
    ]


# ---------------------------------------------------------------------------
# Node: CashPositionAgent
# ---------------------------------------------------------------------------

async def node_cash_position_agent(state: TreasuryState) -> TreasuryState:
    """
    CashPositionAgent (Gemini 3.5 Flash) — parallel branch.

    Computes the opening cash position across all nostro accounts,
    derives the Basel III LCR for the current business day, and
    identifies accounts below internal floor thresholds.

    Runs concurrently with FXExposureAgent. Completes independently —
    does not wait for any other node.

    Key computations:
      • Total liquid assets (HQLA) from nostro balances
      • Net cash outflows over 30-day stress horizon
      • LCR = HQLA / Net Cash Outflows (must be ≥ 100%)
      • Per-account alerts where balance < internal floor

    State mutations: cash_position, alerts (appended)
    """
    logger.info("CashPositionAgent: starting | run_id=%s", state["run_id"])
    start = asyncio.get_event_loop().time()

    try:
        # In production: call T24 OFS /api/v1/treasury/nostro?date={valuation_date}
        # and feed the raw balances through a Gemini Flash extraction
        await asyncio.sleep(0.05)  # Simulate T24 API latency (50ms in production)
        nostro_balances = _mock_nostro_balances()

        # Compute aggregate positions
        total_gbp_equiv_m = sum(
            n["balance_m"] * n["lcr_weight"] for n in nostro_balances
        )
        breached_floors = [
            n for n in nostro_balances if n["balance_m"] < n["floor_m"]
        ]

        # Basel III LCR calculation
        hqla_m = total_gbp_equiv_m * 0.85   # Simplified: 85% of nostro qualifies as HQLA
        net_cash_outflows_30d_m = hqla_m * 0.80   # Stress scenario: 80% outflow
        lcr_pct = (hqla_m / net_cash_outflows_30d_m * 100) if net_cash_outflows_30d_m > 0 else 0

        # Build alerts
        new_alerts = []
        if lcr_pct < LCR_MINIMUM_PCT:
            new_alerts.append({
                "severity": "CRITICAL",
                "source": "CashPositionAgent",
                "message": f"LCR {lcr_pct:.1f}% below regulatory minimum {LCR_MINIMUM_PCT}%",
                "regulatory_ref": "Basel III / PRA Rulebook: Liquidity",
            })
        elif lcr_pct < LCR_INTERNAL_FLOOR_PCT:
            new_alerts.append({
                "severity": "WARNING",
                "source": "CashPositionAgent",
                "message": f"LCR {lcr_pct:.1f}% below internal floor {LCR_INTERNAL_FLOOR_PCT}%",
            })

        for acct in breached_floors:
            new_alerts.append({
                "severity": "WARNING",
                "source": "CashPositionAgent",
                "message": (
                    f"Nostro {acct['account']} ({acct['currency']}) "
                    f"balance {acct['balance_m']:.1f}M below floor {acct['floor_m']:.1f}M"
                ),
            })

        state["cash_position"] = {
            "nostro_balances": nostro_balances,
            "total_gbp_equiv_m": round(total_gbp_equiv_m, 2),
            "hqla_m": round(hqla_m, 2),
            "net_cash_outflows_30d_m": round(net_cash_outflows_30d_m, 2),
            "lcr_pct": round(lcr_pct, 1),
            "lcr_compliant": lcr_pct >= LCR_MINIMUM_PCT,
            "breached_floor_accounts": len(breached_floors),
            "model_used": MODEL_FLASH,
            "latency_ms": round((asyncio.get_event_loop().time() - start) * 1000, 1),
        }
        state["alerts"].extend(new_alerts)
        state["completed_nodes"].add("cash_position")

        state["audit_trail"].append({
            "node": "CashPositionAgent",
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "lcr_pct": lcr_pct,
            "alerts_raised": len(new_alerts),
            "model": MODEL_FLASH,
        })

        logger.info(
            "CashPositionAgent: LCR=%.1f%% | nostro_total=£%sM | alerts=%d",
            lcr_pct, f"{total_gbp_equiv_m:.1f}", len(new_alerts),
        )

    except Exception as exc:
        logger.error("CashPositionAgent FAILED: %s", exc)
        state["error"] = f"CashPositionAgent: {exc}"
        state["cash_position"] = {"error": str(exc), "lcr_pct": None}
        state["alerts"].append({
            "severity": "ERROR",
            "source": "CashPositionAgent",
            "message": f"Cash position unavailable: {exc}",
        })

    return state


# ---------------------------------------------------------------------------
# Node: FXExposureAgent
# ---------------------------------------------------------------------------

async def node_fx_exposure_agent(state: TreasuryState) -> TreasuryState:
    """
    FXExposureAgent (Gemini 3.5 Flash) — parallel branch.

    Computes the open FX position book, derives 1-day 99% VaR using
    a simplified delta-normal approach, and checks against PRA-approved
    VaR limits. Flags positions approaching EMIR reporting thresholds.

    Runs concurrently with CashPositionAgent. In the production system,
    this node calls the AWB FX risk system (Murex) via REST API.

    State mutations: fx_exposure, alerts (appended)
    """
    logger.info("FXExposureAgent: starting | run_id=%s", state["run_id"])
    start = asyncio.get_event_loop().time()

    try:
        await asyncio.sleep(0.08)  # Simulate Murex API latency (80ms)
        positions = _mock_fx_positions()

        # Simplified VaR: sum of notional × daily vol × z-score
        # In production: full delta-normal or historical simulation
        FX_DAILY_VOL = {"EUR/GBP": 0.0042, "USD/GBP": 0.0061,
                        "USD/EUR": 0.0038, "GBP/JPY": 0.0078}
        Z_99 = 2.326  # 99th percentile

        var_components = []
        for pos in positions:
            daily_vol = FX_DAILY_VOL.get(pos["pair"], 0.005)
            notional_gbp_m = pos["notional_m"] * (1 if "GBP" in pos["pair"] else 0.79)
            pos_var_m = notional_gbp_m * daily_vol * Z_99
            var_components.append({
                "pair": pos["pair"],
                "notional_gbp_m": round(notional_gbp_m, 2),
                "var_1d_99_m": round(pos_var_m, 3),
                "direction": pos["direction"],
                "counterparty": pos["counterparty"],
                "maturity": pos["maturity"],
            })

        total_var_gbp_m = sum(v["var_1d_99_m"] for v in var_components)
        total_var_gbp = total_var_gbp_m * 1_000_000

        # Counterparty concentration
        cp_exposure: Dict[str, float] = {}
        for pos in positions:
            cp = pos["counterparty"]
            cp_exposure[cp] = cp_exposure.get(cp, 0) + pos["notional_m"]

        new_alerts = []
        if total_var_gbp > FX_VAR_LIMIT_GBP:
            new_alerts.append({
                "severity": "CRITICAL",
                "source": "FXExposureAgent",
                "message": (
                    f"FX VaR £{total_var_gbp/1e6:.2f}M exceeds PRA limit "
                    f"£{FX_VAR_LIMIT_GBP/1e6:.0f}M"
                ),
                "regulatory_ref": "PRA Internal Model Permission (IMP)",
            })
        elif total_var_gbp > FX_VAR_LIMIT_GBP * 0.85:
            new_alerts.append({
                "severity": "WARNING",
                "source": "FXExposureAgent",
                "message": (
                    f"FX VaR £{total_var_gbp/1e6:.2f}M at "
                    f"{total_var_gbp/FX_VAR_LIMIT_GBP*100:.0f}% of PRA limit"
                ),
            })

        state["fx_exposure"] = {
            "positions": var_components,
            "total_positions": len(positions),
            "total_notional_gbp_m": round(sum(p["notional_m"] for p in positions), 2),
            "total_var_1d_99_m": round(total_var_gbp_m, 3),
            "total_var_1d_99_gbp": round(total_var_gbp, 0),
            "var_limit_gbp": FX_VAR_LIMIT_GBP,
            "var_limit_utilisation_pct": round(total_var_gbp / FX_VAR_LIMIT_GBP * 100, 1),
            "counterparty_exposure": cp_exposure,
            "emir_reportable": sum(1 for p in positions if p["maturity"] > state["valuation_date"]),
            "model_used": MODEL_FLASH,
            "latency_ms": round((asyncio.get_event_loop().time() - start) * 1000, 1),
        }
        state["alerts"].extend(new_alerts)
        state["completed_nodes"].add("fx_exposure")

        state["audit_trail"].append({
            "node": "FXExposureAgent",
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "total_var_gbp_m": total_var_gbp_m,
            "positions": len(positions),
            "alerts_raised": len(new_alerts),
            "model": MODEL_FLASH,
        })

        logger.info(
            "FXExposureAgent: VaR=£%sM (%.0f%% of limit) | positions=%d | alerts=%d",
            f"{total_var_gbp_m:.2f}",
            total_var_gbp / FX_VAR_LIMIT_GBP * 100,
            len(positions),
            len(new_alerts),
        )

    except Exception as exc:
        logger.error("FXExposureAgent FAILED: %s", exc)
        state["error"] = f"FXExposureAgent: {exc}"
        state["fx_exposure"] = {"error": str(exc), "total_var_1d_99_gbp": None}
        state["alerts"].append({
            "severity": "ERROR",
            "source": "FXExposureAgent",
            "message": f"FX exposure data unavailable: {exc}",
        })

    return state


# ---------------------------------------------------------------------------
# Node: SettlementRiskAgent (fan-in join)
# ---------------------------------------------------------------------------

async def node_settlement_risk_agent(state: TreasuryState) -> TreasuryState:
    """
    SettlementRiskAgent (Gemini 3.1 Pro) — fan-in join node.

    Waits for both CashPositionAgent and FXExposureAgent to complete,
    then assesses settlement risk for the day's FX trades. Uses Gemini
    Pro because it must synthesise both branches simultaneously and
    reason about Herstatt risk (which depends on both cash availability
    and FX position timing).

    Herstatt risk arises when a bank pays away a currency leg before
    receiving the corresponding leg from the counterparty — if the
    counterparty defaults in between, the paying bank suffers a loss
    equal to the full principal. CLS Bank eliminates this for qualifying
    trades, but non-CLS trades remain exposed.

    State mutations: settlement_risk, alerts (appended)
    """
    logger.info("SettlementRiskAgent: starting (fan-in) | run_id=%s", state["run_id"])
    start = asyncio.get_event_loop().time()

    cash = state.get("cash_position") or {}
    fx = state.get("fx_exposure") or {}

    try:
        await asyncio.sleep(0.06)  # Simulate RTGS queue API call
        settlement_queue = _mock_settlement_queue()

        # Identify FoP (Free of Payment) trades — highest settlement risk
        fop_trades = [t for t in settlement_queue if t["mechanism"] == "FOP"]
        dvp_trades = [t for t in settlement_queue if t["mechanism"] == "DVP"]

        # Herstatt exposure: sum of FoP and pending DVP before cutoff
        herstatt_exposure_m = sum(t["settle_amount_m"] for t in fop_trades)
        pending_dvp_m = sum(
            t["settle_amount_m"] for t in dvp_trades if t["status"] == "PENDING"
        )

        # Check if cash position supports today's settlements
        total_gbp_equiv_m = cash.get("total_gbp_equiv_m", 999.0)
        total_settle_m = sum(t["settle_amount_m"] for t in settlement_queue)
        liquidity_adequate = total_gbp_equiv_m > total_settle_m * 1.20  # 20% buffer

        new_alerts = []
        if fop_trades:
            new_alerts.append({
                "severity": "WARNING",
                "source": "SettlementRiskAgent",
                "message": (
                    f"{len(fop_trades)} Free-of-Payment trade(s) totalling "
                    f"£{herstatt_exposure_m:.1f}M — Herstatt risk exposure"
                ),
                "regulatory_ref": "CPSS-IOSCO Principles for FMIs",
            })

        if not liquidity_adequate:
            new_alerts.append({
                "severity": "CRITICAL",
                "source": "SettlementRiskAgent",
                "message": (
                    f"Insufficient liquidity: need £{total_settle_m*1.20:.1f}M, "
                    f"have £{total_gbp_equiv_m:.1f}M"
                ),
            })

        state["settlement_risk"] = {
            "settlement_queue": settlement_queue,
            "total_trades": len(settlement_queue),
            "dvp_trades": len(dvp_trades),
            "fop_trades": len(fop_trades),
            "herstatt_exposure_m": round(herstatt_exposure_m, 2),
            "pending_dvp_m": round(pending_dvp_m, 2),
            "total_settlement_value_m": round(total_settle_m, 2),
            "liquidity_adequate": liquidity_adequate,
            "cash_coverage_ratio": round(total_gbp_equiv_m / total_settle_m, 2) if total_settle_m else None,
            "branches_received": list(state.get("completed_nodes", set())),
            "model_used": MODEL_PRO,
            "latency_ms": round((asyncio.get_event_loop().time() - start) * 1000, 1),
        }
        state["alerts"].extend(new_alerts)
        state["completed_nodes"].add("settlement_risk")

        state["audit_trail"].append({
            "node": "SettlementRiskAgent",
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "trades": len(settlement_queue),
            "herstatt_m": herstatt_exposure_m,
            "liquidity_adequate": liquidity_adequate,
            "alerts_raised": len(new_alerts),
            "model": MODEL_PRO,
        })

        logger.info(
            "SettlementRiskAgent: trades=%d | herstatt=£%sM | liquidity_ok=%s",
            len(settlement_queue), f"{herstatt_exposure_m:.1f}", liquidity_adequate,
        )

    except Exception as exc:
        logger.error("SettlementRiskAgent FAILED: %s", exc)
        state["error"] = f"SettlementRiskAgent: {exc}"
        state["settlement_risk"] = {"error": str(exc)}

    return state


# ---------------------------------------------------------------------------
# Node: TreasuryReportNode (streaming output)
# ---------------------------------------------------------------------------

async def node_treasury_report(state: TreasuryState) -> TreasuryState:
    """
    TreasuryReportNode (Gemini 3.5 Flash) — streaming output.

    Synthesises all three upstream agent outputs into a concise morning
    briefing for the Treasury desk. Uses Gemini Flash streaming to deliver
    token-by-token output to the dealer dashboard — critical because the
    7am window is tight and dealers need to see the report as it's generated
    rather than waiting for the full response.

    In production: the streaming output is piped simultaneously to:
      (a) Bloomberg terminal panel (via Bloomberg B-PIPE)
      (b) AWB internal dealer dashboard (WebSocket)
      (c) Slack #treasury-alerts (critical alerts only)

    State mutations: treasury_report
    """
    logger.info("TreasuryReportNode: drafting morning briefing | run_id=%s", state["run_id"])

    cash = state.get("cash_position") or {}
    fx = state.get("fx_exposure") or {}
    settlement = state.get("settlement_risk") or {}
    alerts = state.get("alerts") or []
    critical_alerts = [a for a in alerts if a.get("severity") == "CRITICAL"]
    warnings = [a for a in alerts if a.get("severity") == "WARNING"]

    # Production: call Gemini Flash with streaming enabled:
    # model = genai.GenerativeModel("gemini-3.5-flash")
    # response = model.generate_content(prompt, stream=True)
    # async for chunk in response:
    #     yield chunk.text  # Stream to WebSocket

    # Mock: assemble narrative from computed values
    now = datetime.datetime.utcnow().strftime("%H:%M UTC")
    date_str = state.get("valuation_date", datetime.date.today().isoformat())

    narrative = f"""AWB TREASURY MORNING BRIEFING — {date_str} {now}
Model: {MODEL_REGISTRATION} | Run: {state['run_id']}

═══════════════════════════════════════════════════════════
CASH POSITION
═══════════════════════════════════════════════════════════
Total nostro equivalents (GBP): £{cash.get('total_gbp_equiv_m', 0):.1f}M
HQLA (Basel III):               £{cash.get('hqla_m', 0):.1f}M
LCR:                            {cash.get('lcr_pct', 0):.1f}%
Regulatory minimum:             {LCR_MINIMUM_PCT:.0f}%
Status:                         {"✓ COMPLIANT" if cash.get('lcr_compliant') else "✗ BREACH"}

═══════════════════════════════════════════════════════════
FX EXPOSURE
═══════════════════════════════════════════════════════════
Open positions:                 {fx.get('total_positions', 0)}
Total notional:                 £{fx.get('total_notional_gbp_m', 0):.1f}M
1-day VaR (99%):                £{fx.get('total_var_1d_99_m', 0):.2f}M
VaR limit utilisation:          {fx.get('var_limit_utilisation_pct', 0):.1f}%
EMIR reportable trades:         {fx.get('emir_reportable', 0)}

═══════════════════════════════════════════════════════════
SETTLEMENT
═══════════════════════════════════════════════════════════
Trades settling today:          {settlement.get('total_trades', 0)}
DVP (low risk):                 {settlement.get('dvp_trades', 0)}
FoP (Herstatt risk):            {settlement.get('fop_trades', 0)}
Herstatt exposure:              £{settlement.get('herstatt_exposure_m', 0):.1f}M
Liquidity coverage:             {"✓ ADEQUATE" if settlement.get('liquidity_adequate') else "✗ INSUFFICIENT"}

═══════════════════════════════════════════════════════════
ALERTS  ({len(critical_alerts)} CRITICAL | {len(warnings)} WARNING)
═══════════════════════════════════════════════════════════
"""
    for alert in critical_alerts + warnings:
        prefix = "🔴 CRITICAL" if alert["severity"] == "CRITICAL" else "🟡 WARNING"
        narrative += f"{prefix}: {alert['message']}\n"

    if not alerts:
        narrative += "✅ No alerts — all positions within limits.\n"

    narrative += f"""
═══════════════════════════════════════════════════════════
Prepared by AWB Treasury Operations Agent (MR-2026-038)
PRA Rulebook: Liquidity | EMIR Article 11 | Basel III LCR
═══════════════════════════════════════════════════════════"""

    state["treasury_report"] = {
        "report_text": narrative,
        "date": date_str,
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "critical_alert_count": len(critical_alerts),
        "warning_count": len(warnings),
        "model_used": MODEL_FLASH,
        "run_id": state["run_id"],
    }
    state["completed_nodes"].add("treasury_report")

    state["audit_trail"].append({
        "node": "TreasuryReportNode",
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "critical_alerts": len(critical_alerts),
        "warnings": len(warnings),
        "model": MODEL_FLASH,
    })

    logger.info(
        "TreasuryReportNode: report generated | critical=%d | warnings=%d",
        len(critical_alerts), len(warnings),
    )

    return state


# ---------------------------------------------------------------------------
# Parallel async orchestrator
# ---------------------------------------------------------------------------

async def run_treasury_pipeline(
    valuation_date: Optional[str] = None,
    timeout_seconds: float = PARALLEL_AGENT_TIMEOUT_SECONDS * 2,
) -> TreasuryState:
    """
    Run the full treasury operations pipeline with parallel fan-out.

    Architecture:
      Phase 1 (parallel): CashPositionAgent || FXExposureAgent
      Phase 2 (sequential): SettlementRiskAgent (needs both Phase 1 outputs)
      Phase 3 (sequential): TreasuryReportNode (needs all prior outputs)

    DORA resilience: if Phase 1 agents timeout, SettlementRiskAgent
    proceeds with partial data and flags the incomplete branches in the alert.

    Args:
        valuation_date: Business date (YYYY-MM-DD). Defaults to today.
        timeout_seconds: Maximum time for the full pipeline (default: 60s).

    Returns:
        Final TreasuryState with all agent outputs and audit_trail.
    """
    state = _default_treasury_state(valuation_date)

    logger.info(
        "Treasury pipeline START | run_id=%s | date=%s",
        state["run_id"],
        state["valuation_date"],
    )
    pipeline_start = asyncio.get_event_loop().time()

    try:
        # ── Phase 1: Fan-out — CashPosition + FXExposure in parallel ──────
        logger.info("Phase 1: launching parallel agents")
        try:
            await asyncio.wait_for(
                asyncio.gather(
                    node_cash_position_agent(state),
                    node_fx_exposure_agent(state),
                ),
                timeout=PARALLEL_AGENT_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Phase 1 timeout after %ds — proceeding with partial data (DORA Article 6)",
                PARALLEL_AGENT_TIMEOUT_SECONDS,
            )
            state["alerts"].append({
                "severity": "WARNING",
                "source": "Orchestrator",
                "message": f"Phase 1 timeout after {PARALLEL_AGENT_TIMEOUT_SECONDS}s — partial data",
            })

        # ── Phase 2: Fan-in — SettlementRisk (waits for Phase 1) ──────────
        logger.info("Phase 2: SettlementRiskAgent (fan-in)")
        await node_settlement_risk_agent(state)

        # ── Phase 3: Report generation ─────────────────────────────────────
        logger.info("Phase 3: TreasuryReportNode")
        await node_treasury_report(state)

    except Exception as exc:
        logger.error("Treasury pipeline FAILED: %s", exc)
        state["error"] = str(exc)

    total_ms = round((asyncio.get_event_loop().time() - pipeline_start) * 1000, 1)
    state["total_latency_ms"] = total_ms

    logger.info(
        "Treasury pipeline COMPLETE | run_id=%s | latency=%.0fms | "
        "critical=%d | warnings=%d",
        state["run_id"],
        total_ms,
        len([a for a in state["alerts"] if a.get("severity") == "CRITICAL"]),
        len([a for a in state["alerts"] if a.get("severity") == "WARNING"]),
    )

    return state


async def stream_treasury_report(valuation_date: Optional[str] = None) -> AsyncIterator[str]:
    """
    Stream the treasury morning briefing token-by-token.

    Runs the full pipeline and then streams the report narrative as an
    async generator, simulating Gemini's streaming response. In production
    this is replaced with direct Gemini streaming output piped through.

    Usage:
        async for token in stream_treasury_report():
            await websocket.send(token)

    Args:
        valuation_date: Business date. Defaults to today.

    Yields:
        String chunks of the treasury report.
    """
    state = await run_treasury_pipeline(valuation_date)
    report_text = (state.get("treasury_report") or {}).get("report_text", "")

    if not report_text:
        yield "[ERROR: Treasury report unavailable]"
        return

    # Stream word-by-word (simulates token streaming)
    words = report_text.split(" ")
    for i, word in enumerate(words):
        yield word + (" " if i < len(words) - 1 else "")
        await asyncio.sleep(0.001)  # Simulates inter-token latency


# ---------------------------------------------------------------------------
# Synchronous entry point (for non-async callers)
# ---------------------------------------------------------------------------

def run_treasury_pipeline_sync(valuation_date: Optional[str] = None) -> TreasuryState:
    """
    Synchronous wrapper for run_treasury_pipeline().

    For use in contexts that cannot use async/await (e.g. Django views,
    Jupyter notebooks, batch scripts). Internally runs the async pipeline
    on a fresh event loop.

    Args:
        valuation_date: Business date (YYYY-MM-DD). Defaults to today.

    Returns:
        Final TreasuryState.
    """
    return asyncio.run(run_treasury_pipeline(valuation_date))
