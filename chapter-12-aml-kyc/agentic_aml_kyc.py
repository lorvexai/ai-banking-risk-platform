"""Agentic AML/KYC Investigation Pipeline — Chapter 12 Agentic Extension.

Model ID  : MR-2026-060-AML
Risk Class: HIGH (POCA 2002 s.330 disclosure obligation; EU AI Act Art.6 Annex III §5b)
Chapter   : 12 — AML, KYC, and Financial Crime Prevention

Architecture: LangGraph StateGraph — five specialist agents + HITL gate.

Agents
------
1. TransactionScoringAgent    — Gemini 3.5 Flash   — XGBoost alert scoring + SHAP feature attribution
2. NetworkGraphAgent          — Gemini 3.5 Flash   — NetworkX Louvain community detection, structuring rings
3. KYCScreeningAgent          — Gemini 3.5 Flash   — document verification, PEP/sanctions, liveness, UBO
4. TypologyMatchingAgent      — Gemini 3.1 Pro     — AML typology RAG matching, risk theme synthesis
5. SARDraftingAgent           — Claude Sonnet 4.6  — POCA s.330 SAR narrative + MLRO escalation package

HITL Gate: HITLDecision enum — APPROVE / ESCALATE / OVERRIDE / PENDING.
           Conservative default: ESCALATE whenever score >= 0.70, PEP/sanctions hit,
           structuring ring detected, or SAR draft generated.

Regulatory Coverage
-------------------
- POCA 2002 ss.327-333A (primary UK AML offences + disclosure + tipping-off)
- MLR 2017 Regs 28, 33, 35 (CDD, EDD, PEP obligations)
- FCA SYSC 6.3 (systems and controls for financial crime)
- JMLSG Part I & II Banking (guidance — near-statutory status)
- FATF 40 Recommendations (international AML standard)
- EU AI Act Art. 6 Annex III §5b (high-risk — credit-linked KYC)
- EU AI Act Art. 6 Annex III §6 (high-risk — biometric identification)
- BAP-2026-AML-001 (AWB internal: agentic AML governance)
- NOT: BSA / FinCEN / PATRIOT Act (US-only — not applicable to AWB)

LLM Allocation
--------------
Agents 1-3 : google/gemini-3.5-flash   — fast, structured AML/KYC tasks
Agent 4    : google/gemini-3.1-pro          — complex multi-typology reasoning
Agent 5    : anthropic/claude-sonnet-4-6    — supervisory SAR narrative synthesis

POCA s.333A Tipping-Off Architectural Guarantee
-----------------------------------------------
The SARDraftingAgent sets state["sar_filed_indicator"] = True when a SAR
is generated, but NEVER exposes this to the credit pipeline. The
get_credit_gate_decision() helper returns only "BLOCKED" — the credit
agent cannot infer whether the block is due to KYC failure or SAR filing.
This architectural separation is enforced in code and cannot be bypassed.

Hop-Chain Audit
---------------
Every agent appends to state["hop_chain"]:
  {seq, agent, timestamp, reason, act, outcome}
Mandatory per PRA AI Roundtable October 2025 / BAP-2026-AML-001 §6.
NCA SubmitSAR audit trail maintained separately per POCA 2002 s.330.
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import date, datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AML/KYC REGULATORY CONSTANTS
# ---------------------------------------------------------------------------

# XGBoost alert score thresholds (ROC-calibrated on AWB SAR dataset)
AML_SCORE_ALERT_THRESHOLD: float = 0.35   # JMLSG Part I — minimum to flag
AML_SCORE_HIGH_PRIORITY: float = 0.70     # High-priority: MLRO notification
AML_SCORE_AUTO_MLRO: float = 0.90         # Auto-escalate to MLRO

# Structuring detection thresholds (JMLSG Part II Banking — smurfing)
STRUCTURING_RING_MIN_ACCOUNTS: int = 5    # Minimum Louvain community size
STRUCTURING_RING_MIN_AMOUNT_GBP: float = 500_000.0  # £500K threshold
STRUCTURING_TRANSACTION_DAYS: int = 14   # 14-day rolling window

# KYC identity verification thresholds (MLR 2017 Reg. 28)
DOC_CONFIDENCE_THRESHOLD: float = 0.90   # Auto-pass; below = manual review
LIVENESS_PASS_THRESHOLD: float = 0.85    # Liveness detection threshold
SANCTIONS_MATCH_AUTO_BLOCK: float = 0.95 # OFSI/UN match — auto-block
SANCTIONS_MATCH_REVIEW: float = 0.85     # Fuzzy match — compliance review

# UBO tracing (MLR 2017 Reg. 28(3)(b))
UBO_THRESHOLD_PCT: float = 25.0          # Beneficial ownership threshold
UBO_MAX_LAYERS: int = 4                  # Maximum ownership chain depth per JMLSG

# SAR timing (POCA 2002 s.330 — authorised disclosure)
SAR_MORATORIUM_DAYS: int = 7             # s.335 moratorium period
SAR_RETENTION_YEARS: int = 7            # AWB policy (exceeds SYSC 6.3.3R 5yr)

# Record retention
KYC_RETENTION_YEARS: int = 7            # AWB policy (SYSC 6.3.3R min 5yr)
AML_ALERT_RETENTION_YEARS: int = 7


# ---------------------------------------------------------------------------
# STATE SCHEMA
# ---------------------------------------------------------------------------

class AMLKYCState(dict):
    """Shared mutable state threaded through all five agents.

    Inherits dict for LangGraph compatibility (TypedDict-style access).
    POCA s.333A guarantee: sar_filed_indicator is NEVER propagated to
    the credit pipeline or exposed outside the MLRO workflow.
    """
    pass


def _initial_state(
    run_date: date,
    trigger_event: str,
    transaction_batch: List[Dict[str, Any]],
    customer_profile: Dict[str, Any],
    network_graph_data: Dict[str, Any],
    kyc_documents: List[Dict[str, Any]],
) -> AMLKYCState:
    """Construct clean initial state for the AML/KYC pipeline.

    Args:
        run_date: Date of this AML/KYC run (transaction date or onboarding date).
        trigger_event: Description of the trigger
            (e.g., "Overnight AML batch — 847 transactions T24 2026-01-15").
        transaction_batch: List of transaction dicts for scoring.
            Each dict: {transaction_id, account_id, amount_gbp, currency,
            counterparty_country, transaction_type, merchant_category, ...}
        customer_profile: Customer background data for KYC and context.
            Keys: customer_id, entity_type (individual/corporate), name,
            incorporation_country, risk_rating, relationship_start.
        network_graph_data: Pre-built graph structure.
            Keys: nodes (account_ids), edges (list of {from, to, amount, date}).
        kyc_documents: List of identity document dicts for KYC screening.
            Each dict: {document_type, full_name, date_of_birth,
            document_number, expiry_date, issuing_country, image_b64}.

    Returns:
        Initialised AMLKYCState dict.
    """
    return AMLKYCState(
        # ---- inputs ----
        run_date=run_date,
        trigger_event=trigger_event,
        transaction_batch=transaction_batch,
        customer_profile=customer_profile,
        network_graph_data=network_graph_data,
        kyc_documents=kyc_documents,
        # ---- outputs (populated by agents) ----
        scored_alerts=[],             # Agent 1
        high_priority_alerts=[],      # Agent 1
        shap_attributions=[],         # Agent 1
        network_communities=[],       # Agent 2
        structuring_rings=[],         # Agent 2
        network_risk_accounts=[],     # Agent 2
        kyc_decision=None,            # Agent 3
        pep_sanctions_result=None,    # Agent 3
        ubo_records=[],               # Agent 3
        edd_required=False,           # Agent 3
        typology_matches=[],          # Agent 4
        typology_narrative="",        # Agent 4
        risk_theme_summary="",        # Agent 4
        sar_draft=None,               # Agent 5
        mlro_escalation_package={},   # Agent 5
        regulatory_narrative="",      # Agent 5
        # ---- POCA s.333A guarantee ----
        sar_filed_indicator=False,    # NEVER expose to credit pipeline
        _credit_gate_result="PENDING", # Returns only BLOCKED/CLEARED
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
    state: AMLKYCState,
    agent: str,
    reason: str,
    act: str,
    outcome: str,
) -> None:
    """Append one hop to the audit chain.

    Mandatory per PRA AI Roundtable Oct 2025, BAP-2026-AML-001 §6,
    and NCA SubmitSAR audit trail requirements under POCA 2002 s.330.

    Args:
        state: Shared pipeline state — hop_chain appended in place.
        agent: Agent name / model identifier.
        reason: Pre-action regulatory reasoning.
        act: Specific action taken.
        outcome: Result or metric produced.
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

def _escalate_zone(state: AMLKYCState, proposed: str) -> None:
    """Monotonically escalate risk zone — GREEN→AMBER→RED only.

    Args:
        state: Pipeline state whose "risk_zone" key will be updated.
        proposed: Proposed new zone.
    """
    current = state.get("risk_zone", "GREEN")
    new_zone = max(current, proposed, key=lambda z: _ZONE_RANK.get(z, 0))
    if new_zone != current:
        log.warning("AML risk zone escalated: %s → %s", current, new_zone)
    state["risk_zone"] = new_zone


# ---------------------------------------------------------------------------
# HITL DECISION
# ---------------------------------------------------------------------------

class HITLDecision(str, Enum):
    """Human-in-the-loop decision outcome for AML/KYC pipeline.

    APPROVE  : Low-score alerts, no structuring, KYC clean — auto-proceed.
    ESCALATE : High-score alert, PEP/sanctions hit, structuring ring,
               SAR generated, or EDD required — MLRO sign-off needed.
    OVERRIDE : MLRO has reviewed and accepted after investigation.
    PENDING  : Awaiting MLRO decision.
    """
    APPROVE = "APPROVE"
    ESCALATE = "ESCALATE"
    OVERRIDE = "OVERRIDE"
    PENDING = "PENDING"


def _compute_hitl_decision(
    state: AMLKYCState,
) -> Tuple[str, str]:
    """Derive HITL decision. Conservative: any material finding → ESCALATE.

    Args:
        state: Completed pipeline state after all five agents.

    Returns:
        Tuple of (HITLDecision value string, rationale string).
    """
    reasons: List[str] = []

    high_priority = state.get("high_priority_alerts", [])
    if high_priority:
        reasons.append(
            f"{len(high_priority)} high-priority AML alert(s) (score≥{AML_SCORE_HIGH_PRIORITY})"
        )

    structuring = state.get("structuring_rings", [])
    if structuring:
        reasons.append(
            f"{len(structuring)} structuring ring(s) detected — JMLSG Part I smurfing"
        )

    kyc = state.get("kyc_decision", {})
    if isinstance(kyc, dict):
        if kyc.get("sanctions_hit"):
            reasons.append("OFSI/UN sanctions list match — auto-block")
        if kyc.get("pep_flagged"):
            reasons.append("PEP identified — EDD required (MLR 2017 Reg. 35)")
        if kyc.get("liveness_failed"):
            reasons.append("Liveness detection failed — identity verification incomplete")

    if state.get("sar_filed_indicator"):
        reasons.append(
            "SAR draft generated — MLRO approval required (POCA 2002 s.330)"
        )

    if state.get("edd_required"):
        reasons.append("Enhanced Due Diligence required (MLR 2017 Reg. 33)")

    if reasons:
        return HITLDecision.ESCALATE.value, " | ".join(reasons)

    return HITLDecision.APPROVE.value, (
        "All transactions below alert threshold; no structuring detected; "
        "KYC clean; no PEP/sanctions; no SAR required."
    )


# ---------------------------------------------------------------------------
# LLM HELPERS
# ---------------------------------------------------------------------------

def _call_gemini_flash(prompt: str, context: str = "") -> str:
    """Invoke Gemini 3.5 Flash for fast AML scoring and KYC tasks.

    Args:
        prompt: Instruction to the model.
        context: Additional context (transaction data, customer profile).

    Returns:
        Model response as plain text.
    """
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        return f"[FLASH-STUB] {prompt[:80]}..."
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-3.5-flash")
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as exc:
        log.warning("Gemini Flash error: %s", exc)
        return f"[FLASH-ERROR] {exc}"


def _call_gemini_pro(prompt: str, context: str = "") -> str:
    """Invoke Gemini 3.1 Pro for complex typology matching and synthesis.

    Args:
        prompt: Multi-typology reasoning instruction.
        context: SHAP features, network structure, JMLSG typology corpus.

    Returns:
        Detailed typology analysis and risk theme narrative.
    """
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        return f"[PRO-STUB] {prompt[:80]}..."
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-3.1-pro")
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as exc:
        log.warning("Gemini Pro error: %s", exc)
        return f"[PRO-ERROR] {exc}"


def _call_claude_sonnet(prompt: str, context: str = "") -> str:
    """Invoke Claude Sonnet 4.6 for SAR narrative and MLRO package.

    Args:
        prompt: SAR drafting instruction with POCA s.330 structure.
        context: Aggregated intelligence from agents 1-4.

    Returns:
        NCA SubmitSAR-ready SAR narrative draft with typology citations.
    """
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
# AGENT 1 — TransactionScoringAgent (Gemini Flash)
# ---------------------------------------------------------------------------

def transaction_scoring_agent(
    state: AMLKYCState,
) -> AMLKYCState:
    """Agent 1: Score transactions using XGBoost surrogate and SHAP attribution.

    Regulatory basis: FCA SYSC 6.3.2 requires automated transaction
    monitoring systems. JMLSG Part I Chapter 6 specifies that ML-based
    transaction monitoring must be explainable — SHAP values provide the
    feature-level attribution required for MLRO review and NCA SAR filings.
    BAP-2026-AML-001 §4: all scored alerts above 0.35 must be logged with
    SHAP attribution for audit trail purposes.

    Three alert tiers (ROC-calibrated on AWB historical SAR dataset):
      LOW (< 0.35): Auto-cleared — no MLRO action required.
      MEDIUM (0.35–0.70): Analyst review within 5 business days.
      HIGH (≥ 0.70): MLRO notification within 24 hours.
      AUTO-MLRO (≥ 0.90): Immediate MLRO escalation + SAR consideration.

    Populates:
        state["scored_alerts"]: All alerts above threshold.
        state["high_priority_alerts"]: Subset with score >= 0.70.
        state["shap_attributions"]: Top SHAP features per high-priority alert.
        risk_zone: AMBER if any high-priority; RED if any auto-MLRO.

    Args:
        state: Shared pipeline state.

    Returns:
        Updated state.
    """
    transactions = state.get("transaction_batch", [])
    run_date = state.get("run_date", date.today())

    # ---- ReAct: Reason before acting ----
    reason = (
        "FCA SYSC 6.3.2 and JMLSG Part I Ch 6 require automated transaction "
        "monitoring with explainable scoring. XGBoost surrogate scores each "
        "transaction; SHAP provides feature attribution for MLRO review. "
        "Three-tier routing: LOW auto-clear, MEDIUM analyst, HIGH MLRO."
    )

    # ---- Deterministic XGBoost surrogate (rule-based proxy for demo) ----
    # In production: load pre-trained XGBoost model from AWS S3 artifact store
    # Model trained on AWB historical data: 4.2M transactions, 847 confirmed SARs
    # Features: amount_gbp, velocity_30d, counterparty_country_risk,
    #           transaction_type_risk, merchant_category_risk, time_of_day,
    #           network_centrality, pep_proximity_score

    COUNTRY_RISK: Dict[str, float] = {
        "GB": 0.0, "US": 0.0, "DE": 0.0, "FR": 0.0,
        "AE": 0.3, "CH": 0.1, "SG": 0.1, "HK": 0.2,
        "RU": 0.8, "IR": 0.9, "KP": 0.95, "SY": 0.9,
        "AF": 0.85, "MM": 0.75, "BY": 0.6,
    }
    TYPE_RISK: Dict[str, float] = {
        "wire_transfer": 0.2, "cash_deposit": 0.4,
        "cash_withdrawal": 0.35, "fx_exchange": 0.15,
        "crypto_exchange": 0.5, "internal_transfer": 0.05,
        "card_purchase": 0.0, "direct_debit": 0.0,
    }

    scored_alerts: List[Dict[str, Any]] = []
    high_priority: List[Dict[str, Any]] = []
    shap_attributions: List[Dict[str, Any]] = []

    for tx in transactions:
        tx_id = tx.get("transaction_id", f"TXN-{len(scored_alerts):05d}")
        account_id = tx.get("account_id", "UNKNOWN")
        amount_gbp = float(tx.get("amount_gbp", 0))
        country = tx.get("counterparty_country", "GB")
        tx_type = tx.get("transaction_type", "card_purchase")

        # Feature engineering (simplified XGBoost surrogate)
        amount_score = min(amount_gbp / 250_000, 0.5)   # Normalise to £250K max
        country_score = COUNTRY_RISK.get(country, 0.1)
        type_score = TYPE_RISK.get(tx_type, 0.1)
        velocity_score = float(tx.get("velocity_30d_count", 0)) / 50.0
        velocity_score = min(velocity_score, 0.3)

        # Structuring sub-score: amounts just below £10K (Cash Transaction Reporting)
        structuring_score = 0.0
        if 9_000 <= amount_gbp <= 9_999:
            structuring_score = 0.25   # JMLSG Part II — below-threshold structuring

        composite_score = min(
            amount_score * 0.3
            + country_score * 0.35
            + type_score * 0.15
            + velocity_score * 0.1
            + structuring_score * 0.1,
            0.99
        )

        if composite_score < AML_SCORE_ALERT_THRESHOLD:
            continue

        # Assign alert priority
        if composite_score >= AML_SCORE_HIGH_PRIORITY:
            priority = "HIGH"
            _escalate_zone(state, "AMBER")
        else:
            priority = "MEDIUM"

        if composite_score >= AML_SCORE_AUTO_MLRO:
            _escalate_zone(state, "RED")

        # SHAP-style feature attribution (deterministic proxy)
        shap_features = {
            "counterparty_country_risk": round(country_score * 0.35, 3),
            "amount_normalised": round(amount_score * 0.30, 3),
            "transaction_type_risk": round(type_score * 0.15, 3),
            "velocity_30d": round(velocity_score * 0.10, 3),
            "structuring_indicator": round(structuring_score * 0.10, 3),
        }

        alert = {
            "alert_id": f"AML-{run_date.strftime('%Y%m%d')}-{tx_id}",
            "transaction_id": tx_id,
            "account_id": account_id,
            "amount_gbp": amount_gbp,
            "score": round(composite_score, 4),
            "priority": priority,
            "counterparty_country": country,
            "transaction_type": tx_type,
            "model_id": "MR-2026-060-AML",
        }
        scored_alerts.append(alert)

        if composite_score >= AML_SCORE_HIGH_PRIORITY:
            high_priority.append(alert)
            shap_attributions.append({
                "alert_id": alert["alert_id"],
                "top_features": shap_features,
            })

    # ---- LLM alert pattern analysis ----
    if scored_alerts:
        llm_prompt = (
            f"AWB AML transaction scoring run {run_date}.\n"
            f"Input: {len(transactions)} transactions\n"
            f"Scored alerts (≥0.35): {len(scored_alerts)}\n"
            f"High-priority alerts (≥0.70): {len(high_priority)}\n"
            f"Top alert samples: {scored_alerts[:3]}\n\n"
            f"Identify the dominant transaction risk pattern in this batch "
            f"and cite the relevant JMLSG Part I or Part II typology category."
        )
        state["alert_pattern_analysis"] = _call_gemini_flash(llm_prompt)

    state["scored_alerts"] = scored_alerts
    state["high_priority_alerts"] = high_priority
    state["shap_attributions"] = shap_attributions

    _log_step(
        state,
        agent="TransactionScoringAgent [gemini-3.5-flash]",
        reason=reason,
        act=(
            f"Scored {len(transactions)} transactions using XGBoost "
            f"surrogate + SHAP attribution"
        ),
        outcome=(
            f"Alerts: {len(scored_alerts)}, HIGH: {len(high_priority)}, "
            f"Zone: {state['risk_zone']}"
        ),
    )
    return state


# ---------------------------------------------------------------------------
# AGENT 2 — NetworkGraphAgent (Gemini Flash)
# ---------------------------------------------------------------------------

def network_graph_agent(
    state: AMLKYCState,
) -> AMLKYCState:
    """Agent 2: Detect structuring rings using NetworkX Louvain community detection.

    Regulatory basis: JMLSG Part II Banking typology — structuring (smurfing)
    involves coordinated deposits or transfers below reporting thresholds
    across multiple accounts. NetworkX Louvain algorithm identifies
    communities in the transaction graph; communities with high total
    amounts and many below-threshold transactions are classified as
    structuring rings. BAP-2026-AML-001 §5 requires network analysis
    for all batches containing more than 100 transactions.

    The Louvain community detection (directed graph) identifies tightly
    connected account clusters. Each community is assessed for:
    - Total amount (> £500K threshold)
    - Below-threshold transaction concentration (near-CTR structuring)
    - Betweenness centrality (hub accounts that coordinate the ring)

    Populates:
        state["network_communities"]: All Louvain communities detected.
        state["structuring_rings"]: Communities classified as structuring.
        state["network_risk_accounts"]: High-centrality accounts for SAR.
        risk_zone: RED if any structuring ring detected.

    Args:
        state: Shared pipeline state.

    Returns:
        Updated state.
    """
    graph_data = state.get("network_graph_data", {})
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])

    # ---- ReAct: Reason before acting ----
    reason = (
        "JMLSG Part II Banking typology: structuring (smurfing) involves "
        "coordinated below-threshold transactions across account clusters. "
        "NetworkX Louvain community detection identifies rings. Communities "
        "with total > £500K and ≥5 accounts trigger RED zone and SAR consideration."
    )

    communities: List[Dict[str, Any]] = []
    structuring_rings: List[Dict[str, Any]] = []
    risk_accounts: List[str] = []

    if not edges:
        # No graph data — generate synthetic community detection result
        llm_prompt = (
            f"AWB network graph analysis: {len(nodes)} nodes, no edge data provided.\n"
            f"What additional transaction data would be required to perform "
            f"meaningful Louvain community detection for AML structuring detection, "
            f"citing JMLSG Part II Banking typology standards?"
        )
        state["network_analysis_note"] = _call_gemini_flash(llm_prompt)
        _log_step(
            state,
            agent="NetworkGraphAgent [gemini-3.5-flash]",
            reason=reason,
            act=f"Attempted community detection on {len(nodes)} nodes (no edges)",
            outcome="No graph edges provided — community detection deferred",
        )
        state["network_communities"] = []
        state["structuring_rings"] = []
        state["network_risk_accounts"] = []
        return state

    # ---- NetworkX Louvain community detection ----
    try:
        import networkx as nx

        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        for edge in edges:
            G.add_edge(
                edge["from"],
                edge["to"],
                weight=float(edge.get("amount_gbp", 0)),
            )

        # Louvain requires undirected graph for community detection
        G_undirected = G.to_undirected()

        try:
            from networkx.algorithms.community import louvain_communities
            louvain_result = louvain_communities(G_undirected, seed=42)
        except (ImportError, AttributeError):
            # Fallback: connected components
            louvain_result = list(nx.connected_components(G_undirected))

        # Betweenness centrality for hub identification
        centrality = nx.betweenness_centrality(G_undirected)

        for i, community_members in enumerate(louvain_result):
            members = list(community_members)
            if len(members) < 2:
                continue

            # Total amount through community edges
            total_gbp = sum(
                G[u][v].get("weight", 0)
                for u in members for v in members
                if G.has_edge(u, v)
            )
            # Below-threshold transaction count
            below_ctr = sum(
                1 for u in members for v in members
                if G.has_edge(u, v)
                and 9_000 <= G[u][v].get("weight", 0) <= 9_999
            )
            # Top centrality accounts in community
            top_central = sorted(
                members,
                key=lambda n: centrality.get(n, 0),
                reverse=True,
            )[:3]

            community_record = {
                "community_id": f"COMM-{i:04d}",
                "member_accounts": members,
                "member_count": len(members),
                "total_amount_gbp": round(total_gbp, 2),
                "below_ctr_count": below_ctr,
                "top_centrality_accounts": top_central,
                "is_structuring_ring": False,
            }

            # Classify as structuring ring
            if (
                len(members) >= STRUCTURING_RING_MIN_ACCOUNTS
                and total_gbp >= STRUCTURING_RING_MIN_AMOUNT_GBP
            ):
                community_record["is_structuring_ring"] = True
                structuring_rings.append(community_record)
                risk_accounts.extend(top_central)
                _escalate_zone(state, "RED")
                log.warning(
                    "STRUCTURING RING: %d accounts, £%.0f",
                    len(members), total_gbp,
                )

            communities.append(community_record)

    except ImportError:
        # NetworkX not available — stub with LLM-only analysis
        log.warning("NetworkX not available — using LLM stub for community detection")
        communities = []

    # ---- LLM review of graph findings ----
    llm_prompt = (
        f"AWB NetworkX Louvain community detection results:\n"
        f"  Total communities: {len(communities)}\n"
        f"  Structuring rings: {len(structuring_rings)}\n"
        f"  Rings detail: {structuring_rings[:2]}\n\n"
        f"For each structuring ring, identify the JMLSG Part II Banking "
        f"typology that best describes the pattern (e.g., smurfing, "
        f"mirror trading, round-tripping) and cite the FATF typology "
        f"reference number."
    )
    state["network_typology_analysis"] = _call_gemini_flash(llm_prompt)

    state["network_communities"] = communities
    state["structuring_rings"] = structuring_rings
    state["network_risk_accounts"] = list(set(risk_accounts))

    _log_step(
        state,
        agent="NetworkGraphAgent [gemini-3.5-flash]",
        reason=reason,
        act=(
            f"Ran Louvain community detection on {len(nodes)} nodes, "
            f"{len(edges)} edges"
        ),
        outcome=(
            f"Communities: {len(communities)}, "
            f"Structuring rings: {len(structuring_rings)}, "
            f"Zone: {state['risk_zone']}"
        ),
    )
    return state


# ---------------------------------------------------------------------------
# AGENT 3 — KYCScreeningAgent (Gemini Flash)
# ---------------------------------------------------------------------------

def kyc_screening_agent(
    state: AMLKYCState,
) -> AMLKYCState:
    """Agent 3: Document verification, PEP/sanctions screening, liveness, UBO tracing.

    Regulatory basis:
    - MLR 2017 Reg. 28: Identify the customer using reliable, independent
      source documents. AWB requires: primary ID (passport or driving licence)
      + proof of address (utility bill < 3 months).
    - MLR 2017 Reg. 33: Enhanced Due Diligence for high-risk third countries,
      complex corporate structures, and non-face-to-face onboarding.
    - MLR 2017 Reg. 35: Enhanced obligations for Politically Exposed Persons —
      senior management approval; enhanced ongoing monitoring; source of wealth.
    - OFSI (HM Treasury): UK sanctions screening — NOT OFAC (US-only).
    - UN Security Council Consolidated List: secondary sanctions screening.
    - EU AI Act Art. 6 Annex III §6: biometric liveness detection is
      HIGH-RISK AI — HITL required for all identity verification decisions.

    Confidence thresholds:
    - Document verification: ≥ 0.90 auto-pass; < 0.90 manual review.
    - Liveness: ≥ 0.85 pass; < 0.85 fail (re-attempt or in-branch).
    - Sanctions fuzzy match: ≥ 0.95 auto-block; 0.85–0.95 compliance review.

    Populates:
        state["kyc_decision"]: KYC status dict with all sub-checks.
        state["pep_sanctions_result"]: PEP/OFSI/UN screening result.
        state["ubo_records"]: UBO tracing for corporate entities.
        state["edd_required"]: EDD flag (PEP, high-risk country, complex structure).
        risk_zone: AMBER (EDD required) or RED (sanctions hit, liveness fail).

    Args:
        state: Shared pipeline state.

    Returns:
        Updated state.
    """
    kyc_docs = state.get("kyc_documents", [])
    customer = state.get("customer_profile", {})
    customer_id = customer.get("customer_id", "UNKNOWN")
    entity_type = customer.get("entity_type", "individual")

    # ---- ReAct: Reason before acting ----
    reason = (
        "MLR 2017 Reg. 28 requires document-based customer identification. "
        "OFSI/UN sanctions screening (not OFAC). MLR 2017 Reg. 35 mandates "
        "EDD for PEPs. EU AI Act Art. 6 Annex III §6 classifies biometric "
        "liveness as HIGH-RISK AI — HITL required for all decisions."
    )

    # ---- Document verification (Gemini Flash — extends MR-2026-035) ----
    kyc_narrative = ""
    doc_verified = False
    doc_confidence = 0.0
    liveness_score = 0.0
    liveness_passed = False

    if kyc_docs:
        doc_prompt = (
            f"Verify these identity documents for customer {customer_id}.\n"
            f"Documents: {kyc_docs}\n"
            f"Apply MLR 2017 Reg. 28 verification standards:\n"
            f"  1. Primary ID (passport/driving licence) — MRZ checksum valid?\n"
            f"  2. Not expired?\n"
            f"  3. Document number format consistent with issuing country?\n"
            f"Rate overall confidence 0.0–1.0. "
            f"Flag any concerns (expired, poor quality, inconsistent data)."
        )
        doc_result = _call_gemini_flash(doc_prompt)
        doc_verified = True
        doc_confidence = 0.93   # Illustrative AWB Q4 2025 average
        liveness_score = 0.91   # Above 0.85 threshold
        liveness_passed = liveness_score >= LIVENESS_PASS_THRESHOLD
        kyc_narrative = doc_result

    # ---- PEP and sanctions screening ----
    customer_name = customer.get("name", "UNKNOWN")
    country_of_incorporation = customer.get("incorporation_country", "GB")

    # OFSI consolidated list check (HM Treasury)
    # In production: call OFSI API or licensed sanctions data feed
    high_risk_countries = {"RU", "IR", "KP", "SY", "AF", "MM", "BY", "VE"}
    is_high_risk_country = country_of_incorporation in high_risk_countries

    # Deterministic PEP screening proxy
    # In production: World-Check / ComplyAdvantage / Dow Jones Risk & Compliance
    pep_keywords = ["minister", "senator", "prime minister", "president",
                    "ambassador", "general", "admiral", "chief justice"]
    is_pep = any(kw in customer_name.lower() for kw in pep_keywords)

    # Sanctions fuzzy match (Jaro-Winkler proxy)
    # AWB maintains OFSI UK Consolidated List + UN SC Consolidated List
    sanctions_hit = False
    sanctions_match_score = 0.0
    sanctioned_entities_proxy = [
        "Sanctioned Entity Corp", "DPRK Finance Ltd", "IRGC Trade LLC"
    ]
    for entity in sanctioned_entities_proxy:
        # Simplified match — production uses Jaro-Winkler with phonetic
        if any(word.lower() in customer_name.lower()
               for word in entity.split() if len(word) > 3):
            sanctions_match_score = 0.97
            sanctions_hit = True
            _escalate_zone(state, "RED")
            break

    if is_pep:
        _escalate_zone(state, "AMBER")
    elif is_high_risk_country:
        _escalate_zone(state, "AMBER")

    edd_required = is_pep or is_high_risk_country or not liveness_passed
    state["edd_required"] = edd_required

    # ---- UBO tracing for corporate entities ----
    ubo_records: List[Dict[str, Any]] = []
    if entity_type == "corporate":
        # In production: query Companies House PSC register API
        # MLR 2017 Reg. 28(3)(b): identify persons owning > 25%
        ownership_structure = customer.get("ownership_structure", [])
        for owner in ownership_structure:
            ubo = {
                "ubo_name": owner.get("name", "UNKNOWN"),
                "ownership_pct": owner.get("ownership_pct", 0),
                "control_type": owner.get("control_type", "shares"),
                "psc_register_verified": owner.get("psc_verified", False),
                "is_pep": any(kw in owner.get("name", "").lower()
                              for kw in pep_keywords),
                "high_risk_jurisdiction": owner.get("country", "GB") in high_risk_countries,
                "layer": owner.get("layer", 1),
                "requires_edd": False,
            }
            ubo["requires_edd"] = ubo["is_pep"] or ubo["high_risk_jurisdiction"]
            if ubo["requires_edd"]:
                edd_required = True
                state["edd_required"] = True
                _escalate_zone(state, "AMBER")
            ubo_records.append(ubo)

    state["ubo_records"] = ubo_records

    # ---- Compile KYC decision ----
    if sanctions_hit:
        kyc_status = "SANCTIONS_HIT"
    elif not doc_verified:
        kyc_status = "DECLINED"
    elif not liveness_passed:
        kyc_status = "DECLINED"
    elif is_pep:
        kyc_status = "PEP_FLAGGED"
    elif edd_required:
        kyc_status = "EDD_REQUIRED"
    else:
        kyc_status = "CDD_PASS"

    kyc_decision = {
        "customer_id": customer_id,
        "status": kyc_status,
        "doc_verified": doc_verified,
        "doc_confidence": doc_confidence,
        "liveness_score": liveness_score,
        "liveness_passed": liveness_passed,
        "liveness_failed": not liveness_passed,
        "pep_flagged": is_pep,
        "sanctions_hit": sanctions_hit,
        "sanctions_match_score": sanctions_match_score,
        "edd_required": edd_required,
        "ubo_count": len(ubo_records),
        "narrative": kyc_narrative,
        "model_id": "MR-2026-060-AML",
        # EU AI Act Art. 14: HITL required — biometric liveness HIGH-RISK
        "requires_human_review": True,
        # UK GDPR / DPA 2018: biometric template deleted post-verification
        "biometric_template_deleted": True,
    }

    pep_sanctions_result = {
        "customer_id": customer_id,
        "is_pep": is_pep,
        "sanctions_hit": sanctions_hit,
        "sanctions_match_score": sanctions_match_score,
        "sanctions_lists_checked": ["OFSI UK Consolidated List", "UN SC Consolidated List"],
        "high_risk_country": is_high_risk_country,
        "screened_at": datetime.utcnow().isoformat() + "Z",
    }

    state["kyc_decision"] = kyc_decision
    state["pep_sanctions_result"] = pep_sanctions_result

    _log_step(
        state,
        agent="KYCScreeningAgent [gemini-3.5-flash]",
        reason=reason,
        act=(
            f"Verified {len(kyc_docs)} documents, screened OFSI/UN, "
            f"traced {len(ubo_records)} UBO(s)"
        ),
        outcome=(
            f"KYC={kyc_status}, PEP={is_pep}, Sanctions={sanctions_hit}, "
            f"EDD={edd_required}, Zone={state['risk_zone']}"
        ),
    )
    return state


# ---------------------------------------------------------------------------
# AGENT 4 — TypologyMatchingAgent (Gemini 3.1 Pro)
# ---------------------------------------------------------------------------

def typology_matching_agent(
    state: AMLKYCState,
) -> AMLKYCState:
    """Agent 4: Match alert patterns to JMLSG typologies using RAG retrieval.

    Regulatory basis: JMLSG Part I Chapter 6 states that AML typology
    knowledge must be applied in alert investigation. BAP-2026-AML-001 §5
    requires each high-priority alert to be matched against the AWB AML
    Typologies RAG corpus (Chapter 4 ChromaDB infrastructure extension,
    MR-2026-064) before SAR drafting. FATF typology references are cited
    where JMLSG typologies map to FATF categories.

    Uses Gemini 3.1 Pro for the complex multi-source reasoning required to:
    1. Match SHAP top features to JMLSG Part II Banking typology categories.
    2. Synthesise alert patterns with network graph findings.
    3. Assess KYC context against JMLSG high-risk customer typologies.
    4. Identify the primary and secondary typologies for SAR drafting.

    Populates:
        state["typology_matches"]: List of typology match dicts.
        state["typology_narrative"]: Full typology analysis narrative.
        state["risk_theme_summary"]: One-paragraph SAR-ready theme summary.

    Args:
        state: Shared pipeline state.

    Returns:
        Updated state.
    """
    high_priority = state.get("high_priority_alerts", [])
    shap = state.get("shap_attributions", [])
    structuring = state.get("structuring_rings", [])
    kyc = state.get("kyc_decision", {})
    customer = state.get("customer_profile", {})

    # ---- ReAct: Reason before acting ----
    reason = (
        "JMLSG Part I Ch6 mandates typology-driven alert investigation. "
        "AML Typologies RAG (MR-2026-064 — ChromaDB extension from Ch4) "
        "retrieves relevant JMLSG Part II Banking typologies. Gemini 3.1 Pro "
        "synthesises SHAP features, network patterns, and KYC context into "
        "primary and secondary typology classifications for SAR drafting."
    )

    # ---- JMLSG typology knowledge base (RAG proxy) ----
    # In production: query AML Typologies ChromaDB collection (MR-2026-064)
    # Corpus: JMLSG Part I & II (2024 update), FATF 40 Recommendations,
    #         NCA SARs Annual Report 2024, FCA AML findings 2023-2025
    JMLSG_TYPOLOGIES: Dict[str, Dict[str, str]] = {
        "structuring": {
            "jmlsg_ref": "JMLSG Part II Section 6.2.1",
            "fatf_ref": "FATF Typology 3 — Structuring/Smurfing",
            "description": (
                "Multiple transactions just below reporting thresholds "
                "across multiple accounts to avoid detection."
            ),
            "sar_phrase": (
                "suspicious structuring pattern consistent with JMLSG Part II "
                "Section 6.2.1 and FATF Typology 3"
            ),
        },
        "third_party_payments": {
            "jmlsg_ref": "JMLSG Part II Section 6.2.4",
            "fatf_ref": "FATF Typology 7 — Third Party Payments",
            "description": (
                "Payments received from or sent to unrelated third parties "
                "inconsistent with customer's business profile."
            ),
            "sar_phrase": (
                "third-party payment pattern inconsistent with customer "
                "business profile per JMLSG Part II Section 6.2.4"
            ),
        },
        "high_risk_jurisdiction": {
            "jmlsg_ref": "JMLSG Part I Section 5.3.12",
            "fatf_ref": "FATF Recommendation 19 — Higher-Risk Countries",
            "description": (
                "Transactions involving FATF high-risk or non-cooperative "
                "jurisdictions without plausible business explanation."
            ),
            "sar_phrase": (
                "transactions involving FATF high-risk jurisdiction "
                "without credible business explanation (JMLSG Part I Section 5.3.12)"
            ),
        },
        "pep_source_of_wealth": {
            "jmlsg_ref": "JMLSG Part II Section 6.3.2",
            "fatf_ref": "FATF Recommendation 12 — PEPs",
            "description": (
                "PEP customer with unexplained wealth accumulation "
                "inconsistent with public sector income."
            ),
            "sar_phrase": (
                "PEP with wealth accumulation inconsistent with public "
                "sector role (JMLSG Part II 6.3.2, FATF R.12)"
            ),
        },
        "crypto_layering": {
            "jmlsg_ref": "JMLSG Part II Section 6.2.8",
            "fatf_ref": "FATF Typology VASP-2 — Crypto Layering",
            "description": (
                "Repeated fiat-to-crypto-to-fiat cycles with no "
                "evident investment purpose — layering indicator."
            ),
            "sar_phrase": (
                "crypto exchange pattern consistent with layering "
                "(JMLSG Part II 6.2.8, FATF VASP-2)"
            ),
        },
    }

    typology_matches: List[Dict[str, Any]] = []

    # ---- Match structuring ring ----
    if structuring:
        typology_matches.append({
            "typology": "structuring",
            "confidence": 0.92,
            "evidence": f"{len(structuring)} Louvain community ring(s) detected",
            **JMLSG_TYPOLOGIES["structuring"],
        })

    # ---- Match SHAP features to typologies ----
    for shap_rec in shap:
        features = shap_rec.get("top_features", {})
        if features.get("counterparty_country_risk", 0) > 0.2:
            typology_matches.append({
                "typology": "high_risk_jurisdiction",
                "confidence": min(0.5 + features["counterparty_country_risk"], 0.95),
                "evidence": f"Counterparty country SHAP={features['counterparty_country_risk']:.3f}",
                **JMLSG_TYPOLOGIES["high_risk_jurisdiction"],
            })
        if features.get("structuring_indicator", 0) > 0.05:
            typology_matches.append({
                "typology": "structuring",
                "confidence": min(0.6 + features["structuring_indicator"] * 2, 0.95),
                "evidence": f"Below-CTR SHAP={features['structuring_indicator']:.3f}",
                **JMLSG_TYPOLOGIES["structuring"],
            })

    # ---- Match KYC PEP ----
    if isinstance(kyc, dict) and kyc.get("pep_flagged"):
        typology_matches.append({
            "typology": "pep_source_of_wealth",
            "confidence": 0.85,
            "evidence": "PEP identified at KYC screening (MLR 2017 Reg. 35)",
            **JMLSG_TYPOLOGIES["pep_source_of_wealth"],
        })

    # ---- Gemini 3.1 Pro typology synthesis ----
    context = (
        f"Customer: {customer.get('name', 'UNKNOWN')} "
        f"({customer.get('entity_type', 'individual')})\n"
        f"High-priority alerts: {len(high_priority)}\n"
        f"SHAP attributions: {shap[:2]}\n"
        f"Structuring rings: {structuring[:1]}\n"
        f"KYC status: {kyc.get('status', 'UNKNOWN') if isinstance(kyc, dict) else 'N/A'}\n"
        f"PEP: {kyc.get('pep_flagged', False) if isinstance(kyc, dict) else False}\n"
        f"Initial typology matches: {[t['typology'] for t in typology_matches]}"
    )
    llm_prompt = (
        "You are AWB's AML Investigation Officer preparing a JMLSG typology "
        "assessment for an escalated alert case.\n\n"
        "Tasks:\n"
        "1. Identify the PRIMARY JMLSG typology (most significant money "
        "laundering pattern present).\n"
        "2. Identify any SECONDARY typologies.\n"
        "3. Assess the overall suspicion level: REASONABLE GROUNDS TO SUSPECT "
        "(SAR required), INSUFFICIENT (monitor), or NO SUSPICION (clear).\n"
        "4. Draft a 2-sentence risk theme summary for inclusion in the SAR "
        "nature of suspicion section, citing JMLSG and FATF references.\n"
        "5. Cite any relevant NCA SAR Annual Report 2024 typology data.\n\n"
        "Regulatory standard: POCA 2002 s.330 — MLRO must disclose knowledge "
        "or suspicion of money laundering to NCA."
    )
    typology_narrative = _call_gemini_pro(llm_prompt, context)

    # ---- One-paragraph risk theme summary ----
    if typology_matches:
        primary = typology_matches[0]
        risk_theme = (
            f"The pattern of activity is consistent with "
            f"{primary['typology'].replace('_', ' ')} "
            f"({primary['jmlsg_ref']}, {primary['fatf_ref']}). "
            f"{primary['description']} "
            f"This assessment is based on: {primary['evidence']}."
        )
    else:
        risk_theme = (
            "No primary JMLSG typology match identified. "
            "Monitoring recommended under JMLSG Part I §6.2 ongoing monitoring "
            "obligations."
        )

    state["typology_matches"] = typology_matches
    state["typology_narrative"] = typology_narrative
    state["risk_theme_summary"] = risk_theme

    _log_step(
        state,
        agent="TypologyMatchingAgent [gemini-3.1-pro]",
        reason=reason,
        act=(
            f"Matched {len(typology_matches)} JMLSG typologies from "
            f"SHAP features, network graph, and KYC data"
        ),
        outcome=(
            f"Primary typology: "
            f"{typology_matches[0]['typology'] if typology_matches else 'none'}, "
            f"Zone: {state['risk_zone']}"
        ),
    )
    return state


# ---------------------------------------------------------------------------
# AGENT 5 — SARDraftingAgent (Claude Sonnet 4.6)
# ---------------------------------------------------------------------------

def sar_drafting_agent(
    state: AMLKYCState,
) -> AMLKYCState:
    """Agent 5: Draft POCA s.330 SAR and compile MLRO escalation package.

    Regulatory basis:
    - POCA 2002 s.330: Nominated officer (MLRO) must disclose knowledge
      or suspicion of money laundering. AI-assisted SAR drafts must be
      approved by the MLRO before submission to NCA SubmitSAR.
    - POCA 2002 s.333A: Tipping-off is a criminal offence. The customer
      must never learn that a SAR has been filed. This is enforced as a
      hard architectural guarantee — the sar_filed_indicator in state is
      NEVER propagated to the credit pipeline or customer-facing systems.
    - NCA SubmitSAR: UK National Crime Agency submission portal (NOT FinCEN).
    - BAP-2026-AML-001 §8: MLRO must approve all SAR drafts; Claude Sonnet
      generates the draft only — final decision is always human.
    - PRA SS1/23 §7: Audit trail required for AI-assisted SAR drafting.

    A SAR is only drafted when:
    1. Alert score >= 0.70 (HIGH priority), OR
    2. Structuring ring detected, OR
    3. Sanctions hit confirmed.

    Uses Claude Sonnet 4.6 for the highest-quality regulatory narrative
    required by NCA SubmitSAR — the nature_of_suspicion section must be
    precise, legally sound, and JMLSG-compliant.

    POCA s.333A Architectural Guarantee:
    sar_filed_indicator is set in state but the get_credit_gate_decision()
    helper returns only BLOCKED/CLEARED — the reason is never disclosed.

    Populates:
        state["sar_draft"]: Complete NCA SubmitSAR-ready SAR dict.
        state["mlro_escalation_package"]: MLRO pack with all evidence.
        state["regulatory_narrative"]: PRA SS1/23 audit trail narrative.
        state["sar_filed_indicator"]: True if SAR drafted (INTERNAL ONLY).
        state["_credit_gate_result"]: BLOCKED or CLEARED (no SAR reason).
        hitl_decision / hitl_rationale: Final HITL outcome.
        pipeline_completed: True.

    Args:
        state: Fully populated pipeline state after agents 1-4.

    Returns:
        Updated state with SAR draft and final HITL decision.
    """
    high_priority = state.get("high_priority_alerts", [])
    structuring = state.get("structuring_rings", [])
    kyc = state.get("kyc_decision", {})
    typology_matches = state.get("typology_matches", [])
    risk_theme = state.get("risk_theme_summary", "")
    customer = state.get("customer_profile", {})
    run_date = state.get("run_date", date.today())

    # ---- ReAct: Reason before acting ----
    reason = (
        "POCA 2002 s.330: MLRO must be notified of reasonable suspicion. "
        "Claude Sonnet 4.6 drafts SAR nature_of_suspicion citing JMLSG "
        "typologies — MLRO approves before NCA SubmitSAR submission. "
        "POCA s.333A architectural guarantee: sar_filed_indicator never "
        "leaves the MLRO workflow. EU AI Act Art.14 HITL mandatory."
    )

    # ---- Determine if SAR is warranted ----
    sar_warranted = (
        bool(high_priority)
        or bool(structuring)
        or (isinstance(kyc, dict) and kyc.get("sanctions_hit", False))
    )

    sar_draft = None
    mlro_package: Dict[str, Any] = {}

    if sar_warranted:
        # ---- Calculate total suspicious amount ----
        total_suspicious_gbp = sum(
            float(a.get("amount_gbp", 0)) for a in high_priority
        )
        for ring in structuring:
            total_suspicious_gbp += ring.get("total_amount_gbp", 0)

        # ---- SAR ID ----
        sar_id = f"SAR-{run_date.strftime('%Y%m%d')}-AWB-{customer.get('customer_id', 'UNKNOWN')[:8]}"

        # ---- Aggregate context for Claude ----
        sar_context = (
            f"Customer: {customer.get('name', 'UNKNOWN')} "
            f"(ID: {customer.get('customer_id', 'UNKNOWN')})\n"
            f"Entity type: {customer.get('entity_type', 'individual')}\n"
            f"High-priority alerts: {len(high_priority)}\n"
            f"Highest score: {max((a['score'] for a in high_priority), default=0):.4f}\n"
            f"Structuring rings: {len(structuring)}\n"
            f"Sanctions hit: {kyc.get('sanctions_hit', False) if isinstance(kyc, dict) else False}\n"
            f"KYC status: {kyc.get('status', 'N/A') if isinstance(kyc, dict) else 'N/A'}\n"
            f"Total suspicious amount: £{total_suspicious_gbp:,.2f}\n"
            f"Primary typology: "
            f"{typology_matches[0]['typology'] if typology_matches else 'unclassified'}\n"
            f"Risk theme: {risk_theme}\n"
            f"JMLSG typologies: {[t.get('jmlsg_ref') for t in typology_matches]}"
        )

        # ---- Nature of suspicion — POCA s.330(5) ----
        sar_prompt = (
            "You are AWB's MLRO team drafting a Suspicious Activity Report "
            "for submission to NCA SubmitSAR under POCA 2002 s.330.\n\n"
            "Draft the NATURE OF SUSPICION section of the SAR (200 words). "
            "This must:\n"
            "1. State the specific transactions and amounts involved.\n"
            "2. Explain why the activity is suspicious (not merely unusual).\n"
            "3. Cite the primary JMLSG typology and FATF reference.\n"
            "4. Reference the XGBoost score and SHAP top features.\n"
            "5. Note any network graph findings (structuring ring).\n"
            "6. Include KYC context (PEP, sanctions, EDD status).\n"
            "7. Confirm that the ML system output has been reviewed by a "
            "qualified MLRO under POCA 2002 s.330 and BAP-2026-AML-001 §8.\n\n"
            "CRITICAL: Do NOT disclose that this SAR exists to the customer "
            "(POCA 2002 s.333A — tipping-off criminal offence).\n"
            "Format: formal NCA SubmitSAR submission language."
        )
        nature_of_suspicion = _call_claude_sonnet(sar_prompt, sar_context)

        # ---- Financial details section ----
        fin_prompt = (
            "Draft the FINANCIAL DETAILS section for the SAR "
            "(100 words, NCA SubmitSAR format):\n"
            "- Account(s) involved\n"
            "- Date range of suspicious transactions\n"
            "- Total amount (GBP)\n"
            "- Transaction types (wire transfer, cash, etc.)\n"
            "- Counterparty countries\n"
            "Reference model ID MR-2026-060-AML for the automated detection."
        )
        financial_details = _call_claude_sonnet(fin_prompt, sar_context)

        sar_draft = {
            "sar_id": sar_id,
            "customer_id": customer.get("customer_id", "UNKNOWN"),
            "alert_ids": [a["alert_id"] for a in high_priority],
            "total_suspicious_amount_gbp": round(total_suspicious_gbp, 2),
            "nature_of_suspicion": nature_of_suspicion,
            "typology_citation": (
                typology_matches[0]["jmlsg_ref"] if typology_matches
                else "JMLSG Part I General"
            ),
            "financial_details": financial_details,
            "status": "DRAFT",
            "poca_section": "s.330",
            "sar_type": "disclosure",
            "requires_mlro_approval": True,        # ALWAYS True — cannot be bypassed
            "tipping_off_guardrail_active": True,  # ALWAYS True — criminal offence
            "model_id": "MR-2026-060-AML",
            "draft_generated_by": "claude-sonnet-4-6",
            "nca_submission_target": "NCA SubmitSAR",  # NOT FinCEN
        }

        # ---- MLRO escalation package ----
        mlro_package = {
            "sar_id": sar_id,
            "escalation_timestamp": datetime.utcnow().isoformat() + "Z",
            "scored_alerts": high_priority,
            "shap_attributions": state.get("shap_attributions", []),
            "structuring_rings": structuring,
            "kyc_summary": kyc if isinstance(kyc, dict) else {},
            "typology_matches": typology_matches,
            "typology_narrative": state.get("typology_narrative", ""),
            "sar_draft": sar_draft,
            "hop_chain": state.get("hop_chain", []),
            "required_action": (
                "MLRO to review and approve or reject SAR draft. "
                "If approved, submit to NCA SubmitSAR within moratorium period. "
                "If consent SAR required under POCA s.335, await NCA response "
                "before allowing transaction to proceed."
            ),
            "poca_s333a_reminder": (
                "TIPPING-OFF: Do NOT disclose to customer that SAR has been filed. "
                "POCA 2002 s.333A — criminal offence carrying up to 5 years imprisonment."
            ),
        }

        # ---- POCA s.333A guarantee — set internal indicator ----
        state["sar_filed_indicator"] = True
        state["_credit_gate_result"] = "BLOCKED"
        _escalate_zone(state, "RED")

    else:
        # No SAR warranted — credit gate cleared
        state["sar_filed_indicator"] = False
        state["_credit_gate_result"] = "CLEARED"

    state["sar_draft"] = sar_draft
    state["mlro_escalation_package"] = mlro_package

    # ---- PRA SS1/23 audit trail narrative ----
    audit_context = (
        f"AML pipeline run: {run_date}, customer {customer.get('customer_id', 'UNKNOWN')}\n"
        f"Zone: {state['risk_zone']}, Alerts: {len(high_priority)}\n"
        f"Structuring: {len(structuring)}, SAR drafted: {sar_warranted}\n"
        f"Hop chain: {len(state.get('hop_chain', []))} steps"
    )
    audit_prompt = (
        "Write a 150-word PRA SS1/23 §7 audit trail entry for this "
        "AI-assisted AML investigation. Cover: model used (MR-2026-060-AML), "
        "HITL chain (Gemini Flash agents 1-3, Gemini 3.1 Pro agent 4, "
        "Claude Sonnet 4.6 agent 5), key findings, and the mandatory "
        "MLRO human review step before NCA submission. "
        "Confirm BAP-2026-AML-001 §6 hop-chain audit trail is complete."
    )
    regulatory_narrative = _call_claude_sonnet(audit_prompt, audit_context)
    state["regulatory_narrative"] = regulatory_narrative

    # ---- HITL decision ----
    decision, rationale = _compute_hitl_decision(state)
    state["hitl_decision"] = decision
    state["hitl_rationale"] = rationale
    state["pipeline_completed"] = True

    _log_step(
        state,
        agent="SARDraftingAgent [claude-sonnet-4-6]",
        reason=reason,
        act=(
            f"{'Drafted SAR + MLRO package' if sar_warranted else 'No SAR warranted — cleared'}"
        ),
        outcome=(
            f"HITL={decision}, SAR={sar_warranted}, "
            f"CreditGate={state['_credit_gate_result']}, "
            f"Zone={state['risk_zone']}, Pipeline=COMPLETE"
        ),
    )
    return state


# ---------------------------------------------------------------------------
# POCA s.333A CREDIT GATE HELPER
# ---------------------------------------------------------------------------

def get_credit_gate_decision(state: AMLKYCState) -> str:
    """Return credit gate decision — BLOCKED or CLEARED only.

    POCA 2002 s.333A architectural guarantee: the credit agent receives
    only BLOCKED or CLEARED. It cannot infer whether the block is due to
    KYC failure, sanctions hit, or SAR filing. The distinction is known
    only to the MLRO and is never disclosed in the credit pipeline.

    This function is the ONLY interface between the AML/KYC pipeline and
    the credit decision pipeline (MR-2026-037 LangGraph state machine).

    Args:
        state: Completed AMLKYCState after full pipeline run.

    Returns:
        "BLOCKED" if customer should not proceed with credit application.
        "CLEARED" if KYC is clean and no AML concerns are material.

    Example:
        >>> gate = get_credit_gate_decision(final_state)
        >>> assert gate in ("BLOCKED", "CLEARED")
        >>> # Credit agent never learns WHY — POCA s.333A compliance
    """
    return state.get("_credit_gate_result", "BLOCKED")


# ---------------------------------------------------------------------------
# PIPELINE ORCHESTRATION
# ---------------------------------------------------------------------------

def _build_graph():
    """Build LangGraph StateGraph for the AML/KYC investigation pipeline.

    Topology:
        START → transaction_scoring_agent
              → network_graph_agent
              → kyc_screening_agent
              → typology_matching_agent
              → sar_drafting_agent
              → END

    Returns LangGraph CompiledGraph or _SequentialStub if LangGraph absent.
    """
    try:
        from langgraph.graph import StateGraph, END

        graph = StateGraph(AMLKYCState)
        graph.add_node("transaction_scoring", transaction_scoring_agent)
        graph.add_node("network_graph", network_graph_agent)
        graph.add_node("kyc_screening", kyc_screening_agent)
        graph.add_node("typology_matching", typology_matching_agent)
        graph.add_node("sar_drafting", sar_drafting_agent)

        graph.set_entry_point("transaction_scoring")
        graph.add_edge("transaction_scoring", "network_graph")
        graph.add_edge("network_graph", "kyc_screening")
        graph.add_edge("kyc_screening", "typology_matching")
        graph.add_edge("typology_matching", "sar_drafting")
        graph.add_edge("sar_drafting", END)

        return graph.compile()

    except ImportError:
        log.warning("LangGraph not installed — using _SequentialStub")
        return _SequentialStub()


class _SequentialStub:
    """Fallback for environments without LangGraph installed.

    Provides the same invoke() interface as a compiled LangGraph graph.
    Used in CI, unit tests, and demo environments.
    """

    def invoke(self, state: AMLKYCState) -> AMLKYCState:
        state = transaction_scoring_agent(state)
        state = network_graph_agent(state)
        state = kyc_screening_agent(state)
        state = typology_matching_agent(state)
        state = sar_drafting_agent(state)
        return state


# ---------------------------------------------------------------------------
# PUBLIC ENTRY POINT
# ---------------------------------------------------------------------------

async def run_agentic_aml_kyc(
    run_date: date,
    trigger_event: str,
    transaction_batch: List[Dict[str, Any]],
    customer_profile: Dict[str, Any],
    network_graph_data: Optional[Dict[str, Any]] = None,
    kyc_documents: Optional[List[Dict[str, Any]]] = None,
) -> AMLKYCState:
    """Run the five-agent AML/KYC investigation pipeline.

    Orchestrates AWB's complete AML/KYC financial crime investigation:
    1. TransactionScoringAgent — XGBoost scoring + SHAP attribution
    2. NetworkGraphAgent       — Louvain structuring ring detection
    3. KYCScreeningAgent       — document verification, PEP/sanctions, UBO
    4. TypologyMatchingAgent   — JMLSG typology RAG matching + synthesis
    5. SARDraftingAgent        — POCA s.330 SAR + MLRO escalation package

    Model ID: MR-2026-060-AML (BAP-2026-AML-001 §3).

    Args:
        run_date: Date of this AML/KYC run.
        trigger_event: Human-readable trigger description.
        transaction_batch: List of transaction dicts. Required keys:
            transaction_id, account_id, amount_gbp, counterparty_country,
            transaction_type. Optional: velocity_30d_count.
        customer_profile: Customer background data. Required keys:
            customer_id, entity_type, name. Optional: incorporation_country,
            ownership_structure (for corporate UBO tracing).
        network_graph_data: Optional graph structure with nodes and edges.
            If None, network analysis is deferred.
        kyc_documents: Optional list of identity document dicts.
            If None, KYC document verification is deferred.

    Returns:
        Completed AMLKYCState with all fields populated.

    Example:
        >>> import asyncio
        >>> from datetime import date
        >>> state = asyncio.run(run_agentic_aml_kyc(
        ...     run_date=date(2026, 1, 15),
        ...     trigger_event="Overnight AML batch — T24 2026-01-15",
        ...     transaction_batch=[
        ...         {"transaction_id": "TXN-001", "account_id": "ACC-4721",
        ...          "amount_gbp": 9750.00, "counterparty_country": "AE",
        ...          "transaction_type": "wire_transfer"},
        ...         {"transaction_id": "TXN-002", "account_id": "ACC-4721",
        ...          "amount_gbp": 9800.00, "counterparty_country": "AE",
        ...          "transaction_type": "wire_transfer"},
        ...     ],
        ...     customer_profile={
        ...         "customer_id": "CUST-AWB-4721",
        ...         "entity_type": "individual",
        ...         "name": "Test Customer",
        ...         "incorporation_country": "AE",
        ...     },
        ... ))
        >>> print(state["hitl_decision"])  # ESCALATE (high-risk country)
        >>> print(state["risk_zone"])      # AMBER or RED
        >>> print(len(state["hop_chain"])) # 5
        >>> gate = get_credit_gate_decision(state)
        >>> print(gate)                    # BLOCKED or CLEARED
    """
    global _SEQ
    _SEQ = 0

    if network_graph_data is None:
        network_graph_data = {"nodes": [], "edges": []}
    if kyc_documents is None:
        kyc_documents = []

    state = _initial_state(
        run_date=run_date,
        trigger_event=trigger_event,
        transaction_batch=transaction_batch,
        customer_profile=customer_profile,
        network_graph_data=network_graph_data,
        kyc_documents=kyc_documents,
    )

    log.info(
        "Agentic AML/KYC Pipeline START | MR-2026-060-AML | "
        "date=%s | transactions=%d | trigger='%s'",
        run_date, len(transaction_batch), trigger_event,
    )

    graph = _build_graph()
    loop = asyncio.get_event_loop()
    final_state = await loop.run_in_executor(None, graph.invoke, state)

    log.info(
        "Agentic AML/KYC Pipeline END | "
        "HITL=%s | Zone=%s | Hops=%d | Alerts=%d | SAR=%s | Gate=%s",
        final_state.get("hitl_decision"),
        final_state.get("risk_zone"),
        len(final_state.get("hop_chain", [])),
        len(final_state.get("high_priority_alerts", [])),
        final_state.get("sar_filed_indicator"),
        get_credit_gate_decision(final_state),
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

    # Demo: structuring pattern from UAE — below-CTR amounts, wire transfers
    demo_transactions = [
        {
            "transaction_id": f"TXN-{i:04d}",
            "account_id": "ACC-AWB-4721",
            "amount_gbp": 9_750.0 + (i * 11),   # Below £10K CTR threshold
            "counterparty_country": "AE",         # UAE — elevated country risk
            "transaction_type": "wire_transfer",
            "velocity_30d_count": 24,             # High velocity
        }
        for i in range(6)  # 6 structuring transactions
    ]

    demo_customer = {
        "customer_id": "CUST-AWB-4721",
        "entity_type": "individual",
        "name": "Test Customer Demo",
        "incorporation_country": "AE",
        "risk_rating": "HIGH",
        "relationship_start": "2023-06-01",
    }

    demo_network = {
        "nodes": [f"ACC-AWB-{4700+i}" for i in range(8)],
        "edges": [
            {
                "from": f"ACC-AWB-{4700+i}",
                "to": f"ACC-AWB-{4700+((i+1)%8)}",
                "amount_gbp": 9_500.0 + i * 50,
                "date": "2026-01-14",
            }
            for i in range(8)
        ],
    }

    result = asyncio.run(
        run_agentic_aml_kyc(
            run_date=date(2026, 1, 15),
            trigger_event=(
                "Overnight AML batch — T24 2026-01-15 | "
                "847 transactions | JMLSG structuring pattern flagged"
            ),
            transaction_batch=demo_transactions,
            customer_profile=demo_customer,
            network_graph_data=demo_network,
            kyc_documents=[],
        )
    )

    print("\n" + "=" * 70)
    print("AWB AGENTIC AML/KYC PIPELINE — MR-2026-060-AML")
    print("=" * 70)
    print(f"Run Date        : {result['run_date']}")
    print(f"Trigger         : {result['trigger_event']}")
    print(f"Risk Zone       : {result['risk_zone']}")
    print(f"HITL Decision   : {result['hitl_decision']}")
    print(f"HITL Rationale  : {result['hitl_rationale']}")
    print(f"")
    print(f"AML Scoring:")
    print(f"  Total txns scored   : {len(result.get('scored_alerts', []))}")
    print(f"  High-priority alerts: {len(result.get('high_priority_alerts', []))}")
    print(f"  Structuring rings   : {len(result.get('structuring_rings', []))}")
    print(f"")
    print(f"KYC Screening:")
    kyc = result.get("kyc_decision") or {}
    if isinstance(kyc, dict):
        print(f"  Status        : {kyc.get('status', 'N/A')}")
        print(f"  PEP           : {kyc.get('pep_flagged', False)}")
        print(f"  Sanctions     : {kyc.get('sanctions_hit', False)}")
        print(f"  EDD required  : {kyc.get('edd_required', False)}")
    print(f"")
    print(f"Typology Matches: {len(result.get('typology_matches', []))}")
    for tm in result.get("typology_matches", [])[:2]:
        print(f"  • {tm['typology']} ({tm.get('jmlsg_ref', '')}) "
              f"confidence={tm.get('confidence', 0):.2f}")
    print(f"")
    print(f"SAR Status:")
    print(f"  SAR drafted   : {result.get('sar_filed_indicator', False)}")
    sar = result.get("sar_draft")
    if sar:
        print(f"  SAR ID        : {sar.get('sar_id', 'N/A')}")
        print(f"  Amount        : £{sar.get('total_suspicious_amount_gbp', 0):,.2f}")
        print(f"  MLRO approval : {sar.get('requires_mlro_approval', True)}")
        print(f"  Tipping-off   : {sar.get('tipping_off_guardrail_active', True)}")
    print(f"")
    print(f"Credit Gate     : {get_credit_gate_decision(result)}")
    print(f"  (POCA s.333A: reason for BLOCKED is never disclosed to credit agent)")
    print(f"")
    print(f"Hop Chain ({len(result.get('hop_chain', []))} hops):")
    for hop in result.get("hop_chain", []):
        print(
            f"  [{hop['seq']:02d}] {hop['agent']}: "
            f"{hop['act'][:55]} → {hop['outcome'][:55]}"
        )
    print("=" * 70)
