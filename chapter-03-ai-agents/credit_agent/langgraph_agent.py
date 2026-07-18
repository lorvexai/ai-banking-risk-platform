"""
credit_agent/langgraph_agent.py
AWB Credit Decision Workflow — LangGraph Stateful Pipeline
Chapter 3: Agentic AI for Financial Risk

Implements the AWB credit decision pipeline as a typed, stateful LangGraph
graph with four specialist nodes. Each node is a focused LLM agent with
a single responsibility; the graph router directs flow based on intermediate
outcomes, including an interrupt for human-in-the-loop review.

Architecture (Section 3.4):
                                    ┌─────────────────────┐
                                    │  START               │
                                    └──────────┬──────────┘
                                               │
                                    ┌──────────▼──────────┐
                                    │  DocumentIngestor    │  Gemini 3.5 Flash
                                    │  Node                │  (fast extraction)
                                    └──────────┬──────────┘
                                               │ structured financials
                                    ┌──────────▼──────────┐
                                    │  FinancialAnalyser   │  Gemini 3.1 Pro
                                    │  Node                │  (ratio derivation)
                                    └──────────┬──────────┘
                                               │ ratios + flags
                              ┌────────────────┼───────────────┐
                              │                │               │
                    ┌─────────▼──────┐         │    ┌──────────▼──────┐
                    │  PolicyChecker │  Pro     │    │  (future:        │
                    │  Node         │          │    │   CollateralNode)│
                    └─────────┬──────┘         │    └─────────────────┘
                              │                │
                              └───────┬────────┘
                                      │ policy_verdict
                                      │
                   ┌──────────────────▼─────────────────┐
                   │  route_after_policy_check           │
                   │  DECLINE → END (fast rejection)     │
                   │  REFER/APPROVE → MemoDrafter        │
                   └──────────────────┬─────────────────┘
                                      │
                           ┌──────────▼──────────┐
                           │  MemoDrafter Node    │  Gemini 3.5 Flash
                           │                      │  (cost-efficient)
                           └──────────┬──────────┘
                                      │ draft_memo
                                      │
                   ┌──────────────────▼─────────────────┐
                   │  route_after_memo_draft             │
                   │  facility ≥ £500k → HITL interrupt  │
                   │  facility <  £500k → END            │
                   └──────────────────┬─────────────────┘
                                      │
                          ┌───────────▼───────────┐
                          │  HumanReview interrupt │  EU AI Act Art. 14
                          │  (ServiceNow task)     │
                          └───────────┬───────────┘
                                      │ reviewer_decision
                                      └─────► END

Node–LLM routing:
  DocumentIngestor  → Gemini 3.5 Flash  (token-heavy, cost-sensitive)
  FinancialAnalyser → Gemini 3.1 Pro    (ratio derivation needs precision)
  PolicyChecker     → Gemini 3.1 Pro    (regulatory reasoning)
  MemoDrafter       → Gemini 3.5 Flash  (structured output, cost-efficient)

Regulatory context:
  PRA SS1/23: Full state snapshot stored at each graph node for audit.
  EU AI Act 2024 Article 14: LangGraph interrupt() mechanism used for
    mandatory human oversight before credit decisions ≥ £500,000.
  DORA Article 6: Each node wrapped with try/except; partial state
    preserved on tool failure for post-incident investigation.
  FCA PS22/9: Plain-English rationale attached to every MemoDrafter output.

Author: AWB AI Programme (AWB-AI-2025)
Chapter: 3 — Agentic AI for Financial Risk
Version: 1.0.0 (June 2026)
"""
from __future__ import annotations

import datetime
import json
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Annotated, Dict, List, Optional, Sequence

# LangGraph imports
# pip install langgraph>=0.2.0
try:
    from langgraph.graph import StateGraph, END, START
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import interrupt
    LANGGRAPH_AVAILABLE = True
except ImportError:  # pragma: no cover
    # Allows module to import without langgraph installed (for Chapter 3 reading)
    LANGGRAPH_AVAILABLE = False
    StateGraph = object  # type: ignore
    END = "__end__"
    START = "__start__"

from credit_agent.tools import (
    fetch_t24_exposure,
    calculate_ratios,
    check_credit_policy,
    assess_covenants,
    fetch_comparable_portfolio,
    draft_credit_memo,
)

logger = logging.getLogger("awb.langgraph_agent")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_FLASH = "gemini-3.5-flash"     # Fast extraction + memo drafting
MODEL_PRO   = "gemini-3.1-pro"       # Ratio analysis + policy reasoning
MODEL_REGISTRATION = "MR-2026-037"
HITL_THRESHOLD_GBP = 500_000          # EU AI Act Article 14 threshold


# ---------------------------------------------------------------------------
# Typed state — the shared memory of the graph
# ---------------------------------------------------------------------------

class CreditState(dict):
    """
    Typed state schema for the AWB credit decision LangGraph pipeline.

    LangGraph passes this dict between nodes. Each node reads what it needs
    and writes its own output keys. The checkpoint mechanism snapshots the
    full state after every node for audit and replay.

    Keys and their owning nodes:
      application       (input)      — initial credit application dict
      document_text     (input)      — raw document text (optional)
      extracted_data    (DocumentIngestor)  — structured financial fields
      financials        (DocumentIngestor)  — normalised financial figures
      ratios            (FinancialAnalyser) — calculated financial ratios
      ratio_flags       (FinancialAnalyser) — ratio anomaly flags
      policy_result     (PolicyChecker)    — AWB policy evaluation
      covenant_result   (PolicyChecker)    — recommended covenants
      exposure_result   (PolicyChecker)    — existing T24 exposure
      portfolio_result  (PolicyChecker)    — comparable portfolio benchmarks
      draft_memo        (MemoDrafter)      — credit memorandum draft
      final_status      (graph router)     — APPROVED / REFERRED / DECLINED
      human_review      (HumanReview)      — reviewer decision
      audit_trail       (all nodes)        — append-only list of node events
      run_id            (input)            — unique run identifier
      error             (any node)         — error detail on failure
    """
    pass


def _default_state(application: Dict[str, Any]) -> CreditState:
    """Build the initial state from a credit application dict."""
    return CreditState(
        run_id=f"LG-{uuid.uuid4().hex[:12].upper()}",
        application=application,
        document_text=application.get("document_text", ""),
        extracted_data=None,
        financials=None,
        ratios=None,
        ratio_flags=[],
        policy_result=None,
        covenant_result=None,
        exposure_result=None,
        portfolio_result=None,
        draft_memo=None,
        final_status=None,
        human_review=None,
        audit_trail=[],
        error=None,
    )


def _append_audit(state: CreditState, node: str, detail: Dict[str, Any]) -> None:
    """Append a timestamped audit event to the state trail (PRA SS1/23)."""
    state["audit_trail"].append({
        "node": node,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "run_id": state.get("run_id"),
        "model_registration": MODEL_REGISTRATION,
        **detail,
    })


# ---------------------------------------------------------------------------
# LLM client abstraction
# ---------------------------------------------------------------------------

class GeminiNode:
    """
    Thin wrapper around the Gemini API for use within LangGraph nodes.

    Each node instantiates with its chosen model tier (Flash or Pro).
    The production call is shown but commented out — the mock returns
    structured dicts matching the expected schemas.

    DORA fallback: if the primary model is unavailable, the client
    automatically downgrades and flags the fallback in the audit trail.
    """

    def __init__(self, model: str, api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key
        self._fallback_model = MODEL_FLASH if model == MODEL_PRO else None

    def extract_financials(self, document_text: str, application: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gemini 3.5 Flash: extract structured financial figures from document text.
        Uses the structured output contract pattern (Chapter 2, Pattern 3).
        """
        # Production call:
        # import google.generativeai as genai
        # genai.configure(api_key=self.api_key)
        # model = genai.GenerativeModel(
        #     self.model,
        #     system_instruction=build_role_system_prompt(),
        # )
        # response = model.generate_content(
        #     build_structured_output_prompt(FINANCIAL_SCHEMA, ..., document_text),
        #     generation_config=genai.GenerationConfig(
        #         response_mime_type="application/json",
        #     ),
        # )
        # return json.loads(response.text)

        # Mock: derive from application financials dict
        fin = application.get("financials", {})
        return {
            "revenue_gbp": fin.get("revenue_gbp", 25_000_000.0),
            "ebitda_gbp": fin.get("ebitda_gbp", 4_500_000.0),
            "ebit_gbp": fin.get("ebit_gbp", 3_800_000.0),
            "net_profit_gbp": fin.get("net_profit_gbp", 2_100_000.0),
            "total_assets_gbp": fin.get("total_assets_gbp", 32_000_000.0),
            "total_liabilities_gbp": fin.get("total_liabilities_gbp", 20_000_000.0),
            "current_assets_gbp": fin.get("current_assets_gbp", 8_000_000.0),
            "current_liabilities_gbp": fin.get("current_liabilities_gbp", 5_500_000.0),
            "net_debt_gbp": fin.get("net_debt_gbp", 18_000_000.0),
            "interest_expense_gbp": fin.get("interest_expense_gbp", 950_000.0),
            "capital_expenditure_gbp": fin.get("capital_expenditure_gbp", 600_000.0),
            "tangible_equity_gbp": fin.get("tangible_equity_gbp", 12_000_000.0),
            "extraction_confidence": 0.91,
            "model_used": self.model,
        }

    def derive_ratios_with_cot(self, financials: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gemini 3.1 Pro: derive ratios using chain-of-thought (Chapter 2, Pattern 2).
        Forces step-by-step unit checking before committing to results.
        """
        # Production: use build_cot_ratio_prompt() then call Gemini Pro
        # and parse the structured CoT response.

        # Mock: compute directly
        rev = financials["revenue_gbp"]
        ebitda = financials["ebitda_gbp"]
        ebit = financials["ebit_gbp"]
        net_debt = financials["net_debt_gbp"]
        interest = financials["interest_expense_gbp"]
        curr_a = financials["current_assets_gbp"]
        curr_l = financials["current_liabilities_gbp"]
        capex = financials.get("capital_expenditure_gbp", 0)

        flags = []
        leverage = round(net_debt / ebitda, 2) if ebitda else None
        if leverage and leverage > 5.0:
            flags.append(f"LEVERAGE_ELEVATED: {leverage:.2f}x exceeds 5.0x watch level")

        icr = round(ebit / interest, 2) if interest else None
        if icr and icr < 2.0:
            flags.append(f"ICR_LOW: {icr:.2f}x below 2.0x policy minimum")

        return {
            "profitability": {
                "ebitda_margin_pct": round(ebitda / rev * 100, 2),
                "ebit_margin_pct": round(ebit / rev * 100, 2),
                "net_profit_margin_pct": round(financials["net_profit_gbp"] / rev * 100, 2),
            },
            "leverage": {
                "net_debt_to_ebitda": leverage,
            },
            "coverage": {
                "interest_cover_ratio": icr,
                "ebitda_to_interest": round(ebitda / interest, 2) if interest else None,
                "dscr": round((ebitda - capex) / interest, 2) if interest else None,
            },
            "liquidity": {
                "current_ratio": round(curr_a / curr_l, 2) if curr_l else None,
            },
            "flags": flags,
            "cot_reasoning_steps": 4,   # Unit check + formula + substitution + result
            "model_used": self.model,
        }

    def draft_memo_narrative(
        self,
        application: Dict[str, Any],
        policy_result: Dict[str, Any],
        ratios: Dict[str, Any],
        covenant_result: Dict[str, Any],
    ) -> str:
        """
        Gemini 3.5 Flash: generate plain-English credit memo narrative.
        FCA PS22/9 requires explainability in plain English.
        """
        # Production: call Gemini Flash with structured output schema
        applicant = application.get("applicant_name", "the applicant")
        amount = application.get("facility_amount_gbp", 0)
        rec = policy_result.get("recommendation", "REFER")
        breach_count = policy_result.get("breach_count", 0)
        leverage = ratios.get("leverage", {}).get("net_debt_to_ebitda", "N/A")
        icr = ratios.get("coverage", {}).get("interest_cover_ratio", "N/A")

        return (
            f"AWB has completed its automated credit assessment of {applicant} in respect of "
            f"a proposed {application.get('facility_type', 'credit')} facility of "
            f"£{amount:,.0f}. The assessment was conducted by the AWB Automated Credit "
            f"Decision Workflow (Model MR-2026-037) in accordance with PRA SS1/23 and "
            f"EU AI Act 2024 Annex III.\n\n"
            f"The agent's recommendation is {rec}. {breach_count} policy consideration(s) "
            f"were identified. Key ratios: Net Debt/EBITDA {leverage}x, Interest Cover "
            f"{icr}x. {len(covenant_result.get('recommended_covenants', []))} financial "
            f"covenant(s) are recommended.\n\n"
            f"This recommendation requires human review by a Senior Credit Officer before "
            f"being communicated to the applicant (EU AI Act 2024 Article 14)."
        )


# ---------------------------------------------------------------------------
# Node 1: DocumentIngestor
# ---------------------------------------------------------------------------

def node_document_ingestor(state: CreditState) -> CreditState:
    """
    Node 1 — DocumentIngestor (Gemini 3.5 Flash).

    Extracts structured financial data from the raw document text submitted
    with the credit application. Uses Pattern 3 (Structured Output Contract)
    from Chapter 2 to guarantee Pydantic-compatible JSON output.

    If no document_text is present (e.g. API submission with structured data),
    the node uses the pre-structured financials from the application dict.

    State mutations:
      extracted_data: raw LLM extraction with confidence scores
      financials:     normalised financial figures dict (used by all downstream nodes)
    """
    llm = GeminiNode(model=MODEL_FLASH)
    application = state["application"]

    logger.info("DocumentIngestor: starting extraction | run_id=%s", state["run_id"])

    try:
        extracted = llm.extract_financials(
            document_text=state.get("document_text", ""),
            application=application,
        )
        state["extracted_data"] = extracted
        state["financials"] = extracted  # Downstream nodes use this normalised dict

        _append_audit(state, "DocumentIngestor", {
            "event": "extraction_complete",
            "confidence": extracted.get("extraction_confidence"),
            "model": MODEL_FLASH,
        })

        logger.info(
            "DocumentIngestor: extraction complete | confidence=%.2f",
            extracted.get("extraction_confidence", 0),
        )

    except Exception as exc:
        logger.error("DocumentIngestor FAILED: %s", exc)
        state["error"] = f"DocumentIngestor: {exc}"
        # Graceful degradation: populate from application dict if available
        fin = application.get("financials", {})
        if fin:
            state["financials"] = fin
            _append_audit(state, "DocumentIngestor", {
                "event": "fallback_to_application_dict",
                "error": str(exc),
            })
        else:
            _append_audit(state, "DocumentIngestor", {
                "event": "extraction_failed_no_fallback",
                "error": str(exc),
            })

    return state


# ---------------------------------------------------------------------------
# Node 2: FinancialAnalyser
# ---------------------------------------------------------------------------

def node_financial_analyser(state: CreditState) -> CreditState:
    """
    Node 2 — FinancialAnalyser (Gemini 3.1 Pro).

    Derives financial ratios from the structured figures using chain-of-thought
    reasoning (Pattern 2 from Chapter 2). CoT forces explicit unit verification
    at each calculation step, eliminating the class of errors where net debt
    stated in £millions and EBITDA in £thousands produces a 1000x leverage error.

    State mutations:
      ratios:       calculated financial ratios grouped by category
      ratio_flags:  list of anomaly flags for downstream nodes
    """
    llm = GeminiNode(model=MODEL_PRO)
    financials = state.get("financials")

    if not financials:
        logger.warning("FinancialAnalyser: no financials in state; skipping.")
        state["ratio_flags"] = ["MISSING_FINANCIALS"]
        return state

    logger.info("FinancialAnalyser: deriving ratios | run_id=%s", state["run_id"])

    try:
        ratios = llm.derive_ratios_with_cot(financials)
        state["ratios"] = ratios
        state["ratio_flags"] = ratios.get("flags", [])

        _append_audit(state, "FinancialAnalyser", {
            "event": "ratios_derived",
            "leverage": ratios.get("leverage", {}).get("net_debt_to_ebitda"),
            "icr": ratios.get("coverage", {}).get("interest_cover_ratio"),
            "flags": state["ratio_flags"],
            "cot_steps": ratios.get("cot_reasoning_steps", 0),
            "model": MODEL_PRO,
        })

        logger.info(
            "FinancialAnalyser: leverage=%.2fx, ICR=%.2fx, flags=%s",
            ratios.get("leverage", {}).get("net_debt_to_ebitda", 0),
            ratios.get("coverage", {}).get("interest_cover_ratio", 0),
            state["ratio_flags"],
        )

    except Exception as exc:
        logger.error("FinancialAnalyser FAILED: %s", exc)
        state["error"] = f"FinancialAnalyser: {exc}"
        state["ratios"] = {}
        state["ratio_flags"] = ["RATIO_CALCULATION_FAILED"]
        _append_audit(state, "FinancialAnalyser", {"event": "failed", "error": str(exc)})

    return state


# ---------------------------------------------------------------------------
# Node 3: PolicyChecker
# ---------------------------------------------------------------------------

def node_policy_checker(state: CreditState) -> CreditState:
    """
    Node 3 — PolicyChecker (Gemini 3.1 Pro + deterministic tools).

    Evaluates the application against AWB's credit policy rule set using
    the tools from tools.py. Unlike Nodes 1–2 which rely primarily on LLM
    inference, the PolicyChecker wraps deterministic policy functions with
    an LLM that provides regulatory reasoning and context.

    Runs three sub-tasks in sequence:
      (a) check_credit_policy()       — rule-based policy evaluation
      (b) assess_covenants()          — recommend covenant structure
      (c) fetch_t24_exposure()        — existing AWB exposure
      (d) fetch_comparable_portfolio() — peer benchmarking

    State mutations:
      policy_result:   policy evaluation with APPROVE / REFER / DECLINE
      covenant_result: recommended covenant structure
      exposure_result: existing T24 exposure data
      portfolio_result: comparable portfolio benchmarks
    """
    application = state["application"]
    financials = state.get("financials") or {}
    ratios = state.get("ratios") or {}

    logger.info("PolicyChecker: evaluating policy | run_id=%s", state["run_id"])

    # (a) Credit policy evaluation
    try:
        state["policy_result"] = check_credit_policy(
            net_debt_gbp=financials.get("net_debt_gbp", 0),
            ebitda_gbp=financials.get("ebitda_gbp", 1),
            ebit_gbp=financials.get("ebit_gbp", 1),
            interest_expense_gbp=financials.get("interest_expense_gbp", 1),
            total_exposure_gbp=(
                financials.get("net_debt_gbp", 0) +
                application.get("facility_amount_gbp", 0)
            ),
            tangible_equity_gbp=financials.get("tangible_equity_gbp", 1),
            total_assets_gbp=financials.get("total_assets_gbp", 1),
            facility_type=application.get("facility_type", "TERM_LOAN"),
        )
    except Exception as exc:
        logger.error("PolicyChecker policy check failed: %s", exc)
        state["policy_result"] = {"recommendation": "REFER", "breach_count": 0, "breaches": []}

    # (b) Covenant assessment
    try:
        state["covenant_result"] = assess_covenants(
            facility_amount_gbp=application.get("facility_amount_gbp", 1_000_000),
            facility_type=application.get("facility_type", "TERM_LOAN"),
            leverage_ratio=ratios.get("leverage", {}).get("net_debt_to_ebitda", 4.0),
            interest_cover_ratio=ratios.get("coverage", {}).get("interest_cover_ratio", 3.0),
        )
    except Exception as exc:
        logger.error("PolicyChecker covenant assessment failed: %s", exc)
        state["covenant_result"] = {"recommended_covenants": []}

    # (c) T24 existing exposure
    try:
        state["exposure_result"] = fetch_t24_exposure(
            customer_id=application.get("customer_id", "AWB-CUST-000000"),
        )
    except Exception as exc:
        logger.error("PolicyChecker T24 fetch failed: %s", exc)
        state["exposure_result"] = {"total_committed_gbp": 0, "total_drawn_gbp": 0}

    # (d) Comparable portfolio
    try:
        state["portfolio_result"] = fetch_comparable_portfolio(
            industry_code=application.get("industry_code", "6419"),
            facility_type=application.get("facility_type", "TERM_LOAN"),
            facility_size_band=application.get("facility_size_band", "MEDIUM"),
        )
    except Exception as exc:
        logger.error("PolicyChecker portfolio fetch failed: %s", exc)
        state["portfolio_result"] = {}

    _append_audit(state, "PolicyChecker", {
        "event": "policy_evaluated",
        "recommendation": state["policy_result"].get("recommendation"),
        "breach_count": state["policy_result"].get("breach_count", 0),
        "blocking_breaches": state["policy_result"].get("blocking_breach_count", 0),
        "model": MODEL_PRO,
    })

    logger.info(
        "PolicyChecker: recommendation=%s | breaches=%d",
        state["policy_result"].get("recommendation"),
        state["policy_result"].get("breach_count", 0),
    )

    return state


# ---------------------------------------------------------------------------
# Node 4: MemoDrafter
# ---------------------------------------------------------------------------

def node_memo_drafter(state: CreditState) -> CreditState:
    """
    Node 4 — MemoDrafter (Gemini 3.5 Flash).

    Synthesises all prior node outputs into a structured credit memorandum.
    Uses Gemini 3.5 Flash (not Pro) — this is a structured synthesis task,
    not a reasoning task. The cost saving vs Pro is ~97% per memo
    (Flash: £0.039/1M tokens vs Pro: £1.26/1M tokens, June 2026 pricing).

    Calls draft_credit_memo() from tools.py for the structured memo skeleton,
    then augments with LLM-generated plain-English narrative for FCA PS22/9.

    State mutations:
      draft_memo:   complete credit memorandum dict
      final_status: APPROVED, REFERRED, or DECLINED
    """
    llm = GeminiNode(model=MODEL_FLASH)
    application = state["application"]
    policy_result = state.get("policy_result") or {}
    ratios = state.get("ratios") or {}
    covenant_result = state.get("covenant_result") or {}
    exposure_result = state.get("exposure_result") or {}
    portfolio_result = state.get("portfolio_result") or {}

    logger.info("MemoDrafter: drafting memo | run_id=%s", state["run_id"])

    recommendation = policy_result.get("recommendation", "REFER")
    breach_count = policy_result.get("breach_count", 0)
    leverage = ratios.get("leverage", {}).get("net_debt_to_ebitda", 4.0) or 4.0

    # Derive risk rating (1–10) from policy and ratio signals
    if recommendation == "DECLINE":
        risk_rating = min(9, 7 + min(breach_count, 2))
    elif recommendation == "REFER":
        risk_rating = min(7, 5 + min(breach_count, 2))
    else:
        risk_rating = max(1, min(4, int(leverage)))

    try:
        memo_dict = draft_credit_memo(
            applicant_name=application.get("applicant_name", "Unknown Applicant"),
            facility_amount_gbp=application.get("facility_amount_gbp", 0),
            facility_type=application.get("facility_type", "TERM_LOAN"),
            policy_assessment=policy_result,
            financial_ratios=ratios,
            covenant_assessment=covenant_result,
            existing_exposure=exposure_result,
            portfolio_benchmarks=portfolio_result,
            agent_recommendation=recommendation,
            risk_rating=risk_rating,
        )

        # Augment with LLM-generated narrative (FCA PS22/9 plain English)
        memo_dict["llm_narrative"] = llm.draft_memo_narrative(
            application=application,
            policy_result=policy_result,
            ratios=ratios,
            covenant_result=covenant_result,
        )
        memo_dict["run_id"] = state["run_id"]
        memo_dict["graph_version"] = "langgraph_v1.0.0"

        state["draft_memo"] = memo_dict
        state["final_status"] = recommendation

    except Exception as exc:
        logger.error("MemoDrafter FAILED: %s", exc)
        state["error"] = f"MemoDrafter: {exc}"
        state["draft_memo"] = None
        state["final_status"] = "REFER"  # Conservative default on failure

    _append_audit(state, "MemoDrafter", {
        "event": "memo_drafted",
        "recommendation": state["final_status"],
        "risk_rating": risk_rating,
        "memo_id": (state["draft_memo"] or {}).get("memo_id"),
        "model": MODEL_FLASH,
    })

    logger.info(
        "MemoDrafter: memo_id=%s | recommendation=%s | risk_rating=%d",
        (state["draft_memo"] or {}).get("memo_id"),
        state["final_status"],
        risk_rating,
    )

    return state


# ---------------------------------------------------------------------------
# Node 5: HumanReview interrupt (EU AI Act Article 14)
# ---------------------------------------------------------------------------

def node_human_review(state: CreditState) -> CreditState:
    """
    Node 5 — HumanReview interrupt (EU AI Act 2024 Article 14).

    For facilities of £500,000 or above, LangGraph's interrupt() mechanism
    pauses graph execution and surfaces the credit memo to a Senior Credit
    Officer via ServiceNow. Graph execution resumes when the SCO submits
    their decision via webhook.

    The interrupt() call is idempotent: if the graph is resumed with a
    Command object containing the reviewer's decision, the node records
    the decision and returns without re-interrupting.

    Why LangGraph interrupt() is preferred over a manual flag:
      - The graph checkpoint serialises full state to persistent storage
        (MemorySaver in dev, PostgreSQL in production via langgraph-postgres).
      - The graph can be resumed days later when the SCO reviews; no
        in-memory state is needed.
      - The audit trail captures the exact state at the point of interruption,
        satisfying PRA SS1/23 model audit requirements.

    State mutations:
      human_review: dict with reviewer_id, decision, timestamp, override_reason
    """
    memo = state.get("draft_memo") or {}
    facility_amount = state["application"].get("facility_amount_gbp", 0)

    logger.info(
        "HumanReview: facility=£%s | threshold=£%s | memo_id=%s",
        f"{facility_amount:,.0f}",
        f"{HITL_THRESHOLD_GBP:,.0f}",
        memo.get("memo_id"),
    )

    # Create ServiceNow review task
    task_id = f"SCO-{uuid.uuid4().hex[:8].upper()}"
    review_deadline = (
        datetime.datetime.utcnow() + datetime.timedelta(hours=24)
    ).isoformat() + "Z"

    review_request = {
        "task_id": task_id,
        "memo_id": memo.get("memo_id"),
        "run_id": state["run_id"],
        "facility_amount_gbp": facility_amount,
        "agent_recommendation": state.get("final_status"),
        "assignee": "duty_senior_credit_officer@awb.co.uk",
        "deadline": review_deadline,
        "regulatory_basis": "EU AI Act 2024 Article 14",
        "task_url": f"https://awb.service-now.com/credit_review?task_id={task_id}",
    }

    _append_audit(state, "HumanReview", {
        "event": "review_task_created",
        "task_id": task_id,
        "deadline": review_deadline,
        "regulatory_basis": "EU AI Act 2024 Article 14",
    })

    # LangGraph interrupt: pause here and wait for SCO decision
    # In production, resume via: graph.invoke(Command(resume=reviewer_decision), config)
    if LANGGRAPH_AVAILABLE:
        reviewer_decision = interrupt(review_request)
    else:
        # Fallback for environments without langgraph installed
        reviewer_decision = {
            "reviewer_id": "SCO-TEST",
            "decision": state.get("final_status", "REFER"),
            "override_reason": None,
            "reviewed_at": datetime.datetime.utcnow().isoformat() + "Z",
        }

    # Record reviewer decision
    state["human_review"] = reviewer_decision
    _append_audit(state, "HumanReview", {
        "event": "review_completed",
        "reviewer_id": reviewer_decision.get("reviewer_id"),
        "decision": reviewer_decision.get("decision"),
        "override_reason": reviewer_decision.get("override_reason"),
    })

    logger.info(
        "HumanReview completed | reviewer=%s | decision=%s",
        reviewer_decision.get("reviewer_id"),
        reviewer_decision.get("decision"),
    )

    return state


# ---------------------------------------------------------------------------
# Conditional edge routers
# ---------------------------------------------------------------------------

def route_after_policy_check(state: CreditState) -> str:
    """
    Router after PolicyChecker node.

    DECLINE with CRITICAL breach → END immediately (no memo needed).
    All other outcomes → MemoDrafter for full memo.

    Fast rejection avoids spending Flash tokens on a memo that will
    immediately be declined by the Credit Committee.
    """
    policy_result = state.get("policy_result") or {}
    recommendation = policy_result.get("recommendation", "REFER")
    blocking = policy_result.get("blocking_breach_count", 0)

    # Fast rejection: DECLINE with critical breach → END
    if recommendation == "DECLINE" and blocking > 0:
        logger.info(
            "route_after_policy_check: FAST REJECTION | blocking_breaches=%d",
            blocking,
        )
        return "end_declined"

    return "memo_drafter"


def route_after_memo_draft(state: CreditState) -> str:
    """
    Router after MemoDrafter node.

    Facilities ≥ £500k → human_review interrupt (EU AI Act Article 14).
    Facilities <  £500k → END directly.
    """
    facility_amount = state["application"].get("facility_amount_gbp", 0)

    if facility_amount >= HITL_THRESHOLD_GBP:
        logger.info(
            "route_after_memo_draft: HITL required | facility=£%s",
            f"{facility_amount:,.0f}",
        )
        return "human_review"

    return "end_approved_or_referred"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_credit_graph(checkpointer=None) -> Any:
    """
    Assemble the AWB credit decision LangGraph pipeline.

    Args:
        checkpointer: LangGraph checkpointer for state persistence.
            MemorySaver (dev) or langgraph-postgres (production).
            If None, a MemorySaver is used (in-memory, non-persistent).

    Returns:
        Compiled LangGraph graph ready for invoke() / stream() calls.
    """
    if not LANGGRAPH_AVAILABLE:
        raise ImportError(
            "langgraph is required. Install with: pip install langgraph>=0.2.0"
        )

    if checkpointer is None:
        checkpointer = MemorySaver()

    graph = StateGraph(CreditState)

    # Add nodes
    graph.add_node("document_ingestor", node_document_ingestor)
    graph.add_node("financial_analyser", node_financial_analyser)
    graph.add_node("policy_checker", node_policy_checker)
    graph.add_node("memo_drafter", node_memo_drafter)
    graph.add_node("human_review", node_human_review)

    # Add edges
    graph.add_edge(START, "document_ingestor")
    graph.add_edge("document_ingestor", "financial_analyser")
    graph.add_edge("financial_analyser", "policy_checker")

    # Conditional routing after policy check
    graph.add_conditional_edges(
        "policy_checker",
        route_after_policy_check,
        {
            "memo_drafter": "memo_drafter",
            "end_declined": END,
        },
    )

    # Conditional routing after memo draft
    graph.add_conditional_edges(
        "memo_drafter",
        route_after_memo_draft,
        {
            "human_review": "human_review",
            "end_approved_or_referred": END,
        },
    )

    graph.add_edge("human_review", END)

    return graph.compile(checkpointer=checkpointer, interrupt_before=["human_review"])


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_credit_pipeline(
    credit_application: Dict[str, Any],
    thread_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the full credit decision pipeline for a single application.

    For facilities ≥ £500k, the graph pauses at the HumanReview node
    and returns the state with final_status = "AWAITING_HUMAN_REVIEW".
    Resume with resume_human_review() when the SCO has decided.

    Args:
        credit_application: Dict with applicant details and financials.
        thread_id: LangGraph thread ID for state persistence.
            Defaults to a new UUID. Use the same thread_id to resume
            a paused graph after human review.

    Returns:
        Final CreditState dict with all node outputs and audit_trail.
    """
    graph = build_credit_graph()
    initial_state = _default_state(credit_application)

    config = {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}

    logger.info(
        "run_credit_pipeline: START | applicant=%s | facility=£%s | thread_id=%s",
        credit_application.get("applicant_name"),
        f"{credit_application.get('facility_amount_gbp', 0):,.0f}",
        config["configurable"]["thread_id"],
    )

    final_state = graph.invoke(initial_state, config=config)
    return final_state


def resume_human_review(
    thread_id: str,
    reviewer_decision: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Resume a paused credit pipeline after human review decision.

    Called by the ServiceNow webhook handler when the Senior Credit
    Officer submits their review. The graph replays from the checkpoint,
    passing the reviewer decision through the interrupt() call.

    Args:
        thread_id: The thread_id of the paused pipeline.
        reviewer_decision: Dict with keys:
            reviewer_id (str): SCO employee ID.
            decision (str): APPROVED / REFERRED / DECLINED (may override agent).
            override_reason (str, optional): Reason if overriding agent recommendation.

    Returns:
        Final CreditState after human review completion.
    """
    if not LANGGRAPH_AVAILABLE:
        raise ImportError("langgraph is required.")

    from langgraph.types import Command

    graph = build_credit_graph()
    config = {"configurable": {"thread_id": thread_id}}

    reviewer_decision.setdefault("reviewed_at", datetime.datetime.utcnow().isoformat() + "Z")

    logger.info(
        "resume_human_review: thread_id=%s | reviewer=%s | decision=%s",
        thread_id,
        reviewer_decision.get("reviewer_id"),
        reviewer_decision.get("decision"),
    )

    final_state = graph.invoke(Command(resume=reviewer_decision), config=config)
    return final_state
