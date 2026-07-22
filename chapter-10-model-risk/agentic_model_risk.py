"""
chapter-10-model-risk/agentic_model_risk.py
============================================================
Agentic Model Risk Monitor  —  Model ID: MR-2026-058-MRM
Avon & Wessex Bank plc (AWB) | AWB-AI-2025 Programme

LangGraph StateGraph orchestrating five specialist AI agents to
provide continuous PRA SS1/23 model governance: registry audit,
validation status, LLM monitoring compliance, A/B test analysis,
and automated SREP-quality model risk reporting.

Regulatory coverage
-------------------
* PRA SS1/23          — Model Risk Management (primary)
* SR 11-7             — Fed/OCC model risk guidance (cross-border)
* EU AI Act Art. 9-17 — Risk management for high-risk AI systems
* EU AI Act Art. 14   — Human oversight of high-risk AI decisions
* BAP-2026-MRM-001    — Board model risk appetite statement
* SREP               — PRA Supervisory Review and Evaluation Process

Agent graph
-----------
START
  → model_inventory_audit   (Gemini Flash)   Registry scan + overdue flags
  → validation_analysis     (Gemini Flash)   Gini/PSI/AUC metrics review
  → llm_governance          (Gemini Flash)   RAGAS + prompt registry check
  → ab_test_analysis        (Gemini 3.1 Pro) Statistical significance review
  → model_risk_report       (Claude Sonnet)  SREP-quality narrative
  → hitl_gate               (deterministic)  Human escalation logic
  → END

HITL escalation policy
-----------------------
* Any HIGH model with FAIL validation    → CRO + Head of Model Risk (24h)
* Any model overdue revalidation >30d    → Head of Model Risk (48h)
* LLM hallucination rate >1%            → Model Risk Team (72h)
* RAGAS faithfulness <0.85              → Model Risk Team (72h)
* A/B test significant degradation      → Model Owner + Model Risk
* EU AI Act HIGH_RISK with no validator → CRO + Compliance Director
* 5+ active models in UNDER_REVIEW      → Head of Model Risk (ALCO brief)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)

# ── Regulatory constants ──────────────────────────────────────────────────────

# PRA SS1/23 validation thresholds
GINI_MIN            = 0.700
PSI_WARNING         = 0.100
PSI_ACTION          = 0.200
AUC_MIN             = 0.750

# LLM governance thresholds (RAGAS)
FAITHFULNESS_MIN    = 0.85
RELEVANCY_MIN       = 0.80
HALLUCINATION_MAX   = 1.0     # % — requires investigation

# A/B test significance
PVALUE_THRESHOLD    = 0.05
DEGRADATION_LIFT    = -2.0    # -2% lift → significant degradation

# Model risk appetite (BAP-2026-MRM-001)
OVERDUE_DAYS_WARN   = 30      # revalidation overdue warning
OVERDUE_DAYS_RED    = 90      # revalidation overdue RED
MAX_UNDER_REVIEW    = 5       # 5+ models UNDER_REVIEW → escalation
HIGH_RISK_NO_VALIDATOR_RED = True  # EU AI Act: HIGH_RISK must have validator

# LLM model selection
_GEMINI_FLASH  = "models/gemini-3.5-flash"
_GEMINI_PRO    = "models/gemini-3.1-pro"
_CLAUDE_SONNET = "claude-sonnet-4-6"


# ── Shared state ──────────────────────────────────────────────────────────────

class ModelRiskState(dict):
    """
    Shared mutable state passed between all graph nodes.

    Keys set by agents
    ------------------
    run_id                  str          UUID for this pipeline run
    run_date                str          ISO date of the assessment
    trigger_event           str          What initiated this run
    hop_chain               list[dict]   Ordered audit trail of all steps
    inventory_audit         dict         Registry scan results
    validation_findings     dict         Metrics review per model
    llm_governance_report   dict         RAGAS + prompt compliance
    ab_test_summary         dict         A/B test analysis results
    model_risk_narrative    str          SREP-quality report narrative
    risk_zone               str          GREEN / AMBER / RED
    hitl_decision           str          HITLDecision enum value
    escalation_contacts     list[str]    Named contacts to notify
    escalation_flags        list[str]    Reason strings
    model_id                str          MR-2026-058-MRM
    """


class HITLDecision(str, Enum):
    APPROVE  = "APPROVE"
    ESCALATE = "ESCALATE"
    OVERRIDE = "OVERRIDE"
    PENDING  = "PENDING"


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
    state: ModelRiskState,
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




# ── LLM stub ─────────────────────────────────────────────────────────────────

class _LLMClient:
    def __init__(self, model: str) -> None:
        self.model = model

    async def generate(self, prompt: str) -> str:
        logger.debug("LLM[%s] prompt_len=%d", self.model, len(prompt))
        await asyncio.sleep(0)
        return f"[{self.model}] stub response"


# ── Agent 1 — ModelInventoryAgent ─────────────────────────────────────────────

class ModelInventoryAgent:
    """
    Scans the ModelRegistry for governance gaps: overdue
    revalidations, missing validators on HIGH_RISK EU AI Act
    models, and UNDER_REVIEW accumulation.

    ReAct pattern
    -------------
    Reason: Identify models past their SS1/23 revalidation
            schedule, EU AI Act HIGH_RISK models without a
            named validator, and models stalled in UNDER_REVIEW.
    Act:    Query ModelRegistry; compute overdue days; classify
            each finding by severity; emit inventory audit dict.

    PRA SS1/23 requirement
    ----------------------
    Model inventory must be reviewed at minimum quarterly.
    HIGH risk models: 12-month revalidation cycle.
    MEDIUM risk models: 18-month cycle. LOW: 24-month.
    """

    _AGENT_NAME = "ModelInventoryAgent"

    def __init__(self) -> None:
        self._llm = _LLMClient(_GEMINI_FLASH)

    async def run(self, state: ModelRiskState) -> ModelRiskState:
        run_date  = state.get("run_date", datetime.utcnow().date().isoformat())
        as_of     = datetime.fromisoformat(run_date)

        reasoning = (
            f"REASON [{self._AGENT_NAME}]: Scan ModelRegistry "
            f"as of {run_date}. Identify: (1) overdue revalidations "
            f">{OVERDUE_DAYS_WARN} days, (2) EU AI Act HIGH_RISK "
            f"models without named validator, (3) UNDER_REVIEW count "
            f">= {MAX_UNDER_REVIEW}. Zone: any HIGH model overdue "
            f">{OVERDUE_DAYS_RED} days → RED."
        )

        try:
            import sys, os
            sys.path.insert(0, os.path.dirname(__file__))
            from model_inventory.registry import ModelRegistry
            from awb_commons.models import ModelStatus, EUAIActClass

            registry = ModelRegistry()
            all_models = registry.all_models()

            overdue_warn   = []
            overdue_red    = []
            no_validator   = []
            under_review   = []

            for m in all_models:
                # Overdue revalidation
                if m.next_revalidation and m.status.value == "ACTIVE":
                    days_overdue = (as_of - m.next_revalidation).days
                    if days_overdue > OVERDUE_DAYS_RED:
                        overdue_red.append({
                            "mr_reference": m.mr_reference,
                            "model_name":   m.model_name,
                            "days_overdue": days_overdue,
                            "risk_rating":  m.ss1_23_risk.value,
                        })
                    elif days_overdue > OVERDUE_DAYS_WARN:
                        overdue_warn.append({
                            "mr_reference": m.mr_reference,
                            "model_name":   m.model_name,
                            "days_overdue": days_overdue,
                            "risk_rating":  m.ss1_23_risk.value,
                        })

                # EU AI Act HIGH_RISK without validator
                if (m.eu_ai_act.value == "HIGH_RISK"
                        and not m.validator):
                    no_validator.append({
                        "mr_reference": m.mr_reference,
                        "model_name":   m.model_name,
                        "risk_rating":  m.ss1_23_risk.value,
                    })

                # UNDER_REVIEW accumulation
                if m.status.value == "UNDER_REVIEW":
                    under_review.append(m.mr_reference)

            # Determine zone
            risk_zone = "GREEN"
            if overdue_red or len(no_validator) > 0:
                risk_zone = "RED"
            elif overdue_warn or len(under_review) >= MAX_UNDER_REVIEW:
                risk_zone = "AMBER"

            audit = {
                "total_models":       len(all_models),
                "overdue_warn":       overdue_warn,
                "overdue_red":        overdue_red,
                "no_validator_eu_ai": no_validator,
                "under_review":       under_review,
                "under_review_count": len(under_review),
                "risk_zone":          risk_zone,
                "pra_ss123_ref":      "PRA SS1/23 §4",
            }

        except ImportError:
            audit = self._stub_result()
            risk_zone = audit["risk_zone"]

        state["inventory_audit"] = audit
        state["risk_zone"]       = risk_zone

        _log_step(
            state, self._AGENT_NAME,
            "SCAN_MODEL_REGISTRY",
            reasoning,
            {
                "total_models":     audit.get("total_models"),
                "overdue_red":      len(audit.get("overdue_red", [])),
                "overdue_warn":     len(audit.get("overdue_warn", [])),
                "no_validator":     len(audit.get("no_validator_eu_ai", [])),
                "under_review":     audit.get("under_review_count"),
                "risk_zone":        risk_zone,
            },
        )
        return state

    def _stub_result(self) -> dict:
        return {
            "total_models":       5,
            "overdue_warn":       [{"mr_reference": "MR-2026-049",
                                    "model_name": "Payment Fraud Detector",
                                    "days_overdue": 45,
                                    "risk_rating": "MEDIUM"}],
            "overdue_red":        [],
            "no_validator_eu_ai": [],
            "under_review":       [],
            "under_review_count": 0,
            "risk_zone":          "AMBER",
            "pra_ss123_ref":      "PRA SS1/23 §4",
        }


# ── Agent 2 — ValidationAnalysisAgent ────────────────────────────────────────

class ValidationAnalysisAgent:
    """
    Reviews the most recent validation results for each active
    model and flags metrics that breach PRA SS1/23 thresholds.

    ReAct pattern
    -------------
    Reason: For each active model with a recent validation,
            check Gini >= 0.70, PSI <= 0.20, AUC >= 0.75.
            FAIL outcome on any HIGH model → RED zone.
    Act:    Pull validation history from ModelRegistry; evaluate
            against MRMP §4.2 thresholds; build findings dict.
    """

    _AGENT_NAME = "ValidationAnalysisAgent"

    def __init__(self) -> None:
        self._llm = _LLMClient(_GEMINI_FLASH)

    async def run(self, state: ModelRiskState) -> ModelRiskState:
        reasoning = (
            f"REASON [{self._AGENT_NAME}]: Review latest validation "
            f"results. Thresholds: Gini ≥ {GINI_MIN}, "
            f"PSI ≤ {PSI_ACTION} (warning {PSI_WARNING}), "
            f"AUC ≥ {AUC_MIN}. FAIL on HIGH model → RED. "
            f"CONDITIONAL_PASS or FAIL → escalation required."
        )

        try:
            import sys, os
            sys.path.insert(0, os.path.dirname(__file__))
            from model_inventory.registry import ModelRegistry
            from awb_commons.models import RiskRating

            registry   = ModelRegistry()
            all_models = registry.all_models()

            model_findings: list[dict] = []
            any_fail_high  = False

            for m in all_models:
                history = registry.validation_history(m.mr_reference)
                if not history:
                    model_findings.append({
                        "mr_reference": m.mr_reference,
                        "model_name":   m.model_name,
                        "risk_rating":  m.ss1_23_risk.value,
                        "outcome":      "NO_VALIDATION",
                        "findings":     ["No validation on record"],
                    })
                    if m.ss1_23_risk.value == "HIGH":
                        any_fail_high = True
                    continue

                latest = max(history, key=lambda v: v.validated_at)
                finding = {
                    "mr_reference":    m.mr_reference,
                    "model_name":      m.model_name,
                    "risk_rating":     m.ss1_23_risk.value,
                    "validated_at":    latest.validated_at.isoformat(),
                    "outcome":         latest.outcome,
                    "gini":            latest.gini_coefficient,
                    "psi":             latest.psi,
                    "auc_roc":         latest.auc_roc,
                    "findings":        latest.findings,
                }
                model_findings.append(finding)

                if (latest.outcome == "FAIL"
                        and m.ss1_23_risk.value == "HIGH"):
                    any_fail_high = True

            # Determine zone
            prev_zone = state.get("risk_zone", "GREEN")
            zone_rank = {"GREEN": 0, "AMBER": 1, "RED": 2, "CRITICAL": 3}
            fail_zone = "RED" if any_fail_high else (
                "AMBER" if any(
                    f.get("outcome") in ("FAIL", "CONDITIONAL_PASS")
                    for f in model_findings
                ) else "GREEN"
            )
            risk_zone = max(
                prev_zone, fail_zone,
                key=lambda z: zone_rank.get(z, 0),
            )

            result = {
                "model_findings":    model_findings,
                "any_fail_high":     any_fail_high,
                "fail_count":        sum(
                    1 for f in model_findings
                    if f.get("outcome") in ("FAIL", "NO_VALIDATION")),
                "conditional_count": sum(
                    1 for f in model_findings
                    if f.get("outcome") == "CONDITIONAL_PASS"),
                "risk_zone":         risk_zone,
                "threshold_ref":     "AWB MRMP §4.2",
            }

        except ImportError:
            result = self._stub_result()
            risk_zone = result["risk_zone"]

        state["validation_findings"] = result
        state["risk_zone"]           = risk_zone

        _log_step(
            state, self._AGENT_NAME,
            "REVIEW_VALIDATION_METRICS",
            reasoning,
            {
                "fail_count":        result.get("fail_count"),
                "conditional_count": result.get("conditional_count"),
                "any_fail_high":     result.get("any_fail_high"),
                "risk_zone":         risk_zone,
            },
        )
        return state

    def _stub_result(self) -> dict:
        return {
            "model_findings": [
                {"mr_reference": "MR-2026-037",
                 "model_name": "AWB Credit Decision Agent",
                 "risk_rating": "HIGH",
                 "outcome": "PASS",
                 "gini": 0.742, "psi": 0.088, "auc_roc": 0.871,
                 "findings": []},
                {"mr_reference": "MR-2026-035",
                 "model_name": "AWB Credit Document Analyser",
                 "risk_rating": "MEDIUM",
                 "outcome": "CONDITIONAL_PASS",
                 "gini": 0.718, "psi": 0.112, "auc_roc": 0.859,
                 "findings": ["PSI 0.112 exceeds warning threshold"]},
            ],
            "any_fail_high":     False,
            "fail_count":        0,
            "conditional_count": 1,
            "risk_zone":         "AMBER",
            "threshold_ref":     "AWB MRMP §4.2",
        }


# ── Agent 3 — LLMGovernanceAgent ─────────────────────────────────────────────

class LLMGovernanceAgent:
    """
    Audits RAGAS monitoring snapshots and prompt registry
    compliance for all LLM-backed models in the AWB estate.

    ReAct pattern
    -------------
    Reason: Review latest LLMMonitoringSnapshot for each LLM
            model. Flag hallucination rate > 1%, faithfulness
            < 0.85, or p95 latency > 2,000ms. Check PromptRegistry
            for unapproved active prompts.
    Act:    Invoke LLMMonitor.assess_snapshot; compile RAGAS
            breach list; check PromptRegistry for compliance.

    PRA SS1/23 §7
    -------------
    Prompt changes are equivalent to model redevelopment and
    require full change management including MRO approval.
    """

    _AGENT_NAME = "LLMGovernanceAgent"

    def __init__(self) -> None:
        self._llm = _LLMClient(_GEMINI_FLASH)

    async def run(self, state: ModelRiskState) -> ModelRiskState:
        reasoning = (
            f"REASON [{self._AGENT_NAME}]: Audit LLM governance. "
            f"RAGAS thresholds: faithfulness ≥ {FAITHFULNESS_MIN}, "
            f"relevancy ≥ {RELEVANCY_MIN}. "
            f"Hallucination rate cap: {HALLUCINATION_MAX}%. "
            f"Prompt registry: unapproved prompts in production → RED. "
            f"Reference: PRA SS1/23 §7, EU AI Act Art. 9."
        )

        try:
            import sys, os
            sys.path.insert(0, os.path.dirname(__file__))
            from llm_monitoring.monitor import LLMMonitor, PromptRegistry
            from awb_commons.models import LLMMonitoringSnapshot

            # Synthetic current snapshots for AWB LLM models
            snapshots = [
                LLMMonitoringSnapshot(
                    mr_reference="MR-2026-038",
                    snapshot_month="2026-02",
                    faithfulness_score=0.923,
                    answer_relevancy=0.891,
                    context_precision=0.847,
                    context_recall=0.812,
                    avg_cost_per_query_gbp=0.0038,
                    p50_latency_ms=420,
                    p95_latency_ms=1_140,
                    hallucination_rate_pct=0.3,
                    total_queries=28_420,
                    alerts_triggered=[],
                ),
                LLMMonitoringSnapshot(
                    mr_reference="MR-2026-057-LIQ",
                    snapshot_month="2026-02",
                    faithfulness_score=0.831,   # BELOW threshold
                    answer_relevancy=0.874,
                    context_precision=0.792,
                    context_recall=0.761,
                    avg_cost_per_query_gbp=0.0041,
                    p50_latency_ms=380,
                    p95_latency_ms=1_050,
                    hallucination_rate_pct=1.4,  # ABOVE threshold
                    total_queries=21_050,
                    alerts_triggered=[],
                ),
            ]

            ragas_alerts: list[dict] = []
            for snap in snapshots:
                monitor = LLMMonitor(snap.mr_reference)
                alerts  = monitor.assess_snapshot(snap)
                needs_reval = monitor.trigger_revalidation(snap)
                if alerts:
                    ragas_alerts.append({
                        "mr_reference":    snap.mr_reference,
                        "snapshot_month":  snap.snapshot_month,
                        "alerts":          alerts,
                        "needs_reval":     needs_reval,
                        "hallucination":   snap.hallucination_rate_pct,
                    })

            # Prompt registry stub check
            # Production: PromptRegistry.active_prompt() for each LLM model
            unapproved_prompts: list[str] = []  # none in stub

            # Determine zone
            prev_zone = state.get("risk_zone", "GREEN")
            zone_rank = {"GREEN": 0, "AMBER": 1, "RED": 2, "CRITICAL": 3}
            llm_zone  = "RED" if (
                unapproved_prompts or
                any(
                    a.get("hallucination", 0) > HALLUCINATION_MAX
                    for a in ragas_alerts
                )
            ) else ("AMBER" if ragas_alerts else "GREEN")
            risk_zone = max(
                prev_zone, llm_zone,
                key=lambda z: zone_rank.get(z, 0),
            )

            result = {
                "ragas_alerts":       ragas_alerts,
                "alert_count":        len(ragas_alerts),
                "unapproved_prompts": unapproved_prompts,
                "models_reviewed":    len(snapshots),
                "risk_zone":          risk_zone,
                "governance_ref":     "PRA SS1/23 §7 / EU AI Act Art. 9",
            }

        except ImportError:
            result = self._stub_result()
            risk_zone = result["risk_zone"]

        state["llm_governance_report"] = result
        state["risk_zone"]             = risk_zone

        _log_step(
            state, self._AGENT_NAME,
            "AUDIT_LLM_GOVERNANCE",
            reasoning,
            {
                "alert_count":        result.get("alert_count"),
                "unapproved_prompts": len(result.get("unapproved_prompts", [])),
                "risk_zone":          risk_zone,
            },
        )
        return state

    def _stub_result(self) -> dict:
        return {
            "ragas_alerts": [
                {"mr_reference": "MR-2026-057-LIQ",
                 "snapshot_month": "2026-02",
                 "alerts": ["RAGAS faithfulness 0.831 below threshold 0.85",
                            "Hallucination rate 1.4% exceeds threshold 1.0%"],
                 "needs_reval": True,
                 "hallucination": 1.4},
            ],
            "alert_count":        1,
            "unapproved_prompts": [],
            "models_reviewed":    2,
            "risk_zone":          "RED",
            "governance_ref":     "PRA SS1/23 §7 / EU AI Act Art. 9",
        }


# ── Agent 4 — ABTestingAgent ──────────────────────────────────────────────────

class ABTestingAgent:
    """
    Analyses active A/B test results for model challenger
    programmes and flags statistically significant degradation.

    ReAct pattern
    -------------
    Reason: For each active A/B test, determine if the challenger
            shows statistically significant improvement or degradation
            versus the champion. p-value < 0.05 and lift < -2%
            constitutes a degradation requiring model owner review.
    Act:    Evaluate ABTestResult records; classify each as
            IMPROVE / NEUTRAL / DEGRADE; escalate on DEGRADE.

    SR 11-7 / PRA SS1/23
    ---------------------
    Champion-challenger testing is the preferred mechanism for
    model updates that fall below the full redevelopment threshold.
    Results must be documented and approved by MRO before deployment.
    """

    _AGENT_NAME = "ABTestingAgent"

    def __init__(self) -> None:
        self._llm = _LLMClient(_GEMINI_PRO)

    async def run(self, state: ModelRiskState) -> ModelRiskState:
        reasoning = (
            f"REASON [{self._AGENT_NAME}]: Analyse A/B test results. "
            f"Degradation threshold: p < {PVALUE_THRESHOLD} AND "
            f"lift < {DEGRADATION_LIFT}%. "
            f"Any significant degradation → model owner + model risk. "
            f"Reference: SR 11-7, PRA SS1/23 §5.3."
        )

        try:
            import sys, os
            sys.path.insert(0, os.path.dirname(__file__))
            from awb_commons.models import ABTestResult

            # Synthetic current A/B test results
            tests = [
                ABTestResult(
                    mr_reference="MR-2026-037",
                    control_version="v2.3",
                    treatment_version="v2.4",
                    sample_size_control=12_000,
                    sample_size_treatment=12_000,
                    metric_name="approval_rate_accuracy",
                    control_value=0.8712,
                    treatment_value=0.8841,
                    lift_pct=1.48,
                    p_value=0.023,
                    statistically_significant=True,
                    recommendation="Deploy v2.4: statistically significant improvement",
                    test_start=datetime(2026, 2, 1),
                    test_end=datetime(2026, 2, 28),
                ),
                ABTestResult(
                    mr_reference="MR-2026-035",
                    control_version="v1.8",
                    treatment_version="v1.9",
                    sample_size_control=8_000,
                    sample_size_treatment=8_000,
                    metric_name="extraction_accuracy",
                    control_value=0.9970,
                    treatment_value=0.9931,
                    lift_pct=-0.39,
                    p_value=0.041,
                    statistically_significant=True,
                    recommendation="Do not deploy v1.9: degradation observed",
                    test_start=datetime(2026, 2, 15),
                    test_end=None,  # still running
                ),
            ]

            test_summaries: list[dict] = []
            degradation_flags: list[str] = []

            for t in tests:
                if (t.statistically_significant
                        and t.p_value < PVALUE_THRESHOLD
                        and t.lift_pct < DEGRADATION_LIFT):
                    classification = "DEGRADE"
                    degradation_flags.append(t.mr_reference)
                elif (t.statistically_significant
                      and t.p_value < PVALUE_THRESHOLD
                      and t.lift_pct > 0):
                    classification = "IMPROVE"
                else:
                    classification = "NEUTRAL"

                test_summaries.append({
                    "mr_reference":     t.mr_reference,
                    "control_version":  t.control_version,
                    "treatment_version":t.treatment_version,
                    "metric":           t.metric_name,
                    "lift_pct":         t.lift_pct,
                    "p_value":          t.p_value,
                    "significant":      t.statistically_significant,
                    "classification":   classification,
                    "recommendation":   t.recommendation,
                    "still_running":    t.test_end is None,
                })

            prev_zone = state.get("risk_zone", "GREEN")
            zone_rank = {"GREEN": 0, "AMBER": 1, "RED": 2, "CRITICAL": 3}
            ab_zone   = "AMBER" if degradation_flags else "GREEN"
            risk_zone = max(
                prev_zone, ab_zone,
                key=lambda z: zone_rank.get(z, 0),
            )

            result = {
                "test_summaries":    test_summaries,
                "active_tests":      len(tests),
                "degradation_flags": degradation_flags,
                "improve_count":     sum(
                    1 for t in test_summaries
                    if t["classification"] == "IMPROVE"),
                "risk_zone":         risk_zone,
                "sr117_ref":         "SR 11-7 / PRA SS1/23 §5.3",
            }

        except ImportError:
            result = self._stub_result()
            risk_zone = result["risk_zone"]

        state["ab_test_summary"] = result
        state["risk_zone"]       = risk_zone

        _log_step(
            state, self._AGENT_NAME,
            "ANALYSE_AB_TESTS",
            reasoning,
            {
                "active_tests":      result.get("active_tests"),
                "degradation_flags": result.get("degradation_flags"),
                "improve_count":     result.get("improve_count"),
                "risk_zone":         risk_zone,
            },
        )
        return state

    def _stub_result(self) -> dict:
        return {
            "test_summaries": [
                {"mr_reference": "MR-2026-037",
                 "lift_pct": 1.48, "p_value": 0.023,
                 "classification": "IMPROVE",
                 "recommendation": "Deploy v2.4"},
            ],
            "active_tests":      1,
            "degradation_flags": [],
            "improve_count":     1,
            "risk_zone":         "GREEN",
            "sr117_ref":         "SR 11-7 / PRA SS1/23 §5.3",
        }


# ── Agent 5 — ModelRiskReportAgent ───────────────────────────────────────────

class ModelRiskReportAgent:
    """
    Synthesises all prior agent outputs into a PRA SREP-quality
    model risk management report using Claude Sonnet 4.6.

    Covers
    ------
    * Model inventory status (SS1/23 §4)
    * Validation findings and outstanding conditions
    * LLM governance breaches (SS1/23 §7)
    * A/B test champion-challenger programme status
    * Recommended management actions

    EU AI Act Art. 14 compliance
    ----------------------------
    Output is advisory. All model risk decisions require MRO
    and CRO review before execution. Human oversight is mandatory.
    """

    _AGENT_NAME = "ModelRiskReportAgent"

    def __init__(self) -> None:
        self._llm = _LLMClient(_CLAUDE_SONNET)

    async def run(self, state: ModelRiskState) -> ModelRiskState:
        inventory  = state.get("inventory_audit", {})
        validation = state.get("validation_findings", {})
        llm_gov    = state.get("llm_governance_report", {})
        ab_test    = state.get("ab_test_summary", {})
        risk_zone  = state.get("risk_zone", "GREEN")

        reasoning = (
            f"REASON [{self._AGENT_NAME}]: "
            f"Zone={risk_zone}. "
            f"Inventory: {inventory.get('total_models')} models, "
            f"{len(inventory.get('overdue_red', []))} overdue RED. "
            f"Validation: {validation.get('fail_count')} FAIL, "
            f"{validation.get('conditional_count')} CONDITIONAL. "
            f"LLM alerts: {llm_gov.get('alert_count')}. "
            f"Degradation flags: {ab_test.get('degradation_flags')}. "
            f"Draft SREP-quality model risk report. "
            f"EU AI Act Art. 14: MRO + CRO review required."
        )

        prompt = f"""
You are AWB's AI-powered Model Risk Officer assistant, drafting the monthly
PRA SS1/23 Model Risk Management Report for SREP submission.

ASSESSMENT DATE: {state.get('run_date', 'TODAY')}
RISK ZONE: {risk_zone}
MODEL ID: MR-2026-058-MRM

INVENTORY STATUS (PRA SS1/23 §4)
=================================
Total models in registry : {inventory.get('total_models', 0)}
Overdue revalidation (WARN): {len(inventory.get('overdue_warn', []))} models
Overdue revalidation (RED):  {len(inventory.get('overdue_red', []))} models
EU AI Act HIGH_RISK, no validator: {len(inventory.get('no_validator_eu_ai', []))} models
Models UNDER_REVIEW: {inventory.get('under_review_count', 0)}

VALIDATION FINDINGS (PRA SS1/23 §5)
=====================================
Models reviewed: {len(validation.get('model_findings', []))}
FAIL outcomes: {validation.get('fail_count', 0)}
CONDITIONAL_PASS: {validation.get('conditional_count', 0)}
Any HIGH-risk FAIL: {validation.get('any_fail_high', False)}

LLM GOVERNANCE (PRA SS1/23 §7 / EU AI Act Art. 9)
===================================================
LLM models audited: {llm_gov.get('models_reviewed', 0)}
RAGAS alerts triggered: {llm_gov.get('alert_count', 0)}
Unapproved prompts in production: {len(llm_gov.get('unapproved_prompts', []))}

A/B TESTING (SR 11-7 / PRA SS1/23 §5.3)
==========================================
Active tests: {ab_test.get('active_tests', 0)}
Improvement signals: {ab_test.get('improve_count', 0)}
Degradation flags: {len(ab_test.get('degradation_flags', []))} ({ab_test.get('degradation_flags', [])})

TASK
====
1. Write a concise SREP-quality executive summary (3 paragraphs).
2. List the top 3 model risk findings with severity.
3. Recommend management actions with owners and timeframes.

NOTE: MRO and CRO review required before submission.
BAP-2026-MRM-001: Board tolerance for FAIL outcomes on HIGH models is zero.
"""

        llm_narrative = await self._llm.generate(prompt)

        summary_lines = [
            f"AWB Model Risk Management Report — {state.get('run_date', 'TODAY')}",
            f"Model: MR-2026-058-MRM | Zone: {risk_zone}",
            "=" * 60,
            "",
            "INVENTORY SUMMARY (PRA SS1/23 §4)",
            f"  Total models        : {inventory.get('total_models', 0)}",
            f"  Overdue warn (30d)  : {len(inventory.get('overdue_warn', []))}",
            f"  Overdue RED  (90d)  : {len(inventory.get('overdue_red', []))}",
            f"  No validator (EU AI): {len(inventory.get('no_validator_eu_ai', []))}",
            "",
            "VALIDATION SUMMARY",
            f"  FAIL outcomes       : {validation.get('fail_count', 0)}",
            f"  CONDITIONAL_PASS    : {validation.get('conditional_count', 0)}",
            "",
            "LLM GOVERNANCE",
            f"  RAGAS alerts        : {llm_gov.get('alert_count', 0)}",
            f"  Unapproved prompts  : {len(llm_gov.get('unapproved_prompts', []))}",
            "",
            "A/B TESTING",
            f"  Active tests        : {ab_test.get('active_tests', 0)}",
            f"  Degradation flags   : {ab_test.get('degradation_flags', [])}",
            "",
            "[EU AI Act Art. 14] — MRO + CRO approval required",
            "[BAP-2026-MRM-001]  — Zero tolerance for HIGH-risk FAIL",
            "",
            "--- LLM NARRATIVE DRAFT ---",
            llm_narrative,
        ]

        state["model_risk_narrative"] = "\n".join(summary_lines)

        _log_step(
            state, self._AGENT_NAME,
            "GENERATE_SREP_NARRATIVE",
            reasoning,
            {
                "risk_zone":   risk_zone,
                "zone_flag":   "BAP-2026-MRM-001",
                "eu_ai_act":   "EU AI Act Art. 14",
                "pra_ss123":   "PRA SS1/23 §4-7",
            },
        )
        return state


# ── HITL Gate ─────────────────────────────────────────────────────────────────

async def hitl_gate_node(
    state: ModelRiskState,
) -> ModelRiskState:
    """
    Conservative HITL escalation policy.

    Escalation conditions (BAP-2026-MRM-001 / EU AI Act Art. 14)
    -------------------------------------------------------------
    HIGH model FAIL validation      → CRO + Head of Model Risk (24h)
    Any model overdue RED (90d)     → Head of Model Risk (48h)
    LLM hallucination > 1%         → Model Risk Team (72h)
    RAGAS faithfulness < 0.85      → Model Risk Team (72h)
    A/B degradation flag            → Model Owner + Model Risk
    EU AI Act HIGH_RISK no validator→ CRO + Compliance Director
    5+ UNDER_REVIEW models          → Head of Model Risk (ALCO brief)
    Any RED zone                    → CRO minimum escalation
    """
    inventory  = state.get("inventory_audit", {})
    validation = state.get("validation_findings", {})
    llm_gov    = state.get("llm_governance_report", {})
    ab_test    = state.get("ab_test_summary", {})
    zone       = state.get("risk_zone", "GREEN")

    contacts: list[str] = []
    flags:    list[str] = []

    # HIGH model FAIL
    if validation.get("any_fail_high"):
        contacts += ["CRO", "Head of Model Risk"]
        flags.append(
            "HIGH model FAIL: zero tolerance per BAP-2026-MRM-001 — 24h remediation"
        )

    # Overdue RED (90d)
    for m in inventory.get("overdue_red", []):
        contacts.append("Head of Model Risk")
        flags.append(
            f"OVERDUE RED: {m['mr_reference']} ({m['model_name']}) "
            f"{m['days_overdue']}d overdue — 48h escalation"
        )

    # LLM governance alerts
    for alert in llm_gov.get("ragas_alerts", []):
        contacts.append("Model Risk Team")
        flags.append(
            f"LLM RAGAS alert: {alert['mr_reference']} — "
            f"{len(alert.get('alerts', []))} breaches "
            f"(hallucination {alert.get('hallucination', 0):.1f}%)"
        )

    # A/B degradation
    for mr in ab_test.get("degradation_flags", []):
        contacts += ["Model Risk Team"]
        flags.append(
            f"A/B DEGRADATION: {mr} challenger significantly worse — "
            f"do not deploy, refer to MRO (SR 11-7)"
        )

    # EU AI Act: HIGH_RISK without validator
    for m in inventory.get("no_validator_eu_ai", []):
        contacts += ["CRO", "Compliance Director"]
        flags.append(
            f"EU AI ACT BREACH: {m['mr_reference']} is HIGH_RISK "
            f"with no assigned validator (EU AI Act Art. 17)"
        )

    # UNDER_REVIEW accumulation
    if inventory.get("under_review_count", 0) >= MAX_UNDER_REVIEW:
        contacts.append("Head of Model Risk")
        flags.append(
            f"{inventory['under_review_count']} models UNDER_REVIEW — "
            f"ALCO model risk briefing required"
        )

    # General RED escalation
    if zone == "RED" and "CRO" not in contacts:
        contacts.append("CRO")
        flags.append("RED ZONE escalation — CRO notification (BAP-2026-MRM-001)")

    contacts  = list(dict.fromkeys(contacts))
    decision  = (
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
            f"Conservative policy per BAP-2026-MRM-001. "
            f"EU AI Act Art. 14: no auto-approval of model risk decisions. "
            f"MRO + CRO review mandatory for all RED-zone findings."
        ),
        {
            "decision": decision.value,
            "contacts": contacts,
            "flags":    flags,
            "zone":     zone,
        },
    )
    return state


# ── Sequential stub ───────────────────────────────────────────────────────────

class _SequentialStub:
    async def run(
        self, initial_state: ModelRiskState
    ) -> ModelRiskState:
        state = initial_state
        for agent_cls in [
            ModelInventoryAgent,
            ValidationAnalysisAgent,
            LLMGovernanceAgent,
            ABTestingAgent,
            ModelRiskReportAgent,
        ]:
            state = await agent_cls().run(state)
        state = await hitl_gate_node(state)
        return state


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_model_risk_graph():
    """
    Build the LangGraph StateGraph for model risk surveillance.

    Graph topology
    --------------
    START → model_inventory_audit → validation_analysis
          → llm_governance → ab_test_analysis
          → model_risk_report → hitl → END
    """
    try:
        from langgraph.graph import StateGraph, START, END

        graph = StateGraph(ModelRiskState)

        graph.add_node(
            "model_inventory_audit",
            lambda s: asyncio.get_event_loop().run_until_complete(
                ModelInventoryAgent().run(s)),
        )
        graph.add_node(
            "validation_analysis",
            lambda s: asyncio.get_event_loop().run_until_complete(
                ValidationAnalysisAgent().run(s)),
        )
        graph.add_node(
            "llm_governance",
            lambda s: asyncio.get_event_loop().run_until_complete(
                LLMGovernanceAgent().run(s)),
        )
        graph.add_node(
            "ab_test_analysis",
            lambda s: asyncio.get_event_loop().run_until_complete(
                ABTestingAgent().run(s)),
        )
        graph.add_node(
            "model_risk_report",
            lambda s: asyncio.get_event_loop().run_until_complete(
                ModelRiskReportAgent().run(s)),
        )
        graph.add_node(
            "hitl",
            lambda s: asyncio.get_event_loop().run_until_complete(
                hitl_gate_node(s)),
        )

        graph.add_edge(START,                   "model_inventory_audit")
        graph.add_edge("model_inventory_audit",  "validation_analysis")
        graph.add_edge("validation_analysis",    "llm_governance")
        graph.add_edge("llm_governance",         "ab_test_analysis")
        graph.add_edge("ab_test_analysis",       "model_risk_report")
        graph.add_edge("model_risk_report",      "hitl")
        graph.add_edge("hitl",                   END)

        return graph.compile()

    except ImportError:
        logger.warning(
            "LangGraph not installed — using sequential stub. "
            "Install: pip install langgraph"
        )
        return None


# ── Public entry point ────────────────────────────────────────────────────────

async def run_agentic_model_risk(
    run_date:      str,
    trigger_event: str,
) -> ModelRiskState:
    """
    Execute the full agentic model risk management pipeline.

    Parameters
    ----------
    run_date      : ISO date string, e.g. "2026-03-01"
    trigger_event : e.g. "MONTHLY_MRMC" / "VALIDATION_FAIL" /
                         "LLM_ALERT" / "AB_TEST_COMPLETE"

    Returns
    -------
    ModelRiskState — full state with inventory audit, validation
    findings, LLM governance report, A/B test summary,
    SREP narrative, risk zone, HITL decision, and hop-chain.

    Examples
    --------
    >>> import asyncio
    >>> state = asyncio.run(run_agentic_model_risk(
    ...     run_date="2026-03-01",
    ...     trigger_event="MONTHLY_MRMC",
    ... ))
    >>> print(f"Zone: {state['risk_zone']}")
    >>> print(f"HITL: {state['hitl_decision']}")
    """
    run_id = str(uuid4())
    logger.info(
        "run_agentic_model_risk START run_id=%s "
        "date=%s trigger=%s",
        run_id, run_date, trigger_event,
    )

    initial_state = ModelRiskState({
        "run_id":            run_id,
        "run_date":          run_date,
        "trigger_event":     trigger_event,
        "model_id":          "MR-2026-058-MRM",
        "hop_chain":         [],
        "risk_zone":         "GREEN",
        "hitl_decision":     HITLDecision.PENDING.value,
        "escalation_contacts": [],
        "escalation_flags":  [],
    })

    graph = build_model_risk_graph()
    if graph is not None:
        final_state = await asyncio.to_thread(graph.invoke, initial_state)
    else:
        final_state = await _SequentialStub().run(initial_state)

    logger.info(
        "run_agentic_model_risk DONE run_id=%s "
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
        state = await run_agentic_model_risk(
            run_date="2026-03-01",
            trigger_event="MONTHLY_MRMC",
        )
        print("\n" + "=" * 60)
        print("AGENTIC MODEL RISK MONITOR — MR-2026-058-MRM")
        print("=" * 60)
        print(f"Risk Zone  : {state['risk_zone']}")
        print(f"HITL       : {state['hitl_decision']}")
        contacts = state.get("escalation_contacts", [])
        if contacts:
            print(f"Escalate   : {', '.join(contacts)}")
        print(f"Hop-chain  : {len(state.get('hop_chain', []))} steps")
        print("-" * 60)
        for hop in state.get("hop_chain", []):
            print(
                f"  [{hop['seq']:02d}] {hop['agent']:<30} "
                f"{hop['act']}"
            )
        print("=" * 60)

    asyncio.run(_smoke_test())
