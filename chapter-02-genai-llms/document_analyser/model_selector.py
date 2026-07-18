"""
document_analyser/model_selector.py
AWB Credit Document Analyser — LLM Model Selection Logic

Implements AWB's banking task decision matrix for Gemini model selection.
Referenced in Chapter 2 Section 2.7.2 of:
  "AI for Financial Risk, Compliance and Regulatory Reporting"

Decision logic:
  Gemini 3.5 Flash  — documents < 50 pages, SME credit packs, speed-critical
                       tasks, batch processing overnight runs.
                       Cost: ~£0.039/document (AWB benchmark, June 2026)

  Gemini 3.1 Pro    — documents 50–500 pages, corporate credit packs,
                       tasks requiring deep multi-document cross-referencing,
                       tasks where higher accuracy justifies higher cost.
                       Cost: ~£0.158/document (AWB benchmark, June 2026)

The matrix is parameterised so it can be updated when model pricing
or capability changes without modifying caller code.

Regulatory compliance:
  DORA Art. 6 (ICT Risk Management): LLM provider selection is logged
    for concentration risk monitoring. AWB's DORA register tracks
    Gemini (Google) as a critical third-party provider (CTP-2026-004).
  PRA SS1/23 §5.4: Model selection criteria must be documented and
    reproducible. This module serves as that documentation.

Author: AWB AI Programme (AWB-AI-2025)
Chapter: 2 — LLM Foundations for Financial Document Analysis
Version: 1.1.0  (June 2026)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model catalogue — update when new models become available
# ---------------------------------------------------------------------------

class GeminiModel(str, Enum):
    """Approved Gemini models for AWB production use (June 2026)."""
    FLASH_25       = "gemini-3.5-flash"         # Speed-optimised
    PRO_25         = "gemini-3.1-pro"            # Accuracy-optimised
    FLASH_20       = "gemini-3.5-flash"          # DORA fallback
    PRO_15         = "gemini-3.1-pro"        # DORA fallback


@dataclass
class ModelProfile:
    """
    Capability and cost profile for a single Gemini model.

    Costs are in GBP per million tokens (June 2026, £1 = $1.27).
    Benchmarks are from AWB's internal 200-document evaluation set.
    """
    model_id: GeminiModel
    display_name: str
    context_window_tokens: int          # Maximum input tokens
    input_cost_gbp_per_1m: float        # Input token cost
    output_cost_gbp_per_1m: float       # Output token cost
    avg_latency_seconds: float          # p50 latency on AWB benchmark
    p95_latency_seconds: float          # p95 latency on AWB benchmark
    extraction_accuracy: float          # On AWB 200-doc validation set
    is_production_approved: bool        # PRA SS1/23 validation complete
    dora_fallback_rank: int             # 1 = primary, 2 = first fallback, etc.
    notes: str = ""


# June 2026 model profiles.
# NOTE: input/output cost figures are AWB's *effective* blended £/1M rates after volume,
# batch, and prompt-cache discounts — not standard list prices. For current list pricing
# see docs/appendices/tech-stack.md and vendor pages (ai.google.dev, openai.com, anthropic.com).
MODEL_PROFILES: dict[GeminiModel, ModelProfile] = {
    GeminiModel.FLASH_25: ModelProfile(
        model_id=GeminiModel.FLASH_25,
        display_name="Gemini 3.5 Flash",
        context_window_tokens=1_000_000,
        input_cost_gbp_per_1m=0.039,
        output_cost_gbp_per_1m=0.157,
        avg_latency_seconds=8.2,
        p95_latency_seconds=18.4,
        extraction_accuracy=0.961,
        is_production_approved=True,
        dora_fallback_rank=1,
        notes=(
            "Primary model for SME and small corporate credit packs (<50 pages). "
            "Optimal cost-accuracy trade-off for documents under 80,000 tokens. "
            "DORA primary provider: Google Cloud (CTP-2026-004)."
        ),
    ),
    GeminiModel.PRO_25: ModelProfile(
        model_id=GeminiModel.PRO_25,
        display_name="Gemini 3.1 Pro",
        context_window_tokens=1_000_000,
        input_cost_gbp_per_1m=1.26,
        output_cost_gbp_per_1m=5.04,
        avg_latency_seconds=24.7,
        p95_latency_seconds=41.3,
        extraction_accuracy=0.973,
        is_production_approved=True,
        dora_fallback_rank=1,
        notes=(
            "Primary model for large corporate credit packs (50-500 pages). "
            "Required for multi-document cross-referencing and complex "
            "covenant analysis. 1.2pp accuracy gain over Flash justifies "
            "4x cost increase for high-value facilities (>£50M)."
        ),
    ),
    GeminiModel.FLASH_20: ModelProfile(
        model_id=GeminiModel.FLASH_20,
        display_name="Gemini 3.5 Flash",
        context_window_tokens=1_000_000,
        input_cost_gbp_per_1m=0.031,
        output_cost_gbp_per_1m=0.126,
        avg_latency_seconds=6.1,
        p95_latency_seconds=14.2,
        extraction_accuracy=0.947,
        is_production_approved=True,
        dora_fallback_rank=2,
        notes=(
            "DORA fallback for Flash 2.5. Lower accuracy (1.4pp) and "
            "lower cost. Activates automatically when Flash 2.5 API "
            "availability drops below 99.5% SLA."
        ),
    ),
    GeminiModel.PRO_15: ModelProfile(
        model_id=GeminiModel.PRO_15,
        display_name="Gemini 3 Pro 002",
        context_window_tokens=2_000_000,
        input_cost_gbp_per_1m=0.945,
        output_cost_gbp_per_1m=3.78,
        avg_latency_seconds=31.5,
        p95_latency_seconds=58.0,
        extraction_accuracy=0.958,
        is_production_approved=True,
        dora_fallback_rank=2,
        notes=(
            "DORA fallback for Pro 2.5. 2M token context allows "
            "very large document sets (up to ~1,200 pages). Lower "
            "accuracy (1.5pp) but useful for edge cases where document "
            "volume exceeds Pro 2.5 context window."
        ),
    ),
}


# ---------------------------------------------------------------------------
# Task taxonomy — banking tasks mapped to selection criteria
# ---------------------------------------------------------------------------

class BankingTask(str, Enum):
    """AWB banking task taxonomy for LLM selection."""
    # Document extraction
    SME_FINANCIAL_EXTRACTION     = "sme_financial_extraction"
    CORPORATE_FINANCIAL_EXTRACTION = "corporate_financial_extraction"
    COVENANT_COMPLIANCE_CHECK    = "covenant_compliance_check"
    MULTI_DOCUMENT_CREDIT_PACK   = "multi_document_credit_pack"

    # Regulatory
    REGULATORY_POLICY_QA         = "regulatory_policy_qa"
    COREP_FIELD_MAPPING          = "corep_field_mapping"
    SAR_NARRATIVE_GENERATION     = "sar_narrative_generation"

    # Customer
    COMPLAINT_CLASSIFICATION     = "complaint_classification"
    CUSTOMER_SERVICE_INTENT      = "customer_service_intent"

    # Risk
    CREDIT_MEMO_GENERATION       = "credit_memo_generation"
    RISK_NARRATIVE_SUMMARY       = "risk_narrative_summary"


@dataclass
class TaskRequirements:
    """Requirements profile for a banking task."""
    task: BankingTask
    typical_input_tokens: int       # Typical document size in tokens
    max_input_tokens: int           # Maximum document size
    latency_sensitive: bool         # True if user is waiting in real-time
    accuracy_critical: bool         # True if errors have material financial impact
    batch_eligible: bool            # True if can run overnight in batch
    min_accuracy_threshold: float   # Minimum acceptable extraction accuracy


TASK_REQUIREMENTS: dict[BankingTask, TaskRequirements] = {
    BankingTask.SME_FINANCIAL_EXTRACTION: TaskRequirements(
        task=BankingTask.SME_FINANCIAL_EXTRACTION,
        typical_input_tokens=20_000,
        max_input_tokens=80_000,
        latency_sensitive=True,
        accuracy_critical=True,
        batch_eligible=True,
        min_accuracy_threshold=0.90,
    ),
    BankingTask.CORPORATE_FINANCIAL_EXTRACTION: TaskRequirements(
        task=BankingTask.CORPORATE_FINANCIAL_EXTRACTION,
        typical_input_tokens=120_000,
        max_input_tokens=500_000,
        latency_sensitive=False,
        accuracy_critical=True,
        batch_eligible=False,
        min_accuracy_threshold=0.95,
    ),
    BankingTask.COVENANT_COMPLIANCE_CHECK: TaskRequirements(
        task=BankingTask.COVENANT_COMPLIANCE_CHECK,
        typical_input_tokens=40_000,
        max_input_tokens=120_000,
        latency_sensitive=False,
        accuracy_critical=True,
        batch_eligible=False,
        min_accuracy_threshold=0.97,
    ),
    BankingTask.MULTI_DOCUMENT_CREDIT_PACK: TaskRequirements(
        task=BankingTask.MULTI_DOCUMENT_CREDIT_PACK,
        typical_input_tokens=300_000,
        max_input_tokens=800_000,
        latency_sensitive=False,
        accuracy_critical=True,
        batch_eligible=False,
        min_accuracy_threshold=0.95,
    ),
    BankingTask.REGULATORY_POLICY_QA: TaskRequirements(
        task=BankingTask.REGULATORY_POLICY_QA,
        typical_input_tokens=50_000,
        max_input_tokens=200_000,
        latency_sensitive=True,
        accuracy_critical=True,
        batch_eligible=False,
        min_accuracy_threshold=0.94,
    ),
    BankingTask.COMPLAINT_CLASSIFICATION: TaskRequirements(
        task=BankingTask.COMPLAINT_CLASSIFICATION,
        typical_input_tokens=500,
        max_input_tokens=2_000,
        latency_sensitive=True,
        accuracy_critical=False,
        batch_eligible=True,
        min_accuracy_threshold=0.90,
    ),
    BankingTask.CUSTOMER_SERVICE_INTENT: TaskRequirements(
        task=BankingTask.CUSTOMER_SERVICE_INTENT,
        typical_input_tokens=300,
        max_input_tokens=1_000,
        latency_sensitive=True,
        accuracy_critical=False,
        batch_eligible=False,
        min_accuracy_threshold=0.88,
    ),
    BankingTask.CREDIT_MEMO_GENERATION: TaskRequirements(
        task=BankingTask.CREDIT_MEMO_GENERATION,
        typical_input_tokens=8_000,
        max_input_tokens=30_000,
        latency_sensitive=False,
        accuracy_critical=True,
        batch_eligible=True,
        min_accuracy_threshold=0.95,
    ),
    BankingTask.SAR_NARRATIVE_GENERATION: TaskRequirements(
        task=BankingTask.SAR_NARRATIVE_GENERATION,
        typical_input_tokens=15_000,
        max_input_tokens=60_000,
        latency_sensitive=False,
        accuracy_critical=True,
        batch_eligible=True,
        min_accuracy_threshold=0.95,
    ),
}


# ---------------------------------------------------------------------------
# Selection logic
# ---------------------------------------------------------------------------

@dataclass
class ModelSelectionResult:
    """Result of a model selection decision."""
    primary_model: GeminiModel
    fallback_model: GeminiModel
    rationale: str
    estimated_cost_gbp: float           # Per document estimate
    estimated_latency_seconds: float    # p95 latency
    dora_provider: str = "Google Cloud (CTP-2026-004)"
    log_entry: dict = field(default_factory=dict)


def select_model(
    task: BankingTask,
    actual_input_tokens: Optional[int] = None,
    force_batch_mode: bool = False,
    override_model: Optional[GeminiModel] = None,
) -> ModelSelectionResult:
    """
    Select the optimal Gemini model for a given banking task.

    Decision matrix (in priority order):
      1. If override_model set: use it (requires documented justification).
      2. If task has accuracy_critical=True AND input > 80,000 tokens: use Pro.
      3. If task has accuracy_critical=True AND min_threshold > 0.96: use Pro.
      4. If latency_sensitive=True AND tokens < 80,000: use Flash.
      5. If batch_eligible=True: use Flash (cost optimisation).
      6. Default: Flash for speed, Pro available as first fallback.

    All selection decisions are logged for DORA concentration risk
    monitoring and PRA SS1/23 model usage audit.

    Args:
        task:                Banking task from BankingTask enum.
        actual_input_tokens: Actual token count if known (overrides typical).
        force_batch_mode:    Override to batch-eligible Flash even if not
                             normally batch eligible.
        override_model:      Force a specific model (must be documented).

    Returns:
        ModelSelectionResult with primary model, fallback, rationale, and cost estimate.
    """
    requirements = TASK_REQUIREMENTS.get(task)
    if requirements is None:
        logger.warning(f"Unknown task {task} — defaulting to Flash 2.5")
        return _default_flash_result(task)

    input_tokens = actual_input_tokens or requirements.typical_input_tokens

    # Override path
    if override_model is not None:
        profile = MODEL_PROFILES[override_model]
        fallback = _get_dora_fallback(override_model)
        return ModelSelectionResult(
            primary_model=override_model,
            fallback_model=fallback,
            rationale=f"Manual override to {override_model.value}. Document this in SS1/23 model log.",
            estimated_cost_gbp=_estimate_cost(override_model, input_tokens),
            estimated_latency_seconds=profile.p95_latency_seconds,
        )

    # Rule 1: Large document requiring high accuracy → Pro
    if requirements.accuracy_critical and input_tokens > 80_000:
        model = GeminiModel.PRO_25
        rationale = (
            f"Pro selected: accuracy_critical=True and input_tokens={input_tokens:,} "
            f"> 80,000 token Flash threshold. "
            f"Task: {task.value}. Expected accuracy: {MODEL_PROFILES[model].extraction_accuracy:.1%}."
        )
        return ModelSelectionResult(
            primary_model=model,
            fallback_model=GeminiModel.PRO_15,
            rationale=rationale,
            estimated_cost_gbp=_estimate_cost(model, input_tokens),
            estimated_latency_seconds=MODEL_PROFILES[model].p95_latency_seconds,
        )

    # Rule 2: Very high accuracy threshold → Pro
    if requirements.min_accuracy_threshold > 0.96:
        model = GeminiModel.PRO_25
        rationale = (
            f"Pro selected: min_accuracy_threshold={requirements.min_accuracy_threshold:.2%} "
            f"exceeds Flash capability ({MODEL_PROFILES[GeminiModel.FLASH_25].extraction_accuracy:.1%}). "
            f"Task: {task.value}."
        )
        return ModelSelectionResult(
            primary_model=model,
            fallback_model=GeminiModel.PRO_15,
            rationale=rationale,
            estimated_cost_gbp=_estimate_cost(model, input_tokens),
            estimated_latency_seconds=MODEL_PROFILES[model].p95_latency_seconds,
        )

    # Rule 3: Latency-sensitive + small document → Flash
    if requirements.latency_sensitive and input_tokens <= 80_000:
        model = GeminiModel.FLASH_25
        rationale = (
            f"Flash selected: latency_sensitive=True and input_tokens={input_tokens:,} "
            f"<= 80,000. p95 latency: {MODEL_PROFILES[model].p95_latency_seconds:.1f}s. "
            f"Task: {task.value}."
        )
        return ModelSelectionResult(
            primary_model=model,
            fallback_model=GeminiModel.FLASH_20,
            rationale=rationale,
            estimated_cost_gbp=_estimate_cost(model, input_tokens),
            estimated_latency_seconds=MODEL_PROFILES[model].p95_latency_seconds,
        )

    # Rule 4: Batch eligible → Flash (cost optimisation)
    if requirements.batch_eligible or force_batch_mode:
        model = GeminiModel.FLASH_25
        rationale = (
            f"Flash selected: batch_eligible=True. "
            f"Cost saving vs Pro: {_estimate_cost(GeminiModel.PRO_25, input_tokens) - _estimate_cost(model, input_tokens):.4f} GBP/doc. "
            f"Task: {task.value}."
        )
        return ModelSelectionResult(
            primary_model=model,
            fallback_model=GeminiModel.FLASH_20,
            rationale=rationale,
            estimated_cost_gbp=_estimate_cost(model, input_tokens),
            estimated_latency_seconds=MODEL_PROFILES[model].p95_latency_seconds,
        )

    # Default → Flash
    return _default_flash_result(task, input_tokens)


def _default_flash_result(
    task: BankingTask,
    input_tokens: int = 20_000,
) -> ModelSelectionResult:
    model = GeminiModel.FLASH_25
    return ModelSelectionResult(
        primary_model=model,
        fallback_model=GeminiModel.FLASH_20,
        rationale=f"Default Flash selection. Task: {task.value if hasattr(task, 'value') else task}.",
        estimated_cost_gbp=_estimate_cost(model, input_tokens),
        estimated_latency_seconds=MODEL_PROFILES[model].p95_latency_seconds,
    )


def _estimate_cost(model: GeminiModel, input_tokens: int) -> float:
    """Estimate per-document cost in GBP based on typical input/output split."""
    profile = MODEL_PROFILES[model]
    # Assume output is ~5% of input (typical for structured extraction)
    output_tokens = max(500, int(input_tokens * 0.05))
    cost = (
        (input_tokens / 1_000_000) * profile.input_cost_gbp_per_1m
        + (output_tokens / 1_000_000) * profile.output_cost_gbp_per_1m
    )
    return round(cost, 6)


def _get_dora_fallback(model: GeminiModel) -> GeminiModel:
    """Return the DORA-registered fallback for a given model."""
    fallbacks = {
        GeminiModel.FLASH_25: GeminiModel.FLASH_20,
        GeminiModel.PRO_25:   GeminiModel.PRO_15,
        GeminiModel.FLASH_20: GeminiModel.PRO_15,
        GeminiModel.PRO_15:   GeminiModel.FLASH_20,
    }
    return fallbacks.get(model, GeminiModel.FLASH_20)


# ---------------------------------------------------------------------------
# Convenience: get model for document size
# ---------------------------------------------------------------------------

def select_model_for_document(
    page_count: int,
    token_count: Optional[int] = None,
    is_corporate: bool = True,
) -> ModelSelectionResult:
    """
    Simplified entry point: select model based on document size and type.

    This is the function called by the CDA pipeline (extractor.py) to
    determine whether to use Flash or Pro for each incoming credit pack.

    AWB rule of thumb:
      < 50 pages  → Flash 2.5  (< 80,000 tokens at 1,600 tokens/page)
      50-500 pages → Pro 2.5   (80,000 to 800,000 tokens)
      > 500 pages  → Pro 2.5 + document chunking strategy

    Args:
        page_count:   Number of pages in the document.
        token_count:  Actual token count if pre-computed.
        is_corporate: True for corporate credit packs, False for SME.

    Returns:
        ModelSelectionResult with selected model and rationale.
    """
    TOKENS_PER_PAGE = 1_600   # AWB empirical benchmark
    estimated_tokens = token_count or (page_count * TOKENS_PER_PAGE)

    task = (
        BankingTask.CORPORATE_FINANCIAL_EXTRACTION
        if is_corporate
        else BankingTask.SME_FINANCIAL_EXTRACTION
    )

    result = select_model(task=task, actual_input_tokens=estimated_tokens)

    logger.info(
        "Model selected for document",
        extra={
            "page_count": page_count,
            "estimated_tokens": estimated_tokens,
            "selected_model": result.primary_model.value,
            "fallback_model": result.fallback_model.value,
            "estimated_cost_gbp": result.estimated_cost_gbp,
            "rationale": result.rationale,
        },
    )
    return result
