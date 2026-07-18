"""
AWB Credit Document Analyser — Chapter 2
AI for Financial Risk, Compliance and Regulatory Reporting

PRA SS1/23 Model: MR-2026-035 | Risk Rating: MEDIUM
EU AI Act: Annex III §5b | Classification: HIGH-RISK
DORA ICT Asset: DA-2026-002

Modules:
  extractor.py       — Core LLM extraction pipeline (Gemini 3.1 Pro/Flash)
  validator.py       — Post-extraction Pydantic validation and range checks
  audit_log.py       — PRA SS1/23 compliant audit trail
  prompt_patterns.py — Four LLM prompt engineering patterns (Section 2.6.1)
  model_selector.py  — Banking task decision matrix for model selection (Section 2.7.2)
"""
from .prompt_patterns import (
    build_role_system_prompt,
    build_cot_ratio_prompt,
    build_structured_output_prompt,
    build_few_shot_edge_case_prompt,
    build_full_extraction_prompt,
    PROMPT_TEMPLATE_VERSION,
)
from .model_selector import (
    GeminiModel,
    ModelProfile,
    BankingTask,
    ModelSelectionResult,
    select_model,
    select_model_for_document,
    MODEL_PROFILES,
    TASK_REQUIREMENTS,
)
