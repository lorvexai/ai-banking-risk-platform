"""
exercises/sme_extraction.py
AWB Credit Document Analyser — Exercise 2.1 Starter

Exercise 2.1: Extract financials from a synthetic SME credit pack
Difficulty: ★★☆☆☆ | Estimated time: 20 minutes

Task:
    Build a Gemini 3.5 Flash extraction prompt for 5 financial fields
    from a synthetic AWB SME credit pack.

    Target: precision >= 0.90 on the provided test set (20 documents).

Fields to extract:
    1. revenue_gbp          — Annual revenue (£)
    2. ebitda_gbp           — EBITDA (£)
    3. net_debt_gbp         — Net debt (£)
    4. interest_cover_ratio — EBITDA / Net interest expense
    5. leverage_ratio       — Net debt / EBITDA

Regulatory context:
    PRA SS1/23 — all extractions must carry confidence scores
    EU AI Act Annex III §5b — HIGH-RISK: source citations mandatory

Author: AWB AI Programme (AWB-AI-2025)
Chapter: 2 — LLM Foundations for Financial Document Analysis
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── Pydantic schema for extraction ───────────────────────────────────────────


class FieldExtract(BaseModel):
    """Single extracted financial field with confidence."""

    value: Optional[float] = Field(
        None, description="Extracted numeric value"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Confidence score 0-1"
    )
    source_page: Optional[int] = Field(
        None, description="Page number where value found"
    )
    source_text: Optional[str] = Field(
        None, description="Verbatim text excerpt (<=50 chars)"
    )


class SMEExtraction(BaseModel):
    """Structured output for SME financial extraction."""

    revenue_gbp: FieldExtract
    ebitda_gbp: FieldExtract
    net_debt_gbp: FieldExtract
    interest_cover_ratio: FieldExtract
    leverage_ratio: FieldExtract
    analyst_review_required: bool = Field(
        False,
        description=(
            "True if any field confidence < 0.80"
        )
    )


# ── TODO: Complete the extraction function ────────────────────────────────────


def extract_sme_financials(
    document_text: str,
    model_id: str = "gemini-3.5-flash",
) -> SMEExtraction:
    """
    Extract 5 financial fields from an SME credit pack.

    Args:
        document_text: Full text of the credit pack document.
        model_id: LLM model to use for extraction.

    Returns:
        SMEExtraction with all 5 fields populated.

    TODO:
        1. Build a structured extraction prompt that:
           - Instructs the model to extract each of the 5 fields
           - Requires source_page and source_text for every field
           - Requests confidence scores based on clarity of source
           - Uses the SMEExtraction Pydantic schema for output

        2. Call the Gemini API with:
           - temperature=0.1 (deterministic extraction)
           - response_mime_type="application/json"
           - response_schema matching SMEExtraction

        3. Parse and return the structured response.

    Hint: See Section 2.3 in the chapter for the full prompt pattern.
    The complete solution is in solutions/sme_extraction_solution.py
    """
    # YOUR CODE HERE
    raise NotImplementedError(
        "Complete the extraction function. "
        "See chapter Section 2.3 for guidance."
    )


# ── Evaluation harness ────────────────────────────────────────────────────────


def evaluate_extraction(
    predictions: list[SMEExtraction],
    ground_truth: list[dict],
    tolerance: float = 0.02,
) -> dict:
    """
    Evaluate extraction precision against ground truth.

    Args:
        predictions: List of SMEExtraction results.
        ground_truth: List of dicts with true field values.
        tolerance: Relative tolerance for numeric match (2%).

    Returns:
        Dict with per-field and overall precision scores.
    """
    fields = [
        "revenue_gbp",
        "ebitda_gbp",
        "net_debt_gbp",
        "interest_cover_ratio",
        "leverage_ratio",
    ]
    results: dict[str, dict] = {f: {"correct": 0, "total": 0}
                                 for f in fields}

    for pred, truth in zip(predictions, ground_truth):
        for field in fields:
            true_val = truth.get(field)
            pred_field: FieldExtract = getattr(pred, field)
            pred_val = pred_field.value

            results[field]["total"] += 1
            if true_val is None and pred_val is None:
                results[field]["correct"] += 1
            elif true_val is not None and pred_val is not None:
                rel_err = abs(pred_val - true_val) / (
                    abs(true_val) + 1e-9
                )
                if rel_err <= tolerance:
                    results[field]["correct"] += 1

    summary = {}
    total_correct = 0
    total_fields = 0
    for field, counts in results.items():
        prec = (
            counts["correct"] / counts["total"]
            if counts["total"] > 0 else 0.0
        )
        summary[field] = round(prec, 4)
        total_correct += counts["correct"]
        total_fields += counts["total"]

    overall = (
        total_correct / total_fields
        if total_fields > 0 else 0.0
    )
    summary["overall_precision"] = round(overall, 4)
    summary["target_met"] = overall >= 0.90

    return summary


# ── Quick test with synthetic data ────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    data_dir = Path(__file__).parent.parent / "data"
    test_files = list(data_dir.glob("*_credit_pack.txt"))

    if not test_files:
        print(
            "No test files found in data/. "
            "Run data/generate_sample_credit_pack.py first."
        )
        sys.exit(1)

    print(f"Found {len(test_files)} test documents.")
    print("Implement extract_sme_financials() then re-run.")
    print(
        "Target: overall_precision >= 0.90 on the test set."
    )
