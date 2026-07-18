"""
exercises/exercise_2.py
AWB Credit Document Analyser — Exercise 2.2 Starter

Exercise 2.2: Build a hallucination detection harness
              for MR-2026-035
Difficulty: ★★★★☆ | Estimated time: 45 minutes

Task:
    Using the AWB Credit Document Analyser (MR-2026-035), build a
    systematic hallucination detection harness that:

    1. Runs the CDA against 20 synthetic credit packs with known
       ground-truth values.

    2. Measures field-level precision and recall.

    3. Flags any extraction where the model cites a page number
       that does NOT contain the figure in the source document.

    4. Produces a validation report suitable for submission as
       PRA SS1/23 evidence (model card format).

    Targets:
        - Field extraction accuracy >= 97.2%
          (AWB MR-2026-035 benchmark)
        - Hallucination rate (false citations) <= 0.5%

Regulatory context:
    PRA SS1/23 §5.3: Models must be validated against a labelled
    test set before production deployment.
    EU AI Act Art. 72: HIGH-RISK systems require post-market
    monitoring. This harness forms the monitoring baseline.

Author: AWB AI Programme (AWB-AI-2025)
Chapter: 2 — LLM Foundations for Financial Document Analysis
"""
from __future__ import annotations

import datetime
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ── Data structures ───────────────────────────────────────────────────────────


@dataclass
class GroundTruth:
    """Known correct values for a single credit pack."""

    document_id: str
    revenue_gbp: Optional[float]
    ebitda_gbp: Optional[float]
    net_debt_gbp: Optional[float]
    interest_cover_ratio: Optional[float]
    leverage_ratio: Optional[float]
    # Page numbers where each field appears in source doc
    revenue_page: Optional[int] = None
    ebitda_page: Optional[int] = None
    net_debt_page: Optional[int] = None
    interest_cover_page: Optional[int] = None
    leverage_page: Optional[int] = None


@dataclass
class HallucinationFlag:
    """Records a detected citation hallucination."""

    document_id: str
    field_name: str
    cited_page: int
    correct_page: Optional[int]
    extracted_value: Optional[float]
    true_value: Optional[float]
    severity: str  # "CITATION_ERROR" | "VALUE_ERROR" | "GHOST"


@dataclass
class ValidationReport:
    """
    PRA SS1/23 model validation report for MR-2026-035.

    This is the primary evidence document for the model card.
    """

    model_id: str = "MR-2026-035"
    model_name: str = "AWB Credit Document Analyser"
    validation_date: str = field(
        default_factory=lambda: datetime.date.today().isoformat()
    )
    test_set_size: int = 0
    total_fields_evaluated: int = 0
    correct_extractions: int = 0
    field_accuracy: float = 0.0
    hallucination_count: int = 0
    hallucination_rate: float = 0.0
    flags: list[HallucinationFlag] = field(default_factory=list)
    per_field_accuracy: dict[str, float] = field(
        default_factory=dict
    )
    target_accuracy_met: bool = False
    target_hallucination_met: bool = False
    pra_ss1_23_compliant: bool = False
    notes: str = ""


# ── TODO: Implement the harness ───────────────────────────────────────────────


def load_ground_truth(
    ground_truth_path: Path,
) -> list[GroundTruth]:
    """
    Load ground truth values from JSON file.

    TODO: Parse ground_truth.json from the data directory.
    Each entry should map to a GroundTruth dataclass.
    """
    raise NotImplementedError(
        "Implement ground truth loader."
    )


def check_citation_accuracy(
    extracted_page: Optional[int],
    true_page: Optional[int],
    document_text_pages: dict[int, str],
    extracted_value: Optional[float],
    field_name: str,
    document_id: str,
) -> Optional[HallucinationFlag]:
    """
    Verify that the cited page actually contains the extracted value.

    Args:
        extracted_page: Page number cited by the model.
        true_page: Correct page number from ground truth.
        document_text_pages: Dict mapping page_num -> page_text.
        extracted_value: The numeric value that was extracted.
        field_name: Name of the financial field.
        document_id: Source document identifier.

    Returns:
        HallucinationFlag if a citation error is detected,
        None if citation is valid.

    TODO:
        1. If extracted_page is None → return None (no citation)
        2. If extracted_page != true_page → CITATION_ERROR flag
        3. If page exists but value not found in text → GHOST flag
        4. Use string matching with ±2% tolerance for value check
    """
    raise NotImplementedError(
        "Implement citation accuracy checker."
    )


def run_validation_harness(
    test_documents: list[tuple[str, str]],
    ground_truth_list: list[GroundTruth],
    document_pages: dict[str, dict[int, str]],
) -> ValidationReport:
    """
    Run the full validation harness against the test set.

    Args:
        test_documents: List of (document_id, full_text) pairs.
        ground_truth_list: Ground truth for each document.
        document_pages: Per-document page text lookup.

    Returns:
        Completed ValidationReport for PRA SS1/23 submission.

    TODO:
        1. Import extract_financial_data from the CDA module.
        2. For each document, run extraction and compare to GT.
        3. Call check_citation_accuracy for each field.
        4. Compute per-field accuracy and overall hallucination rate.
        5. Set pra_ss1_23_compliant = True if both targets met.

    Targets:
        field_accuracy    >= 0.972  (97.2% AWB benchmark)
        hallucination_rate <= 0.005  (0.5% max)
    """
    raise NotImplementedError(
        "Implement the full validation harness."
    )


def format_pra_report(report: ValidationReport) -> str:
    """
    Format the ValidationReport as a PRA SS1/23 model card.

    Returns markdown string suitable for inclusion in the
    MR-2026-035 model card documentation.

    TODO: Format the report fields into a readable markdown
    table with pass/fail indicators for each metric.
    """
    raise NotImplementedError(
        "Implement PRA report formatter."
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    data_dir = (
        Path(__file__).parent.parent / "data"
    )
    print("AWB MR-2026-035 — Hallucination Detection Harness")
    print("=" * 52)
    print(f"Data directory: {data_dir}")
    print()
    print("Steps to complete this exercise:")
    print(
        "  1. Implement load_ground_truth()"
    )
    print(
        "  2. Implement check_citation_accuracy()"
    )
    print(
        "  3. Implement run_validation_harness()"
    )
    print(
        "  4. Implement format_pra_report()"
    )
    print()
    print(
        "Targets: accuracy >= 97.2% | "
        "hallucination rate <= 0.5%"
    )
    print()
    print(
        "Solution: "
        "github.com/lorvenio/ai-banking-risk-platform"
        "/chapter_02/solutions/"
    )
    sys.exit(0)
