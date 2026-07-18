"""
exercises/exercise_2.py
Exercise 4.2: Build a RAGAS Evaluation Harness

Exercise: Run a full RAGAS evaluation against MR-2026-038 and produce a
PRA SS1/23 model card summary.
Difficulty: ★★★★☆ | Estimated time: 45 minutes

Task:
  1. Load the 150-question test set from evaluation/test_set.json
     (or use the 10-question sample provided in this file for quick testing)
  2. Run all four RAGAS metrics against MR-2026-038
  3. Identify which question category scores lowest
  4. Generate a model card summary in the format used by Section 4.8

Success criterion:
  faithfulness >= 0.85 and answer_relevancy >= 0.80 on the full test set.

Solution: github.com/lorvenio/ai-banking-risk-platform/chapter_04/solutions/

Note on API keys:
  The AWBRagasEvaluator supports use_mock=True for offline testing.
  Set use_mock=False and provide a GOOGLE_API_KEY to run against the
  real MR-2026-038 RKA engine.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

# Make sure imports work from exercises/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.ragas_evaluator import (
    AWBRagasEvaluator,
    RAGASThresholds,
    ValidationResult,
)


# ── Sample test set for quick offline testing ─────────────────────────────────
# The full 150-question test set lives at evaluation/test_set.json
# This 10-question sample lets you test your harness without loading all data.

SAMPLE_TEST_SET = [
    {
        "question": "What is the PRA SS1/23 risk rating for MR-2026-038?",
        "expected_answer": "MR-2026-038 (AWB Regulatory Knowledge Assistant) "
                           "has a LOW risk rating under PRA SS1/23 because it "
                           "is decision-support only and takes no autonomous action.",
        "category": "credit",
        "question_id": "q001",
    },
    {
        "question": "What are the CRR3 Article 153 supervisory slotting categories?",
        "expected_answer": "CRR3 Article 153 defines five supervisory slotting "
                           "categories for specialised lending: Strong, Good, "
                           "Satisfactory, Weak, and Default.",
        "category": "capital",
        "question_id": "q002",
    },
    {
        "question": "When must AWB submit its CoRep LCR return?",
        "expected_answer": "The LCR CoRep return (template C 72.00) must be "
                           "submitted monthly, within 15 business days of "
                           "the reporting reference date.",
        "category": "reporting",
        "question_id": "q003",
    },
    {
        "question": "What is the FCA Consumer Duty PS22/9 outcome for products?",
        "expected_answer": "FCA PS22/9 requires that products and services "
                           "deliver good outcomes for retail customers. Products "
                           "must be designed to meet the needs of the target market.",
        "category": "credit",
        "question_id": "q004",
    },
    {
        "question": "What is the minimum leverage ratio under CRR3 Article 429?",
        "expected_answer": "The minimum leverage ratio under CRR3 Article 429 "
                           "is 3% of Tier 1 Capital to Total Leverage Exposure "
                           "Measure. G-SIBs must maintain 3.5%.",
        "category": "capital",
        "question_id": "q005",
    },
    {
        "question": "How does DORA classify AWB's AI systems?",
        "expected_answer": "Under DORA Article 9, AI systems used in regulatory "
                           "decision support must be registered as ICT assets "
                           "in the ICT asset inventory and subject to ICT "
                           "risk management requirements.",
        "category": "reporting",
        "question_id": "q006",
    },
    {
        "question": "What is the EU AI Act Annex III classification for credit AI?",
        "expected_answer": "EU AI Act Annex III Section 5(b) classifies AI "
                           "systems used for creditworthiness assessment as "
                           "high-risk. The conformity assessment obligation "
                           "takes effect 2 August 2026.",
        "category": "capital",
        "question_id": "q007",
    },
    {
        "question": "What LCR components does CRR3 require AWB to report?",
        "expected_answer": "CRR3 requires reporting of High Quality Liquid Assets "
                           "(HQLA), net liquidity outflows over 30 days, and "
                           "the resulting LCR ratio. Template C 72.00 covers "
                           "28 data elements.",
        "category": "reporting",
        "question_id": "q008",
    },
    {
        "question": "What is the PRA SS1/23 requirement for model audit trails?",
        "expected_answer": "PRA SS1/23 requires that model outputs be traceable "
                           "to documented sources and retained for audit. AWB "
                           "maintains a 7-year audit log for MR-2026-038 "
                           "query responses under FCA COBS 9.",
        "category": "credit",
        "question_id": "q009",
    },
    {
        "question": "What RAGAS faithfulness threshold does AWB use for MR-2026-038?",
        "expected_answer": "AWB's faithfulness threshold for MR-2026-038 is 0.85. "
                           "A sustained drop below 0.80 in production monitoring "
                           "triggers a formal model performance review.",
        "category": "credit",
        "question_id": "q010",
    },
]


def write_sample_test_set() -> str:
    """Write the sample test set to a temporary file and return the path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix="_test_set.json", delete=False
    )
    json.dump(SAMPLE_TEST_SET, tmp, indent=2)
    tmp.close()
    return tmp.name


# ── TODO: Implement the evaluation pipeline ───────────────────────────────────

def run_evaluation(
    test_set_path: str,
    use_mock: bool = True,
    rag_engine=None,
) -> ValidationResult:
    """
    TODO: Run a full RAGAS evaluation and return ValidationResult.

    Steps:
      1. Create an AWBRagasEvaluator with AWB's standard thresholds
      2. Call evaluator.validate() with model_id="MR-2026-038"
      3. Return the ValidationResult

    Args:
        test_set_path: Path to JSON test set file.
        use_mock: If True, uses mock scores (no API key needed).
        rag_engine: Live RAG engine (required when use_mock=False).

    Returns:
        ValidationResult for model card / audit log.
    """
    raise NotImplementedError("TODO: implement run_evaluation()")


def identify_weakest_category(result: ValidationResult) -> str:
    """
    TODO: Return the category name with the lowest faithfulness score.

    Use result.by_category to compare per-category scores.
    If by_category is empty, return "unknown".

    Args:
        result: ValidationResult from run_evaluation().

    Returns:
        Category name ("credit", "capital", or "reporting").
    """
    raise NotImplementedError("TODO: implement identify_weakest_category()")


def print_model_card(result: ValidationResult) -> None:
    """
    TODO: Print a PRA SS1/23 model card summary.

    Expected output format:
      ========================================
      AWB MODEL CARD — MR-2026-038
      AWB Regulatory Knowledge Assistant
      ========================================
      Status:             PASS / FAIL
      Evaluated:          2026-03-27
      Test set size:      150 questions

      RAGAS Scores:
        Faithfulness:       0.XXX  (threshold: 0.85)  ✅ / ❌
        Answer Relevancy:   0.XXX  (threshold: 0.80)  ✅ / ❌
        Context Precision:  0.XXX  (threshold: 0.75)  ✅ / ❌
        Context Recall:     0.XXX  (threshold: 0.70)  ✅ / ❌

      By Category:
        Credit:    faithfulness=0.XXX
        Capital:   faithfulness=0.XXX
        Reporting: faithfulness=0.XXX

      Weakest category: XXXX
      ========================================

    Args:
        result: ValidationResult from run_evaluation().
    """
    raise NotImplementedError("TODO: implement print_model_card()")


# ── Main runner ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Chapter 4 — Exercise 4.2: RAGAS Evaluation Harness")
    print("=" * 50)

    # Use the sample test set (10 questions) for quick offline testing
    # Replace with "evaluation/test_set.json" for the full 150-question run
    test_set = write_sample_test_set()
    print(f"Using sample test set: {test_set}")
    print("(Set use_mock=False and provide GOOGLE_API_KEY for live run)")
    print()

    # Step 1: Run evaluation
    try:
        result = run_evaluation(test_set, use_mock=True)
        print(f"✅ Evaluation complete: {result.summary()}")
    except NotImplementedError:
        print("❌ run_evaluation() not yet implemented")
        exit(1)

    # Step 2: Identify weakest category
    try:
        weak = identify_weakest_category(result)
        print(f"Weakest category: {weak}")
    except NotImplementedError:
        print("❌ identify_weakest_category() not yet implemented")

    # Step 3: Print model card
    print()
    try:
        print_model_card(result)
    except NotImplementedError:
        print("❌ print_model_card() not yet implemented")

    print("\nSolution at:")
    print("github.com/lorvenio/ai-banking-risk-platform/chapter_04/solutions/")
