# exercises/exercise_2.py | Exercise 14.2 starter
# Chapter 14 | AWB LLMOps
# Implement a prompt version upgrade with A/B testing
# and RAGAS validation for MR-2026-038.
#
# TASK:
# Part 1: Create a MINOR version bump for the
#   Regulatory Knowledge Assistant (MR-2026-038).
#   Add IFRS 9 staging classification as a new
#   output field. Bump 1.2.0 -> 1.3.0.
#
# Part 2: Run the RAGAS robustness test suite against
#   the new prompt. faithfulness >= 0.85 on golden set.
#
# Part 3: Configure A/B test routing:
#   - 10% to candidate version
#   - 90% to production version
#   - Auto-rollback if faithfulness < 0.80 in 3hr window
#
# SUCCESS CRITERION (pytest tests):
#   test_version_bumped_to_1_3_0
#   test_ragas_faithfulness_above_threshold
#   test_ab_proxy_routes_10pct_to_candidate
#   test_rollback_fires_below_threshold
#
# See solutions/ for reference implementation.
from __future__ import annotations
from pathlib import Path

# Use the prompt registry from llmops/
from chapter_14.llmops.prompt_registry import (
    PromptRegistry,
    PromptVersion,
    ChangeType,
)
from chapter_14.llmops.ragas_monitor import RAGASMonitor

SERVICE_ID = "MR-2026-038"
CURRENT_VERSION = "1.2.0"
NEW_VERSION = "1.3.0"   # MINOR bump

# TODO Part 1: Define the updated prompt text
# Add IFRS 9 staging classification as output field.
UPDATED_PROMPT = """
You are AWB's Regulatory Knowledge Assistant.
Given a compliance query, retrieve relevant regulatory
guidance and provide a cited answer.

Output JSON with fields:
- answer: str
- citations: list[str]
- confidence: float
- ifrs9_staging: str | null  # TODO: add this field

...
"""  # TODO: complete the prompt


def register_new_version(
    registry: PromptRegistry,
) -> PromptVersion:
    """TODO: Register v1.3.0 as MINOR change.

    Should use ChangeType.MINOR, set requires_mrc=False,
    ab_test_days=7, and set git_tag appropriately.
    """
    raise NotImplementedError


def run_ragas_validation(
    prompt_text: str,
    golden_set_path: Path,
) -> float:
    """TODO: Run RAGAS on golden test set.

    Should return mean faithfulness score.
    Raise AssertionError if below 0.85.
    """
    raise NotImplementedError


def configure_ab_test(
    service_id: str,
    candidate_version: str,
    production_version: str,
    candidate_pct: float = 0.10,
) -> dict:
    """TODO: Return A/B test routing config.

    Should return dict with keys:
    - candidate_version: str
    - production_version: str
    - candidate_pct: float
    - rollback_threshold: float (0.80)
    - window_hours: int (3)
    """
    raise NotImplementedError


if __name__ == "__main__":
    registry = PromptRegistry()
    version = register_new_version(registry)
    print(f"Registered: {SERVICE_ID} v{NEW_VERSION}")

    faithfulness = run_ragas_validation(
        UPDATED_PROMPT,
        Path("chapter_14/tests/golden_set.json"),
    )
    print(f"RAGAS faithfulness: {faithfulness:.3f}")

    config = configure_ab_test(
        SERVICE_ID,
        NEW_VERSION,
        CURRENT_VERSION,
    )
    print(f"A/B config: {config}")
