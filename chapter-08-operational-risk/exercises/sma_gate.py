"""Exercise 8.1 — Add a Basel III event-type gate to the Op Loss pipeline.

Difficulty: 3/5 | Estimated time: 45 minutes

Extend the Op Loss Event Detection pipeline's Stage 1 keyword pre-filter
with a NEW Basel III event-type category of your choice, then re-run the
two-stage pipeline against the synthetic document sample below.

Success criterion: overall precision >= 0.85 while recall on your new
category reaches >= 0.80, and every extracted event carries a confidence
score and Basel category in the output schema.

Solution: solutions/sma_gate_solution.py
"""
from __future__ import annotations

import random
from dataclasses import dataclass

# Basel III level-1 operational risk event types already gated in Stage 1.
STAGE1_KEYWORDS: dict[str, list[str]] = {
    "internal_fraud": ["unauthorised trading", "employee theft", "forged"],
    "external_fraud": ["phishing", "card fraud", "cyber attack"],
    "execution_delivery": ["settlement fail", "data entry error", "missed deadline"],
    # TODO: add ONE new Basel III category with 3+ keywords, e.g.
    # "clients_products_business_practices", "damage_physical_assets",
    # "business_disruption", or "employment_practices".
}


@dataclass(frozen=True)
class OpLossEvent:
    doc_id: str
    basel_category: str
    confidence: float


def make_sample(n: int = 200, seed: int = 8) -> list[tuple[str, str, str]]:
    """(doc_id, text, true_category) triples; ~25% are non-events."""
    rng = random.Random(seed)
    phrases = {
        "internal_fraud": "unauthorised trading position concealed by desk head",
        "external_fraud": "customer accounts drained via phishing campaign",
        "execution_delivery": "settlement fail on gilt trade due to data entry error",
        "business_disruption": "core banking outage halted payment processing",
        "none": "quarterly town hall scheduled for the Bristol office",
    }
    cats = list(phrases)
    return [
        (f"DOC-{i:04d}", phrases[c], c)
        for i, c in enumerate(rng.choices(cats, weights=[2, 2, 2, 2, 3], k=n))
    ]


def stage1_gate(text: str) -> str | None:
    """TODO: return the Basel category whose keywords match, else None."""
    raise NotImplementedError("Exercise 8.1")


def stage2_extract(doc_id: str, text: str, category: str) -> OpLossEvent:
    """TODO: emit OpLossEvent with a confidence score (mock LLM stage)."""
    raise NotImplementedError("Exercise 8.1")


def run_pipeline() -> None:
    """TODO: run both stages over make_sample(); print precision/recall."""
    raise NotImplementedError("Exercise 8.1")


if __name__ == "__main__":
    run_pipeline()
