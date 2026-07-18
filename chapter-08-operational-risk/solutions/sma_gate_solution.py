"""Solution — Exercise 8.1: Basel III event-type gate (business_disruption)."""
from __future__ import annotations

import random
from dataclasses import dataclass

STAGE1_KEYWORDS: dict[str, list[str]] = {
    "internal_fraud": ["unauthorised trading", "employee theft", "forged"],
    "external_fraud": ["phishing", "card fraud", "cyber attack"],
    "execution_delivery": ["settlement fail", "data entry error", "missed deadline"],
    # New category added for the exercise:
    "business_disruption": ["outage", "system failure", "halted payment"],
}


@dataclass(frozen=True)
class OpLossEvent:
    doc_id: str
    basel_category: str
    confidence: float


def make_sample(n: int = 200, seed: int = 8) -> list[tuple[str, str, str]]:
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
    for category, keywords in STAGE1_KEYWORDS.items():
        if any(k in text for k in keywords):
            return category
    return None


def stage2_extract(doc_id: str, text: str, category: str) -> OpLossEvent:
    # Mock LLM confirmation stage: confidence from keyword density.
    hits = sum(text.count(k) for k in STAGE1_KEYWORDS[category])
    return OpLossEvent(doc_id, category, confidence=min(0.99, 0.75 + 0.1 * hits))


def run_pipeline() -> None:
    sample = make_sample()
    events = []
    for doc_id, text, _true in sample:
        cat = stage1_gate(text)
        if cat is not None:
            events.append(stage2_extract(doc_id, text, cat))
    truth = {d: t for d, _x, t in sample}
    tp = sum(1 for e in events if truth[e.doc_id] == e.basel_category)
    precision = tp / len(events)
    bd_truth = [d for d, _x, t in sample if t == "business_disruption"]
    bd_found = [e.doc_id for e in events if e.basel_category == "business_disruption"]
    recall_bd = len(set(bd_found) & set(bd_truth)) / len(bd_truth)
    print(f"events extracted: {len(events)}")
    print(f"overall precision: {precision:.3f} (target >= 0.85)")
    print(f"business_disruption recall: {recall_bd:.3f} (target >= 0.80)")
    assert precision >= 0.85 and recall_bd >= 0.80
    print("Success criterion met")


if __name__ == "__main__":
    run_pipeline()
