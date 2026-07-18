"""Solution — Exercise 2.2: hallucination detection harness (MR-2026-035).

Runs the Credit Document Analyser against synthetic credit packs with
known ground truth; measures field-level precision/recall and flags any
extraction citing a page that does not contain the figure.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

FIELDS = ["revenue_gbp", "ebitda_gbp", "net_debt_gbp", "dscr", "facility_gbp"]


@dataclass(frozen=True)
class Extraction:
    pack_id: str
    field: str
    value: float
    cited_page: int


def make_ground_truth(n_packs: int = 20, seed: int = 22):
    rng = random.Random(seed)
    truth, page_index = {}, {}
    for i in range(n_packs):
        pid = f"PACK-{i:03d}"
        for f in FIELDS:
            value = round(rng.uniform(1, 60) * 1e6, 2)
            page = rng.randint(1, 40)
            truth[(pid, f)] = value
            page_index[(pid, f)] = page
    return truth, page_index


def mock_cda_run(truth, page_index, seed: int = 23) -> list[Extraction]:
    """Mock CDA: mostly correct, with seeded hallucinations."""
    rng = random.Random(seed)
    out = []
    for (pid, f), value in truth.items():
        roll = rng.random()
        if roll < 0.985:  # correct extraction, correct citation
            out.append(Extraction(pid, f, value, page_index[(pid, f)]))
        elif roll < 0.99:  # hallucinated value, wrong page
            out.append(Extraction(pid, f, value * rng.uniform(1.5, 3), 41))
        # else: field missed entirely (affects recall)
    return out


def evaluate() -> None:
    truth, page_index = make_ground_truth()
    extractions = mock_cda_run(truth, page_index)
    correct = [
        e for e in extractions
        if abs(e.value - truth[(e.pack_id, e.field)]) < 0.01
    ]
    page_hallucinations = [
        e for e in extractions if e.cited_page != page_index[(e.pack_id, e.field)]
    ]
    precision = len(correct) / len(extractions)
    recall = len(correct) / len(truth)
    print("PRA SS1/23 validation report — MR-2026-035 hallucination harness")
    print(f"packs: 20 | fields/pack: {len(FIELDS)} | extractions: {len(extractions)}")
    print(f"field-level precision: {precision:.3f}")
    print(f"field-level recall:    {recall:.3f}")
    print(f"page-citation hallucinations flagged: {len(page_hallucinations)}")
    assert precision >= 0.972, "below the 97.2% accuracy benchmark"
    print("Benchmark met: >= 97.2% field extraction accuracy")


if __name__ == "__main__":
    evaluate()
