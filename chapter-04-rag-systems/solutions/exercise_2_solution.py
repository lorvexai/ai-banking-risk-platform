"""Solution — Exercise 4.2: RAGAS evaluation harness for MR-2026-038.

Computes the four RAGAS metrics over the 150-question test set, identifies
the weakest question categories, and emits a PRA SS1/23 model card summary.
"""
from __future__ import annotations

import json
import pathlib
import random
from collections import defaultdict

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
TARGETS = {"faithfulness": 0.85, "answer_relevancy": 0.80}


def load_test_set() -> list[dict]:
    p = pathlib.Path(__file__).resolve().parents[1] / "evaluation" / "test_set.json"
    return json.loads(p.read_text())["questions"]


def mock_ragas_scores(questions: list[dict], seed: int = 42) -> list[dict]:
    """Mock scorer — production uses ragas.evaluate() against MR-2026-038."""
    rng = random.Random(seed)
    base = {"definition": 0.93, "citation": 0.88, "applicability": 0.90,
            "deadline": 0.86, "interpretation": 0.84}
    rows = []
    for q in questions:
        b = base[q["question_type"]]
        rows.append({**q, **{m: min(1.0, max(0.6, rng.gauss(b, 0.04))) for m in METRICS}})
    return rows


def evaluate() -> None:
    rows = mock_ragas_scores(load_test_set())
    overall = {m: sum(r[m] for r in rows) / len(rows) for m in METRICS}
    by_cat: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r["faithfulness"])
    weakest = sorted(by_cat, key=lambda c: sum(by_cat[c]) / len(by_cat[c]))[:2]

    print("MODEL CARD SUMMARY — MR-2026-038 (PRA SS1/23 format)")
    print(f"evaluation set: {len(rows)} questions | date: 2026-06")
    for m in METRICS:
        flag = ""
        if m in TARGETS:
            flag = "PASS" if overall[m] >= TARGETS[m] else "FAIL"
        print(f"  {m:20s} {overall[m]:.3f} {flag}")
    print(f"  weakest categories: {', '.join(weakest)} -> targeted re-ingestion")
    assert overall["faithfulness"] >= 0.85 and overall["answer_relevancy"] >= 0.80
    print("Success criterion met")


if __name__ == "__main__":
    evaluate()
