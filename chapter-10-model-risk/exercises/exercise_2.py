"""Exercise 10.2 — Build the PSI Monthly Monitoring Job.

Difficulty: 3/5 | Estimated time: 30 minutes

Using the starter below, build the monthly job that calculates the
Population Stability Index for a registered model's score distribution
and raises a revalidation trigger when PSI exceeds 0.2 (Section 10.6).

Companion starter (registry wiring): ex_10_2_registry.py
Solution: solutions/ (ex_10_2 solution files)
"""
from __future__ import annotations

import math
import random

BINS = 10


def make_scores(shift: float = 0.0, n: int = 5_000, seed: int = 10) -> list[float]:
    rng = random.Random(seed)
    return [min(0.999, max(0.001, rng.gauss(0.45 + shift, 0.15))) for _ in range(n)]


def bin_shares(scores: list[float]) -> list[float]:
    counts = [0] * BINS
    for s in scores:
        counts[min(BINS - 1, int(s * BINS))] += 1
    return [max(c / len(scores), 1e-6) for c in counts]


def psi(expected: list[float], actual: list[float]) -> float:
    """TODO: implement PSI = sum((a - e) * ln(a / e)) over the bins."""
    raise NotImplementedError("Exercise 10.2")


def monthly_job() -> None:
    """TODO: compare the development baseline against this month's scores;
    print PSI and raise/print a REVALIDATION_TRIGGER event when PSI > 0.2."""
    raise NotImplementedError("Exercise 10.2")


if __name__ == "__main__":
    baseline = bin_shares(make_scores())
    drifted = bin_shares(make_scores(shift=0.12, seed=11))
    print(f"PSI (stable):  {psi(baseline, bin_shares(make_scores(seed=12))):.3f}")
    print(f"PSI (drifted): {psi(baseline, drifted):.3f}  -> expect > 0.2 trigger")
