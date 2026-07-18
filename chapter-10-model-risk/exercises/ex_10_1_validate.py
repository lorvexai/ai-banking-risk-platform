"""Exercise 10.1 — Credit Model Validation.

AWB Chapter 10 | Difficulty: ★★★☆☆ | ~35 minutes

TASK
----
Complete the CreditModelValidator class below so that
all four validation metrics are computed correctly for
the synthetic AWB loan data provided.

SCORE CONVENTION
----------------
Scores represent P(default): a HIGH score means the
borrower is HIGH RISK.  Defaulters therefore have high
scores and non-defaulters have low scores.

AWB PRA SS1/23 thresholds (must all PASS):
  AUC-ROC           >= 0.750
  Gini coefficient  >= 0.700
  KS statistic      >= 0.300
  PSI (vs reference) < 0.100

Run your solution:
  cd chapter_10
  pytest exercises/ex_10_1_validate.py -v

Solution:
  github.com/lorvenio/ai-banking-risk-platform
  /chapter_10/solutions/sol_10_1_validate.py
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass


# ── Synthetic AWB loan data (fixed seed for reproducibility) ──
# Score = P(default): high score -> high credit risk

def _make_dataset(
    n: int = 2_000,
    seed: int = 42,
) -> tuple[list[float], list[int]]:
    """Return (scores, outcomes) for a synthetic portfolio.

    Scores  : float in [0, 1] — higher = higher default risk
    Outcomes: 0 = no default, 1 = default within 12 months
    """
    rng = random.Random(seed)
    scores, outcomes = [], []
    for _ in range(n):
        default = rng.random() < 0.10   # 10% base default rate
        if default:
            # Defaulters cluster around HIGH scores (high risk)
            score = max(0.0, min(1.0, rng.gauss(0.72, 0.15)))
        else:
            score = max(0.0, min(1.0, rng.gauss(0.28, 0.18)))
        scores.append(score)
        outcomes.append(int(default))
    return scores, outcomes


SCORES, OUTCOMES = _make_dataset()

# Reference score distribution (training population — 20 bins)
_REF_SEED_SCORES, _ = _make_dataset(n=5_000, seed=0)
_BINS = 20


def _bin_proportions(
    scores: list[float],
) -> list[float]:
    """Return proportion of scores in each of 20 equal bins."""
    counts = [0] * _BINS
    for s in scores:
        idx = min(int(s * _BINS), _BINS - 1)
        counts[idx] += 1
    total = len(scores)
    return [c / total for c in counts]


REF_PROPS = _bin_proportions(_REF_SEED_SCORES)


# ── Data class for results ─────────────────────────────────────

@dataclass
class ValidationMetrics:
    gini: float
    ks_statistic: float
    auc_roc: float
    psi: float

    GINI_MIN: float = 0.700
    KS_MIN:   float = 0.300
    AUC_MIN:  float = 0.750
    PSI_WARN: float = 0.100

    def pass_fail(self) -> dict[str, bool]:
        return {
            "gini": self.gini >= self.GINI_MIN,
            "ks_statistic": self.ks_statistic >= self.KS_MIN,
            "auc_roc": self.auc_roc >= self.AUC_MIN,
            "psi_ok": self.psi < self.PSI_WARN,
        }

    def all_pass(self) -> bool:
        return all(self.pass_fail().values())


# ── YOUR IMPLEMENTATION ────────────────────────────────────────

class CreditModelValidator:
    """AWB PRA SS1/23 credit model validation suite.

    Complete each method marked TODO.
    Do NOT change the method signatures.

    Score convention: score = P(default).
    Higher score = higher default risk = outcome likely 1.
    """

    def __init__(
        self,
        scores: list[float],
        outcomes: list[int],
        ref_props: list[float],
    ) -> None:
        self.scores = scores
        self.outcomes = outcomes
        self.ref_props = ref_props

    # ── TODO 1 ────────────────────────────────────────────────
    def auc_roc(self) -> float:
        """Compute Area Under the ROC Curve (AUC-ROC).

        Sort by score DESCENDING (highest risk first).
        Treat outcome=1 (default) as the positive class.
        Apply the trapezoidal rule to accumulate the AUC.
        AUC must be in [0.5, 1.0] for a useful model.
        """
        # TODO: implement AUC-ROC
        raise NotImplementedError

    # ── TODO 2 ────────────────────────────────────────────────
    def gini_coefficient(self) -> float:
        """Gini = 2 * AUC-ROC - 1."""
        # TODO: implement using self.auc_roc()
        raise NotImplementedError

    # ── TODO 3 ────────────────────────────────────────────────
    def ks_statistic(self) -> float:
        """Kolmogorov-Smirnov statistic.

        KS = max |F_bad(t) - F_good(t)| over all thresholds t,
        where F_bad and F_good are the empirical CDFs of scores
        for defaulters (outcome=1) and non-defaulters (outcome=0).
        """
        # TODO: implement KS statistic
        raise NotImplementedError

    # ── TODO 4 ────────────────────────────────────────────────
    def psi(self) -> float:
        """Population Stability Index vs reference distribution.

        PSI = sum over bins of (A_i - E_i) * ln(A_i / E_i)
        where A_i = actual proportion (current scores)
              E_i = expected proportion (ref_props)

        Add epsilon=1e-6 to avoid log(0).
        Use the global _bin_proportions() helper.
        """
        # TODO: implement PSI
        raise NotImplementedError

    # ── Orchestration (do not modify) ─────────────────────────
    def run(self) -> ValidationMetrics:
        return ValidationMetrics(
            gini=round(self.gini_coefficient(), 4),
            ks_statistic=round(self.ks_statistic(), 4),
            auc_roc=round(self.auc_roc(), 4),
            psi=round(self.psi(), 4),
        )


# ── Tests ─────────────────────────────────────────────────────

def test_auc_roc_above_threshold() -> None:
    v = CreditModelValidator(SCORES, OUTCOMES, REF_PROPS)
    result = v.run()
    assert result.auc_roc >= 0.750, (
        f"AUC-ROC {result.auc_roc:.3f} below threshold 0.750"
    )


def test_gini_above_threshold() -> None:
    v = CreditModelValidator(SCORES, OUTCOMES, REF_PROPS)
    result = v.run()
    assert result.gini >= 0.700, (
        f"Gini {result.gini:.3f} below threshold 0.700"
    )


def test_ks_above_threshold() -> None:
    v = CreditModelValidator(SCORES, OUTCOMES, REF_PROPS)
    result = v.run()
    assert result.ks_statistic >= 0.300, (
        f"KS {result.ks_statistic:.3f} below threshold 0.300"
    )


def test_psi_below_warning() -> None:
    v = CreditModelValidator(SCORES, OUTCOMES, REF_PROPS)
    result = v.run()
    assert result.psi < 0.100, (
        f"PSI {result.psi:.4f} above warning threshold 0.10"
    )


def test_all_pass() -> None:
    v = CreditModelValidator(SCORES, OUTCOMES, REF_PROPS)
    result = v.run()
    assert result.all_pass(), (
        f"Not all metrics pass: {result.pass_fail()}"
    )


def test_gini_equals_2_auc_minus_1() -> None:
    v = CreditModelValidator(SCORES, OUTCOMES, REF_PROPS)
    result = v.run()
    expected = round(2 * result.auc_roc - 1, 4)
    assert abs(result.gini - expected) < 0.001, (
        "Gini must equal 2*AUC-ROC - 1"
    )
