"""Solution — Exercise 10.1: Credit Model Validation.

AWB Chapter 10 | Reference solution (do not peek until done!)
"""
from __future__ import annotations
import math
from exercises.ex_10_1_validate import (
    CreditModelValidator,
    ValidationMetrics,
    SCORES,
    OUTCOMES,
    REF_PROPS,
    _bin_proportions,
)


class SolCreditModelValidator(CreditModelValidator):
    """Reference implementation of all four metrics.

    Score convention: score = P(default).
    High score -> high default risk -> outcome=1.
    Sort descending so highest-risk applicants rank first.
    """

    def auc_roc(self) -> float:
        paired = sorted(
            zip(self.scores, self.outcomes),
            key=lambda x: x[0],
            reverse=True,           # highest default-risk first
        )
        n_pos = sum(self.outcomes)  # defaulters
        n_neg = len(self.outcomes) - n_pos
        if n_pos == 0 or n_neg == 0:
            return 0.5
        tp = fp = prev_tp = prev_fp = 0
        auc = 0.0
        prev_score = None
        for score, label in paired:
            if score != prev_score and prev_score is not None:
                auc += (fp - prev_fp) * (tp + prev_tp) / 2
                prev_tp, prev_fp = tp, fp
            if label == 1:
                tp += 1
            else:
                fp += 1
            prev_score = score
        auc += (fp - prev_fp) * (tp + prev_tp) / 2
        return auc / (n_pos * n_neg)

    def gini_coefficient(self) -> float:
        return 2 * self.auc_roc() - 1

    def ks_statistic(self) -> float:
        bads  = sorted(
            s for s, o in zip(self.scores, self.outcomes)
            if o == 1
        )
        goods = sorted(
            s for s, o in zip(self.scores, self.outcomes)
            if o == 0
        )
        n_bad, n_good = len(bads), len(goods)
        ks = 0.0
        for t in sorted(set(self.scores)):
            f_bad  = sum(1 for s in bads  if s <= t) / n_bad
            f_good = sum(1 for s in goods if s <= t) / n_good
            ks = max(ks, abs(f_bad - f_good))
        return ks

    def psi(self) -> float:
        actual = _bin_proportions(self.scores)
        eps = 1e-6
        return sum(
            (a - e) * math.log((a + eps) / (e + eps))
            for a, e in zip(actual, self.ref_props)
        )


if __name__ == "__main__":
    v = SolCreditModelValidator(SCORES, OUTCOMES, REF_PROPS)
    r = v.run()
    print(f"AUC-ROC : {r.auc_roc:.4f}")
    print(f"Gini    : {r.gini:.4f}")
    print(f"KS      : {r.ks_statistic:.4f}")
    print(f"PSI     : {r.psi:.4f}")
    print(f"All pass: {r.all_pass()}")
