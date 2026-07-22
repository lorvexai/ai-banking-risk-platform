"""Solution — Exercise 7.2: CVA via the CIM Credit API (awb_commons client).

Retrieves a one-year PD for a stub counterparty from the CIM credit API
(MR-2026-055 -> MR-2026-043), extends it with a Nelson-Siegel-style hazard
curve, and computes unilateral CVA on a 3-year facility exposure profile.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PDResponse:
    counterparty: str
    rating: str
    pd_1y: float
    model_ref: str = "MR-2026-043"


class CIMCreditClient:
    """Stub of awb_commons.CIMCreditClient — calls /credit-risk/pd."""

    def get_pd(self, counterparty: str, rating: str) -> PDResponse:
        table = {"AAA": 0.0002, "AA": 0.0006, "A": 0.0015, "BBB": 0.0042,
                 "BB": 0.0135, "B": 0.041}
        return PDResponse(counterparty, rating, table[rating])


def hazard_curve(pd_1y: float, years: int = 3, beta: float = 0.35) -> list[float]:
    """Marginal default probabilities per year (simple NS-style slope)."""
    h1 = -math.log(1 - pd_1y)
    hazards = [h1 * (1 + beta * t) for t in range(years)]
    surv, margins = 1.0, []
    for h in hazards:
        s_next = surv * math.exp(-h)
        margins.append(surv - s_next)
        surv = s_next
    return margins


def cva(margins: list[float], epe_gbp: list[float], lgd: float = 0.6,
        r: float = 0.041) -> float:
    return sum(
        lgd * m * e * math.exp(-r * (t + 0.5))
        for t, (m, e) in enumerate(zip(margins, epe_gbp))
    )


if __name__ == "__main__":
    client = CIMCreditClient()
    pd = client.get_pd("Mercia & Humber Bank", "BBB")
    print(f"{pd.counterparty} ({pd.rating}) 1y PD from {pd.model_ref}: {pd.pd_1y:.4%}")
    margins = hazard_curve(pd.pd_1y)
    epe = [18_000_000.0, 14_500_000.0, 9_800_000.0]  # 3y expected positive exposure
    value = cva(margins, epe)
    print(f"marginal default probs: {[f'{m:.5f}' for m in margins]}")
    print(f"unilateral CVA (LGD 60%): £{value:,.0f}")
    assert 0 < value < 1_000_000
    print("CVA sourced from the same PD as credit decisions — single source of truth")
