"""
chapter_07/exercises/var_exercise.py
Exercise 7.1: Monte Carlo VaR for a 5-Asset Portfolio

Difficulty: ★★★☆☆ | Estimated time: 45 minutes

Task: Implement a Monte Carlo VaR engine for AWB's
5-asset portfolio (FTSE 100, GBP/USD, 10Y Gilt,
LIBOR swap, FTSE 100 call option).

Your implementation should:
  1. Build a 5x5 correlation matrix from AWB data
  2. Generate 10,000 correlated scenarios
  3. Calculate 1-day 99% VaR and ES(99)
  4. Produce component VaR by asset class

Target: VaR(99) within 10% of reference answer.
Note: Option uses delta approximation (hint below).

Complete solution:
  github.com/lorvenio/ai-banking-risk-platform/
  chapter_07/solutions/

AWB | awb_commons | MR-2026-046 exercise
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict


# ── AWB 5-Asset Portfolio (June 2026) ───────────────────────
# Positions in GBP sensitivity units (DV01 / delta)
ASSET_NAMES = [
    "FTSE100_EQ",
    "GBPUSD_FX",
    "GILT_10Y",
    "LIBOR_SWAP",
    "FTSE100_CALL",
]

# Daily volatilities (annualised / sqrt(252))
DAILY_VOLS = np.array([
    0.0120,   # FTSE 100 equity: 19% annual
    0.0065,   # GBP/USD FX: 10.3% annual
    0.0055,   # 10Y Gilt: 8.7% annual
    0.0040,   # LIBOR swap: 6.3% annual
    0.0095,   # FTSE call delta: proxy vol
])

# Correlation matrix (AWB internal, June 2026)
CORRELATION = np.array([
    [1.00,  0.25,  0.30,  0.20,  0.85],  # FTSE100
    [0.25,  1.00,  0.10,  0.05,  0.20],  # GBPUSD
    [0.30,  0.10,  1.00,  0.75,  0.25],  # Gilt
    [0.20,  0.05,  0.75,  1.00,  0.15],  # Swap
    [0.85,  0.20,  0.25,  0.15,  1.00],  # Call
])

# Position sensitivities (GBP per unit move)
# Equity: £ delta | FX: £ notional | Rates: DV01
SENSITIVITIES = np.array([
    180_000,   # FTSE 100 long equity: £180K delta
    95_000,    # GBP/USD long: £95K
    -45_000,   # Gilt short: -£45K DV01
    30_000,    # Swap receive-fixed: £30K DV01
    22_000,    # FTSE call: £22K delta (hint: this
               # uses delta approx — see below)
])

N_SCENARIOS = 10_000


@dataclass
class PortfolioVaRResult:
    """5-asset portfolio VaR output."""

    var_95_gbp: float
    var_99_gbp: float
    es_99_gbp: float
    component_var: Dict[str, float]
    n_scenarios: int


def calculate_var() -> PortfolioVaRResult:
    """
    Calculate Monte Carlo VaR for AWB 5-asset portfolio.

    HINT for FTSE 100 call option:
      Delta approximation: dP ≈ delta * dS
      The option delta (0.65) is already embedded in
      the SENSITIVITIES array above. You do NOT need
      to reprice the option — treat it as a linear
      position for this exercise.

    Returns:
        PortfolioVaRResult with VaR 95/99 and ES
    """
    # TODO: Step 1 — Cholesky decompose CORRELATION
    # L = np.linalg.cholesky(CORRELATION)

    # TODO: Step 2 — Generate N_SCENARIOS standard
    # normals, apply Cholesky, scale by DAILY_VOLS
    # z = np.random.standard_normal((N_SCENARIOS, 5))
    # corr_z = z @ L.T
    # scenarios = corr_z * DAILY_VOLS

    # TODO: Step 3 — Calculate portfolio P&L
    # pnl = scenarios @ SENSITIVITIES

    # TODO: Step 4 — Calculate VaR 95, VaR 99, ES
    # var_95 = -np.percentile(pnl, 5)
    # var_99 = -np.percentile(pnl, 1)
    # tail = pnl[pnl <= np.percentile(pnl, 1)]
    # es_99 = -tail.mean()

    # TODO: Step 5 — Component VaR
    # corr(factor_i, portfolio) * VaR_i per factor

    raise NotImplementedError(
        "Complete the TODO steps above."
    )


if __name__ == "__main__":
    np.random.seed(42)
    result = calculate_var()
    print(
        f"VaR 99: £{result.var_99_gbp:,.0f}\n"
        f"ES  99: £{result.es_99_gbp:,.0f}\n"
        f"Component VaR: {result.component_var}"
    )
