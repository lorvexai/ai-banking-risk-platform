"""AWB Corporate PD — CRR3 Art. 153 IRB RWA Calculator.

Implements:
  - CRR3 Art. 153: IRB RWA formula for corporate exposures
  - CRR3 Art. 160: PD floor 0.03%
  - CRR3 Art. 161: LGD floor 25% (unsecured corporate F-IRB)
  - CRR3 Art. 465: Output floor (72.5% of SA RWA, phased in)

Source: CRR3 Regulation (EU) 2024/1623.

Used by:
  - Chapter 3 RWAForecastAgent
  - Chapter 11 COREP C 07.00 generator

Output floor phase-in schedule (configure via env var):
  2025: 50% | 2026: 55% | 2027: 60% | 2028+: 72.5%
"""
from __future__ import annotations

import logging
import math
import os
from datetime import datetime

from awb_commons.schemas import RWAResult

log = logging.getLogger(__name__)

# AWB Tier 1 capital ratio (minimum CET1 + conservation buffer)
TIER1_RATIO = float(os.getenv("AWB_TIER1_RATIO", "0.105"))

# CRR3 Art. 465 — output floor phase-in
# Override via env: CRR3_FLOOR_RATE=0.55 for 2026
_DEFAULT_FLOOR = 0.50   # 2025 transitional rate
CRR3_FLOOR_RATE = float(
    os.getenv("CRR3_FLOOR_RATE", str(_DEFAULT_FLOOR))
)

# CRR3 Art. 161 — F-IRB LGD floors for corporate
LGD_FLOOR_UNSECURED = 0.25
LGD_FLOOR_SECURED   = 0.20


class CRR3RWACalculator:
    """CRR3 Art. 153 IRB RWA with output floor (Art. 465).

    Usage::

        calc = CRR3RWACalculator()               # 2025: 50% floor
        calc = CRR3RWACalculator(floor_rate=0.725) # 2028: full floor

        result = calc.calculate(
            facility_id="F-1234",
            pd=0.0215, lgd=0.45, ead=8_500_000,
            maturity=3.5, sa_rwa=6_800_000,
        )
        print(result.rwa_effective)  # max(IRB RWA, floor RWA)
        print(result.floor_binds)    # True if floor is binding
    """

    def __init__(
        self, floor_rate: float = CRR3_FLOOR_RATE
    ) -> None:
        self.floor_rate = floor_rate

    def calculate(
        self,
        facility_id: str,
        pd: float,
        lgd: float,
        ead: float,
        maturity: float,
        sa_rwa: float,
        secured: bool = False,
    ) -> RWAResult:
        """Compute IRB RWA with CRR3 output floor.

        Args:
            facility_id: AWB facility reference.
            pd:          Calibrated PD from MR-2026-040 (0–1).
            lgd:         Loss Given Default (0–1).
            ead:         Exposure at Default (£).
            maturity:    Remaining contractual maturity (years).
            sa_rwa:      Standardised Approach RWA (£).
                         Required for output floor calculation.
            secured:     True for property-secured exposures
                         (lower LGD floor applies).

        Returns:
            RWAResult with IRB, SA, floor, and effective RWA.

        Notes:
            PD floored at 0.0003 (CRR3 Art. 160).
            LGD floored at 0.25 unsecured / 0.20 secured
            (CRR3 Art. 161 F-IRB).
        """
        # Apply regulatory floors
        pd  = max(pd, 0.0003)       # CRR3 Art. 160
        lgd = max(
            lgd,
            LGD_FLOOR_SECURED if secured else LGD_FLOOR_UNSECURED
        )

        rho = self._correlation(pd)
        b   = self._maturity_adj_b(pd)
        k   = self._capital_requirement(pd, lgd, rho, b, maturity)

        rwa_irb   = ead * 1.06 * k
        rwa_floor = sa_rwa * self.floor_rate
        floor_binds = rwa_irb < rwa_floor
        rwa_eff   = max(rwa_irb, rwa_floor)
        cap_req   = rwa_eff * TIER1_RATIO

        log.debug(
            "RWA fac=%s pd=%.4f k=%.4f irb=£%.0f "
            "floor=£%.0f eff=£%.0f binds=%s",
            facility_id, pd, k,
            rwa_irb, rwa_floor, rwa_eff, floor_binds,
        )
        return RWAResult(
            facility_id   = facility_id,
            pd            = pd,
            lgd           = lgd,
            ead           = ead,
            maturity      = maturity,
            rwa_irb       = round(rwa_irb, 2),
            rwa_sa        = round(sa_rwa, 2),
            rwa_floor     = round(rwa_floor, 2),
            rwa_effective = round(rwa_eff, 2),
            floor_binds   = floor_binds,
            capital_req   = round(cap_req, 2),
            floor_rate    = self.floor_rate,
        )

    def batch_calculate(
        self,
        facilities: list[dict],
    ) -> list[RWAResult]:
        """Score a portfolio batch for COREP C 07.00.

        Args:
            facilities: List of dicts with keys:
                facility_id, pd, lgd, ead, maturity, sa_rwa
                (optional: secured)
        """
        return [self.calculate(**f) for f in facilities]

    def output_floor_impact(
        self, results: list[RWAResult]
    ) -> dict:
        """Summarise output floor impact across a portfolio.

        Returns dict with:
          total_rwa_irb:     sum of pure IRB RWA (£)
          total_rwa_eff:     sum of effective RWA (£)
          floor_addition:    total RWA added by floor (£)
          floor_bound_count: number of floor-bound facilities
          floor_bound_pct:   % of portfolio floor-bound
        """
        total_irb = sum(r.rwa_irb for r in results)
        total_eff = sum(r.rwa_effective for r in results)
        bound_n   = sum(1 for r in results if r.floor_binds)
        return {
            "total_rwa_irb":     round(total_irb, 2),
            "total_rwa_eff":     round(total_eff, 2),
            "floor_addition":    round(total_eff - total_irb, 2),
            "floor_bound_count": bound_n,
            "floor_bound_pct":   round(
                bound_n / len(results) * 100, 1
            ) if results else 0.0,
        }

    # ── CRR3 Art. 153 formula implementation ─────────────────────

    def _correlation(self, pd: float) -> float:
        """Asset correlation ρ — CRR3 Art. 153(1)."""
        e50 = math.exp(-50 * pd)
        base = 1 - math.exp(-50)
        return 0.12 * (1 - e50) / base + 0.24 * (
            1 - (1 - e50) / base
        )

    def _maturity_adj_b(self, pd: float) -> float:
        """Maturity adjustment b — CRR3 Art. 153(1)."""
        return (0.11852 - 0.05478 * math.log(pd)) ** 2

    def _capital_requirement(
        self,
        pd: float,
        lgd: float,
        rho: float,
        b: float,
        maturity: float,
    ) -> float:
        """K — capital requirement formula — CRR3 Art. 153(1).

        K = [LGD × N((N⁻¹(PD) + √ρ × N⁻¹(0.999)) / √(1−ρ))
             − PD × LGD]
            × (1 + (M − 2.5) × b) / (1 − 1.5 × b)
        """
        from scipy.stats import norm

        n_inv_pd  = norm.ppf(pd)
        n_inv_999 = norm.ppf(0.999)
        sqrt_rho  = math.sqrt(rho)
        sqrt_1mr  = math.sqrt(1 - rho)

        conditional_pd = norm.cdf(
            (n_inv_pd + sqrt_rho * n_inv_999) / sqrt_1mr
        )
        base_k = lgd * conditional_pd - pd * lgd
        mat_adj = (
            (1 + (maturity - 2.5) * b) / (1 - 1.5 * b)
        )
        return base_k * mat_adj
