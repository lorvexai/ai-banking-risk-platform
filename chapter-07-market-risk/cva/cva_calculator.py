"""
chapter_07/cva/cva_calculator.py
AWB CVA Calculator — Credit Valuation Adjustment
Model: MR-2026-048 | MEDIUM Risk PRA SS1/23
Primary Thread: integrates MR-2026-043 (Ch 6 PD model)
awb_commons
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import logging

log = logging.getLogger(__name__)

# CRR3 Art. 274 default recovery rate for unsecured
DEFAULT_RECOVERY_RATE = 0.40

# SA-CVA CRR3 Art. 383 systemic factor
SA_CVA_RHO = 0.50


@dataclass
class ExposureProfile:
    """Expected Exposure term structure."""

    counterparty_id: str
    time_steps: List[float]     # Years (e.g. 0.25, 0.5…)
    expected_exposure: List[float]   # EE in GBP at each t
    peak_exposure_975: List[float]   # PE 97.5% at each t


@dataclass
class CVAResult:
    """CVA calculation output."""

    counterparty_id: str
    cva_gbp: float             # CVA in GBP
    dva_gbp: float             # DVA (regulatory info only)
    sa_cva_capital_gbp: float  # SA-CVA capital CRR3 383
    recovery_rate: float
    lgd: float
    n_time_steps: int
    model_id: str = "MR-2026-048"


class CVACalculator:
    """
    CVA calculator using CRR3 SA-CVA methodology.

    CVA = (1-R) x sum[ EE(t_i) x PD(t_{i-1}, t_i)
                       x DF(t_i) ]

    Integrates with Chapter 6 MR-2026-043 Corporate PD
    Model via PDModelTool for counterparty PD inputs.

    Model: MR-2026-048 | MEDIUM risk PRA SS1/23
    DORA ICT Asset: CVA-2026-048

    Args:
        recovery_rate: Recovery rate R (default 40%
            per CRR3 Art. 274 for unsecured)
        ois_discount_factors: Dict {time_years: df}
    """

    def __init__(
        self,
        recovery_rate: float = DEFAULT_RECOVERY_RATE,
        ois_discount_factors: Optional[
            Dict[float, float]
        ] = None,
    ) -> None:
        if not 0 < recovery_rate < 1:
            raise ValueError(
                f"recovery_rate must be in (0,1), "
                f"got {recovery_rate}"
            )
        self.recovery_rate = recovery_rate
        self.lgd = 1.0 - recovery_rate
        self.ois_dfs = ois_discount_factors or {}
        log.info(
            "MR-2026-048 CVACalculator: "
            "R=%.0f%% LGD=%.0f%%",
            recovery_rate * 100, self.lgd * 100,
        )

    def calculate_cva(
        self,
        exposure: ExposureProfile,
        pd_term_structure: Dict[float, float],
    ) -> CVAResult:
        """
        Calculate bilateral CVA and SA-CVA capital.

        Args:
            exposure: EE profile from MC simulation
            pd_term_structure: {time_years: cumulative_pd}
                from MR-2026-043 PDModelTool (Ch 6)
        Returns:
            CVAResult with CVA, DVA, SA-CVA capital
        Raises:
            ValueError: If exposure and PD steps mismatch
        """
        times = exposure.time_steps
        ee = exposure.expected_exposure
        if len(times) != len(ee):
            raise ValueError(
                "EE profile length mismatch"
            )
        cva = 0.0
        prev_cumulative_pd = 0.0
        for i, t in enumerate(times):
            cumulative_pd = self._get_cumulative_pd(
                t, pd_term_structure
            )
            marginal_pd = max(
                cumulative_pd - prev_cumulative_pd,
                0.0,
            )
            df = self._get_discount_factor(t)
            cva += ee[i] * marginal_pd * df
            prev_cumulative_pd = cumulative_pd
        cva_gbp = self.lgd * cva
        # DVA: calculated but not recognised for reg capital
        dva_gbp = cva_gbp * 0.15   # simplified AWB estimate
        sa_cva_capital = self._calculate_sa_cva_capital(
            cva_gbp
        )
        log.info(
            "MR-2026-048: %s CVA=£%.0f DVA=£%.0f "
            "SA_CVA_cap=£%.0f",
            exposure.counterparty_id,
            cva_gbp, dva_gbp, sa_cva_capital,
        )
        return CVAResult(
            counterparty_id=exposure.counterparty_id,
            cva_gbp=round(cva_gbp, 0),
            dva_gbp=round(dva_gbp, 0),
            sa_cva_capital_gbp=round(sa_cva_capital, 0),
            recovery_rate=self.recovery_rate,
            lgd=self.lgd,
            n_time_steps=len(times),
        )

    def _get_marginal_pd(
        self,
        time_years: float,
        pd_term_structure: Dict[float, float],
        prev_time: float = 0.0,
    ) -> float:
        """Extract marginal PD between two time steps."""
        cum_pd_t = self._get_cumulative_pd(
            time_years, pd_term_structure
        )
        cum_pd_prev = self._get_cumulative_pd(
            prev_time, pd_term_structure
        )
        return max(cum_pd_t - cum_pd_prev, 0.0)

    def _get_cumulative_pd(
        self,
        t: float,
        pd_ts: Dict[float, float],
    ) -> float:
        """Interpolate cumulative PD at time t."""
        keys = sorted(pd_ts.keys())
        if t <= keys[0]:
            return pd_ts[keys[0]] * (t / keys[0])
        if t >= keys[-1]:
            return pd_ts[keys[-1]]
        for i in range(len(keys) - 1):
            if keys[i] <= t <= keys[i + 1]:
                alpha = (t - keys[i]) / (
                    keys[i + 1] - keys[i]
                )
                return (
                    pd_ts[keys[i]] * (1 - alpha)
                    + pd_ts[keys[i + 1]] * alpha
                )
        return pd_ts[keys[-1]]

    def _get_discount_factor(self, t: float) -> float:
        """OIS discount factor at time t."""
        if not self.ois_dfs:
            # Flat 4.5% OIS rate (UK base rate approx)
            return np.exp(-0.045 * t)
        keys = sorted(self.ois_dfs.keys())
        if t in self.ois_dfs:
            return self.ois_dfs[t]
        # Linear interpolation
        for i in range(len(keys) - 1):
            if keys[i] <= t <= keys[i + 1]:
                alpha = (t - keys[i]) / (
                    keys[i + 1] - keys[i]
                )
                return (
                    self.ois_dfs[keys[i]] * (1 - alpha)
                    + self.ois_dfs[keys[i + 1]] * alpha
                )
        return self.ois_dfs[keys[-1]]

    def _calculate_sa_cva_capital(
        self, cva_gbp: float
    ) -> float:
        """
        SA-CVA capital per CRR3 Art. 383 (simplified).

        Full SA-CVA requires aggregation across sectors
        with prescribed correlation matrix. This uses
        the single-counterparty formula as approximation.

        K_CVA = rho * CVA_hedged + sqrt(1-rho^2) * CVA_idio
        For unhedged positions: CVA_hedged = CVA_idio = CVA
        """
        rho = SA_CVA_RHO
        k_cva = np.sqrt(
            rho**2 * cva_gbp**2
            + (1 - rho**2) * cva_gbp**2
        )
        return k_cva

    def aggregate_portfolio_cva(
        self,
        results: List[CVAResult],
    ) -> Dict[str, float]:
        """
        Aggregate CVA across all counterparties.

        Args:
            results: List of individual CVAResult
        Returns:
            Dict with total CVA, DVA, and SA-CVA capital
        """
        total_cva = sum(r.cva_gbp for r in results)
        total_dva = sum(r.dva_gbp for r in results)
        total_sa = sum(
            r.sa_cva_capital_gbp for r in results
        )
        log.info(
            "MR-2026-048 Portfolio: CVA=£%.0f "
            "DVA=£%.0f SA_capital=£%.0f "
            "(%d counterparties)",
            total_cva, total_dva, total_sa,
            len(results),
        )
        return {
            "total_cva_gbp": round(total_cva, 0),
            "total_dva_gbp": round(total_dva, 0),
            "total_sa_cva_capital_gbp": round(
                total_sa, 0
            ),
            "n_counterparties": len(results),
            "model_id": "MR-2026-048",
        }
