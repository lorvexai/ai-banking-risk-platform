"""
chapter_07/var_engine/mc_var_engine.py
AWB Real-Time Monte Carlo VaR Engine
Model: MR-2026-046 | HIGH Risk PRA SS1/23 | awb_commons
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import logging
import json
from datetime import datetime

log = logging.getLogger(__name__)

# FRTB back-test traffic light thresholds (CRR3 Art 325bg)
BACKTEST_GREEN_LIMIT = 4
BACKTEST_AMBER_LIMIT = 9


@dataclass
class VaRResult:
    """VaR calculation output."""

    var_95: float          # 1-day 95% VaR in GBP
    var_99: float          # 1-day 99% VaR in GBP
    expected_shortfall_99: float
    component_var: Dict[str, float]
    n_scenarios: int
    computed_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )


@dataclass
class BackTestResult:
    """FRTB 250-day back-test result."""

    date: str
    var_99: float
    actual_loss: float
    is_exception: bool
    rolling_exceptions: int
    traffic_light: str     # GREEN | AMBER | RED
    capital_multiplier: float


class MonteCarloVaREngine:
    """
    Monte Carlo VaR engine with Cholesky decomposition.

    Computes 10,000-scenario VaR for AWB's £800M trading
    book. Supports intraday recalculation and FRTB
    back-testing (CRR3 Part Three Title IV, Ch 1b).

    Model: MR-2026-046 | HIGH risk PRA SS1/23
    DORA ICT Asset: VAR-2026-046

    Args:
        correlation_matrix: n_factors x n_factors array
        volatilities: Daily vol per risk factor
        risk_factor_names: Names for component VaR
    """

    N_SCENARIOS = 10_000

    def __init__(
        self,
        correlation_matrix: np.ndarray,
        volatilities: np.ndarray,
        risk_factor_names: Optional[List[str]] = None,
    ) -> None:
        if not np.allclose(
            correlation_matrix,
            correlation_matrix.T,
        ):
            raise ValueError(
                "Correlation matrix must be symmetric"
            )
        self.L = np.linalg.cholesky(correlation_matrix)
        self.vols = volatilities
        n = len(volatilities)
        self.factor_names = risk_factor_names or [
            f"factor_{i}" for i in range(n)
        ]
        log.info(
            "MR-2026-046 initialised: %d risk factors, "
            "%d scenarios",
            n, self.N_SCENARIOS,
        )

    def generate_scenarios(self) -> np.ndarray:
        """
        Generate correlated return scenarios.

        Uses Cholesky decomposition to produce correlated
        standard normals, then scales to current vols.

        Returns:
            ndarray of shape (N_SCENARIOS, n_factors)
        """
        z = np.random.standard_normal(
            (self.N_SCENARIOS, len(self.vols))
        )
        corr_z = z @ self.L.T
        return corr_z * self.vols

    def calculate_portfolio_var(
        self,
        pnl_scenarios: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Calculate portfolio VaR at 95% and 99%.

        Args:
            pnl_scenarios: (N_SCENARIOS,) P&L array in GBP
        Returns:
            Tuple of (var_95, var_99) in GBP (positive)
        """
        var_95 = float(
            -np.percentile(pnl_scenarios, 5)
        )
        var_99 = float(
            -np.percentile(pnl_scenarios, 1)
        )
        log.info(
            "MR-2026-046: VaR95=£%.0f VaR99=£%.0f",
            var_95, var_99,
        )
        return var_95, var_99

    def get_component_var(
        self,
        factor_pnls: np.ndarray,
        portfolio_pnl: np.ndarray,
    ) -> Dict[str, float]:
        """
        Component VaR by risk factor.

        CVaR_i = corr(factor_i, portfolio) * VaR_i
        Sum of component VaRs = portfolio VaR
        (diversification check).

        Args:
            factor_pnls: (N_SCENARIOS, n_factors) array
            portfolio_pnl: (N_SCENARIOS,) total P&L
        Returns:
            Dict mapping factor name to component VaR GBP
        """
        components = {}
        for i, name in enumerate(self.factor_names):
            factor = factor_pnls[:, i]
            corr = float(np.corrcoef(
                factor, portfolio_pnl
            )[0, 1])
            factor_var = float(
                -np.percentile(factor, 1)
            )
            components[name] = round(
                corr * factor_var, 0
            )
        total = sum(components.values())
        log.debug(
            "Component VaR sum=£%.0f "
            "(diversification benefit embedded)",
            total,
        )
        return components

    def compute_full_var(
        self,
        position_sensitivities: np.ndarray,
    ) -> VaRResult:
        """
        End-to-end VaR calculation.

        Args:
            position_sensitivities: (n_positions, n_factors)
                DV01/delta sensitivity matrix
        Returns:
            VaRResult with all metrics
        """
        scenarios = self.generate_scenarios()
        factor_pnls = scenarios @ position_sensitivities.T
        portfolio_pnl = factor_pnls.sum(axis=1)
        var_95, var_99 = self.calculate_portfolio_var(
            portfolio_pnl
        )
        es_99 = float(
            -portfolio_pnl[
                portfolio_pnl <= np.percentile(
                    portfolio_pnl, 1
                )
            ].mean()
        )
        component_var = self.get_component_var(
            factor_pnls, portfolio_pnl
        )
        return VaRResult(
            var_95=var_95,
            var_99=var_99,
            expected_shortfall_99=es_99,
            component_var=component_var,
            n_scenarios=self.N_SCENARIOS,
        )


class VaRBackTester:
    """
    FRTB 250-day back-testing engine (CRR3 Art. 325bf).

    Compares actual 1-day desk P&L against prior-day
    99% VaR. Maintains rolling exception count and
    calculates FRTB traffic light status.

    Capital multiplier under CRR3 Art. 325bg:
        Green (0-4): multiplier = 1.5
        Amber (5-9): multiplier = 1.7 - 2.5 (sliding)
    Red (10+): multiplier up to 4.0 (PRA discretion)
    """

    MULTIPLIER_GREEN = 1.5
    MULTIPLIER_RED   = 4.0

    def __init__(self, lookback_days: int = 250) -> None:
        self.lookback = lookback_days
        self._history: List[BackTestResult] = []

    def run_daily_backtest(
        self,
        date: str,
        var_99: float,
        actual_pnl: float,
    ) -> BackTestResult:
        """
        Run one day's back-test comparison.

        An exception occurs when actual_loss > var_99.
        Positive actual_pnl = profit (no exception risk).
        Negative actual_pnl = loss for comparison.

        Args:
            date: ISO date string (YYYY-MM-DD)
            var_99: Prior-day 99% VaR in GBP (positive)
            actual_pnl: Actual desk P&L in GBP
        Returns:
            BackTestResult with traffic light status
        Raises:
            ValueError: If var_99 is not positive
        """
        if var_99 <= 0:
            raise ValueError(
                f"var_99 must be positive, got {var_99}"
            )
        actual_loss = -actual_pnl
        is_exception = actual_loss > var_99
        # Rolling 250-day count
        recent = self._history[-self.lookback:]
        rolling_exc = sum(
            r.is_exception for r in recent
        ) + (1 if is_exception else 0)
        tl, mult = self.get_traffic_light(rolling_exc)
        result = BackTestResult(
            date=date,
            var_99=var_99,
            actual_loss=actual_loss,
            is_exception=is_exception,
            rolling_exceptions=rolling_exc,
            traffic_light=tl,
            capital_multiplier=mult,
        )
        self._history.append(result)
        if is_exception:
            log.warning(
                "MR-2026-046 BACKTEST EXCEPTION %s: "
                "loss=£%.0f var99=£%.0f rolling=%d %s",
                date, actual_loss, var_99,
                rolling_exc, tl,
            )
        if tl == "RED":
            log.error(
                "FRTB RED ZONE: %d exceptions. "
                "PRA notification required within "
                "5 business days.",
                rolling_exc,
            )
        return result

    def get_traffic_light(
        self, exceptions: int
    ) -> Tuple[str, float]:
        """
        Return FRTB traffic light and capital multiplier.

        Args:
            exceptions: Rolling 250-day exception count
        Returns:
            Tuple of (status, capital_multiplier)
        """
        if exceptions <= BACKTEST_GREEN_LIMIT:
            return "GREEN", self.MULTIPLIER_GREEN
        elif exceptions <= BACKTEST_AMBER_LIMIT:
            # Sliding scale 1.7 to 2.5 across amber zone
            amber_pos = (
                (exceptions - BACKTEST_GREEN_LIMIT)
                / (BACKTEST_AMBER_LIMIT
                   - BACKTEST_GREEN_LIMIT)
            )
            mult = 1.7 + (amber_pos * 0.8)
            return "AMBER", round(mult, 2)
        else:
            return "RED", self.MULTIPLIER_RED

    def get_250_day_summary(self) -> dict:
        """Return summary statistics for monitoring."""
        recent = self._history[-self.lookback:]
        exc_count = sum(r.is_exception for r in recent)
        tl, mult = self.get_traffic_light(exc_count)
        return {
            "period_days": len(recent),
            "exceptions": exc_count,
            "traffic_light": tl,
            "capital_multiplier": mult,
            "model_id": "MR-2026-046",
        }
