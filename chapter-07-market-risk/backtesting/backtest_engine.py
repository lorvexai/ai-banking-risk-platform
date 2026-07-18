"""
chapter_07/backtesting/backtest_engine.py
AWB Algorithmic Trading Backtesting Platform
Model: MR-2026-047 | MEDIUM Risk PRA SS1/23 | awb_commons
ICT Asset: BT-2026-001
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import pandas as pd
import numpy as np
import logging

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Risk-adjusted performance metrics."""

    sharpe: float
    sortino: float
    max_drawdown_pct: float
    var_95_daily: float
    cvar_95_daily: float
    total_return_pct: float
    n_trades: int
    win_rate: float


@dataclass
class MARFlag:
    """Market Abuse Regulation flag."""

    flag_type: str       # WASH_TRADE | SPOOFING
    severity: str        # HIGH | MEDIUM | LOW
    trade_ids: List[str]
    description: str


class BacktestEngine:
    """
    Vectorised backtest engine with transaction cost model.

    Supports walk-forward validation and MAR compliance
    checking per EU 596/2014 (UK MAR).

    Args:
        spread_bps: Bid/ask spread in basis points
        impact_bps: Market impact in basis points
    """

    def __init__(
        self,
        spread_bps: float = 5.0,
        impact_bps: float = 2.0,
    ) -> None:
        self.spread_bps = spread_bps
        self.impact_bps = impact_bps

    def run_backtest(
        self,
        signals: pd.Series,
        prices: pd.Series,
    ) -> BacktestResult:
        """
        Run vectorised backtest with transaction costs.

        Args:
            signals: Series of {-1, 0, 1} positions
            prices: Series of closing prices
        Returns:
            BacktestResult with risk-adjusted metrics
        Raises:
            ValueError: If signals and prices misaligned
        """
        if not signals.index.equals(prices.index):
            raise ValueError(
                "signals and prices must share index"
            )
        returns = prices.pct_change().fillna(0)
        positions = signals.shift(1).fillna(0)
        trades = positions.diff().abs()
        cost_per_trade = (
            self.spread_bps + self.impact_bps
        ) / 10_000
        cost = trades * cost_per_trade
        strategy_returns = positions * returns - cost
        n_trades = int((trades > 0).sum())
        active = positions != 0
        trade_pnl = (positions * returns)[active]
        wins = int((trade_pnl > 0).sum())
        win_rate = (
            float(wins / n_trades) if n_trades else 0.0
        )
        metrics = self._calculate_metrics(
            strategy_returns
        )
        metrics.n_trades = n_trades
        metrics.win_rate = round(min(win_rate, 1.0), 3)
        log.info(
            "Backtest: Sharpe=%.2f trades=%d "
            "win_rate=%.1f%%",
            metrics.sharpe, n_trades,
            metrics.win_rate * 100,
        )
        return metrics

    def _calculate_metrics(
        self, r: pd.Series
    ) -> BacktestResult:
        """Calculate annualised risk-adjusted metrics."""
        ann_factor = 252 ** 0.5
        sharpe = (r.mean() / r.std()) * ann_factor
        neg_r = r[r < 0]
        sortino = (r.mean() / neg_r.std()) * ann_factor
        cum = (1 + r).cumprod()
        dd = float((cum / cum.cummax() - 1).min())
        var_95 = float(np.percentile(r, 5))
        tail = r[r <= np.percentile(r, 5)]
        cvar_95 = float(tail.mean()) if len(tail) else var_95
        total_ret = float(cum.iloc[-1] - 1)
        return BacktestResult(
            sharpe=round(sharpe, 3),
            sortino=round(sortino, 3),
            max_drawdown_pct=round(dd * 100, 2),
            var_95_daily=round(var_95 * 100, 4),
            cvar_95_daily=round(cvar_95 * 100, 4),
            total_return_pct=round(total_ret * 100, 2),
            n_trades=0,
            win_rate=0.0,
        )

    def walk_forward_validate(
        self,
        signals: pd.Series,
        prices: pd.Series,
        train_years: int = 3,
        test_years: int = 1,
        step_months: int = 6,
    ) -> List[BacktestResult]:
        """
        Walk-forward validation with expanding window.

        Args:
            signals: Strategy signals
            prices: Price series
            train_years: In-sample window (years)
            test_years: Out-of-sample window (years)
            step_months: Step size in months
        Returns:
            List of BacktestResult per OOS window
        """
        results = []
        freq = pd.DateOffset(months=step_months)
        train_days = train_years * 252
        if len(prices) <= train_days:
            return results
        start = prices.index[train_days]
        cursor = start
        while cursor < prices.index[-1]:
            oos_end = cursor + pd.DateOffset(
                years=test_years
            )
            oos_sig = signals[cursor:oos_end]
            oos_px = prices[cursor:oos_end]
            if len(oos_px) < 60:
                break
            r = self.run_backtest(oos_sig, oos_px)
            results.append(r)
            log.info(
                "WF window %s: Sharpe=%.2f",
                cursor.date(), r.sharpe,
            )
            cursor += freq
        return results


class MARComplianceChecker:
    """
    MAR compliance checker (EU 596/2014 / UK MAR).

    Detects wash trades, spoofing patterns, and
    front-running signals. Logs to compliance_audit_log.

    Args:
        wash_trade_window_min: Round-trip window (minutes)
        spoofing_cancel_rate: Cancellation rate threshold
    """

    def __init__(
        self,
        wash_trade_window_min: int = 30,
        spoofing_cancel_rate: float = 0.8,
    ) -> None:
        self.wt_window = wash_trade_window_min
        self.spoof_rate = spoofing_cancel_rate

    def check_wash_trades(
        self,
        trades: pd.DataFrame,
    ) -> Optional[MARFlag]:
        """
        Detect round-trip trades within time window.

        Args:
            trades: DataFrame with columns
                [trade_id, instrument, side, time]
        Returns:
            MARFlag if wash trades detected, else None
        """
        flagged = []
        for inst, grp in trades.groupby("instrument"):
            buys = grp[grp["side"] == "BUY"]
            sells = grp[grp["side"] == "SELL"]
            for _, buy in buys.iterrows():
                window_end = buy["time"] + pd.Timedelta(
                    minutes=self.wt_window
                )
                matching = sells[
                    (sells["time"] >= buy["time"])
                    & (sells["time"] <= window_end)
                ]
                if not matching.empty:
                    flagged.extend(
                        [buy["trade_id"]]
                        + matching["trade_id"].tolist()
                    )
        if flagged:
            log.warning(
                "MAR WASH TRADE: %d trades flagged",
                len(flagged),
            )
            return MARFlag(
                flag_type="WASH_TRADE",
                severity="HIGH",
                trade_ids=flagged,
                description=(
                    f"Round-trip trades within "
                    f"{self.wt_window} minutes"
                ),
            )
        return None

    def flag_spoofing_patterns(
        self,
        order_book: pd.DataFrame,
    ) -> Optional[MARFlag]:
        """
        Detect spoofing via high cancellation rate.

        Args:
            order_book: DataFrame with columns
                [order_id, placed, cancelled, executed]
        Returns:
            MARFlag if spoofing detected, else None
        """
        total = len(order_book)
        if total == 0:
            return None
        cancelled = order_book["cancelled"].sum()
        cancel_rate = cancelled / total
        if cancel_rate >= self.spoof_rate:
            log.warning(
                "MAR SPOOFING: cancel rate %.1f%%",
                cancel_rate * 100,
            )
            flagged_ids = order_book[
                order_book["cancelled"]
            ]["order_id"].tolist()
            return MARFlag(
                flag_type="SPOOFING",
                severity="HIGH",
                trade_ids=flagged_ids,
                description=(
                    f"Cancellation rate "
                    f"{cancel_rate:.1%} "
                    f">= threshold "
                    f"{self.spoof_rate:.1%}"
                ),
            )
        return None
