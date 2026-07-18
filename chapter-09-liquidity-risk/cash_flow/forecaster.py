"""
cash_flow/forecaster.py — AWB Cash Flow Forecasting System.
Model ID: MR-2026-052 | PRA SS1/23 Risk: MEDIUM
Avon & Wessex Bank plc (AWB) — AWB-AI-2025 programme.

LSTM-based 30-day cash flow forecast for treasury operations.
Feeds daily into LCR/NSFR calculation engine and PRA reporting.
"""
from __future__ import annotations
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
from awb_commons.models import CashFlowForecast

logger = logging.getLogger(__name__)

MINIMUM_BUFFER_GBP = 35_000_000_000.0  # £35B regulatory floor
CONFIDENCE_LEVEL   = 0.95


@dataclass
class TreasuryInputs:
    """Daily treasury inputs from T24 and Reuters feeds."""
    current_position_gbp: float
    scheduled_inflows_gbp: float
    scheduled_outflows_gbp: float
    uncommitted_facilities_gbp: float
    fx_exposure_gbp: float
    wholesale_maturing_7d_gbp: float
    retail_deposit_base_gbp: float
    forecast_date: datetime


class CashFlowForecaster:
    """
    30-day net cash flow forecaster for AWB treasury.

    Uses LSTM model trained on 4 years of AWB daily cash flows
    (2021-2024) combined with macroeconomic features. Provides
    point forecast plus 95% confidence interval for each horizon.

    PRA: output feeds daily ILAS (Individual Liquidity Adequacy
    Standard) assessment and LCR intraday monitoring.
    MR-2026-052: MEDIUM risk (drives treasury decisions).
    """

    def __init__(
        self,
        model_version: str = "lstm-v2.1-2025",
        horizon_days: int = 30,
    ) -> None:
        self.model_version = model_version
        self.horizon_days = horizon_days
        logger.info(
            "CashFlowForecaster initialised: "
            "model=%s horizon=%dd",
            model_version, horizon_days,
        )

    def forecast(
        self, inputs: TreasuryInputs
    ) -> list[CashFlowForecast]:
        """
        Generate daily net cash position forecasts.

        Args:
            inputs: Current treasury state from T24.

        Returns:
            List of CashFlowForecast, one per forecast day.

        Raises:
            ValueError: If current position is negative.
        """
        if inputs.current_position_gbp < 0:
            raise ValueError(
                "current_position_gbp must be non-negative"
            )
        forecasts: list[CashFlowForecast] = []
        position = inputs.current_position_gbp
        for day in range(1, self.horizon_days + 1):
            net_flow = self._estimate_daily_flow(
                inputs, day
            )
            position = position + net_flow
            ci_width = self._confidence_interval(day)
            forecast = CashFlowForecast(
                forecast_date=(
                    inputs.forecast_date
                    + timedelta(days=day)
                ),
                horizon_days=day,
                net_position_gbp=round(position, 2),
                confidence_lower_gbp=round(
                    position - ci_width, 2
                ),
                confidence_upper_gbp=round(
                    position + ci_width, 2
                ),
                model_version=self.model_version,
            )
            forecasts.append(forecast)
        logger.info(
            "Generated %d-day cash flow forecast: "
            "D+1=£%.1fB D+30=£%.1fB",
            self.horizon_days,
            forecasts[0].net_position_gbp / 1e9,
            forecasts[-1].net_position_gbp / 1e9,
        )
        return forecasts

    def flag_buffer_breaches(
        self,
        forecasts: list[CashFlowForecast],
        buffer_gbp: float = MINIMUM_BUFFER_GBP,
    ) -> list[int]:
        """
        Return horizon days where lower CI breaches buffer.
        Used for PRA early warning trigger (ILAA).
        """
        return [
            f.horizon_days
            for f in forecasts
            if f.confidence_lower_gbp < buffer_gbp
        ]

    # ── Private helpers ───────────────────────────────────────────

    def _estimate_daily_flow(
        self, inputs: TreasuryInputs, day: int
    ) -> float:
        """
        Deterministic flow estimate (stub for testing).
        Production: LSTM inference from MLflow registry.
        """
        base_flow = (
            inputs.scheduled_inflows_gbp
            - inputs.scheduled_outflows_gbp
        )
        decay = math.exp(-0.02 * day)
        day_pattern = math.sin(2 * math.pi * day / 5) * 0.3e9
        return base_flow * decay + day_pattern

    def _confidence_interval(self, day: int) -> float:
        """CI widens with horizon: ±£900M at day 1, ±£3.2B at day 30."""
        return (900_000_000 + day * 77_000_000)
