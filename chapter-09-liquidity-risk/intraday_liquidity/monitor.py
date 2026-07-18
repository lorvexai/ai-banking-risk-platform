"""
intraday_liquidity/monitor.py — AWB Intraday Liquidity Monitor.
Model ID: MR-2026-054 | PRA SS1/23 Risk: LOW
BCBS 248 / PRA supervisory expectations for intraday liquidity.
Avon & Wessex Bank plc (AWB) — AWB-AI-2025 programme.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from datetime import datetime
from awb_commons.models import IntradayAlert

logger = logging.getLogger(__name__)

# PRA threshold: alert when buffer < 20% of daily peak usage
BUFFER_ALERT_PCT   = 0.20
# DORA: system degradation alert threshold
DORA_ALERT_PCT     = 0.10


@dataclass
class IntradayPosition:
    """Real-time intraday liquidity position from SWIFT/T24."""
    timestamp: datetime
    opening_balance_gbp: float
    gross_settlements_gbp: float
    gross_receipts_gbp: float
    central_bank_facility_gbp: float
    peak_usage_today_gbp: float
    available_facility_gbp: float


class IntradayLiquidityMonitor:
    """
    Real-time intraday liquidity monitoring for AWB.

    Monitors peak usage against available facilities throughout
    the CHAPS/SWIFT settlement day (07:00–18:00 UK time).
    Triggers alerts when buffer falls below PRA thresholds.

    BCBS 248: banks must monitor and manage intraday liquidity
    to meet payment obligations on time under normal and stress.
    PRA: intraday liquidity data required in ILAAP submissions.
    """

    def __init__(
        self,
        alert_threshold_pct: float = BUFFER_ALERT_PCT,
    ) -> None:
        self.alert_threshold_pct = alert_threshold_pct
        logger.info(
            "IntradayLiquidityMonitor initialised: "
            "alert_threshold=%.0f%%",
            alert_threshold_pct * 100,
        )

    def assess(
        self, position: IntradayPosition
    ) -> IntradayAlert:
        """
        Assess current intraday liquidity position.

        Args:
            position: Real-time position snapshot.

        Returns:
            IntradayAlert with action recommendation.

        Raises:
            ValueError: If position data is inconsistent.
        """
        self._validate(position)
        net_position = (
            position.opening_balance_gbp
            + position.gross_receipts_gbp
            - position.gross_settlements_gbp
        )
        utilisation_pct = (
            position.peak_usage_today_gbp
            / max(position.available_facility_gbp, 1.0)
        ) * 100.0
        buffer_pct = 1.0 - (utilisation_pct / 100.0)
        requires_action = (
            buffer_pct < self.alert_threshold_pct
        )
        action = self._recommend_action(
            buffer_pct, position
        )
        alert = IntradayAlert(
            alert_time=position.timestamp,
            peak_usage_gbp=position.peak_usage_today_gbp,
            available_buffer_gbp=(
                position.available_facility_gbp
                - position.peak_usage_today_gbp
            ),
            utilisation_pct=round(utilisation_pct, 2),
            requires_action=requires_action,
            recommended_action=action,
        )
        if requires_action:
            logger.warning(
                "Intraday alert: utilisation=%.1f%% "
                "buffer=%.1f%% action=%s",
                utilisation_pct, buffer_pct * 100, action,
            )
        return alert

    def daily_peak_summary(
        self, positions: list[IntradayPosition]
    ) -> dict:
        """
        Summarise peak intraday usage across the settlement day.
        Used for BCBS 248 daily monitoring report to treasury.
        """
        if not positions:
            return {}
        peaks = [p.peak_usage_today_gbp for p in positions]
        utilisations = [
            p.peak_usage_today_gbp / max(
                p.available_facility_gbp, 1.0
            )
            for p in positions
        ]
        return {
            "peak_usage_gbp": max(peaks),
            "average_usage_gbp": sum(peaks) / len(peaks),
            "max_utilisation_pct": max(utilisations) * 100,
            "alert_count": sum(
                1 for u in utilisations
                if u > (1 - self.alert_threshold_pct)
            ),
            "monitoring_date": positions[0].timestamp.date(),
        }

    # ── Private helpers ───────────────────────────────────────────

    def _validate(self, pos: IntradayPosition) -> None:
        if pos.opening_balance_gbp < 0:
            raise ValueError(
                "opening_balance_gbp cannot be negative"
            )
        if pos.available_facility_gbp <= 0:
            raise ValueError(
                "available_facility_gbp must be positive"
            )

    def _recommend_action(
        self,
        buffer_pct: float,
        pos: IntradayPosition,
    ) -> str:
        if buffer_pct < DORA_ALERT_PCT:
            return (
                "CRITICAL: Activate central bank facility "
                "immediately. Notify treasury director and PRA."
            )
        elif buffer_pct < self.alert_threshold_pct:
            return (
                "ALERT: Defer non-urgent outgoing payments. "
                "Accelerate receipt collection. "
                "Notify head of treasury."
            )
        else:
            return "NORMAL: No action required."
