"""AWB Portfolio — Sector Concentration Monitor (HHI).

Computes Herfindahl-Hirschman Index across 4 dimensions:
  - Industry sector (12 AWB internal taxonomy sectors)
  - UK geography (9 regions)
  - Obligor size band (micro/small/mid-market/large)
  - Collateral type (unsecured/property/receivables)

AWB internal limits: HHI < 0.20 (sector), < 0.15 (geography).
Alert at 90% of limit (0.18 / 0.135).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

# AWB internal concentration limits
LIMITS = {
    "sector":    {"alert": 0.18, "breach": 0.20},
    "geography": {"alert": 0.135, "breach": 0.15},
    "size_band": {"alert": 0.18, "breach": 0.20},
    "collateral":{"alert": 0.18, "breach": 0.20},
}

# Single-name limit: 2% of Tier 1 capital (£72M at AWB)
SINGLE_NAME_LIMIT_GBP = float(72_000_000)


@dataclass
class ConcentrationAlert:
    """Alert for a concentration approaching or breaching limit."""
    dimension: str
    entity: str
    hhi: float
    limit: float
    alert_type: str     # "alert" | "breach"


@dataclass
class ConcentrationReport:
    """Monthly portfolio concentration report."""
    report_date: str
    total_exposure_gbp: float
    hhi_by_dimension: dict = field(default_factory=dict)
    alerts: list[ConcentrationAlert] = field(default_factory=list)
    single_name_breaches: list[dict] = field(default_factory=list)

    @property
    def has_alerts(self) -> bool:
        return bool(self.alerts or self.single_name_breaches)


class ConcentrationMonitor:
    """HHI-based portfolio concentration monitor.

    Usage::

        monitor = ConcentrationMonitor()
        portfolio_df = pd.DataFrame(...)  # facility data
        report = monitor.analyse(portfolio_df, "2026-06-30")
    """

    def analyse(
        self,
        portfolio: pd.DataFrame,
        report_date: str,
    ) -> ConcentrationReport:
        """Compute HHI and raise concentration alerts.

        Args:
            portfolio: DataFrame with columns:
                facility_id, ead_gbp, sector, geography,
                size_band, collateral_type, obligor_id
            report_date: Report date (YYYY-MM-DD).

        Returns:
            ConcentrationReport with HHI scores and alerts.
        """
        total = portfolio["ead_gbp"].sum()
        report = ConcentrationReport(
            report_date        = report_date,
            total_exposure_gbp = float(total),
        )

        dimensions = {
            "sector":     "sector",
            "geography":  "geography",
            "size_band":  "size_band",
            "collateral": "collateral_type",
        }

        for dim_name, col in dimensions.items():
            if col not in portfolio.columns:
                continue
            hhi_result = self._hhi_by_dimension(portfolio, col, total)
            report.hhi_by_dimension[dim_name] = hhi_result

            # Check for alerts
            for entity, hhi_val in hhi_result.items():
                limit_cfg = LIMITS[dim_name]
                if hhi_val >= limit_cfg["breach"]:
                    report.alerts.append(ConcentrationAlert(
                        dimension  = dim_name,
                        entity     = entity,
                        hhi        = round(hhi_val, 4),
                        limit      = limit_cfg["breach"],
                        alert_type = "breach",
                    ))
                    log.warning(
                        "Concentration BREACH: %s %s HHI=%.3f >= %.2f",
                        dim_name, entity, hhi_val,
                        limit_cfg["breach"],
                    )
                elif hhi_val >= limit_cfg["alert"]:
                    report.alerts.append(ConcentrationAlert(
                        dimension  = dim_name,
                        entity     = entity,
                        hhi        = round(hhi_val, 4),
                        limit      = limit_cfg["alert"],
                        alert_type = "alert",
                    ))
                    log.warning(
                        "Concentration ALERT: %s %s HHI=%.3f >= %.2f",
                        dim_name, entity, hhi_val,
                        limit_cfg["alert"],
                    )

        # Single-name concentration
        if "obligor_id" in portfolio.columns:
            single_name = (
                portfolio.groupby("obligor_id")["ead_gbp"]
                .sum()
                .reset_index()
            )
            breaches = single_name[
                single_name["ead_gbp"] >= SINGLE_NAME_LIMIT_GBP
            ]
            for _, row in breaches.iterrows():
                report.single_name_breaches.append({
                    "obligor_id":  row["obligor_id"],
                    "ead_gbp":     float(row["ead_gbp"]),
                    "limit_gbp":   SINGLE_NAME_LIMIT_GBP,
                })
                log.warning(
                    "Single-name concentration: %s EAD=£%.1fM",
                    row["obligor_id"],
                    row["ead_gbp"] / 1_000_000,
                )

        return report

    # ── Internal ─────────────────────────────────────────────────

    def _hhi_by_dimension(
        self,
        portfolio: pd.DataFrame,
        col: str,
        total: float,
    ) -> dict:
        """Compute HHI for each group in a dimension."""
        grouped = (
            portfolio.groupby(col)["ead_gbp"]
            .sum()
        )
        shares = grouped / total
        hhi_total = float((shares ** 2).sum())
        result = {"_portfolio_hhi": round(hhi_total, 4)}
        for entity, share in shares.items():
            result[str(entity)] = round(float(share ** 2), 4)
        return result
