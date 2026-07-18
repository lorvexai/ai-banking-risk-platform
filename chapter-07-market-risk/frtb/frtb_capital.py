"""
chapter_07/frtb/frtb_capital.py
AWB SA-FRTB Capital Calculator
CRR3 Part Three Title IV Chapter 1b | January 2025
awb_commons

AWB SA-FRTB capital breakdown (June 2026):
  Equity SbM:        £12M
  Rates (GIRR) SbM:  £18M
  FX SbM:             £8M
  Credit Spread DRC:  £4M
  TOTAL:             £42M
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import logging

log = logging.getLogger(__name__)

# CRR3 Art. 325ag risk class weights (simplified)
EQUITY_BUCKET_WEIGHT = 0.20    # Large-cap developed
GIRR_TENOR_WEIGHT   = 0.0165  # 10-year SONIA tenor
FX_WEIGHT           = 0.15     # GBPUSD / GBPEUR
CREDIT_SPREAD_WEIGHT = 0.01    # IG corporate


@dataclass
class SbMResult:
    """Sensitivity-based method capital by risk class."""

    girr_capital_gbp: float
    equity_capital_gbp: float
    fx_capital_gbp: float
    credit_spread_capital_gbp: float
    total_sbm_gbp: float
    drc_gbp: float
    rrao_gbp: float
    total_sa_frtb_gbp: float


class SaFrtbCalculator:
    """
    Simplified SA-FRTB capital calculator.

    Implements the three-component SA-FRTB structure:
    1. Sensitivity-Based Method (SbM): delta, vega,
       curvature per risk class
    2. Default Risk Charge (DRC): jump-to-default
    3. Residual Risk Add-On (RRAO): exotic residuals

    CRR3 Part Three Title IV Chapter 1b (Art. 325a–325bh)
    PRA PS17/23 UK implementation (January 2025)

    AWB context: £800M trading book, SA-FRTB chosen
    over IMA (see Decision Point 7.2). Capital £42M.
    """

    def calculate_sbm(
        self,
        girr_dv01_gbp: float,
        equity_delta_gbp: float,
        fx_delta_gbp: float,
        credit_delta_gbp: float,
    ) -> Dict[str, float]:
        """
        Calculate SbM capital for each risk class.

        Uses simplified bucket-level aggregation.
        Full CRR3 requires intra-bucket and
        inter-bucket correlation matrices.

        Args:
            girr_dv01_gbp: GIRR DV01 in GBP
            equity_delta_gbp: Equity delta in GBP
            fx_delta_gbp: FX net open position GBP
            credit_delta_gbp: Credit spread delta GBP
        Returns:
            Dict of capital by risk class in GBP
        """
        girr_cap = abs(girr_dv01_gbp) * 100.0
        equity_cap = abs(equity_delta_gbp) * 0.20
        fx_cap = abs(fx_delta_gbp) * 0.15
        credit_cap = abs(credit_delta_gbp) * 0.01
        result = {
            "girr": round(girr_cap, 0),
            "equity": round(equity_cap, 0),
            "fx": round(fx_cap, 0),
            "credit_spread": round(credit_cap, 0),
            "total_sbm": round(
                girr_cap + equity_cap
                + fx_cap + credit_cap, 0
            ),
        }
        log.info(
            "SA-FRTB SbM: GIRR=£%.0f EQ=£%.0f "
            "FX=£%.0f CS=£%.0f TOTAL=£%.0f",
            result["girr"], result["equity"],
            result["fx"], result["credit_spread"],
            result["total_sbm"],
        )
        return result

    def calculate_drc(
        self,
        gross_jtd_gbp: float,
        net_jtd_gbp: float,
    ) -> float:
        """
        Default Risk Charge per CRR3 Art. 325w.

        DRC = max(0, net_JtD) * risk_weight
        Simplified: 8% weight for AWB IG equity book.

        Args:
            gross_jtd_gbp: Gross jump-to-default GBP
            net_jtd_gbp: Net JtD after hedging GBP
        Returns:
            DRC in GBP
        """
        drc = max(0.0, net_jtd_gbp) * 0.08
        log.info(
            "SA-FRTB DRC: gross=£%.0f net=£%.0f "
            "DRC=£%.0f",
            gross_jtd_gbp, net_jtd_gbp, drc,
        )
        return round(drc, 0)

    def calculate_rrao(
        self,
        exotic_notional_gbp: float,
        rrao_rate: float = 0.01,
    ) -> float:
        """
        Residual Risk Add-On per CRR3 Art. 325u.

        RRAO = notional * rate (1% for most exotics)
        Applies to instruments with gap risk,
        correlation risk, or behavioural optionality.

        Args:
            exotic_notional_gbp: Exotic notional GBP
            rrao_rate: RRAO rate (default 1%)
        Returns:
            RRAO capital in GBP
        """
        rrao = exotic_notional_gbp * rrao_rate
        log.info(
            "SA-FRTB RRAO: notional=£%.0f "
            "rate=%.1f%% RRAO=£%.0f",
            exotic_notional_gbp,
            rrao_rate * 100, rrao,
        )
        return round(rrao, 0)

    def calculate_total(
        self,
        girr_dv01: float,
        equity_delta: float,
        fx_delta: float,
        credit_delta: float,
        gross_jtd: float,
        net_jtd: float,
        exotic_notional: float,
    ) -> SbMResult:
        """
        Full SA-FRTB capital calculation.

        Args:
            girr_dv01: GIRR DV01 sensitivity GBP
            equity_delta: Equity delta GBP
            fx_delta: FX delta GBP
            credit_delta: Credit spread delta GBP
            gross_jtd: Gross JtD exposure GBP
            net_jtd: Net JtD after hedging GBP
            exotic_notional: Exotic instruments GBP
        Returns:
            SbMResult with full capital breakdown
        """
        sbm = self.calculate_sbm(
            girr_dv01, equity_delta,
            fx_delta, credit_delta,
        )
        drc = self.calculate_drc(gross_jtd, net_jtd)
        rrao = self.calculate_rrao(exotic_notional)
        total = sbm["total_sbm"] + drc + rrao
        log.info(
            "SA-FRTB TOTAL: SbM=£%.0f DRC=£%.0f "
            "RRAO=£%.0f TOTAL=£%.0f",
            sbm["total_sbm"], drc, rrao, total,
        )
        return SbMResult(
            girr_capital_gbp=sbm["girr"],
            equity_capital_gbp=sbm["equity"],
            fx_capital_gbp=sbm["fx"],
            credit_spread_capital_gbp=sbm[
                "credit_spread"
            ],
            total_sbm_gbp=sbm["total_sbm"],
            drc_gbp=drc,
            rrao_gbp=rrao,
            total_sa_frtb_gbp=round(total, 0),
        )
