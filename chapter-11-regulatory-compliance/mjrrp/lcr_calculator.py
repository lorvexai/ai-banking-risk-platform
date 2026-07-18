"""LCR Calculator — CRR3 Articles 411-428.

COREP return: C 72.00 (monthly, 15 business day deadline)
Formula: Adjusted HQLA / Net Cash Outflows >= 100%
"""
from __future__ import annotations
from datetime import date
from decimal import Decimal
from typing import Dict
import logging

from awb_commons.models import LCRResult

log = logging.getLogger(__name__)

HQLA_L2A_HAIRCUT = Decimal("0.15")  # 15% haircut
HQLA_L2B_HAIRCUT = Decimal("0.25")  # Min 25% haircut


class LCRCalculator:
    """Calculate LCR per CRR3 Arts 411-428 and EBA ITS C 72.00."""

    def calculate_hqla(
        self,
        level1_gbp: Decimal,
        level2a_gbp: Decimal,
        level2b_gbp: Decimal,
    ) -> Decimal:
        """Calculate adjusted HQLA after haircuts.

        Args:
            level1_gbp: Central bank reserves and govt bonds.
            level2a_gbp: Covered bonds, IG corporate bonds.
            level2b_gbp: RMBS, equities (min 25% haircut).

        Returns:
            Adjusted HQLA in £GBP.
        """
        adjusted = (
            level1_gbp
            + level2a_gbp * (1 - HQLA_L2A_HAIRCUT)
            + level2b_gbp * (1 - HQLA_L2B_HAIRCUT)
        )
        log.info(
            "HQLA: L1=£%sB L2A=£%sB L2B=£%sB adj=£%sB",
            round(float(level1_gbp)/1e9, 2),
            round(float(level2a_gbp)/1e9, 2),
            round(float(level2b_gbp)/1e9, 2),
            round(float(adjusted)/1e9, 2),
        )
        return adjusted

    def calculate_net_cash_outflows(
        self,
        retail_deposits_gbp: Decimal,
        retail_runoff_rate: Decimal,
        wholesale_deposits_gbp: Decimal,
        wholesale_runoff_rate: Decimal,
        derivative_outflows_gbp: Decimal,
        cash_inflows_gbp: Decimal,
        inflow_cap_pct: Decimal = Decimal("0.75"),
    ) -> Decimal:
        """Calculate 30-day net stressed cash outflows.

        Applies CRR3 Art. 425 inflow cap (max 75% of outflows).

        Returns:
            Net cash outflows for LCR denominator.
        """
        gross_outflows = (
            retail_deposits_gbp * retail_runoff_rate
            + wholesale_deposits_gbp * wholesale_runoff_rate
            + derivative_outflows_gbp
        )
        capped_inflows = min(
            cash_inflows_gbp,
            gross_outflows * inflow_cap_pct,
        )
        net = gross_outflows - capped_inflows
        log.info(
            "Net outflows: gross=£%sB inflows_capped=£%sB "
            "net=£%sB",
            round(float(gross_outflows)/1e9, 2),
            round(float(capped_inflows)/1e9, 2),
            round(float(net)/1e9, 2),
        )
        return net

    def calculate(
        self,
        reporting_date: date,
        hqla_l1: Decimal,
        hqla_l2a: Decimal,
        hqla_l2b: Decimal,
        net_outflows: Decimal,
    ) -> LCRResult:
        """Assemble and return LCRResult."""
        return LCRResult(
            reporting_date=reporting_date,
            hqla_level1_gbp=hqla_l1,
            hqla_level2a_gbp=hqla_l2a,
            hqla_level2b_gbp=hqla_l2b,
            net_cash_outflows_30d_gbp=net_outflows,
        )
