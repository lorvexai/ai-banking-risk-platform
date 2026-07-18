"""NSFR Calculator — CRR3 Articles 428a-428au.

COREP return: C 80.00 (quarterly, 15 business day deadline)
Formula: Available Stable Funding / Required Stable Funding >= 100%
"""
from __future__ import annotations
from datetime import date
from decimal import Decimal
import logging

from awb_commons.models import NSFRResult

log = logging.getLogger(__name__)

# ASF factors (CRR3 Art. 428d - 428l)
ASF_TIER1_FACTOR = Decimal("1.00")
ASF_RETAIL_LT1YR = Decimal("0.90")  # stable retail
ASF_WHOLESALE_GE6M = Decimal("0.50")

# RSF factors (CRR3 Art. 428m - 428au)
RSF_LOANS_GT1YR = Decimal("0.65")   # unencumbered loans
RSF_HQLA_L1 = Decimal("0.05")
RSF_UNDRAWN_COMMIT = Decimal("0.05")


class NSFRCalculator:
    """Calculate NSFR per CRR3 Arts 428a-428au."""

    def calculate_asf(
        self,
        tier1_capital_gbp: Decimal,
        retail_deposits_lt1yr_gbp: Decimal,
        wholesale_ge6m_gbp: Decimal,
    ) -> Decimal:
        """Calculate Available Stable Funding.

        Returns:
            ASF in £GBP per CRR3 prescribed factors.
        """
        asf = (
            tier1_capital_gbp * ASF_TIER1_FACTOR
            + retail_deposits_lt1yr_gbp * ASF_RETAIL_LT1YR
            + wholesale_ge6m_gbp * ASF_WHOLESALE_GE6M
        )
        log.info("ASF: £%sB", round(float(asf)/1e9, 2))
        return asf

    def calculate_rsf(
        self,
        loans_gt1yr_gbp: Decimal,
        hqla_l1_gbp: Decimal,
        undrawn_commitments_gbp: Decimal,
    ) -> Decimal:
        """Calculate Required Stable Funding.

        Returns:
            RSF in £GBP per CRR3 prescribed factors.
        """
        rsf = (
            loans_gt1yr_gbp * RSF_LOANS_GT1YR
            + hqla_l1_gbp * RSF_HQLA_L1
            + undrawn_commitments_gbp * RSF_UNDRAWN_COMMIT
        )
        log.info("RSF: £%sB", round(float(rsf)/1e9, 2))
        return rsf

    def calculate(
        self,
        quarter_end: date,
        asf_gbp: Decimal,
        rsf_gbp: Decimal,
    ) -> NSFRResult:
        """Assemble NSFRResult."""
        return NSFRResult(
            quarter_end=quarter_end,
            available_stable_funding_gbp=asf_gbp,
            required_stable_funding_gbp=rsf_gbp,
        )
