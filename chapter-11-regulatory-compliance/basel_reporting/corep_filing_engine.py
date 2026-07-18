"""Basel Credit Risk COREP Filing Engine.

Model ID: MR-2026-049 | Risk: MEDIUM
Returns: C 02.00 (SA credit), C 08.00 (IRB credit)
Source: MR-2026-040 (Ch 6 Corporate PD Model) via PDModelTool
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Dict, List
import logging

from awb_commons.models import (
    RWAResult, XBRLInstance, LeverageRatioResult,
)

log = logging.getLogger(__name__)

# CRR3 Art. 112-141 SA risk weights by exposure class
SA_RISK_WEIGHTS: Dict[str, Decimal] = {
    "central_govts": Decimal("0.00"),
    "regional_govts": Decimal("0.20"),
    "public_sector": Decimal("0.20"),
    "institutions": Decimal("0.20"),
    "corporates": Decimal("1.00"),
    "retail": Decimal("0.75"),
    "residential_mortgage": Decimal("0.35"),
    "commercial_re": Decimal("0.50"),
    "sme": Decimal("0.765"),  # 76.5% CRR3 supporting factor
    "in_default": Decimal("1.50"),
    "high_risk": Decimal("1.50"),
}


@dataclass
class SAExposureData:
    """Standardised approach exposure by class."""
    exposure_class: str
    exposure_gbp: Decimal
    risk_weight: Decimal

    @property
    def rwa_gbp(self) -> Decimal:
        return self.exposure_gbp * self.risk_weight


@dataclass
class IRBExposureData:
    """IRB approach portfolio data from MR-2026-040."""
    pd_band: str          # e.g., "<0.5%", "0.5-1%"
    ead_gbp: Decimal
    lgd: Decimal          # e.g., 0.45 for senior unsecured
    rwa_gbp: Decimal      # From MR-2026-040 IRB formula


class COREPFilingEngine:
    """Generate C 02.00 and C 08.00 COREP returns.

    Consumes MR-2026-040 (Chapter 6 Corporate PD Model)
    output for IRB approach, and T24 position data for SA.

    Args:
        quarter_end: Reporting quarter end date.
        filer: EBAXBRLFiler instance for XML generation.
    """

    def __init__(
        self,
        quarter_end: date,
    ) -> None:
        self._quarter_end = quarter_end

    def generate_c0200(
        self,
        sa_exposures: List[SAExposureData],
    ) -> XBRLInstance:
        """Generate COREP C 02.00 (SA credit risk).

        Args:
            sa_exposures: Exposure by class from T24.

        Returns:
            XBRLInstance for C 02.00.
        """
        total_rwa = sum(e.rwa_gbp for e in sa_exposures)
        total_exp = sum(e.exposure_gbp for e in sa_exposures)
        data: Dict[str, str] = {
            "eba_c_02.00_r010_c010": str(
                int(total_exp)
            ),
            "eba_c_02.00_r010_c060": str(int(total_rwa)),
        }
        for exp in sa_exposures:
            key = f"eba_c_02.00_{exp.exposure_class}_rwa"
            data[key] = str(int(exp.rwa_gbp))
        log.info(
            "C 02.00: SA RWA=£%sB exposures=%d",
            round(float(total_rwa)/1e9, 2),
            len(sa_exposures),
        )
        return XBRLInstance(
            return_code="C 02.00",
            xml_content=str(data),
        )

    def generate_c0800(
        self,
        irb_exposures: List[IRBExposureData],
    ) -> XBRLInstance:
        """Generate COREP C 08.00 (IRB credit risk).

        Source: MR-2026-040 IRB formula output from Chapter 6.
        PD × LGD × EAD × 1.06 × f(correlation, maturity).

        Args:
            irb_exposures: IRB portfolio from Ch 6 MR-2026-040.

        Returns:
            XBRLInstance for C 08.00.
        """
        total_irb_rwa = sum(e.rwa_gbp for e in irb_exposures)
        total_ead = sum(e.ead_gbp for e in irb_exposures)
        data: Dict[str, str] = {
            "eba_c_08.00_r010_c010": str(int(total_ead)),
            "eba_c_08.00_r010_c060": str(int(total_irb_rwa)),
        }
        log.info(
            "C 08.00: IRB RWA=£%sB EAD=£%sB "
            "source=MR-2026-040",
            round(float(total_irb_rwa)/1e9, 2),
            round(float(total_ead)/1e9, 2),
        )
        return XBRLInstance(
            return_code="C 08.00",
            xml_content=str(data),
        )

    def validate_cross_return_consistency(
        self,
        c0200_sa_rwa: Decimal,
        c0800_irb_rwa: Decimal,
        rwa_result: RWAResult,
    ) -> bool:
        """Validate C 02.00 + C 08.00 sum matches RWA total.

        PRA PS17/23: cross-return inconsistency is a Tier 1
        data quality error requiring restatement.

        Args:
            c0200_sa_rwa: SA credit RWA from C 02.00.
            c0800_irb_rwa: IRB credit RWA from C 08.00.
            rwa_result: Overall RWAResult for comparison.

        Returns:
            True if consistent (within 0.1% tolerance).

        Raises:
            ValueError: If inconsistency exceeds tolerance.
        """
        computed_total = c0200_sa_rwa + c0800_irb_rwa
        reference_total = (
            rwa_result.credit_risk_sa_gbp
            + rwa_result.credit_risk_irb_gbp
        )
        if reference_total == 0:
            return True
        discrepancy_pct = abs(float(
            (computed_total - reference_total)
            / reference_total * 100
        ))
        if discrepancy_pct > 0.1:
            raise ValueError(
                f"Cross-return inconsistency: C02.00+C08.00 "
                f"credit RWA £{float(computed_total)/1e9:.2f}B "
                f"vs reference £{float(reference_total)/1e9:.2f}B"
                f" ({discrepancy_pct:.2f}% — PRA Tier 1 error)"
            )
        log.info(
            "Cross-return validation passed: discrepancy=%.4f%%",
            discrepancy_pct,
        )
        return True
