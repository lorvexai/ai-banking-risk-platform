"""AWB Regulatory Reporting — COREP C 07.00 Generator.

Generates the quarterly COREP C 07.00 (Credit Risk — IRB)
return for submission to the PRA via EBA XBRL/XML taxonomy.

Inputs:
  - AWB facility data from T24
  - MR-2026-040 PD scores (calibrated)
  - CRR3RWACalculator outputs (including output floor)

Output:
  - COREP C 07.00 summary CSV (for review)
  - XBRL XML stub (for EBA reporting taxonomy)

Filing: quarterly, within 30 days of quarter-end.
"""
from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from awb_commons.schemas import RWAResult

log = logging.getLogger(__name__)

# CRR3 IRB exposure classes
EXPOSURE_CLASSES = {
    "corporate":    "CRR3_Art_147(2)(c)",
    "retail":       "CRR3_Art_147(2)(d)",
    "institution":  "CRR3_Art_147(2)(b)",
    "sovereign":    "CRR3_Art_147(2)(a)",
}

# PD bands for COREP C 07.00 row structure
PD_BANDS = [
    (0.0000, 0.0003, "PD_band_01"),
    (0.0003, 0.0010, "PD_band_02"),
    (0.0010, 0.0025, "PD_band_03"),
    (0.0025, 0.0050, "PD_band_04"),
    (0.0050, 0.0100, "PD_band_05"),
    (0.0100, 0.0250, "PD_band_06"),
    (0.0250, 0.0500, "PD_band_07"),
    (0.0500, 0.1000, "PD_band_08"),
    (0.1000, 0.2000, "PD_band_09"),
    (0.2000, 0.9999, "PD_band_10"),
    (0.9999, 1.0000, "PD_band_11_default"),
]


@dataclass
class C0700Row:
    """Single row in COREP C 07.00 IRB report."""
    exposure_class: str
    pd_band: str
    pd_band_lower: float
    pd_band_upper: float
    number_of_obligors: int
    ead_pre_ccf_gbp: float
    ead_post_ccf_gbp: float
    average_pd: float
    average_lgd: float
    average_maturity_years: float
    rwa_irb_gbp: float
    rwa_effective_gbp: float    # after output floor
    floor_addition_gbp: float
    expected_loss_gbp: float


@dataclass
class C0700Report:
    """COREP C 07.00 quarterly report."""
    reporting_entity: str      # AWB LEI or name
    reference_date: str        # YYYY-MM-DD (quarter-end)
    currency: str = "GBP"
    rows: list[C0700Row] = field(default_factory=list)

    @property
    def total_rwa_effective(self) -> float:
        return sum(r.rwa_effective_gbp for r in self.rows)

    @property
    def total_floor_addition(self) -> float:
        return sum(r.floor_addition_gbp for r in self.rows)

    @property
    def floor_bound_rwa_pct(self) -> float:
        total = self.total_rwa_effective
        if total == 0:
            return 0.0
        irb = sum(r.rwa_irb_gbp for r in self.rows)
        return round((total - irb) / total * 100, 2)


class CorePC0700Generator:
    """Generate COREP C 07.00 IRB credit risk return.

    Usage::

        gen = CorePC0700Generator()
        report = gen.generate(rwa_results, reference_date)
        csv_text = gen.to_csv(report)
    """

    REPORTING_ENTITY = "Avon & Wessex Bank plc"

    def generate(
        self,
        rwa_results: list[dict],
        reference_date: str,
        exposure_class: str = "corporate",
    ) -> C0700Report:
        """Build COREP C 07.00 from RWA results.

        Args:
            rwa_results: List of dicts with keys from RWAResult
                plus: actual_default (bool).
            reference_date: Quarter-end date (YYYY-MM-DD).
            exposure_class: CRR3 exposure class.

        Returns:
            C0700Report with PD-band rows.
        """
        report = C0700Report(
            reporting_entity = self.REPORTING_ENTITY,
            reference_date   = reference_date,
        )

        for band_lo, band_hi, band_id in PD_BANDS:
            band_items = [
                r for r in rwa_results
                if band_lo <= r["pd"] < band_hi
            ]
            if not band_items:
                continue

            n    = len(band_items)
            ead  = sum(r["ead"] for r in band_items)
            avg_pd  = sum(r["pd"]  for r in band_items) / n
            avg_lgd = sum(r["lgd"] for r in band_items) / n
            avg_mat = sum(r["maturity"] for r in band_items) / n
            rwa_irb = sum(r["rwa_irb"] for r in band_items)
            rwa_eff = sum(r["rwa_effective"] for r in band_items)
            el      = sum(
                r["pd"] * r["lgd"] * r["ead"]
                for r in band_items
            )

            report.rows.append(C0700Row(
                exposure_class      = exposure_class,
                pd_band             = band_id,
                pd_band_lower       = band_lo,
                pd_band_upper       = band_hi,
                number_of_obligors  = n,
                ead_pre_ccf_gbp     = round(ead, 2),
                ead_post_ccf_gbp    = round(ead, 2),
                average_pd          = round(avg_pd, 6),
                average_lgd         = round(avg_lgd, 4),
                average_maturity_years = round(avg_mat, 2),
                rwa_irb_gbp         = round(rwa_irb, 2),
                rwa_effective_gbp   = round(rwa_eff, 2),
                floor_addition_gbp  = round(rwa_eff - rwa_irb, 2),
                expected_loss_gbp   = round(el, 2),
            ))

        log.info(
            "COREP C 07.00 generated: %d rows, "
            "total RWA £%.1fM, floor addition £%.1fM",
            len(report.rows),
            report.total_rwa_effective / 1_000_000,
            report.total_floor_addition / 1_000_000,
        )
        return report

    def to_csv(self, report: C0700Report) -> str:
        """Serialise report to CSV for manual review."""
        output = io.StringIO()
        writer = csv.writer(output)

        writer.writerow([
            "reporting_entity", "reference_date",
            "exposure_class", "pd_band",
            "pd_band_lower", "pd_band_upper",
            "number_of_obligors",
            "ead_post_ccf_gbp",
            "average_pd", "average_lgd",
            "average_maturity_years",
            "rwa_irb_gbp", "rwa_effective_gbp",
            "floor_addition_gbp",
            "expected_loss_gbp",
        ])

        for row in report.rows:
            writer.writerow([
                report.reporting_entity,
                report.reference_date,
                row.exposure_class,
                row.pd_band,
                row.pd_band_lower,
                row.pd_band_upper,
                row.number_of_obligors,
                row.ead_post_ccf_gbp,
                row.average_pd,
                row.average_lgd,
                row.average_maturity_years,
                row.rwa_irb_gbp,
                row.rwa_effective_gbp,
                row.floor_addition_gbp,
                row.expected_loss_gbp,
            ])

        return output.getvalue()
