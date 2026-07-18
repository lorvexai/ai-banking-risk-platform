# exercises/exercise_2.py
# Exercise 15.2: BCBS 239 P6 cross-risk query
# Difficulty: ★★★★☆  |  Estimated time: 45 minutes
# AWB-AI-2025 | chapter_15/exercises/
#
# TASK: Write a SQL query that answers the BCBS 239 P6
# test scenario used by the PRA in AWB's 2024 SREP:
#   "Total credit exposure by industry sector (2-digit SIC)
#    for all AWB corporate borrowers, across all three
#    business divisions, as at a given reporting date."
#
# Requirements:
#   - Join ENTITY_MASTER, CREDIT_EXPOSURE, FACILITY_TERMS
#   - Apply CRR3 on-balance-sheet netting rules
#   - Complete in < 10 seconds on the 28,000-row dataset
#   - Grand total of all sectors within £1M of portfolio sum
#
# Before ERDW: this query took 4 days (manual, 3 systems)
# After ERDW:  90 seconds (single SQL, verified in production)
#
# Solution: chapter_15/solutions/ex2_bcbs239_p6.py
from datetime import date
from typing import Optional
import logging

logger = logging.getLogger(__name__)


P6_QUERY_TEMPLATE = """
-- BCBS 239 P6 sector exposure query
-- AWB ERDW | chapter_15/exercises/exercise_2.py
-- TODO: Complete this query

SELECT
    -- TODO: Add SIC 2-digit sector grouping
    -- ft.sic_code_2digit        AS sector_code,
    -- ft.sector_description     AS sector_name,
    -- TODO: Add exposure aggregation
    -- SUM(ce.ead_on_balance_sheet_gbp) AS total_exposure_gbp,
    -- SUM(ce.ead_on_balance_sheet_gbp)
    --     / SUM(SUM(ce.ead_on_balance_sheet_gbp))
    --         OVER () AS pct_of_portfolio,
    -- COUNT(DISTINCT em.awb_customer_id) AS borrower_count
FROM
    entity_master em
    -- TODO: Join CREDIT_EXPOSURE
    -- JOIN credit_risk.credit_exposure ce
    --   ON em.awb_customer_id = ce.awb_customer_id
    -- TODO: Join FACILITY_TERMS for SIC codes
    -- JOIN credit_risk.facility_terms ft
    --   ON ce.facility_id = ft.facility_id
WHERE
    -- TODO: Filter to reporting date
    -- ce.reporting_date = %(reporting_date)s
    -- TODO: Exclude intraday positions
    -- AND ce.is_intraday = FALSE
    1 = 1
-- TODO: Group and order
-- GROUP BY ft.sic_code_2digit, ft.sector_description
-- ORDER BY total_exposure_gbp DESC
"""


def run_p6_sector_query(
    db_conn,
    reporting_date: date,
) -> list:
    """Execute BCBS 239 P6 sector exposure query.

    Target: complete in < 10 seconds on 28,000 exposures.
    Grand total must reconcile to portfolio total ±£1M.

    Args:
        db_conn: PostgreSQL connection to ERDW.
        reporting_date: COREP reporting date.

    Returns:
        List of dicts: sector_code, sector_name,
        total_exposure_gbp, pct_of_portfolio, borrower_count.

    Raises:
        ValueError: If reconciliation check fails (> £1M gap).
    """
    logger.info(
        "P6 sector query: reporting_date=%s", reporting_date
    )
    # TODO: Execute P6_QUERY_TEMPLATE with reporting_date
    # cursor = db_conn.cursor()
    # cursor.execute(
    #     P6_QUERY_TEMPLATE,
    #     {"reporting_date": reporting_date}
    # )
    # rows = cursor.fetchall()
    # _validate_reconciliation(rows, db_conn, reporting_date)
    # return rows
    raise NotImplementedError(
        "Complete this function — see exercise instructions"
    )


def _validate_reconciliation(
    rows: list,
    db_conn,
    reporting_date: date,
    tolerance_gbp: float = 1_000_000.0,
) -> None:
    """Verify sector sum reconciles to portfolio total.

    BCBS 239 P3 (Accuracy) requires that any aggregated
    view reconciles to the source data within tolerance.
    £1M tolerance on a £28B portfolio = 3.6bps.

    Args:
        rows: Sector exposure rows from P6 query.
        db_conn: PostgreSQL connection for total check.
        reporting_date: Must match the query date.
        tolerance_gbp: Maximum permitted variance.

    Raises:
        ValueError: If variance exceeds tolerance.
    """
    sector_total = sum(r["total_exposure_gbp"] for r in rows)
    # TODO: Fetch portfolio total from ERDW and compare
    raise NotImplementedError(
        "Implement reconciliation check — see exercise"
    )
