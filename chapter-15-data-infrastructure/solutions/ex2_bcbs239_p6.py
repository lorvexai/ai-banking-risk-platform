"""Solution — Exercise 15.2: BCBS 239 P6 query (exposure by SIC sector).

Builds the 28,000-row synthetic dataset in-memory SQLite, runs the P6
query with CRR3 netting, and verifies sector totals reconcile to the
portfolio grand total (P6 accuracy principle).
"""
from __future__ import annotations

import random
import sqlite3
import time

DDL_LITE = """
CREATE TABLE ENTITY_MASTER (awb_customer_id TEXT PRIMARY KEY, legal_name TEXT,
  sic_code TEXT, division TEXT);
CREATE TABLE FACILITY_TERMS (facility_id TEXT PRIMARY KEY, awb_customer_id TEXT,
  committed_gbp REAL, netting_agreement INTEGER);
CREATE TABLE CREDIT_EXPOSURE (exposure_id INTEGER PRIMARY KEY, facility_id TEXT,
  reporting_date TEXT, drawn_gbp REAL, undrawn_gbp REAL, cash_collateral_gbp REAL);
"""

P6_QUERY = """
SELECT substr(e.sic_code, 1, 2)                       AS sector,
       e.division,
       ROUND(SUM(MAX(0, c.drawn_gbp - CASE WHEN f.netting_agreement = 1
                 THEN c.cash_collateral_gbp ELSE 0 END)), 2) AS total_exposure_gbp
FROM   CREDIT_EXPOSURE c
JOIN   FACILITY_TERMS  f ON f.facility_id = c.facility_id
JOIN   ENTITY_MASTER   e ON e.awb_customer_id = f.awb_customer_id
WHERE  c.reporting_date = ?
GROUP  BY sector, e.division
"""


def build(conn: sqlite3.Connection, seed: int = 15) -> None:
    rng = random.Random(seed)
    conn.executescript(DDL_LITE)
    divisions = ["CORPORATE", "COMMERCIAL", "RETAIL_BUSINESS"]
    custs = [(f"AWB-{i:05d}", f"Entity {i}", f"{rng.randint(10, 98)}000",
              rng.choice(divisions)) for i in range(4_000)]
    conn.executemany("INSERT INTO ENTITY_MASTER VALUES (?,?,?,?)", custs)
    facs = [(f"FAC-{i:05d}", custs[i % len(custs)][0],
             rng.uniform(1e5, 5e7), rng.random() < 0.3) for i in range(7_000)]
    conn.executemany("INSERT INTO FACILITY_TERMS VALUES (?,?,?,?)", facs)
    rows = [(i, facs[i % len(facs)][0], "2026-06-30", rng.uniform(5e4, 3e7),
             rng.uniform(0, 5e6), rng.uniform(0, 2e6)) for i in range(28_000)]
    conn.executemany("INSERT INTO CREDIT_EXPOSURE VALUES (?,?,?,?,?,?)", rows)


if __name__ == "__main__":
    conn = sqlite3.connect(":memory:")
    build(conn)
    t0 = time.perf_counter()
    result = conn.execute(P6_QUERY, ("2026-06-30",)).fetchall()
    elapsed = time.perf_counter() - t0
    sector_sum = sum(r[2] for r in result)
    grand = conn.execute(
        """SELECT SUM(MAX(0, c.drawn_gbp - CASE WHEN f.netting_agreement = 1
             THEN c.cash_collateral_gbp ELSE 0 END))
           FROM CREDIT_EXPOSURE c JOIN FACILITY_TERMS f
             ON f.facility_id = c.facility_id
           WHERE c.reporting_date = '2026-06-30'"""
    ).fetchone()[0]
    print(f"sector/division rows: {len(result)} | query time: {elapsed*1000:.0f}ms "
          "(target < 10s)")
    print(f"sum of sectors £{sector_sum:,.0f} vs grand total £{grand:,.0f}")
    assert elapsed < 10 and abs(sector_sum - grand) < 1.0
    print("P6 reconciliation holds: sector totals equal portfolio grand total")
