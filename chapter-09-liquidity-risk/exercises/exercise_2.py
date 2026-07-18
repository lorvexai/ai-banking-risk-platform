"""
exercises/exercise_2.py — Exercise 9.2 starter.
Chapter 9: Liquidity Risk — AWB-AI-2025 Programme.

Task: Simulate a month-end intraday liquidity stress scenario.
Connect CashFlowForecaster → IntradayLiquidityMonitor → alerts.

Target: Detect AMBER and RED alerts on simulated quarter-end day.
"""
from __future__ import annotations
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cash_flow.forecaster import CashFlowForecaster, TreasuryInputs
from intraday_liquidity.monitor import (
    IntradayLiquidityMonitor,
    IntradayPosition,
)
from awb_commons.models import StressScenario


# ── Scenario: AWB quarter-end (31 June 2026) ────────────────────
# On quarter-end, peak intraday usage reached £4.3B (78% of £5.5B)
# Simulate an adverse scenario where usage reaches £4.7B (85%)

QUARTER_END = datetime(2026, 3, 31, 16, 0)  # 16:00 peak window
FACILITY_GBP = 5_500_000_000               # AWB BoE facility

intraday_positions = [
    IntradayPosition(
        timestamp=datetime(2026, 3, 31, 9, 0),
        opening_balance_gbp=3_200_000_000,
        gross_settlements_gbp=800_000_000,
        gross_receipts_gbp=900_000_000,
        central_bank_facility_gbp=FACILITY_GBP,
        peak_usage_today_gbp=800_000_000,    # 07-09h: £0.8B
        available_facility_gbp=FACILITY_GBP,
    ),
    IntradayPosition(
        timestamp=datetime(2026, 3, 31, 12, 0),
        opening_balance_gbp=3_100_000_000,
        gross_settlements_gbp=1_100_000_000,
        gross_receipts_gbp=800_000_000,
        central_bank_facility_gbp=FACILITY_GBP,
        peak_usage_today_gbp=1_900_000_000,  # 09-12h running peak
        available_facility_gbp=FACILITY_GBP,
    ),
    IntradayPosition(
        timestamp=datetime(2026, 3, 31, 15, 30),
        opening_balance_gbp=2_800_000_000,
        gross_settlements_gbp=2_500_000_000,
        gross_receipts_gbp=1_800_000_000,
        central_bank_facility_gbp=FACILITY_GBP,
        peak_usage_today_gbp=4_700_000_000,  # 85%: RED threshold
        available_facility_gbp=FACILITY_GBP,
    ),
]

# AWB treasury inputs for 30-day cash flow forecast
treasury_inputs = TreasuryInputs(
    current_position_gbp=38_000_000_000,
    scheduled_inflows_gbp=2_400_000_000,
    scheduled_outflows_gbp=2_200_000_000,
    uncommitted_facilities_gbp=800_000_000,
    fx_exposure_gbp=300_000_000,
    wholesale_maturing_7d_gbp=1_200_000_000,
    retail_deposit_base_gbp=18_000_000_000,
    forecast_date=datetime(2026, 3, 31),
)


def exercise_stress_simulation() -> None:
    """
    TODO: Build end-to-end liquidity stress simulation.

    Steps:
    1. Run CashFlowForecaster on treasury_inputs (30-day horizon)
    2. Flag any buffer breaches using flag_buffer_breaches()
    3. Run IntradayLiquidityMonitor on each intraday_position
    4. Collect all alerts where requires_action == True
    5. Assert: at least one AMBER and one RED/CRITICAL alert fired
    6. Generate daily_peak_summary and print the peak usage

    Bonus: Modify the peak_usage_today_gbp in the 15:30 position
    to £5.2B (95% — CRITICAL threshold) and verify the system
    fires the CRITICAL recommended_action response.
    """
    # YOUR CODE HERE

    # Step 1: cash flow forecast
    # fc = CashFlowForecaster(horizon_days=30)
    # forecasts = fc.forecast(treasury_inputs)
    # print(f"D+1: £{forecasts[0].net_position_gbp/1e9:.1f}B")

    # Step 2: buffer breach detection (use £35B buffer)
    # breaches = fc.flag_buffer_breaches(forecasts, 35_000_000_000)

    # Step 3: intraday monitoring
    # mon = IntradayLiquidityMonitor()
    # alerts = [mon.assess(pos) for pos in intraday_positions]

    # Step 4: find action-required alerts
    # action_alerts = [a for a in alerts if a.requires_action]

    # Step 5: assertions
    # assert len(action_alerts) >= 1

    # Step 6: peak summary
    # summary = mon.daily_peak_summary(intraday_positions)
    # print(f"Peak: £{summary['peak_usage_gbp']/1e9:.1f}B "
    #       f"({summary['max_utilisation_pct']:.1f}%)")

    print("Exercise 9.2: implement the steps above.")
    print("Solution: chapter_09/solutions/exercise_2_sol.py")
    print(
        "GitHub: github.com/lorvenio/"
        "ai-banking-risk-platform/chapter_09/solutions/"
    )


if __name__ == "__main__":
    exercise_stress_simulation()
