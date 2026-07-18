"""Tests — Portfolio Monitoring (MR-2026-042).

Coverage:
  - EarlyWarningSystem: all 12 triggers individually
  - EarlyWarningSystem: RAG status assignment
  - EarlyWarningSystem: HIGH trigger → immediate RED
  - EarlyWarningSystem: 3+ triggers → RED
  - EarlyWarningSystem: 1-2 triggers → AMBER
  - EarlyWarningSystem: no triggers → GREEN
  - EarlyWarningSystem: score range 0–10
  - EarlyWarningSystem: news scan disabled in tests
  - EarlyWarningSystem: portfolio batch scoring
  - ConcentrationMonitor: HHI computation
  - ConcentrationMonitor: sector breach alert
  - ConcentrationMonitor: single-name breach
  - ConcentrationMonitor: no alerts for healthy portfolio
  - COREP C 07.00: row generation, CSV output

Run: pytest chapter_06/tests/test_portfolio.py -v
"""
import pytest
import pandas as pd
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../"))

from awb_commons.schemas import RAGStatus
from chapter_06.portfolio.ews import (
    EarlyWarningSystem, FacilityData, HIGH_TRIGGERS,
)
from chapter_06.portfolio.concentration import (
    ConcentrationMonitor, LIMITS, SINGLE_NAME_LIMIT_GBP,
)
from chapter_06.reporting.corep_c0700 import (
    CorePC0700Generator,
)


# ── EWS Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def ews():
    return EarlyWarningSystem()


@pytest.fixture
def clean_facility():
    """No triggers fired — GREEN."""
    return FacilityData(
        facility_id="F-CLEAN",
        revenue_yoy_change=0.05,
        ebitda_margin_qoq_change=50,
        covenant_breach=False,
        interest_cover=4.0,
        days_past_due=0,
        utilisation_rate=0.45,
        missed_reporting=False,
        waiver_request_received=False,
        sector_pmi=52.0,
        ch_filing_anomaly=False,
        director_disqualification=False,
    )


@pytest.fixture
def stressed_facility():
    """All triggers fired — RED."""
    return FacilityData(
        facility_id="F-STRESSED",
        revenue_yoy_change=-0.35,      # trigger 1
        ebitda_margin_qoq_change=-600, # trigger 2
        covenant_breach=True,          # trigger 3 (HIGH)
        interest_cover=1.2,            # trigger 4 (HIGH)
        days_past_due=10,              # trigger 5
        utilisation_rate=0.95,         # trigger 6
        missed_reporting=True,         # trigger 7
        waiver_request_received=True,  # trigger 8 (HIGH)
        sector_pmi=42.0,               # trigger 9
        ch_filing_anomaly=True,        # trigger 10
        director_disqualification=True, # trigger 11 (HIGH)
    )


@pytest.fixture
def amber_facility():
    """Two medium triggers — AMBER."""
    return FacilityData(
        facility_id="F-AMBER",
        revenue_yoy_change=0.02,
        ebitda_margin_qoq_change=-200,
        covenant_breach=False,
        interest_cover=3.0,
        days_past_due=6,               # trigger 1
        utilisation_rate=0.92,         # trigger 2
        missed_reporting=False,
        waiver_request_received=False,
        sector_pmi=50.0,
        ch_filing_anomaly=False,
        director_disqualification=False,
    )


# ── EWS Tests ─────────────────────────────────────────────────────

class TestEarlyWarningSystem:

    def test_clean_facility_green(self, ews, clean_facility):
        result = ews.score("F-CLEAN", clean_facility, scan_news=False)
        assert result.status == RAGStatus.GREEN
        assert result.triggers_fired == []

    def test_stressed_facility_red(self, ews, stressed_facility):
        result = ews.score("F-STR", stressed_facility, scan_news=False)
        assert result.status == RAGStatus.RED

    def test_amber_facility_amber(self, ews, amber_facility):
        result = ews.score("F-AMB", amber_facility, scan_news=False)
        assert result.status == RAGStatus.AMBER
        assert len(result.triggers_fired) == 2

    # ── Individual trigger tests ──────────────────────────────────

    def test_trigger_revenue_decline(self, ews, clean_facility):
        clean_facility.revenue_yoy_change = -0.25
        result = ews.score("F-REV", clean_facility, scan_news=False)
        assert "revenue_decline_20pct" in result.triggers_fired

    def test_trigger_ebitda_margin(self, ews, clean_facility):
        clean_facility.ebitda_margin_qoq_change = -600
        result = ews.score("F-EBITDA", clean_facility, scan_news=False)
        assert "ebitda_margin_decline_500bps" in result.triggers_fired

    def test_trigger_covenant_breach_is_high(
        self, ews, clean_facility
    ):
        clean_facility.covenant_breach = True
        result = ews.score("F-COV", clean_facility, scan_news=False)
        assert "covenant_breach" in result.triggers_fired
        assert result.status == RAGStatus.RED   # HIGH trigger → RED

    def test_trigger_interest_cover_is_high(
        self, ews, clean_facility
    ):
        clean_facility.interest_cover = 1.3   # below 1.5 floor
        result = ews.score("F-IC", clean_facility, scan_news=False)
        assert "interest_cover_floor" in result.triggers_fired
        assert result.status == RAGStatus.RED

    def test_trigger_days_past_due(self, ews, clean_facility):
        clean_facility.days_past_due = 6
        result = ews.score("F-DPD", clean_facility, scan_news=False)
        assert "days_past_due" in result.triggers_fired

    def test_no_trigger_days_past_due_threshold(
        self, ews, clean_facility
    ):
        clean_facility.days_past_due = 5   # exactly at threshold
        result = ews.score("F-DPD5", clean_facility, scan_news=False)
        # Threshold is >, not >= — 5 should NOT trigger
        assert "days_past_due" not in result.triggers_fired

    def test_trigger_utilisation_spike(self, ews, clean_facility):
        clean_facility.utilisation_rate = 0.91
        result = ews.score("F-UTIL", clean_facility, scan_news=False)
        assert "utilisation_spike" in result.triggers_fired

    def test_trigger_missed_reporting(self, ews, clean_facility):
        clean_facility.missed_reporting = True
        result = ews.score("F-MR", clean_facility, scan_news=False)
        assert "missed_reporting" in result.triggers_fired

    def test_trigger_waiver_request_high(self, ews, clean_facility):
        clean_facility.waiver_request_received = True
        result = ews.score("F-WR", clean_facility, scan_news=False)
        assert "waiver_request" in result.triggers_fired
        assert result.status == RAGStatus.RED  # HIGH trigger

    def test_trigger_sector_pmi(self, ews, clean_facility):
        clean_facility.sector_pmi = 44.9
        result = ews.score("F-PMI", clean_facility, scan_news=False)
        assert "sector_pmi_contractionary" in result.triggers_fired

    def test_trigger_ch_anomaly(self, ews, clean_facility):
        clean_facility.ch_filing_anomaly = True
        result = ews.score("F-CH", clean_facility, scan_news=False)
        assert "ch_filing_anomaly" in result.triggers_fired

    def test_trigger_director_disqualification_high(
        self, ews, clean_facility
    ):
        clean_facility.director_disqualification = True
        result = ews.score(
            "F-DIR", clean_facility, scan_news=False
        )
        assert "director_disqualification" in result.triggers_fired
        assert result.status == RAGStatus.RED

    # ── RAG logic tests ───────────────────────────────────────────

    def test_three_medium_triggers_red(self, ews, clean_facility):
        """3+ triggers → RED regardless of trigger type."""
        clean_facility.days_past_due = 6
        clean_facility.utilisation_rate = 0.91
        clean_facility.sector_pmi = 44.0
        result = ews.score("F-3TRIG", clean_facility, scan_news=False)
        assert result.status == RAGStatus.RED
        assert len(result.triggers_fired) >= 3

    def test_single_medium_trigger_amber(self, ews, clean_facility):
        clean_facility.days_past_due = 6
        result = ews.score("F-1MED", clean_facility, scan_news=False)
        assert result.status == RAGStatus.AMBER

    def test_ews_score_range(self, ews, stressed_facility):
        result = ews.score(
            "F-SCORE", stressed_facility, scan_news=False
        )
        assert 0.0 <= result.ews_score <= 10.0

    def test_ews_score_zero_for_clean(self, ews, clean_facility):
        result = ews.score("F-SCORE0", clean_facility, scan_news=False)
        assert result.ews_score == 0.0

    def test_facility_id_preserved(self, ews, clean_facility):
        result = ews.score("F-ID-TEST", clean_facility, scan_news=False)
        assert result.facility_id == "F-ID-TEST"

    def test_news_flag_none_when_scan_disabled(
        self, ews, clean_facility
    ):
        result = ews.score("F-NF", clean_facility, scan_news=False)
        assert result.news_flag is None

    def test_portfolio_batch_returns_all(self, ews):
        facilities = [
            (f"F-B{i}", FacilityData(
                facility_id=f"F-B{i}",
                revenue_yoy_change=0.02, ebitda_margin_qoq_change=0,
                covenant_breach=False, interest_cover=4.0,
                days_past_due=0, utilisation_rate=0.4,
                missed_reporting=False, waiver_request_received=False,
                sector_pmi=51.0, ch_filing_anomaly=False,
                director_disqualification=False,
            )) for i in range(10)
        ]
        results = ews.score_portfolio(facilities)
        assert len(results) == 10
        for r in results:
            assert r.status == RAGStatus.GREEN

    def test_high_triggers_constant(self):
        """HIGH_TRIGGERS set must contain the 4 documented triggers."""
        assert "covenant_breach"         in HIGH_TRIGGERS
        assert "interest_cover_floor"    in HIGH_TRIGGERS
        assert "director_disqualification" in HIGH_TRIGGERS
        assert "waiver_request"          in HIGH_TRIGGERS


# ── Concentration Monitor Tests ───────────────────────────────────

class TestConcentrationMonitor:

    @pytest.fixture
    def monitor(self):
        return ConcentrationMonitor()

    @pytest.fixture
    def healthy_portfolio(self):
        """10 equally-weighted sectors — HHI = 0.10."""
        sectors = [f"S{i}" for i in range(10)]
        return pd.DataFrame({
            "facility_id": [f"F{i}" for i in range(100)],
            "obligor_id":  [f"O{i}" for i in range(100)],
            "ead_gbp":     [1_000_000.0] * 100,
            "sector":      [sectors[i % 10] for i in range(100)],
            "geography":   ["London"] * 50 + ["Midlands"] * 50,
            "size_band":   ["mid_market"] * 100,
            "collateral_type": ["unsecured"] * 100,
        })

    @pytest.fixture
    def concentrated_portfolio(self):
        """45% in one sector — HHI > 0.20."""
        return pd.DataFrame({
            "facility_id": [f"F{i}" for i in range(100)],
            "obligor_id":  [f"O{i}" for i in range(100)],
            "ead_gbp": (
                [9_000_000.0] * 45 +   # 45% in manufacturing
                [1_000_000.0] * 55     # 55% spread
            ),
            "sector": (
                ["manufacturing"] * 45 +
                [f"sector_{i%5}" for i in range(55)]
            ),
            "geography":   ["London"] * 100,
            "size_band":   ["mid_market"] * 100,
            "collateral_type": ["unsecured"] * 100,
        })

    def test_healthy_portfolio_no_alerts(
        self, monitor, healthy_portfolio
    ):
        report = monitor.analyse(healthy_portfolio, "2026-06-30")
        sector_alerts = [
            a for a in report.alerts if a.dimension == "sector"
        ]
        assert len(sector_alerts) == 0

    def test_concentrated_portfolio_has_alert(
        self, monitor, concentrated_portfolio
    ):
        report = monitor.analyse(concentrated_portfolio, "2026-06-30")
        assert report.has_alerts
        sector_alerts = [
            a for a in report.alerts if a.dimension == "sector"
        ]
        assert len(sector_alerts) >= 1

    def test_breach_alert_type_set(
        self, monitor, concentrated_portfolio
    ):
        report = monitor.analyse(concentrated_portfolio, "2026-06-30")
        breach_alerts = [
            a for a in report.alerts
            if a.alert_type == "breach"
        ]
        assert len(breach_alerts) >= 1

    def test_single_name_breach_detected(self, monitor):
        """Single obligor > £72M triggers single-name alert."""
        portfolio = pd.DataFrame({
            "facility_id": ["F1", "F2"],
            "obligor_id":  ["O-BIG", "O-SMALL"],
            "ead_gbp":     [80_000_000.0, 1_000_000.0],
            "sector":      ["manufacturing", "retail"],
            "geography":   ["London", "London"],
            "size_band":   ["large", "small"],
            "collateral_type": ["unsecured", "unsecured"],
        })
        report = monitor.analyse(portfolio, "2026-06-30")
        assert len(report.single_name_breaches) == 1
        assert report.single_name_breaches[0]["obligor_id"] == "O-BIG"

    def test_total_exposure_computed(self, monitor, healthy_portfolio):
        report = monitor.analyse(healthy_portfolio, "2026-06-30")
        assert report.total_exposure_gbp == pytest.approx(
            100 * 1_000_000.0
        )

    def test_hhi_sum_of_squared_shares(self, monitor):
        """HHI for 4 equal sectors = 4 × (0.25)² = 0.25."""
        portfolio = pd.DataFrame({
            "facility_id": [f"F{i}" for i in range(40)],
            "obligor_id":  [f"O{i}" for i in range(40)],
            "ead_gbp":     [1_000_000.0] * 40,
            "sector":      ["A"] * 10 + ["B"] * 10 + ["C"] * 10 + ["D"] * 10,
            "geography":   ["L"] * 40,
            "size_band":   ["mid"] * 40,
            "collateral_type": ["u"] * 40,
        })
        report = monitor.analyse(portfolio, "2026-06-30")
        sector_hhi = report.hhi_by_dimension["sector"]["_portfolio_hhi"]
        assert sector_hhi == pytest.approx(0.25, rel=0.01)


# ── COREP C 07.00 Tests ───────────────────────────────────────────

class TestCorePC0700Generator:

    @pytest.fixture
    def gen(self):
        return CorePC0700Generator()

    @pytest.fixture
    def sample_rwa_results(self):
        """10 facilities across PD bands."""
        return [
            {"facility_id": f"F-C{i}",
             "pd": 0.005 + i * 0.015, "lgd": 0.45,
             "ead": 2_000_000.0, "maturity": 3.0,
             "rwa_irb": 800_000.0, "rwa_effective": 1_000_000.0,
             "sa_rwa": 1_600_000.0, "actual_default": False}
            for i in range(10)
        ]

    def test_report_has_rows(self, gen, sample_rwa_results):
        report = gen.generate(sample_rwa_results, "2026-06-30")
        assert len(report.rows) > 0

    def test_total_rwa_positive(self, gen, sample_rwa_results):
        report = gen.generate(sample_rwa_results, "2026-06-30")
        assert report.total_rwa_effective > 0

    def test_csv_output_has_header(self, gen, sample_rwa_results):
        report = gen.generate(sample_rwa_results, "2026-06-30")
        csv_text = gen.to_csv(report)
        first_line = csv_text.split("\n")[0]
        assert "exposure_class" in first_line
        assert "rwa_irb_gbp" in first_line

    def test_csv_rows_match_report(self, gen, sample_rwa_results):
        report = gen.generate(sample_rwa_results, "2026-06-30")
        csv_text = gen.to_csv(report)
        lines = [l for l in csv_text.split("\n") if l.strip()]
        # Header + data rows
        assert len(lines) == len(report.rows) + 1

    def test_reporting_entity_awb(self, gen, sample_rwa_results):
        report = gen.generate(sample_rwa_results, "2026-06-30")
        assert "Avon & Wessex Bank" in report.reporting_entity

    def test_reference_date_stored(self, gen, sample_rwa_results):
        report = gen.generate(sample_rwa_results, "2026-06-30")
        assert report.reference_date == "2026-06-30"

    def test_floor_addition_non_negative(
        self, gen, sample_rwa_results
    ):
        report = gen.generate(sample_rwa_results, "2026-06-30")
        for row in report.rows:
            assert row.floor_addition_gbp >= 0
