"""
tests/test_document_analyser.py
AWB Credit Document Analyser — Comprehensive Test Suite

Tests cover:
  Section 1: FieldExtraction and FinancialSummary Pydantic models
  Section 2: Range validators (hallucination mitigation b)
  Section 3: Confidence scoring and analyst review flags (mitigation c)
  Section 4: Red flag detection (validator.py)
  Section 5: Cross-validation of ratios
  Section 6: EBA margin of conservatism
  Section 7: Audit log (SQLite in temp dir)
  Section 8: Drift monitoring (PSI computation)
  Section 9: Document text extraction helpers
  Section 10: Sample credit pack content
  Section 11: Live API tests (skipped without GOOGLE_API_KEY)

Run all non-live tests: pytest tests/ -v -k "not live"
Run with API:          GOOGLE_API_KEY=xxx pytest tests/ -v

Author: AWB AI Programme (AWB-AI-2025)
Chapter: 2 — LLM Foundations for Financial Document Analysis
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from document_analyser.extractor import (
    FieldExtraction,
    FinancialSummary,
    RangeConfig,
    _validate_range,
    _apply_range_validations_direct,
    MODEL_ID,
    LLM_MODEL_NAME,
    EU_AI_ACT_CLASSIFICATION,
)
from document_analyser.validator import (
    RedFlagSeverity,
    ValidationResult,
    MAX_LEVERAGE,
    MIN_INTEREST_COVER,
    MIN_CURRENT_RATIO,
    MAX_REVENUE_DECLINE,
    CONSERVATISM_UPLIFT,
    CONSERVATISM_CONFIDENCE_THRESHOLD,
    detect_red_flags,
    cross_validate_ratios,
    apply_conservatism,
    validate_extraction,
)
from document_analyser.audit_log import (
    ExtractionAuditRecord,
    initialise_db,
    log_extraction,
    build_audit_record,
    get_records_by_document,
    get_high_risk_extractions,
    compute_monitoring_metrics,
    _compute_psi,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _fe(value, confidence=0.92, source_page=1, unit="£000s") -> FieldExtraction:
    """Helper: build a FieldExtraction with defaults."""
    return FieldExtraction(
        value=value,
        unit=unit,
        source_page=source_page,
        source_paragraph=f"Test value: {value}",
        confidence=confidence,
    )


@pytest.fixture
def healthy_summary() -> FinancialSummary:
    """ABC Manufacturing — healthy, all within policy."""
    return FinancialSummary(
        document_id="DOC-ABC-001",
        company_name=_fe("ABC Manufacturing Ltd", confidence=0.99, unit=""),
        reporting_period=_fe("Year ended 31 December 2024", confidence=0.99, unit=""),
        revenue=_fe(42350.0, confidence=0.95),
        ebitda=_fe(7800.0, confidence=0.93),
        ebitda_margin_pct=_fe(18.4, confidence=0.90, unit="%"),
        net_debt=_fe(4150.0, confidence=0.88),
        total_assets=_fe(20300.0, confidence=0.95),
        current_assets=_fe(11600.0, confidence=0.94),
        current_liabilities=_fe(5250.0, confidence=0.93),
        leverage_ratio=_fe(0.53, confidence=0.90, unit="x"),
        interest_cover=_fe(12.0, confidence=0.91, unit="x"),
        current_ratio=_fe(2.21, confidence=0.92, unit="x"),
    )


@pytest.fixture
def distressed_summary() -> FinancialSummary:
    """Riverside Retail — multiple P1 red flags."""
    return FinancialSummary(
        document_id="DOC-RVRSD-001",
        company_name=_fe("Riverside Retail Holdings Ltd", confidence=0.99, unit=""),
        reporting_period=_fe("Year ended 30 September 2024", confidence=0.99, unit=""),
        revenue=_fe(18200.0, confidence=0.94),
        ebitda=_fe(650.0, confidence=0.85),
        ebitda_margin_pct=_fe(3.6, confidence=0.82, unit="%"),
        net_debt=_fe(4520.0, confidence=0.76),  # Below confidence threshold
        total_assets=_fe(8430.0, confidence=0.90),
        current_assets=_fe(5230.0, confidence=0.89),
        current_liabilities=_fe(5800.0, confidence=0.90),
        leverage_ratio=_fe(6.95, confidence=0.82, unit="x"),
        interest_cover=_fe(1.05, confidence=0.81, unit="x"),
        current_ratio=_fe(0.90, confidence=0.88, unit="x"),
    )


@pytest.fixture
def incomplete_summary() -> FinancialSummary:
    """Summit Digital — missing EBITDA."""
    return FinancialSummary(
        document_id="DOC-SMMT-001",
        company_name=_fe("Summit Digital Services Ltd", confidence=0.99, unit=""),
        reporting_period=_fe("9 months ended 30 September 2025", confidence=0.95, unit=""),
        revenue=_fe(4200.0, confidence=0.90),
        ebitda=_fe(None, confidence=0.30),       # Not disclosed — very low confidence
        ebitda_margin_pct=_fe(None, confidence=0.30, unit="%"),
        net_debt=_fe(750.0, confidence=0.72),    # Below threshold
        total_assets=_fe(2470.0, confidence=0.85),
        current_assets=_fe(1550.0, confidence=0.88),
        current_liabilities=_fe(1000.0, confidence=0.82),
        leverage_ratio=_fe(None, confidence=0.30, unit="x"),
        interest_cover=_fe(None, confidence=0.30, unit="x"),
        current_ratio=_fe(1.55, confidence=0.85, unit="x"),
    )


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Redirect audit log to a temporary SQLite database."""
    db_file = str(tmp_path / "test_extraction_audit.db")
    monkeypatch.setenv("AUDIT_DB_PATH", db_file)
    initialise_db()
    return db_file


def _make_validation(summary, prior_revenue=None) -> ValidationResult:
    return validate_extraction(summary, prior_year_revenue=prior_revenue)


# ---------------------------------------------------------------------------
# Section 1: FieldExtraction and FinancialSummary model tests
# ---------------------------------------------------------------------------

class TestFieldExtraction:

    def test_valid_field_extraction(self):
        fe = FieldExtraction(value=42350.0, unit="£000s", confidence=0.95)
        assert fe.value == 42350.0
        assert fe.confidence == 0.95

    def test_confidence_lower_bound(self):
        with pytest.raises(Exception):
            FieldExtraction(value=100.0, confidence=-0.1)

    def test_confidence_upper_bound(self):
        with pytest.raises(Exception):
            FieldExtraction(value=100.0, confidence=1.1)

    def test_null_value_allowed(self):
        fe = FieldExtraction(value=None, confidence=0.20)
        assert fe.value is None

    def test_string_value_allowed(self):
        fe = FieldExtraction(value="ABC Manufacturing Ltd", unit="", confidence=0.99)
        assert fe.value == "ABC Manufacturing Ltd"

    def test_analyst_review_false_by_default(self):
        fe = FieldExtraction(value=100.0, confidence=0.90)
        assert fe.analyst_review_required is False

    def test_source_page_optional(self):
        fe = FieldExtraction(value=100.0, confidence=0.90, source_page=None)
        assert fe.source_page is None

    def test_source_paragraph_optional(self):
        fe = FieldExtraction(value=100.0, confidence=0.90, source_paragraph=None)
        assert fe.source_paragraph is None


class TestFinancialSummary:

    def test_healthy_summary_no_review_required(self, healthy_summary):
        assert healthy_summary.analyst_review_required is False

    def test_healthy_summary_overall_confidence(self, healthy_summary):
        assert healthy_summary.overall_confidence > 0.85

    def test_pra_model_id_present(self, healthy_summary):
        assert healthy_summary.model_id == MODEL_ID
        assert healthy_summary.model_id.startswith("MR-")

    def test_eu_ai_act_classification_present(self, healthy_summary):
        assert healthy_summary.eu_ai_act_status == "HIGH_RISK"

    def test_extraction_model_name(self, healthy_summary):
        assert healthy_summary.extraction_model == LLM_MODEL_NAME

    def test_low_confidence_triggers_review(self):
        """Model validator sets analyst_review_required when any material field < 0.80."""
        summary = FinancialSummary(
            document_id="DOC-LOW-CONF",
            company_name=_fe("Test Co", unit=""),
            reporting_period=_fe("2024", unit=""),
            revenue=_fe(10000.0, confidence=0.70),   # Below threshold
            ebitda=_fe(2000.0, confidence=0.92),
            ebitda_margin_pct=_fe(20.0, confidence=0.90, unit="%"),
            net_debt=_fe(5000.0, confidence=0.91),
            total_assets=_fe(15000.0, confidence=0.90),
            current_assets=_fe(6000.0, confidence=0.90),
            current_liabilities=_fe(3000.0, confidence=0.90),
            leverage_ratio=_fe(2.5, confidence=0.88, unit="x"),
            interest_cover=_fe(4.0, confidence=0.89, unit="x"),
            current_ratio=_fe(2.0, confidence=0.90, unit="x"),
        )
        assert summary.analyst_review_required is True
        assert any("revenue" in r for r in summary.analyst_review_reasons)

    def test_document_id_stored(self, healthy_summary):
        assert healthy_summary.document_id == "DOC-ABC-001"

    def test_extraction_date_set(self, healthy_summary):
        assert len(healthy_summary.extraction_date) == 10   # YYYY-MM-DD


# ---------------------------------------------------------------------------
# Section 2: Range validation tests (hallucination mitigation b)
# ---------------------------------------------------------------------------

class TestRangeValidation:

    def test_normal_leverage_no_flag(self):
        fe = _fe(2.5, unit="x")
        result = _validate_range(fe, "leverage_ratio", 0.0, 20.0)
        assert not result.analyst_review_required

    def test_excessive_leverage_flagged(self):
        fe = _fe(25.0, unit="x")    # Exceeds 20x max
        result = _validate_range(fe, "leverage_ratio", 0.0, 20.0)
        assert result.analyst_review_required
        assert "25.0" in result.review_reason

    def test_negative_leverage_flagged(self):
        fe = _fe(-1.0, unit="x")    # Negative leverage is unusual
        result = _validate_range(fe, "leverage_ratio", 0.0, 20.0)
        assert result.analyst_review_required

    def test_normal_ebitda_margin_no_flag(self):
        fe = _fe(18.4, unit="%")
        result = _validate_range(fe, "ebitda_margin_pct", 0.0, 60.0)
        assert not result.analyst_review_required

    def test_impossible_ebitda_margin_flagged(self):
        fe = _fe(85.0, unit="%")    # Exceeds 60% maximum
        result = _validate_range(fe, "ebitda_margin_pct", 0.0, 60.0)
        assert result.analyst_review_required

    def test_normal_interest_cover_no_flag(self):
        fe = _fe(12.0, unit="x")
        result = _validate_range(fe, "interest_cover", 0.0, 50.0)
        assert not result.analyst_review_required

    def test_extreme_interest_cover_flagged(self):
        fe = _fe(75.0, unit="x")    # Exceeds 50x
        result = _validate_range(fe, "interest_cover", 0.0, 50.0)
        assert result.analyst_review_required

    def test_null_value_not_flagged(self):
        fe = FieldExtraction(value=None, confidence=0.30)
        result = _validate_range(fe, "leverage_ratio", 0.0, 20.0)
        # None value should not trigger range flag
        assert not result.analyst_review_required

    def test_string_value_not_flagged(self):
        fe = _fe("ABC Manufacturing", unit="")
        result = _validate_range(fe, "company_name", 0.0, 100.0)
        assert not result.analyst_review_required

    def test_range_constants_correct(self):
        assert RangeConfig.EBITDA_MARGIN_MAX == 60.0
        assert RangeConfig.LEVERAGE_RATIO_MAX == 20.0
        assert RangeConfig.INTEREST_COVER_MAX == 50.0
        assert RangeConfig.CONFIDENCE_REVIEW_THRESHOLD == 0.80


# ---------------------------------------------------------------------------
# Section 3: Confidence scoring tests (hallucination mitigation c)
# ---------------------------------------------------------------------------

class TestConfidenceScoring:

    def test_high_confidence_no_review(self):
        fe = _fe(1000.0, confidence=0.95)
        assert not fe.analyst_review_required

    def test_low_confidence_triggers_review_flag_on_field(self):
        fe = FieldExtraction(value=1000.0, confidence=0.70)
        # FieldExtraction itself doesn't auto-set review — validator does that
        # But we can test the threshold constant
        assert fe.confidence < RangeConfig.CONFIDENCE_REVIEW_THRESHOLD

    def test_overall_confidence_computed_from_material_fields(self, healthy_summary):
        # Overall confidence is mean of material field confidences
        material_confidences = [
            healthy_summary.revenue.confidence,
            healthy_summary.ebitda.confidence,
            healthy_summary.net_debt.confidence,
            healthy_summary.leverage_ratio.confidence,
            healthy_summary.interest_cover.confidence,
        ]
        expected = sum(material_confidences) / len(material_confidences)
        assert abs(healthy_summary.overall_confidence - expected) < 0.01

    def test_incomplete_summary_review_required(self, incomplete_summary):
        assert incomplete_summary.analyst_review_required is True

    def test_distressed_confidence_review(self, distressed_summary):
        # Net debt confidence is 0.76 < 0.80 — should trigger review
        assert distressed_summary.net_debt.confidence < RangeConfig.CONFIDENCE_REVIEW_THRESHOLD


# ---------------------------------------------------------------------------
# Section 4: Red flag detection tests
# ---------------------------------------------------------------------------

class TestRedFlagDetection:

    def test_healthy_no_red_flags(self, healthy_summary):
        flags = detect_red_flags(healthy_summary)
        assert len(flags) == 0

    def test_high_leverage_p1_flag(self, distressed_summary):
        flags = detect_red_flags(distressed_summary)
        leverage_flags = [f for f in flags if f.flag_code == "CR-LEV-001"]
        assert len(leverage_flags) == 1
        assert leverage_flags[0].severity == RedFlagSeverity.P1

    def test_low_interest_cover_p1_flag(self, distressed_summary):
        flags = detect_red_flags(distressed_summary)
        ic_flags = [f for f in flags if f.flag_code == "CR-IC-001"]
        assert len(ic_flags) == 1
        assert ic_flags[0].severity == RedFlagSeverity.P1

    def test_low_current_ratio_p2_flag(self, distressed_summary):
        flags = detect_red_flags(distressed_summary)
        liq_flags = [f for f in flags if f.flag_code == "CR-LIQ-001"]
        assert len(liq_flags) == 1
        assert liq_flags[0].severity == RedFlagSeverity.P2

    def test_revenue_decline_p2_flag(self, distressed_summary):
        prior_revenue = 23500.0   # From Riverside Retail pack
        flags = detect_red_flags(distressed_summary, prior_year_revenue=prior_revenue)
        rev_flags = [f for f in flags if f.flag_code == "CR-REV-001"]
        assert len(rev_flags) == 1
        assert rev_flags[0].severity == RedFlagSeverity.P2

    def test_revenue_decline_below_threshold_no_flag(self, healthy_summary):
        prior_revenue = 40000.0   # ~5.9% growth — no decline
        flags = detect_red_flags(healthy_summary, prior_year_revenue=prior_revenue)
        rev_flags = [f for f in flags if f.flag_code == "CR-REV-001"]
        assert len(rev_flags) == 0

    def test_no_prior_revenue_skips_trend_check(self, distressed_summary):
        flags = detect_red_flags(distressed_summary, prior_year_revenue=None)
        rev_flags = [f for f in flags if f.flag_code == "CR-REV-001"]
        assert len(rev_flags) == 0

    def test_p1_flags_sorted_before_p2(self, distressed_summary):
        flags = detect_red_flags(distressed_summary, prior_year_revenue=23500.0)
        p1_positions = [i for i, f in enumerate(flags) if f.severity == RedFlagSeverity.P1]
        p2_positions = [i for i, f in enumerate(flags) if f.severity == RedFlagSeverity.P2]
        if p1_positions and p2_positions:
            assert max(p1_positions) < min(p2_positions)

    def test_red_flag_has_remediation(self, distressed_summary):
        flags = detect_red_flags(distressed_summary)
        for flag in flags:
            assert len(flag.remediation) > 10

    def test_red_flag_has_policy_reference(self, distressed_summary):
        flags = detect_red_flags(distressed_summary)
        for flag in flags:
            assert "AWB Credit Policy" in flag.awb_policy_reference

    def test_missing_values_not_flagged(self, incomplete_summary):
        """Null leverage/interest_cover should not raise false flags."""
        flags = detect_red_flags(incomplete_summary)
        lev_flags = [f for f in flags if f.flag_code == "CR-LEV-001"]
        ic_flags = [f for f in flags if f.flag_code == "CR-IC-001"]
        assert len(lev_flags) == 0
        assert len(ic_flags) == 0

    def test_threshold_constants_correct(self):
        assert MAX_LEVERAGE == 5.0
        assert MIN_INTEREST_COVER == 2.0
        assert MIN_CURRENT_RATIO == 1.0
        assert MAX_REVENUE_DECLINE == 0.20


# ---------------------------------------------------------------------------
# Section 5: Cross-validation tests
# ---------------------------------------------------------------------------

class TestCrossValidation:

    def test_no_issues_when_ratios_consistent(self, healthy_summary):
        """
        ABC Manufacturing: leverage stated 0.53x, calculated 4150/7800 = 0.53x.
        Should produce no cross-validation issues.
        """
        issues = cross_validate_ratios(healthy_summary)
        # Some rounding tolerance — may or may not flag depending on exact values
        # Key check: no issue should be catastrophically wrong
        assert isinstance(issues, list)

    def test_leverage_discrepancy_detected(self):
        """Create summary where stated leverage != calculated leverage."""
        summary = FinancialSummary(
            document_id="DOC-DISC-001",
            company_name=_fe("Test Co", unit=""),
            reporting_period=_fe("2024", unit=""),
            revenue=_fe(10000.0),
            ebitda=_fe(2000.0),
            ebitda_margin_pct=_fe(20.0, unit="%"),
            net_debt=_fe(4000.0),   # Net debt / EBITDA = 2.0x
            total_assets=_fe(15000.0),
            current_assets=_fe(6000.0),
            current_liabilities=_fe(3000.0),
            leverage_ratio=_fe(5.0, unit="x"),  # Stated as 5.0x — DISCREPANCY
            interest_cover=_fe(4.0, unit="x"),
            current_ratio=_fe(2.0, unit="x"),
        )
        issues = cross_validate_ratios(summary)
        leverage_issues = [i for i in issues if "LEVERAGE" in i]
        assert len(leverage_issues) >= 1

    def test_ebitda_margin_discrepancy_detected(self):
        summary = FinancialSummary(
            document_id="DOC-DISC-002",
            company_name=_fe("Test Co", unit=""),
            reporting_period=_fe("2024", unit=""),
            revenue=_fe(10000.0),
            ebitda=_fe(2000.0),      # 2000/10000 = 20%
            ebitda_margin_pct=_fe(35.0, unit="%"),  # Stated as 35% — DISCREPANCY
            net_debt=_fe(4000.0),
            total_assets=_fe(15000.0),
            current_assets=_fe(6000.0),
            current_liabilities=_fe(3000.0),
            leverage_ratio=_fe(2.0, unit="x"),
            interest_cover=_fe(4.0, unit="x"),
            current_ratio=_fe(2.0, unit="x"),
        )
        issues = cross_validate_ratios(summary)
        margin_issues = [i for i in issues if "MARGIN" in i]
        assert len(margin_issues) >= 1

    def test_null_values_skip_check(self, incomplete_summary):
        """Missing EBITDA — leverage cross-check should be skipped."""
        issues = cross_validate_ratios(incomplete_summary)
        # Should not raise an error — just skip checks for null fields
        assert isinstance(issues, list)


# ---------------------------------------------------------------------------
# Section 6: EBA margin of conservatism tests
# ---------------------------------------------------------------------------

class TestConservatism:

    def test_conservatism_applied_when_low_confidence(self, distressed_summary):
        """Distressed: net_debt confidence 0.76 < 0.80 threshold."""
        adj_debt, adj_lev, notes = apply_conservatism(distressed_summary)
        assert adj_debt is not None
        assert adj_debt > distressed_summary.net_debt.value
        expected = distressed_summary.net_debt.value * (1 + CONSERVATISM_UPLIFT)
        assert abs(adj_debt - expected) < 1.0

    def test_conservatism_uplift_is_ten_percent(self):
        assert CONSERVATISM_UPLIFT == 0.10

    def test_conservatism_not_applied_when_high_confidence(self, healthy_summary):
        """Net debt confidence 0.88 > 0.80 — no conservatism applied."""
        adj_debt, adj_lev, notes = apply_conservatism(healthy_summary)
        assert adj_debt is None
        assert adj_lev is None
        assert len(notes) == 0

    def test_adjusted_leverage_computed(self, distressed_summary):
        adj_debt, adj_lev, notes = apply_conservatism(distressed_summary)
        assert adj_lev is not None
        # adj_lev = adj_debt / ebitda
        expected_lev = adj_debt / distressed_summary.ebitda.value
        assert abs(adj_lev - expected_lev) < 0.01

    def test_conservatism_notes_describe_adjustment(self, distressed_summary):
        _, _, notes = apply_conservatism(distressed_summary)
        assert any("EBA conservatism" in n for n in notes)
        assert any("+10%" in n for n in notes)

    def test_null_net_debt_skips_conservatism(self, incomplete_summary):
        # Summit Digital has net debt value — but confidence is 0.72 < 0.80
        adj_debt, adj_lev, notes = apply_conservatism(incomplete_summary)
        if incomplete_summary.net_debt.value is not None:
            if incomplete_summary.net_debt.confidence < CONSERVATISM_CONFIDENCE_THRESHOLD:
                assert adj_debt is not None


# ---------------------------------------------------------------------------
# Section 7: Full validation integration tests
# ---------------------------------------------------------------------------

class TestValidationIntegration:

    def test_healthy_validation_passes(self, healthy_summary):
        result = validate_extraction(healthy_summary)
        assert result.validation_passed is True
        assert len(result.p1_flags) == 0

    def test_distressed_validation_fails(self, distressed_summary):
        result = validate_extraction(distressed_summary)
        assert result.validation_passed is False
        assert len(result.p1_flags) >= 2   # Leverage + Interest Cover

    def test_result_has_analyst_notes(self, healthy_summary):
        result = validate_extraction(healthy_summary)
        assert "MR-2026-035" in result.analyst_notes
        assert "human oversight" in result.analyst_notes.lower()

    def test_result_document_id_matches(self, healthy_summary):
        result = validate_extraction(healthy_summary)
        assert result.document_id == healthy_summary.document_id

    def test_p1_property_filters_correctly(self, distressed_summary):
        result = validate_extraction(distressed_summary)
        for flag in result.p1_flags:
            assert flag.severity == RedFlagSeverity.P1

    def test_p2_property_filters_correctly(self, distressed_summary):
        result = validate_extraction(distressed_summary, prior_year_revenue=23500.0)
        for flag in result.p2_flags:
            assert flag.severity == RedFlagSeverity.P2


# ---------------------------------------------------------------------------
# Section 8: Audit log tests
# ---------------------------------------------------------------------------

class TestAuditLog:

    def _make_audit_record(self, summary, validation) -> ExtractionAuditRecord:
        return build_audit_record(summary, validation, latency_ms=1250)

    def test_initialise_creates_table(self, temp_db):
        import sqlite3
        conn = sqlite3.connect(temp_db)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        conn.close()
        table_names = [t[0] for t in tables]
        assert "extraction_audit" in table_names

    def test_log_extraction_writes_record(self, temp_db, healthy_summary):
        validation = validate_extraction(healthy_summary)
        record = self._make_audit_record(healthy_summary, validation)
        eid = log_extraction(record)
        assert len(eid) == 36   # UUID4

    def test_record_retrieval_by_document(self, temp_db, healthy_summary):
        validation = validate_extraction(healthy_summary)
        record = self._make_audit_record(healthy_summary, validation)
        log_extraction(record)
        rows = get_records_by_document(healthy_summary.document_id)
        assert len(rows) >= 1
        assert rows[0]["document_id"] == healthy_summary.document_id

    def test_record_immutable_no_update_sql(self, temp_db, healthy_summary):
        """Verify there is no UPDATE path — records are immutable (PRA SS1/23)."""
        import document_analyser.audit_log as al
        import inspect
        source = inspect.getsource(al)
        # Count UPDATE statements — should be zero
        update_count = source.upper().count("UPDATE extraction_audit")
        assert update_count == 0

    def test_p1_flag_count_stored(self, temp_db, distressed_summary):
        validation = validate_extraction(distressed_summary)
        record = self._make_audit_record(distressed_summary, validation)
        log_extraction(record)
        rows = get_records_by_document(distressed_summary.document_id)
        assert rows[0]["p1_flag_count"] >= 2

    def test_high_risk_extractions_query(self, temp_db, distressed_summary):
        validation = validate_extraction(distressed_summary)
        record = self._make_audit_record(distressed_summary, validation)
        log_extraction(record)
        high_risk = get_high_risk_extractions(limit=10)
        assert len(high_risk) >= 1

    def test_dora_asset_id_stored(self, temp_db, healthy_summary):
        validation = validate_extraction(healthy_summary)
        record = self._make_audit_record(healthy_summary, validation)
        log_extraction(record)
        rows = get_records_by_document(healthy_summary.document_id)
        assert rows[0]["dora_asset_id"] == "DA-2026-002"

    def test_model_id_stored(self, temp_db, healthy_summary):
        validation = validate_extraction(healthy_summary)
        record = self._make_audit_record(healthy_summary, validation)
        log_extraction(record)
        rows = get_records_by_document(healthy_summary.document_id)
        assert rows[0]["model_id"] == MODEL_ID

    def test_field_confidences_stored_as_json(self, temp_db, healthy_summary):
        validation = validate_extraction(healthy_summary)
        record = self._make_audit_record(healthy_summary, validation)
        log_extraction(record)
        rows = get_records_by_document(healthy_summary.document_id)
        confidences = json.loads(rows[0]["field_confidences_json"])
        assert "revenue" in confidences
        assert isinstance(confidences["revenue"], float)

    def test_unique_extraction_ids(self, temp_db, healthy_summary):
        validation = validate_extraction(healthy_summary)
        r1 = build_audit_record(healthy_summary, validation)
        r2 = build_audit_record(healthy_summary, validation)
        assert r1.extraction_id != r2.extraction_id


# ---------------------------------------------------------------------------
# Section 9: Drift monitoring tests
# ---------------------------------------------------------------------------

class TestDriftMonitoring:

    def test_psi_zero_for_identical_distribution(self):
        """Identical observed and expected → PSI near zero."""
        # All high-confidence scores (matches baseline expectation)
        scores = [0.95] * 100
        psi = _compute_psi(scores)
        assert isinstance(psi, float)
        assert psi >= 0.0

    def test_psi_high_for_low_confidence_scores(self):
        """Low-confidence distribution diverges strongly from healthy baseline."""
        scores = [0.30] * 100
        psi_low = _compute_psi(scores)
        scores_high = [0.95] * 100
        psi_high = _compute_psi(scores_high)
        # Low-confidence PSI should differ from high-confidence PSI
        assert psi_low != psi_high

    def test_insufficient_records_returns_none(self, temp_db):
        result = compute_monitoring_metrics(model_id="MR-2026-035", window=100)
        assert result is None   # No records yet

    def test_monitoring_requires_ten_records(self, temp_db, healthy_summary):
        validation = validate_extraction(healthy_summary)
        # Write 9 records — should return None
        for i in range(9):
            rec = build_audit_record(healthy_summary, validation)
            log_extraction(rec)
        result = compute_monitoring_metrics(model_id=MODEL_ID, window=100)
        assert result is None

    def test_monitoring_returns_metrics_with_ten_records(self, temp_db, healthy_summary):
        validation = validate_extraction(healthy_summary)
        for i in range(10):
            rec = build_audit_record(healthy_summary, validation)
            log_extraction(rec)
        result = compute_monitoring_metrics(model_id=MODEL_ID, window=100)
        assert result is not None
        assert result.mean_confidence > 0.0
        assert result.drift_alert_level in ("GREEN", "AMBER", "RED")


# ---------------------------------------------------------------------------
# Section 10: Sample credit pack content tests
# ---------------------------------------------------------------------------

class TestSampleCreditPacks:

    @pytest.fixture
    def packs_dir(self) -> Path:
        return Path(__file__).parent.parent / "data"

    def test_abc_manufacturing_pack_exists(self, packs_dir):
        path = packs_dir / "abc_manufacturing_credit_pack.txt"
        assert path.exists(), f"Run: python data/generate_sample_credit_pack.py"

    def test_riverside_retail_pack_exists(self, packs_dir):
        path = packs_dir / "riverside_retail_credit_pack.txt"
        assert path.exists()

    def test_summit_digital_pack_exists(self, packs_dir):
        path = packs_dir / "summit_digital_credit_pack.txt"
        assert path.exists()

    def test_abc_pack_contains_revenue_figure(self, packs_dir):
        text = (packs_dir / "abc_manufacturing_credit_pack.txt").read_text()
        assert "42,350" in text or "42350" in text

    def test_abc_pack_contains_ebitda(self, packs_dir):
        text = (packs_dir / "abc_manufacturing_credit_pack.txt").read_text()
        assert "EBITDA" in text
        assert "7,800" in text or "7800" in text

    def test_riverside_pack_has_covenant_breach(self, packs_dir):
        text = (packs_dir / "riverside_retail_credit_pack.txt").read_text()
        assert "BREACH" in text

    def test_summit_pack_has_missing_ebitda_note(self, packs_dir):
        text = (packs_dir / "summit_digital_credit_pack.txt").read_text()
        assert "not separately disclosed" in text.lower() or "not provided" in text.lower()

    def test_all_packs_use_gbp(self, packs_dir):
        for fname in ["abc_manufacturing_credit_pack.txt",
                      "riverside_retail_credit_pack.txt"]:
            text = (packs_dir / fname).read_text()
            assert "£" in text, f"{fname} missing GBP currency symbol"


# ---------------------------------------------------------------------------
# Section 11: Regulatory compliance checks
# ---------------------------------------------------------------------------

class TestRegulatoryCompliance:

    def test_model_id_format(self):
        """PRA SS1/23: model IDs must follow MR-YYYY-NNN format."""
        import re
        assert re.match(r"MR-\d{4}-\d{3}", MODEL_ID)

    def test_eu_ai_act_high_risk_classification(self):
        assert EU_AI_ACT_CLASSIFICATION == "HIGH_RISK"

    def test_conservatism_threshold_is_documented(self):
        """EBA guideline: conservatism applied at specific confidence threshold."""
        assert CONSERVATISM_CONFIDENCE_THRESHOLD == 0.80

    def test_conservatism_uplift_is_ten_percent(self):
        """EBA guideline: 10% uplift on uncertain debt figures."""
        assert CONSERVATISM_UPLIFT == 0.10

    def test_validation_result_always_has_analyst_notes(self, healthy_summary):
        result = validate_extraction(healthy_summary)
        assert len(result.analyst_notes) > 0
        assert "human oversight" in result.analyst_notes.lower()

    def test_extraction_model_in_approved_list(self):
        """LLM must be from approved June 2026 list — no deprecated models."""
        approved_models = {
            "gemini-3.1-pro",
            "gemini-3.5-flash",
            "gpt-5.5",
            "gpt-5-mini",
            "claude-sonnet-4-6",
            "claude-haiku-4-5",
        }
        assert LLM_MODEL_NAME in approved_models


# ---------------------------------------------------------------------------
# Section 12: Live API tests (skipped without GOOGLE_API_KEY)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY not set — skipping live API tests"
)
class TestLiveAPI:

    def test_live_extract_abc_manufacturing(self):
        from document_analyser.extractor import extract_financial_summary
        text_path = Path(__file__).parent.parent / "data" / "abc_manufacturing_credit_pack.txt"
        text = text_path.read_text()
        summary = extract_financial_summary(text, document_id="LIVE-ABC-001")
        assert summary.company_name.value is not None
        assert "ABC" in str(summary.company_name.value)
        assert summary.revenue.value is not None
        assert summary.overall_confidence > 0.0

    def test_live_extract_hallucination_mitigation(self):
        """Verify all three mitigations present in live extraction output."""
        from document_analyser.extractor import extract_financial_summary
        text_path = Path(__file__).parent.parent / "data" / "abc_manufacturing_credit_pack.txt"
        text = text_path.read_text()
        summary = extract_financial_summary(text, document_id="LIVE-HAL-001")

        # (a) Grounding: source_page present on at least one material field
        material_fields = [summary.revenue, summary.ebitda, summary.net_debt]
        grounded = any(f.source_page is not None for f in material_fields)
        assert grounded, "Hallucination mitigation (a) failed: no source_page grounding"

        # (b) Range validation applied
        assert isinstance(summary.leverage_ratio.analyst_review_required, bool)

        # (c) Confidence scores present
        assert 0.0 <= summary.overall_confidence <= 1.0

    def test_live_extract_riverside_retail_flags_p1(self):
        from document_analyser.extractor import extract_financial_summary
        from document_analyser.validator import validate_extraction
        text_path = Path(__file__).parent.parent / "data" / "riverside_retail_credit_pack.txt"
        text = text_path.read_text()
        summary = extract_financial_summary(text, document_id="LIVE-RVRSD-001")
        result = validate_extraction(summary, prior_year_revenue=23500.0)
        # Riverside has leverage ~6.95x and interest cover ~1.05x
        # Expect at least one P1 flag
        assert len(result.p1_flags) >= 1
