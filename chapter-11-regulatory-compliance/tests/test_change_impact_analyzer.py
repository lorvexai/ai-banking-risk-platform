"""Tests for the Regulatory Change Impact Analyzer (Section 11.4A,
MR-2026-069-REG). Exercises the pipeline offline (no Gemini API key
required): the summary step falls back to a deterministic template.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from change_impact.analyzer import (
    RegulatoryChangeAnalyzer,
    RegulatoryChangeEvent,
    DocumentStatus,
    CrossReferenceTable,
    CrossReferenceRow,
    CitationType,
    MODEL_ID,
)


@pytest.fixture
def analyzer() -> RegulatoryChangeAnalyzer:
    return RegulatoryChangeAnalyzer()


class TestCrossReferenceTable:
    def test_default_rows_include_output_floor(self):
        table = CrossReferenceTable()
        rows = table.find_affected("CRR3 Art. 92a")
        assert len(rows) == 3
        system_ids = {r.affected_system_id for r in rows}
        assert system_ids == {"MR-2026-043", "MR-2026-071", "MR-2026-072"}

    def test_unknown_citation_returns_empty(self):
        table = CrossReferenceTable()
        assert table.find_affected("CRR3 Art. 999") == []

    def test_add_row(self):
        table = CrossReferenceTable(rows=[])
        table.add_row(
            CrossReferenceRow(
                citation_type=CitationType.PRA_RULEBOOK,
                citation_ref="PRA Rulebook Ch. 4",
                affected_system_id="MR-2026-047",
            )
        )
        assert len(table.find_affected("PRA Rulebook Ch. 4")) == 1


class TestAnalyzeChange:
    def test_output_floor_amendment_identifies_all_affected_systems(self, analyzer):
        # Mirrors the Q1 2026 CRR3 output floor scenario in Section 11.4A.3:
        # the Basel Credit Risk Reporting module, the MJRRP RWA Calculation
        # Engine, and the Chapter 6 IRB PD model all affected, alongside the
        # COREP C 02.00 template.
        event = RegulatoryChangeEvent(
            document_id="CRR3_output_floor_2026Q1",
            from_status=DocumentStatus.CONSULTATION,
            to_status=DocumentStatus.FINAL,
        )
        report = analyzer.analyze_change(
            event,
            citation_ref="CRR3 Art. 92a",
            old_text="Output floor phased in at 65% from 1 January 2026.",
            new_text="Output floor phased in at 70% from 1 January 2026.",
        )
        assert report.affected_system_ids == [
            "MR-2026-043", "MR-2026-071", "MR-2026-072",
        ]
        assert "COREP C 02.00" in report.affected_report_templates
        assert "COREP C 08.00" in report.affected_report_templates
        assert report.model_id == MODEL_ID
        assert MODEL_ID in report.summary_text

    def test_draft_status_raises(self, analyzer):
        event = RegulatoryChangeEvent(
            document_id="DRAFT-001",
            from_status=DocumentStatus.DRAFT,
            to_status=DocumentStatus.CONSULTATION,
        )
        with pytest.raises(ValueError):
            analyzer.analyze_change(
                event, citation_ref="CRR3 Art. 92a", old_text="a", new_text="b"
            )

    def test_no_matching_citation_returns_empty_lists(self, analyzer):
        event = RegulatoryChangeEvent(
            document_id="UNRELATED-001",
            from_status=DocumentStatus.CONSULTATION,
            to_status=DocumentStatus.FINAL,
        )
        report = analyzer.analyze_change(
            event, citation_ref="CRR3 Art. 000", old_text="a", new_text="b"
        )
        assert report.affected_system_ids == []
        assert report.affected_report_templates == []


class TestHumanGate:
    def test_new_report_requires_review(self, analyzer):
        event = RegulatoryChangeEvent(
            document_id="CRR3_output_floor_2026Q1",
            from_status=DocumentStatus.CONSULTATION,
            to_status=DocumentStatus.FINAL,
        )
        report = analyzer.analyze_change(
            event, citation_ref="CRR3 Art. 92a", old_text="a", new_text="b"
        )
        assert report.requires_review is True

    def test_route_for_review_does_not_clear_requires_review(self, analyzer):
        event = RegulatoryChangeEvent(
            document_id="CRR3_output_floor_2026Q1",
            from_status=DocumentStatus.CONSULTATION,
            to_status=DocumentStatus.FINAL,
        )
        report = analyzer.analyze_change(
            event, citation_ref="CRR3 Art. 92a", old_text="a", new_text="b"
        )
        analyzer.route_for_review(report)
        assert report.requires_review is True  # not yet confirmed reviewed

    def test_confirm_reviewed_clears_requires_review(self, analyzer):
        event = RegulatoryChangeEvent(
            document_id="CRR3_output_floor_2026Q1",
            from_status=DocumentStatus.CONSULTATION,
            to_status=DocumentStatus.FINAL,
        )
        report = analyzer.analyze_change(
            event, citation_ref="CRR3 Art. 92a", old_text="a", new_text="b"
        )
        analyzer.route_for_review(report)
        analyzer.confirm_reviewed(report, reviewer="J. Okafor", notes="Confirmed scope")
        assert report.requires_review is False
        assert report.reviewed_by == "J. Okafor"
