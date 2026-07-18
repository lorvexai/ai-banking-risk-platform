# tests/test_ch15.py
# Chapter 15 — Data Infrastructure test suite
# AWB-AI-2025 | PRA SS1/23 | Target: 48 tests
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from datetime import date
from unittest.mock import MagicMock, patch

from customer_360.profile_builder import (
    Customer360Builder,
    CustomerRole,
    CustomerProfile,
)
from customer_360.identity_resolver import (
    IdentityResolver,
    MatchCandidate,
    levenshtein_distance,
)
from risk_warehouse.bcbs239_monitor import (
    BCBS239ComplianceMonitor,
    BCBS239Score,
    AWB_Q1_2026_SCORES,
    BCBS_PRINCIPLES,
)
from governance.retention_policy import (
    DataCategory,
    RetentionPolicy,
    get_retention_policy,
    RETENTION_SCHEDULE,
)


# ── Levenshtein Distance Tests ─────────────────────────────
class TestLevenshteinDistance:
    def test_identical_zero(self):
        assert levenshtein_distance("SMITH", "SMITH") == 0

    def test_one_substitution(self):
        assert levenshtein_distance("SMYTH", "SMITH") == 1

    def test_one_insertion(self):
        assert levenshtein_distance("SMIT", "SMITH") == 1

    def test_one_deletion(self):
        assert levenshtein_distance("SMITHS", "SMITH") == 1

    def test_two_differences(self):
        assert levenshtein_distance("SMIHTS", "SMITH") == 2

    def test_empty_both(self):
        assert levenshtein_distance("", "") == 0

    def test_empty_one_side(self):
        assert levenshtein_distance("ABC", "") == 3

    def test_transposition_is_two(self):
        # Levenshtein counts transposition as 2 ops
        assert levenshtein_distance("SMIHT", "SMITH") == 2


# ── Identity Resolver Tests ────────────────────────────────
class TestIdentityResolver:
    def setup_method(self):
        self.resolver = IdentityResolver(
            db_client=MagicMock()
        )

    def _make_candidate(
        self,
        lev: int,
        score: float,
    ) -> MatchCandidate:
        return MatchCandidate(
            record_id_a="A1",
            record_id_b="B1",
            levenshtein_dist=lev,
            jaro_winkler_score=score,
            dob_match=True,
            postcode_match_level="exact",
            combined_score=score,
            decision="",
        )

    def test_high_levenshtein_rejects_regardless_score(self):
        c = self._make_candidate(lev=3, score=0.99)
        assert self.resolver.resolve_candidate(c) == "reject"

    def test_lev_2_rejects(self):
        c = self._make_candidate(lev=2, score=0.95)
        assert self.resolver.resolve_candidate(c) == "reject"

    def test_lev_1_high_score_auto_links(self):
        c = self._make_candidate(lev=1, score=0.94)
        assert self.resolver.resolve_candidate(c) == "auto_link"

    def test_lev_0_high_score_auto_links(self):
        c = self._make_candidate(lev=0, score=0.95)
        assert self.resolver.resolve_candidate(c) == "auto_link"

    def test_mid_score_manual_review(self):
        c = self._make_candidate(lev=1, score=0.88)
        assert (
            self.resolver.resolve_candidate(c) == "manual_review"
        )

    def test_boundary_0_92_auto_links(self):
        c = self._make_candidate(lev=1, score=0.92)
        assert self.resolver.resolve_candidate(c) == "auto_link"

    def test_boundary_0_85_manual_review(self):
        c = self._make_candidate(lev=1, score=0.85)
        assert (
            self.resolver.resolve_candidate(c) == "manual_review"
        )

    def test_below_review_threshold_rejects(self):
        c = self._make_candidate(lev=1, score=0.60)
        assert self.resolver.resolve_candidate(c) == "reject"

    def test_custom_thresholds_respected(self):
        resolver = IdentityResolver(
            db_client=MagicMock(),
            auto_link_threshold=0.95,
            review_threshold=0.90,
        )
        c = self._make_candidate(lev=1, score=0.93)
        assert resolver.resolve_candidate(c) == "manual_review"


# ── BCBS 239 Monitor Tests ─────────────────────────────────
class TestBCBS239Score:
    def test_compliant_at_90(self):
        s = BCBS239Score("P1", 90.0, date(2026, 3, 31))
        assert s.compliant is True

    def test_non_compliant_below_90(self):
        s = BCBS239Score("P6", 85.0, date(2026, 3, 31))
        assert s.compliant is False

    def test_gap_calculated_correctly(self):
        s = BCBS239Score("P6", 85.0, date(2026, 3, 31))
        assert s.gap == pytest.approx(5.0)

    def test_gap_zero_when_compliant(self):
        s = BCBS239Score("P1", 95.0, date(2026, 3, 31))
        assert s.gap == 0.0


class TestBCBS239ComplianceMonitor:
    def setup_method(self):
        self.monitor = BCBS239ComplianceMonitor(
            db_client=MagicMock(),
            assessment_date=date(2026, 3, 31),
        )

    def test_all_11_principles_in_schedule(self):
        assert len(BCBS_PRINCIPLES) == 11

    def test_all_11_in_awb_scores(self):
        assert len(AWB_Q1_2026_SCORES) == 11

    def test_overall_awb_score_above_85(self):
        # Simple avg = 89.1%; weighted score cited in book
        # (9.2/10) uses PRA priority weighting for P3/P5/P6.
        avg = sum(AWB_Q1_2026_SCORES.values()) / 11
        assert avg >= 85.0, f"AWB avg {avg:.1f}% < 85%"

    def test_p5_timeliness_highest(self):
        max_p = max(
            AWB_Q1_2026_SCORES,
            key=AWB_Q1_2026_SCORES.get,
        )
        assert max_p == "P5-Timeliness"

    def test_p11_distribution_lowest(self):
        min_p = min(
            AWB_Q1_2026_SCORES,
            key=AWB_Q1_2026_SCORES.get,
        )
        assert min_p == "P11-Distribution"

    def test_p11_score_is_74(self):
        assert AWB_Q1_2026_SCORES["P11-Distribution"] == 74.0

    def test_p6_adaptability_is_85(self):
        assert AWB_Q1_2026_SCORES["P6-Adaptability"] == 85.0

    def test_alert_detects_p11_below_80(self):
        scorecard = {
            "P11-Distribution": BCBS239Score(
                "P11-Distribution", 74.0, date(2026, 3, 31)
            )
        }
        failures = self.monitor.alert_if_below_threshold(
            scorecard, 80.0
        )
        assert "P11-Distribution" in failures

    def test_alert_empty_when_all_above_threshold(self):
        scorecard = {
            "P3-Accuracy": BCBS239Score(
                "P3-Accuracy", 94.0, date(2026, 3, 31)
            )
        }
        failures = self.monitor.alert_if_below_threshold(
            scorecard, 80.0
        )
        assert failures == []

    def test_overall_score_calculation(self):
        scorecard = {
            "P1": BCBS239Score("P1", 90.0, date.today()),
            "P2": BCBS239Score("P2", 80.0, date.today()),
        }
        score = self.monitor.overall_score(scorecard)
        assert score == pytest.approx(85.0)

    def test_overall_score_empty_returns_zero(self):
        assert self.monitor.overall_score({}) == 0.0

    def test_gap_actions_for_p6(self):
        actions = self.monitor._get_gap_actions(
            "P6-Adaptability", 85.0
        )
        assert len(actions) >= 1
        assert any("query" in a.lower() for a in actions)

    def test_no_gap_actions_when_compliant(self):
        actions = self.monitor._get_gap_actions("P3-Accuracy", 95.0)
        assert actions == []


# ── Retention Policy Tests ─────────────────────────────────
class TestRetentionPolicy:
    def test_credit_decisions_7_years(self):
        p = get_retention_policy(DataCategory.CREDIT_DECISIONS)
        assert p.retention_years == 7

    def test_credit_decisions_s3_lock(self):
        p = get_retention_policy(DataCategory.CREDIT_DECISIONS)
        assert p.requires_s3_object_lock is True

    def test_credit_basis_is_cobs9(self):
        p = get_retention_policy(DataCategory.CREDIT_DECISIONS)
        assert "COBS 9" in p.regulatory_basis

    def test_sar_records_5_years(self):
        p = get_retention_policy(DataCategory.SAR_RECORDS)
        assert p.retention_years == 5

    def test_sar_records_s3_lock(self):
        p = get_retention_policy(DataCategory.SAR_RECORDS)
        assert p.requires_s3_object_lock is True

    def test_sar_basis_is_mlr2017(self):
        p = get_retention_policy(DataCategory.SAR_RECORDS)
        assert "MLR 2017" in p.regulatory_basis

    def test_model_outputs_7_years(self):
        p = get_retention_policy(DataCategory.MODEL_OUTPUTS)
        assert p.retention_years == 7

    def test_model_outputs_s3_lock(self):
        p = get_retention_policy(DataCategory.MODEL_OUTPUTS)
        assert p.requires_s3_object_lock is True

    def test_model_basis_is_ss123(self):
        p = get_retention_policy(DataCategory.MODEL_OUTPUTS)
        assert "SS1/23" in p.regulatory_basis

    def test_audit_logs_7_years(self):
        p = get_retention_policy(DataCategory.AUDIT_LOGS)
        assert p.retention_years == 7

    def test_training_datasets_5_years(self):
        p = get_retention_policy(
            DataCategory.TRAINING_DATASETS
        )
        assert p.retention_years == 5

    def test_training_datasets_no_s3_lock(self):
        p = get_retention_policy(
            DataCategory.TRAINING_DATASETS
        )
        assert p.requires_s3_object_lock is False

    def test_vector_embeddings_5_years(self):
        p = get_retention_policy(
            DataCategory.VECTOR_EMBEDDINGS
        )
        assert p.retention_years == 5

    def test_s3_lifecycle_days_7yr(self):
        p = get_retention_policy(DataCategory.CREDIT_DECISIONS)
        assert p.to_s3_lifecycle_days() == 2555

    def test_s3_lifecycle_days_5yr(self):
        p = get_retention_policy(DataCategory.SAR_RECORDS)
        assert p.to_s3_lifecycle_days() == 1825

    def test_all_categories_have_schedule_entry(self):
        for cat in DataCategory:
            p = get_retention_policy(cat)
            assert p.retention_years >= 5

    def test_regulated_categories_have_s3_lock(self):
        regulated = [
            DataCategory.CREDIT_DECISIONS,
            DataCategory.AUDIT_LOGS,
            DataCategory.MODEL_OUTPUTS,
            DataCategory.KYC_RECORDS,
            DataCategory.SAR_RECORDS,
            DataCategory.REGULATORY_POSITIONS,
        ]
        for cat in regulated:
            p = get_retention_policy(cat)
            assert p.requires_s3_object_lock is True, (
                f"{cat.value} should require S3 Object Lock"
            )
