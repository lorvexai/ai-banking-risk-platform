"""
tests/test_model_governance.py
Comprehensive test suite for AWB AI Governance Platform
Chapter 5: Model Risk Management (PRA SS1/23)

Test coverage:
- Model inventory and ID validation (12 tests)
- Validation framework (11 tests)
- Monitoring and PSI drift detection (9 tests)
- Incident management (10 tests)
- Fairness testing (11 tests)
- Sample data generation (3 tests)

Total: 56 tests
No LLM calls — pure Python governance framework.

Run with: pytest tests/test_model_governance.py -v
"""

from __future__ import annotations

import datetime
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model_governance.model_inventory import (
    ModelCard, ModelInventory, RiskRating, EUAIActClassification,
    ValidationStatus, ModelStatus, validate_model_id, build_awb_model_inventory,
)
from model_governance.validation_framework import (
    ValidationTest, ValidationReport, ValidationResult, TestType, OverallResult,
    ModelValidator,
    run_gini_coefficient_test, run_psi_test, run_accuracy_test,
    run_documentation_test, run_demographic_parity_test,
)
from model_governance.monitoring import (
    ModelMonitor, MonitoringRegistry, AlertLevel, classify_psi,
    PSI_GREEN_THRESHOLD, PSI_RED_THRESHOLD,
)
from model_governance.incident_management import (
    AIIncident, IncidentLog, IncidentSeverity, IncidentStatus,
    classify_incident_severity, SEVERITY_NOTIFICATION_MAP,
)
from model_governance.fairness_testing import (
    FairnessReport, FairnessTestResult, FairnessAssessor, GroupStatistics,
    FairnessResult, FairnessMetric,
    run_disparate_impact_test, run_demographic_parity_test, run_equal_opportunity_test,
    DISPARATE_IMPACT_THRESHOLD,
)
from data.generate_sample_model_inventory import build_extended_inventory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_model_card() -> ModelCard:
    return ModelCard(
        model_id="MR-2026-001",
        model_name="Test Credit Model",
        version="1.0.0",
        purpose="Test credit scoring model for unit tests.",
        model_type="Logistic Regression",
        inputs=["Income", "Credit score"],
        outputs=["Approval probability"],
        limitations=["Limited to UK applicants"],
        risk_rating=RiskRating.HIGH,
        eu_ai_act_classification=EUAIActClassification.HIGH_RISK,
        owner="Head of Credit Risk",
        monitoring_plan="Monthly PSI; quarterly GINI review.",
    )


@pytest.fixture
def inventory() -> ModelInventory:
    return ModelInventory()


@pytest.fixture
def populated_inventory() -> ModelInventory:
    return build_awb_model_inventory()


@pytest.fixture
def monitor() -> ModelMonitor:
    return ModelMonitor("MR-2026-001", "Test Model")


@pytest.fixture
def incident_log() -> IncidentLog:
    return IncidentLog()


@pytest.fixture
def group_stats_passing() -> list:
    """Group stats where disparate impact ratio > 0.90 (PASS) and parity gap < 5pp."""
    return [
        GroupStatistics("Group_A", "gender", approval_rate=0.70),
        GroupStatistics("Group_B", "gender", approval_rate=0.67),  # 0.67/0.70 = 0.957 > 0.90; gap = 3pp < 5pp
    ]


@pytest.fixture
def group_stats_failing() -> list:
    """Group stats where disparate impact ratio < 0.80 (FAIL)."""
    return [
        GroupStatistics("Group_A", "gender", approval_rate=0.70),
        GroupStatistics("Group_B", "gender", approval_rate=0.50),  # 0.50/0.70 = 0.714 < 0.80
    ]


# ===========================================================================
# SECTION 1: Model Inventory and ID Validation (12 tests)
# ===========================================================================

class TestModelIdValidation:

    def test_valid_id_accepted(self):
        validate_model_id("MR-2026-001")  # Should not raise

    def test_valid_id_accepted_different_year(self):
        validate_model_id("MR-2022-999")

    def test_invalid_id_wrong_prefix_raises(self):
        with pytest.raises(ValueError, match="MR-YYYY-NNN"):
            validate_model_id("SR-2026-001")  # SR is US SR 11-7; not AWB format

    def test_invalid_id_missing_year_raises(self):
        with pytest.raises(ValueError):
            validate_model_id("MR-26-001")

    def test_invalid_id_two_digit_seq_raises(self):
        with pytest.raises(ValueError):
            validate_model_id("MR-2026-01")   # Must be 3 digits

    def test_invalid_id_empty_raises(self):
        with pytest.raises(ValueError):
            validate_model_id("")

    def test_model_card_rejects_invalid_id(self):
        with pytest.raises(ValueError, match="MR-YYYY-NNN"):
            ModelCard(
                model_id="INVALID-ID",
                model_name="Test", version="1.0", purpose="Test",
                model_type="LR", inputs=["x"], outputs=["y"],
                limitations=[], risk_rating=RiskRating.LOW,
                eu_ai_act_classification=EUAIActClassification.MINIMAL,
                owner="Test",
            )

    def test_model_card_blank_name_raises(self):
        with pytest.raises(ValueError, match="model_name"):
            ModelCard(
                model_id="MR-2026-001",
                model_name="   ", version="1.0", purpose="Test",
                model_type="LR", inputs=["x"], outputs=["y"],
                limitations=[], risk_rating=RiskRating.LOW,
                eu_ai_act_classification=EUAIActClassification.MINIMAL,
                owner="Test",
            )


class TestModelInventory:

    def test_register_model(self, inventory, valid_model_card):
        inventory.register(valid_model_card)
        assert "MR-2026-001" in inventory

    def test_register_duplicate_raises(self, inventory, valid_model_card):
        inventory.register(valid_model_card)
        with pytest.raises(ValueError, match="already registered"):
            inventory.register(valid_model_card)

    def test_get_by_risk_rating(self, populated_inventory):
        high_risk = populated_inventory.get_by_risk_rating(RiskRating.HIGH)
        assert len(high_risk) > 0
        for m in high_risk:
            assert m.risk_rating == RiskRating.HIGH

    def test_overdue_validation_detection(self, populated_inventory):
        """MR-2022-015 has next_validation_due in the past."""
        overdue = populated_inventory.get_overdue_validations()
        overdue_ids = [m.model_id for m in overdue]
        assert "MR-2022-015" in overdue_ids

    def test_export_to_dict_returns_list(self, populated_inventory):
        data = populated_inventory.export_to_dict()
        assert isinstance(data, list)
        assert len(data) == len(populated_inventory)

    def test_inventory_len(self, populated_inventory):
        assert len(populated_inventory) == 7  # Pre-populated with 7 models (Chs 1-5)

    def test_update_status(self, inventory, valid_model_card):
        inventory.register(valid_model_card)
        updated = inventory.update_status(
            "MR-2026-001",
            validation_status=ValidationStatus.VALIDATED,
            change_note="Validation completed by Model Risk.",
        )
        assert updated.validation_status == ValidationStatus.VALIDATED
        assert len(updated.change_log) > 0

    def test_update_nonexistent_raises(self, inventory):
        with pytest.raises(KeyError):
            inventory.update_status("MR-9999-999")


# ===========================================================================
# SECTION 2: Validation Framework (11 tests)
# ===========================================================================

class TestGiniCoefficient:

    def test_high_gini_passes(self):
        result = run_gini_coefficient_test(0.65)
        assert result.result == ValidationResult.PASS

    def test_low_gini_fails(self):
        result = run_gini_coefficient_test(0.30)
        assert result.result == ValidationResult.FAIL

    def test_boundary_gini_passes(self):
        result = run_gini_coefficient_test(0.40)  # Exactly at threshold
        assert result.result == ValidationResult.PASS

    def test_gini_test_type_is_performance(self):
        result = run_gini_coefficient_test(0.55)
        assert result.test_type == TestType.PERFORMANCE


class TestPSIValidation:

    def test_low_psi_passes(self):
        result = run_psi_test(0.05)
        assert result.result == ValidationResult.PASS

    def test_moderate_psi_is_conditional(self):
        result = run_psi_test(0.15)
        assert result.result == ValidationResult.CONDITIONAL

    def test_high_psi_fails(self):
        result = run_psi_test(0.25)
        assert result.result == ValidationResult.FAIL

    def test_psi_test_type_is_stability(self):
        result = run_psi_test(0.08)
        assert result.test_type == TestType.STABILITY


class TestDocumentationCompleteness:

    def test_complete_card_passes(self, valid_model_card):
        result = run_documentation_test(valid_model_card)
        assert result.result == ValidationResult.PASS

    def test_missing_monitoring_plan_fails(self):
        card = ModelCard(
            model_id="MR-2026-002", model_name="Incomplete Model",
            version="1.0", purpose="Test", model_type="LR",
            inputs=["x"], outputs=["y"], limitations=[],
            risk_rating=RiskRating.LOW,
            eu_ai_act_classification=EUAIActClassification.MINIMAL,
            owner="Test", monitoring_plan="",  # Blank
        )
        result = run_documentation_test(card)
        assert result.result == ValidationResult.FAIL


class TestValidationRunner:

    def test_standard_suite_produces_report(self, valid_model_card):
        report = ModelValidator.run_standard_suite(
            model_card=valid_model_card,
            gini=0.58, psi=0.07, accuracy=0.85,
        )
        assert isinstance(report, ValidationReport)
        assert report.model_id == "MR-2026-001"

    def test_failing_gini_produces_fail_report(self, valid_model_card):
        report = ModelValidator.run_standard_suite(
            model_card=valid_model_card,
            gini=0.25,  # Below threshold
        )
        assert report.overall_result == OverallResult.FAIL

    def test_conditional_psi_produces_conditional_report(self, valid_model_card):
        report = ModelValidator.run_standard_suite(
            model_card=valid_model_card,
            psi=0.15,  # AMBER
        )
        assert report.overall_result == OverallResult.CONDITIONAL


# ===========================================================================
# SECTION 3: Monitoring and PSI Drift (9 tests)
# ===========================================================================

class TestPSIClassification:

    def test_psi_below_010_is_green(self):
        assert classify_psi(0.05) == AlertLevel.GREEN

    def test_psi_exactly_010_is_amber(self):
        assert classify_psi(0.10) == AlertLevel.AMBER

    def test_psi_between_010_020_is_amber(self):
        assert classify_psi(0.15) == AlertLevel.AMBER

    def test_psi_above_020_is_red(self):
        assert classify_psi(0.25) == AlertLevel.RED

    def test_negative_psi_raises(self):
        with pytest.raises(ValueError, match="negative"):
            classify_psi(-0.01)


class TestModelMonitor:

    def test_record_psi_green(self, monitor):
        entry = monitor.record_psi(0.05)
        assert entry.alert_level == AlertLevel.GREEN
        assert not entry.requires_action

    def test_record_psi_amber_requires_action(self, monitor):
        entry = monitor.record_psi(0.12)
        assert entry.alert_level == AlertLevel.AMBER
        assert entry.requires_action

    def test_record_psi_red_requires_action(self, monitor):
        entry = monitor.record_psi(0.30)
        assert entry.alert_level == AlertLevel.RED
        assert entry.requires_action

    def test_get_alerts_filters_by_level(self, monitor):
        monitor.record_psi(0.05)   # GREEN
        monitor.record_psi(0.15)   # AMBER
        monitor.record_psi(0.25)   # RED
        amber_plus = monitor.get_alerts(AlertLevel.AMBER)
        assert len(amber_plus) == 2  # AMBER + RED

    def test_has_red_alert_true(self, monitor):
        monitor.record_psi(0.30)
        assert monitor.has_red_alert()

    def test_has_red_alert_false(self, monitor):
        monitor.record_psi(0.05)
        assert not monitor.has_red_alert()


# ===========================================================================
# SECTION 4: Incident Management (10 tests)
# ===========================================================================

class TestIncidentSeverityClassifier:

    def test_customer_harm_is_p1(self):
        assert classify_incident_severity(customer_harm=True) == IncidentSeverity.P1

    def test_regulatory_breach_is_p1(self):
        assert classify_incident_severity(regulatory_breach=True) == IncidentSeverity.P1

    def test_performance_degraded_is_p2(self):
        assert classify_incident_severity(performance_degraded=True) == IncidentSeverity.P2

    def test_monitoring_alert_is_p3(self):
        assert classify_incident_severity(monitoring_alert_only=True) == IncidentSeverity.P3

    def test_no_flags_is_p4(self):
        assert classify_incident_severity() == IncidentSeverity.P4


class TestAIIncident:

    def test_p1_incident_sets_regulatory_notification(self):
        incident = AIIncident(
            model_id="MR-2026-036",
            severity=IncidentSeverity.P1,
            title="Credit model produced biased outputs",
            description="P1 customer harm incident.",
        )
        assert incident.regulatory_notification_required is True

    def test_p4_incident_no_regulatory_notification(self):
        incident = AIIncident(
            model_id="MR-2026-030",
            severity=IncidentSeverity.P4,
            title="Minor latency spike",
            description="Latency increased by 200ms for 10 minutes.",
        )
        assert incident.regulatory_notification_required is False

    def test_p1_notification_deadline_4_hours(self):
        detected = datetime.datetime(2026, 3, 10, 9, 0, 0)
        incident = AIIncident(
            model_id="MR-2026-036",
            severity=IncidentSeverity.P1,
            title="Test P1",
            description="Test.",
            detected_at=detected,
        )
        expected_deadline = detected + datetime.timedelta(hours=4)
        assert incident.notification_deadline == expected_deadline

    def test_mark_resolved_updates_status(self):
        incident = AIIncident(
            model_id="MR-2026-036",
            severity=IncidentSeverity.P2,
            title="Test",
            description="Test P2.",
        )
        incident.mark_resolved(
            root_cause="Data quality issue in input pipeline",
            remediation_actions=["Fix ETL pipeline", "Reprocess affected records"],
        )
        assert incident.status == IncidentStatus.RESOLVED
        assert incident.resolved_at is not None

    def test_eu_ai_act_serious_incident_flag(self):
        incident = AIIncident(
            model_id="MR-2026-036",
            severity=IncidentSeverity.P1,
            title="Test",
            description="Test.",
            eu_ai_act_serious_incident=True,
        )
        recipients = incident.required_notification_recipients
        from model_governance.incident_management import NotificationRecipient
        assert NotificationRecipient.EU_AI_OFFICE in recipients


class TestIncidentLog:

    def test_create_and_retrieve_incident(self, incident_log):
        inc = incident_log.create_incident(
            model_id="MR-2026-036",
            model_name="Test Model",
            severity=IncidentSeverity.P2,
            title="Performance degradation",
            description="GINI dropped from 0.62 to 0.45.",
        )
        retrieved = incident_log.get(inc.incident_id)
        assert retrieved.incident_id == inc.incident_id

    def test_get_open_incidents(self, incident_log):
        incident_log.create_incident(
            "MR-2026-036", "Test", IncidentSeverity.P3, "Test", "Test"
        )
        open_incidents = incident_log.get_open()
        assert len(open_incidents) >= 1


# ===========================================================================
# SECTION 5: Fairness Testing (11 tests)
# ===========================================================================

class TestDisparateImpact:

    def test_passing_ratio_returns_pass(self, group_stats_passing):
        result = run_disparate_impact_test(group_stats_passing, "Group_A", "gender")
        assert result.result == FairnessResult.PASS

    def test_failing_ratio_returns_fail(self, group_stats_failing):
        result = run_disparate_impact_test(group_stats_failing, "Group_A", "gender")
        assert result.result == FairnessResult.FAIL

    def test_disparate_impact_below_080_fails(self):
        stats = [
            GroupStatistics("Majority", "ethnicity", approval_rate=0.80),
            GroupStatistics("Minority", "ethnicity", approval_rate=0.60),  # 0.75 < 0.80
        ]
        result = run_disparate_impact_test(stats, "Majority", "ethnicity")
        assert result.result == FairnessResult.FAIL
        assert "Minority" in result.failing_groups

    def test_reference_group_missing_raises(self, group_stats_passing):
        with pytest.raises(ValueError, match="Reference group"):
            run_disparate_impact_test(group_stats_passing, "NonExistent", "gender")

    def test_worst_ratio_computed_correctly(self):
        stats = [
            GroupStatistics("G1", "age", approval_rate=0.80),
            GroupStatistics("G2", "age", approval_rate=0.72),  # 0.90
            GroupStatistics("G3", "age", approval_rate=0.68),  # 0.85
        ]
        result = run_disparate_impact_test(stats, "G1", "age")
        # Worst ratio is 0.68/0.80 = 0.85
        assert abs(result.worst_ratio - 0.85) < 0.01


class TestDemographicParity:

    def test_similar_rates_pass(self):
        stats = [
            GroupStatistics("Male", "gender", approval_rate=0.70),
            GroupStatistics("Female", "gender", approval_rate=0.68),  # Within ±5pp
        ]
        result = run_demographic_parity_test(stats, "Male", "gender")
        assert result.result == FairnessResult.PASS

    def test_large_gap_fails(self):
        stats = [
            GroupStatistics("Male", "gender", approval_rate=0.70),
            GroupStatistics("Female", "gender", approval_rate=0.55),  # 15pp gap
        ]
        result = run_demographic_parity_test(stats, "Male", "gender")
        assert result.result == FairnessResult.FAIL


class TestEqualOpportunity:

    def test_equal_tpr_passes(self):
        stats = [
            GroupStatistics("G1", "age", approval_rate=0.70, true_positive_rate=0.80),
            GroupStatistics("G2", "age", approval_rate=0.65, true_positive_rate=0.78),
        ]
        result = run_equal_opportunity_test(stats, "G1", "age")
        assert result.result == FairnessResult.PASS

    def test_missing_tpr_raises(self):
        stats = [
            GroupStatistics("G1", "age", approval_rate=0.70),  # No TPR
            GroupStatistics("G2", "age", approval_rate=0.65),
        ]
        with pytest.raises(ValueError, match="true_positive_rate"):
            run_equal_opportunity_test(stats, "G1", "age")


class TestFairnessAssessor:

    def test_full_assessment_failing(self, group_stats_failing):
        assessor = FairnessAssessor()
        report = assessor.run_assessment(
            model_id="MR-2026-036",
            model_name="Test",
            characteristics={"gender": (group_stats_failing, "Group_A")},
        )
        assert report.overall_result == FairnessResult.FAIL
        assert report.remediation_required

    def test_full_assessment_passing(self, group_stats_passing):
        assessor = FairnessAssessor()
        report = assessor.run_assessment(
            model_id="MR-2026-036",
            model_name="Test",
            characteristics={"gender": (group_stats_passing, "Group_A")},
        )
        assert report.overall_result in (FairnessResult.PASS, FairnessResult.REVIEW_REQUIRED)

    def test_report_has_regulatory_references(self, group_stats_passing):
        assessor = FairnessAssessor()
        report = assessor.run_assessment(
            model_id="MR-2026-001",
            model_name="Test",
            characteristics={"gender": (group_stats_passing, "Group_A")},
        )
        assert len(report.regulatory_references) > 0
        # EU AI Act must be referenced for credit models
        refs_str = " ".join(report.regulatory_references)
        assert "EU AI Act" in refs_str


# ===========================================================================
# SECTION 6: EU AI Act Classification (4 tests)
# ===========================================================================

class TestEUAIActClassification:

    def test_credit_scoring_is_high_risk(self, populated_inventory):
        """SME credit scoring model must be classified HIGH_RISK per Annex III."""
        model = populated_inventory.get("MR-2022-015")
        assert model.eu_ai_act_classification == EUAIActClassification.HIGH_RISK

    def test_chatbot_is_limited_risk(self, populated_inventory):
        """Chatbot with transparency obligations is LIMITED risk."""
        model = populated_inventory.get("MR-2026-030")
        assert model.eu_ai_act_classification == EUAIActClassification.LIMITED

    def test_high_risk_requires_conformity_assessment(self, valid_model_card):
        assert valid_model_card.requires_eu_ai_act_conformity_assessment()

    def test_minimal_risk_no_conformity_required(self):
        card = ModelCard(
            model_id="MR-2026-003", model_name="Low Risk Tool",
            version="1.0", purpose="Internal analytics tool.",
            model_type="SQL queries", inputs=["GL data"], outputs=["Reports"],
            limitations=[], risk_rating=RiskRating.LOW,
            eu_ai_act_classification=EUAIActClassification.MINIMAL,
            owner="Finance",
        )
        assert not card.requires_eu_ai_act_conformity_assessment()


# ===========================================================================
# SECTION 7: Sample Data Generation (3 tests)
# ===========================================================================

class TestSampleInventoryGeneration:

    def test_extended_inventory_has_8_models(self):
        inv = build_extended_inventory()
        assert len(inv) == 8

    def test_all_required_risk_ratings_present(self):
        inv = build_extended_inventory()
        # Directly read risk_rating from ModelCard objects
        ratings = {inv.get(mid).risk_rating for mid in [
            "MR-2022-015",   # CRITICAL
            "MR-2023-021",   # HIGH
            "MR-2026-039",   # MEDIUM
            "MR-2026-040",   # LOW
        ]}
        assert RiskRating.CRITICAL in ratings
        assert RiskRating.HIGH in ratings
        assert RiskRating.MEDIUM in ratings
        assert RiskRating.LOW in ratings

    def test_all_model_ids_valid_format(self):
        inv = build_extended_inventory()
        for model in inv.export_to_dict():
            validate_model_id(model["model_id"])
