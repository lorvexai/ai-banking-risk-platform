"""
AWB AI Governance Platform
Chapter 5: Model Risk Management (PRA SS1/23)

Avon & Wessex Bank plc — PRA SS1/23 compliant governance framework.
"""

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
    ModelMonitor, MonitoringRegistry, MonitoringLogEntry, AlertLevel,
    classify_psi, PSI_GREEN_THRESHOLD, PSI_RED_THRESHOLD,
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

__version__ = "1.0.0"
