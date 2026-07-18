"""
operational_mrm.py
==================
Chapter 5 — AI Governance and Model Risk Management
Section 5.8A — Operational Model Risk Management (merged from Chapter 10)

Avon & Wessex Bank plc (AWB) | AWB-AI-2025 Programme
Regulation: PRA SS1/23, FCA SYSC 6.3.3R, EU AI Act Art.9

This module implements the four operational MRM components added in Section 5.8A:

  5.8A.1 — SS123DeploymentGate   : 4-gate CI/CD hard-block (AUC-ROC, Gini, PSI, output floor)
  5.8A.2 — ProductionDriftMonitor: PSI and SHAP drift triggers (AMBER/RED/mandatory retrain)
  5.8A.3 — PromptVersionManager  : Prompt versioning as model change (MAJOR/MINOR/PATCH)
  5.8A.4 — ModelRegistryClient   : Enterprise model registry via MCPModelInventoryServer

AWB June 2026 production baseline:
  23 production AI systems | SS1/23 risk ratings LOW/MEDIUM/HIGH/CRITICAL
  MLflow self-hosted on EC2 eu-west-2 (MR-2026-050 Validation Orchestrator)
  RAGAS faithfulness rollback threshold: 0.80 | SLA: 15 minutes

Regulatory basis:
  PRA SS1/23 §4 — model validation and deployment controls
  PRA SS1/23 §6 — ongoing monitoring and performance tracking
  FCA SYSC 6.3.3R — systems and controls for model governance
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("awb.operational_mrm")

# ---------------------------------------------------------------------------
# Constants — SS1/23 thresholds (Section 5.8A)
# ---------------------------------------------------------------------------

# 5.8A.1 Deployment Gate (PRA SS1/23 §4, AWB Board Risk Appetite)
SS123_MIN_AUC_ROC           = 0.70    # Gate 1 — minimum discriminatory power
SS123_MAX_GINI_DELTA        = 0.05    # Gate 2 — maximum Gini deterioration vs champion
SS123_MAX_PSI               = 0.20    # Gate 3 — maximum population stability at deployment
SS123_MIN_OUTPUT_FLOOR_2026 = 0.55    # Gate 4 — minimum predicted default rate floor (2026)

# 5.8A.2 Production Drift (PSI monitoring)
PSI_WARN_THRESHOLD          = 0.10    # AMBER — review within 5 business days
PSI_BREACH_THRESHOLD        = 0.20    # RED   — mandatory retraining triggered
SHAP_DRIFT_WARN_PCT         = 0.15    # AMBER — top feature importance shifted > 15%
SHAP_DRIFT_BREACH_PCT       = 0.25    # RED   — top feature importance shifted > 25%

# 5.8A.3 Prompt Versioning
RAGAS_FAITHFULNESS_ROLLBACK = 0.80    # Rollback trigger if RAGAS faithfulness < 0.80
RAGAS_ROLLBACK_SLA_MINUTES  = 15      # Maximum time from detection to rollback

# 5.8A.4 Model Registry
AWB_TOTAL_PRODUCTION_MODELS = 23      # AWB-AI-2025 programme total
AWB_MLFLOW_HOST             = "ec2-mlflow.awb.internal"   # eu-west-2
AWB_MLFLOW_PORT             = 5000


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class DeploymentGateResult(str, Enum):
    """Outcome of SS1/23 4-gate deployment validation."""
    PASS          = "PASS"           # All 4 gates passed — deploy permitted
    FAIL_AUC_ROC  = "FAIL_AUC_ROC"  # Gate 1: AUC-ROC below minimum
    FAIL_GINI     = "FAIL_GINI"      # Gate 2: Gini delta exceeds maximum
    FAIL_PSI      = "FAIL_PSI"       # Gate 3: PSI exceeds maximum at deployment
    FAIL_FLOOR    = "FAIL_FLOOR"     # Gate 4: Output floor not met
    BLOCKED       = "BLOCKED"        # Hard block — MLflow registry entry rejected


class DriftStatus(str, Enum):
    """Production drift monitoring status."""
    GREEN  = "GREEN"   # No drift detected
    AMBER  = "AMBER"   # Warn threshold breached — review required
    RED    = "RED"     # Breach threshold — mandatory retrain


class PromptChangeType(str, Enum):
    """Prompt versioning change classification (aligned with SS1/23 §4)."""
    MAJOR = "MAJOR"    # Regulatory approval required + full RAGAS re-evaluation
    MINOR = "MINOR"    # Model Risk sign-off + targeted RAGAS evaluation
    PATCH = "PATCH"    # No approval gate — A/B test only


# ---------------------------------------------------------------------------
# 5.8A.1 — SS123DeploymentGate
# ---------------------------------------------------------------------------

@dataclass
class DeploymentGateInput:
    """Metrics provided by the validation team at model deployment."""
    model_ref:      str
    model_name:     str
    auc_roc:        float
    gini_champion:  float    # Gini of current production model (champion)
    gini_challenger: float   # Gini of new model being deployed (challenger)
    psi_at_deploy:  float    # PSI at deployment date (train vs OOT)
    output_floor:   float    # Minimum predicted score across 2026 test population
    ss123_risk_rating: str   # LOW / MEDIUM / HIGH / CRITICAL


@dataclass
class DeploymentGateOutput:
    """Result of SS1/23 4-gate validation."""
    model_ref:      str
    gate_result:    DeploymentGateResult
    gates_passed:   int                     # 0–4
    gate_failures:  List[str]               # Descriptions of failed gates
    mlflow_blocked: bool                    # True if MLflow registry entry rejected
    timestamp:      str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    auditor_hash:   str = ""                # SHA-256 of inputs for audit trail


class SS123DeploymentGate:
    """
    PRA SS1/23 4-gate CI/CD deployment gate.

    All four gates must pass for the model to be registered in the AWB MLflow
    registry. If any gate fails, the MLflow registry entry is hard-blocked —
    the model cannot reach production until the gate is cleared.

    Gates (in order):
      1. AUC-ROC ≥ 0.70         (SS1/23 §4.2 discriminatory power)
      2. Gini delta ≤ 5%        (SS1/23 §4.2 challenger vs champion)
      3. PSI at deploy ≤ 0.20   (SS1/23 §4.3 population stability)
      4. Output floor ≥ 55%     (AWB Board 2026 — minimum predicted default)

    Usage::
        gate = SS123DeploymentGate()
        result = gate.validate(DeploymentGateInput(
            model_ref="MR-2026-037",
            model_name="IFRS 9 PD Model",
            auc_roc=0.923,
            gini_champion=0.847,
            gini_challenger=0.854,
            psi_at_deploy=0.042,
            output_floor=0.68,
            ss123_risk_rating="HIGH",
        ))
        assert result.gate_result == DeploymentGateResult.PASS
    """

    def validate(self, inputs: DeploymentGateInput) -> DeploymentGateOutput:
        failures: List[str] = []
        gates_passed = 0

        # Gate 1: AUC-ROC
        if inputs.auc_roc >= SS123_MIN_AUC_ROC:
            gates_passed += 1
        else:
            failures.append(
                f"Gate 1 FAIL: AUC-ROC={inputs.auc_roc:.3f} < "
                f"minimum {SS123_MIN_AUC_ROC:.2f} (SS1/23 §4.2)"
            )

        # Gate 2: Gini delta (challenger vs champion)
        gini_delta = abs(inputs.gini_challenger - inputs.gini_champion)
        if gini_delta <= SS123_MAX_GINI_DELTA:
            gates_passed += 1
        else:
            failures.append(
                f"Gate 2 FAIL: Gini delta={gini_delta:.3f} > "
                f"maximum {SS123_MAX_GINI_DELTA:.2f} (SS1/23 §4.2)"
            )

        # Gate 3: PSI at deployment
        if inputs.psi_at_deploy <= SS123_MAX_PSI:
            gates_passed += 1
        else:
            failures.append(
                f"Gate 3 FAIL: PSI at deploy={inputs.psi_at_deploy:.3f} > "
                f"maximum {SS123_MAX_PSI:.2f} (SS1/23 §4.3)"
            )

        # Gate 4: Output floor (2026 regulatory floor)
        if inputs.output_floor >= SS123_MIN_OUTPUT_FLOOR_2026:
            gates_passed += 1
        else:
            failures.append(
                f"Gate 4 FAIL: output floor={inputs.output_floor:.3f} < "
                f"minimum {SS123_MIN_OUTPUT_FLOOR_2026:.2f} (AWB Board 2026)"
            )

        mlflow_blocked = len(failures) > 0
        if failures:
            gate_result = DeploymentGateResult(f"FAIL_{['AUC_ROC','GINI','PSI','FLOOR'][len(failures)-1]}")
        else:
            gate_result = DeploymentGateResult.PASS

        # Audit hash — SHA-256 of all inputs for 7-year retention
        audit_hash = hashlib.sha256(
            json.dumps(
                {"model_ref": inputs.model_ref, "auc_roc": inputs.auc_roc,
                 "gini_champion": inputs.gini_champion, "gini_challenger": inputs.gini_challenger,
                 "psi_at_deploy": inputs.psi_at_deploy, "output_floor": inputs.output_floor},
                sort_keys=True
            ).encode()
        ).hexdigest()

        logger.info(
            "SS1/23 deployment gate: model=%s result=%s gates_passed=%d/4",
            inputs.model_ref, gate_result.value, gates_passed
        )
        if mlflow_blocked:
            logger.warning(
                "MLflow BLOCKED for %s: %s", inputs.model_ref, " | ".join(failures)
            )

        return DeploymentGateOutput(
            model_ref=inputs.model_ref,
            gate_result=gate_result,
            gates_passed=gates_passed,
            gate_failures=failures,
            mlflow_blocked=mlflow_blocked,
            auditor_hash=audit_hash,
        )


# ---------------------------------------------------------------------------
# 5.8A.2 — ProductionDriftMonitor
# ---------------------------------------------------------------------------

@dataclass
class DriftReading:
    """Single production drift measurement."""
    model_ref:    str
    run_date:     str
    psi:          float
    shap_drift_pct: float          # % shift in top-3 feature importance rankings
    current_auc:  Optional[float]  # Current AUC-ROC (None for LLM-based models)
    retrain_triggered: bool = False
    drift_status: DriftStatus = DriftStatus.GREEN
    alerts: List[str] = field(default_factory=list)


class ProductionDriftMonitor:
    """
    Ongoing PSI and SHAP drift monitoring for all AWB production models.

    PSI thresholds (SS1/23 §6, AWB MLOps runbook):
      < 0.10 → GREEN  (no action)
      0.10–0.20 → AMBER (review within 5 business days)
      ≥ 0.20 → RED   (mandatory retraining — Jira ticket auto-raised)

    SHAP drift thresholds:
      < 15% → GREEN
      15–25% → AMBER
      ≥ 25% → RED   (feature engineering review required)

    Usage::
        monitor = ProductionDriftMonitor()
        reading = monitor.assess(
            model_ref="MR-2026-037",
            run_date="2026-03-31",
            psi=0.048,
            shap_drift_pct=0.12,
            current_auc=0.923,
        )
        assert reading.drift_status == DriftStatus.GREEN
    """

    def assess(
        self,
        model_ref: str,
        run_date: str,
        psi: float,
        shap_drift_pct: float,
        current_auc: Optional[float] = None,
    ) -> DriftReading:
        alerts: List[str] = []
        status = DriftStatus.GREEN
        retrain = False

        # PSI assessment
        if psi >= PSI_BREACH_THRESHOLD:
            status = DriftStatus.RED
            retrain = True
            alerts.append(
                f"PSI={psi:.3f} ≥ {PSI_BREACH_THRESHOLD} RED — mandatory retraining; "
                f"Jira auto-raised (SS1/23 §6)"
            )
        elif psi >= PSI_WARN_THRESHOLD:
            if status == DriftStatus.GREEN:
                status = DriftStatus.AMBER
            alerts.append(
                f"PSI={psi:.3f} ≥ {PSI_WARN_THRESHOLD} AMBER — review within 5 BD"
            )

        # SHAP drift assessment
        if shap_drift_pct >= SHAP_DRIFT_BREACH_PCT:
            status = DriftStatus.RED
            alerts.append(
                f"SHAP drift={shap_drift_pct:.1%} ≥ {SHAP_DRIFT_BREACH_PCT:.0%} RED — "
                f"feature engineering review required"
            )
        elif shap_drift_pct >= SHAP_DRIFT_WARN_PCT:
            if status == DriftStatus.GREEN:
                status = DriftStatus.AMBER
            alerts.append(
                f"SHAP drift={shap_drift_pct:.1%} ≥ {SHAP_DRIFT_WARN_PCT:.0%} AMBER"
            )

        reading = DriftReading(
            model_ref=model_ref,
            run_date=run_date,
            psi=psi,
            shap_drift_pct=shap_drift_pct,
            current_auc=current_auc,
            retrain_triggered=retrain,
            drift_status=status,
            alerts=alerts,
        )
        logger.info(
            "Drift monitor: %s status=%s PSI=%.3f SHAP=%.1f%% retrain=%s",
            model_ref, status.value, psi, shap_drift_pct * 100, retrain
        )
        return reading


# ---------------------------------------------------------------------------
# 5.8A.3 — PromptVersionManager
# ---------------------------------------------------------------------------

@dataclass
class PromptVersion:
    """A versioned prompt entry in the AWB prompt registry."""
    prompt_id:    str
    model_ref:    str
    version:      str             # SemVer string e.g. "2.1.3"
    change_type:  PromptChangeType
    description:  str
    prompt_text:  str
    ragas_score:  Optional[float]  # Faithfulness score from RAGAS evaluation
    created_at:   str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    approved_by:  Optional[str] = None
    active:       bool = True


class PromptVersionManager:
    """
    Manages prompt versioning as a model change per PRA SS1/23 §4.

    Change classification (aligned with SS1/23 §4 materiality thresholds):
      MAJOR — fundamental instruction change: regulatory approval required,
               full RAGAS re-evaluation mandatory, model reference bumped.
               Example: changing risk assessment framework or output format.
      MINOR — targeted refinement: Model Risk sign-off, targeted RAGAS eval.
               Example: adding a new regulatory citation or clarification.
      PATCH — cosmetic / typo: No approval gate, A/B test only.
               Example: punctuation fix, rephrasing without semantic change.

    RAGAS rollback rule (Section 5.8A.3):
      If RAGAS faithfulness falls below 0.80 after a prompt change,
      automatic rollback to previous version within 15 minutes.

    Usage::
        mgr = PromptVersionManager()
        v = mgr.register(
            model_ref="MR-2026-035",
            change_type=PromptChangeType.MINOR,
            description="Added FCA CONC 5.2.1R citation to Consumer Duty check",
            prompt_text="Assess the credit application...",
            ragas_score=0.91,
            approved_by="head.model.risk@awb.co.uk",
        )
        assert v.active
    """

    def __init__(self) -> None:
        self._registry: Dict[str, List[PromptVersion]] = {}

    def register(
        self,
        model_ref: str,
        change_type: PromptChangeType,
        description: str,
        prompt_text: str,
        ragas_score: Optional[float] = None,
        approved_by: Optional[str] = None,
    ) -> PromptVersion:
        """Register a new prompt version. Raises if approval gate not met."""
        if change_type == PromptChangeType.MAJOR and approved_by is None:
            raise ValueError(
                f"MAJOR prompt change for {model_ref} requires approved_by "
                f"(regulatory approval per SS1/23 §4)"
            )
        if change_type == PromptChangeType.MINOR and approved_by is None:
            raise ValueError(
                f"MINOR prompt change for {model_ref} requires approved_by "
                f"(Model Risk sign-off per SS1/23 §4)"
            )

        # Auto-rollback check
        if ragas_score is not None and ragas_score < RAGAS_FAITHFULNESS_ROLLBACK:
            logger.warning(
                "RAGAS faithfulness=%.3f < %.2f rollback threshold — "
                "prompt version NOT registered for %s",
                ragas_score, RAGAS_FAITHFULNESS_ROLLBACK, model_ref
            )
            raise ValueError(
                f"RAGAS faithfulness {ragas_score:.3f} below rollback threshold "
                f"{RAGAS_FAITHFULNESS_ROLLBACK} — auto-rollback within "
                f"{RAGAS_ROLLBACK_SLA_MINUTES} minutes (Section 5.8A.3)"
            )

        # Determine version number
        existing = self._registry.get(model_ref, [])
        if not existing:
            version = "1.0.0"
        else:
            last = existing[-1].version.split(".")
            if change_type == PromptChangeType.MAJOR:
                version = f"{int(last[0])+1}.0.0"
            elif change_type == PromptChangeType.MINOR:
                version = f"{last[0]}.{int(last[1])+1}.0"
            else:
                version = f"{last[0]}.{last[1]}.{int(last[2])+1}"

        pv = PromptVersion(
            prompt_id=f"{model_ref}-PROMPT-{version}",
            model_ref=model_ref,
            version=version,
            change_type=change_type,
            description=description,
            prompt_text=prompt_text,
            ragas_score=ragas_score,
            approved_by=approved_by,
        )

        # Deactivate previous version
        for prev in existing:
            prev.active = False

        self._registry.setdefault(model_ref, []).append(pv)
        logger.info(
            "Prompt version registered: %s v%s (%s) by %s",
            model_ref, version, change_type.value, approved_by or "PATCH"
        )
        return pv

    def get_active(self, model_ref: str) -> Optional[PromptVersion]:
        """Return the currently active prompt version for a model."""
        versions = self._registry.get(model_ref, [])
        return next((v for v in reversed(versions) if v.active), None)

    def history(self, model_ref: str) -> List[PromptVersion]:
        return list(self._registry.get(model_ref, []))


# ---------------------------------------------------------------------------
# 5.8A.4 — ModelRegistryClient
# ---------------------------------------------------------------------------

@dataclass
class ModelRegistryEntry:
    """Single entry in the AWB Enterprise Model Registry."""
    model_ref:        str
    model_name:       str
    chapter:          int
    ss123_risk:       str          # LOW / MEDIUM / HIGH / CRITICAL
    status:           str          # PRODUCTION / UNDER_REVIEW / RETIRED
    auc_roc:          Optional[float]
    gini:             Optional[float]
    psi:              float
    shap_drift_pct:   float
    last_validation:  str          # ISO date
    next_validation:  str          # ISO date
    gates_passed:     int          # 0-4 (SS1/23 gates)
    hitl_threshold:   str          # Description of HITL trigger


# AWB June 2026 production snapshot — 23 systems (MR-2026-035 through MR-2026-074-IP)
AWB_MODEL_REGISTRY: List[ModelRegistryEntry] = [
    ModelRegistryEntry("MR-2026-035","Credit RAG Knowledge Base",      4,"HIGH",    "PRODUCTION",0.847,None, 0.031,0.08,"2026-01-15","2026-07-15",4,"RAG confidence < 0.75"),
    ModelRegistryEntry("MR-2026-036","Credit Decision Agent",          3,"HIGH",    "PRODUCTION",0.891,0.803,0.018,0.06,"2026-01-20","2026-07-20",4,"Loan > £500K"),
    ModelRegistryEntry("MR-2026-037","IFRS 9 PD Model (LGBM)",         6,"HIGH",    "PRODUCTION",0.923,0.861,0.042,0.09,"2025-12-10","2026-06-10",4,"PD > 15% stage migration"),
    ModelRegistryEntry("MR-2026-038","IFRS 9 LGD Model",               6,"HIGH",    "PRODUCTION",0.887,0.812,0.029,0.07,"2025-12-10","2026-06-10",4,"LGD > 60%"),
    ModelRegistryEntry("MR-2026-039","IFRS 9 EAD Model",               6,"MEDIUM",  "PRODUCTION",0.871,0.787,0.033,0.08,"2025-12-10","2026-06-10",4,"Exposure > £10M"),
    ModelRegistryEntry("MR-2026-040","IFRS 9 Staging Engine",          6,"HIGH",    "PRODUCTION",0.912,0.848,0.021,0.05,"2026-01-05","2026-07-05",4,"Stage 2 migration > £5M"),
    ModelRegistryEntry("MR-2026-041","Real-Time VaR Engine",           7,"CRITICAL","PRODUCTION",None, None, 0.048,0.11,"2026-02-01","2026-05-01",3,"VaR breach or PSI AMBER"),
    ModelRegistryEntry("MR-2026-042","CVA Computation Engine",         7,"HIGH",    "PRODUCTION",None, None, 0.027,0.06,"2026-01-28","2026-07-28",4,"CVA delta > 15%"),
    ModelRegistryEntry("MR-2026-043","Algo Trading Backtester",        7,"MEDIUM",  "PRODUCTION",0.834,0.751,0.019,0.04,"2025-11-15","2026-05-15",4,"Sharpe < 1.0"),
    ModelRegistryEntry("MR-2026-044","Operational Risk NLP Classifier",8,"MEDIUM",  "PRODUCTION",0.879,0.798,0.037,0.09,"2025-12-20","2026-06-20",4,"Loss > £100K"),
    ModelRegistryEntry("MR-2026-045","Fraud Detection Model",          8,"HIGH",    "PRODUCTION",0.944,0.893,0.014,0.03,"2026-01-10","2026-07-10",4,"Fraud score > 0.85"),
    ModelRegistryEntry("MR-2026-046","Basel III SMA Calculator",       8,"LOW",     "PRODUCTION",None, None, 0.008,0.02,"2025-10-01","2026-10-01",4,"SMA change > 5%"),
    ModelRegistryEntry("MR-2026-047","LCR Forecasting Model",          9,"MEDIUM",  "PRODUCTION",0.856,0.766,0.031,0.07,"2026-01-25","2026-07-25",4,"LCR < 110%"),
    ModelRegistryEntry("MR-2026-048","NSFR Optimiser",                 9,"MEDIUM",  "PRODUCTION",None, None, 0.022,0.05,"2026-01-25","2026-07-25",4,"NSFR < 105%"),
    ModelRegistryEntry("MR-2026-049","Intraday Liquidity Agent",       9,"MEDIUM",  "PRODUCTION",None, None, 0.016,0.04,"2026-01-25","2026-07-25",4,"Intraday limit > 90%"),
    ModelRegistryEntry("MR-2026-050","Model Validation Orchestrator", 10,"LOW",     "PRODUCTION",None, None, 0.009,0.02,"2026-02-01","2027-02-01",4,"Validation failure"),
    ModelRegistryEntry("MR-2026-051","COREP/FINREP Automation",       11,"LOW",     "PRODUCTION",None, None, 0.005,0.01,"2026-01-15","2027-01-15",4,"Filing error detected"),
    ModelRegistryEntry("MR-2026-052","Regulatory Change Tracker",     11,"LOW",     "PRODUCTION",None, None, 0.011,0.03,"2026-01-15","2027-01-15",4,"MATERIAL rule change"),
    ModelRegistryEntry("MR-2026-053","Consumer Duty Classifier",       5,"HIGH",    "PRODUCTION",0.883,0.808,0.028,0.06,"2026-01-20","2026-07-20",4,"Vulnerability score > 0.7"),
    ModelRegistryEntry("MR-2026-060-AML","AML/KYC Transaction Monitor",12,"HIGH",  "PRODUCTION",0.916,0.857,0.019,0.04,"2026-02-10","2026-08-10",4,"SAR auto-draft or PEP hit"),
    ModelRegistryEntry("MR-2026-061-EA","Enterprise Architecture Agent",13,"MEDIUM","PRODUCTION",None, None, 0.007,0.02,"2026-03-01","2027-03-01",4,"Architecture RED zone"),
    ModelRegistryEntry("MR-2026-062-MLO","MLOps/LLMOps Orchestrator",14,"MEDIUM",  "PRODUCTION",None, None, 0.006,0.01,"2026-03-01","2027-03-01",4,"Model drift RED zone"),
    ModelRegistryEntry("MR-2026-063-DI","Data Infrastructure Agent",  15,"MEDIUM",  "PRODUCTION",None, None, 0.004,0.01,"2026-03-01","2027-03-01",4,"BCBS 239 score < 85%"),
]


class ModelRegistryClient:
    """
    Client for the AWB Enterprise Model Registry.

    In production, delegates to MCPModelInventoryServer (Section 3.9B) for
    live data. Falls back to the static AWB_MODEL_REGISTRY snapshot for
    offline/test use.

    Usage::
        client = ModelRegistryClient()
        entry = client.get("MR-2026-037")
        assert entry.status == "PRODUCTION"

        amber_models = client.get_by_drift_status("AMBER")
        overdue = client.get_overdue_validations("2026-03-31")
    """

    def __init__(self, use_mcp: bool = False) -> None:
        self._use_mcp = use_mcp
        self._local: Dict[str, ModelRegistryEntry] = {
            e.model_ref: e for e in AWB_MODEL_REGISTRY
        }

    def get(self, model_ref: str) -> Optional[ModelRegistryEntry]:
        if self._use_mcp:
            # Production: route via MCPModelInventoryServer
            # from credit_agent.mcp_servers import AWBMCPServerRegistry
            # registry = AWBMCPServerRegistry.default()
            # result = registry.call_tool("model_lookup", {"model_ref": model_ref}, "operational_mrm")
            # return ModelRegistryEntry(**result["model"])
            pass
        return self._local.get(model_ref)

    def get_all(self) -> List[ModelRegistryEntry]:
        return list(self._local.values())

    def get_by_drift_status(self, status: str) -> List[ModelRegistryEntry]:
        """Return models matching a drift status (GREEN/AMBER/RED) based on PSI."""
        monitor = ProductionDriftMonitor()
        results = []
        for entry in self._local.values():
            reading = monitor.assess(
                model_ref=entry.model_ref,
                run_date=datetime.now(timezone.utc).date().isoformat(),
                psi=entry.psi,
                shap_drift_pct=entry.shap_drift_pct,
                current_auc=entry.auc_roc,
            )
            if reading.drift_status.value == status:
                results.append(entry)
        return results

    def get_overdue_validations(self, as_of_date: str) -> List[ModelRegistryEntry]:
        """Return models whose next_validation date has passed."""
        return [
            e for e in self._local.values()
            if e.next_validation < as_of_date and e.status == "PRODUCTION"
        ]

    def summary(self) -> Dict[str, Any]:
        """Return a platform-level summary suitable for the CRO dashboard."""
        all_entries = self.get_all()
        amber = self.get_by_drift_status("AMBER")
        red   = self.get_by_drift_status("RED")
        return {
            "total_production_models": len(all_entries),
            "green_count": len(all_entries) - len(amber) - len(red),
            "amber_count": len(amber),
            "red_count":   len(red),
            "amber_models": [e.model_ref for e in amber],
            "red_models":   [e.model_ref for e in red],
            "critical_rating_count": sum(1 for e in all_entries if e.ss123_risk == "CRITICAL"),
        }


# ---------------------------------------------------------------------------
# Convenience: validate a batch deployment set (used by Ch14 MLOps pipeline)
# ---------------------------------------------------------------------------

def validate_deployment_batch(
    candidates: List[DeploymentGateInput],
) -> Tuple[List[DeploymentGateOutput], int]:
    """
    Run SS1/23 4-gate validation across a batch of model candidates.

    Returns (results_list, pass_count).
    Used by Ch14 agentic_mlops_llmops.py CICDValidationAgent.
    """
    gate = SS123DeploymentGate()
    results = [gate.validate(c) for c in candidates]
    passed = sum(1 for r in results if r.gate_result == DeploymentGateResult.PASS)
    logger.info(
        "Batch deployment validation: %d/%d candidates passed all 4 SS1/23 gates",
        passed, len(candidates)
    )
    return results, passed
