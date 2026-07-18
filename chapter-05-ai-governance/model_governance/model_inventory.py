"""
model_governance/model_inventory.py
AWB AI Governance Platform — Model Inventory
Chapter 5: Model Risk Management (PRA SS1/23)

Implements AWB's model inventory in compliance with PRA SS1/23 Section 2.1:
"Firms are expected to maintain a complete and current model inventory."

Model ID format: MR-YYYY-NNN
  MR  = Model Registration
  YYYY = registration year
  NNN  = sequential three-digit number

Pre-populated with 6 AWB AI systems from Chapters 1–5.

Regulatory context:
- PRA SS1/23 Section 2.1: Complete model inventory, maintained by model risk function
- PRA SS1/23 Section 5.1: Inventory available to PRA within 5 business days
- PRA AI/ML Roundtables (Oct 2025, Feb 2026): explicit agentic AI governance and
  board-level risk appetite requirements; traditional validation must be adapted
  for generative and agentic systems (Bank of England, November 2025)
- EU AI Act 2024 Annex III: High-risk classification for credit-scoring models
  UPDATED: EU AI Act Omnibus (political agreement 7 May 2026) extends Annex III
  obligations from 2 August 2026 to 2 December 2027 for standalone high-risk systems
- DORA Article 8: ICT asset inventory must include AI systems; CTPP list published
  by EBA/EIOPA/ESMA November 2025 — review critical third-party AI providers
- FCA AI Input Zone (May 2026): evidence-based AI assurance replacing attestations

British English throughout. GBP primary currency.
"""

from __future__ import annotations

import datetime
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class RiskRating(str, Enum):
    """
    PRA SS1/23 model risk ratings.
    Assigned by the model risk function; reviewed annually and on material change.
    """
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class EUAIActClassification(str, Enum):
    """
    EU AI Act 2024 risk classifications (Article 6 + Annex III).
    HIGH_RISK includes all credit-scoring AI (Annex III Item 5.2).

    OMNIBUS UPDATE (7 May 2026): Political agreement reached to postpone
    Annex III high-risk obligations from 2 August 2026 to 2 December 2027.
    HIGH_RISK models should use ai_act_omnibus_revised_deadline=date(2027, 12, 2).
    Prudent firms should maintain August 2026 internal target as best practice.
    """
    MINIMAL = "MINIMAL"          # General purpose tools, no regulated use
    LIMITED = "LIMITED"          # Chatbots with transparency obligations
    HIGH_RISK = "HIGH_RISK"      # Annex III — credit scoring, employment, etc.
    PROHIBITED = "PROHIBITED"    # Article 5 banned practices


class ValidationStatus(str, Enum):
    """Current validation state of the model."""
    NOT_VALIDATED = "NOT_VALIDATED"
    PENDING = "PENDING"
    VALIDATED = "VALIDATED"
    CONDITIONAL = "CONDITIONAL"   # Approved with conditions
    FAILED = "FAILED"
    OVERDUE = "OVERDUE"


class ModelStatus(str, Enum):
    """Operational status of the model."""
    IN_DEVELOPMENT = "IN_DEVELOPMENT"
    STAGING = "STAGING"
    PRODUCTION = "PRODUCTION"
    RETIRED = "RETIRED"
    SUSPENDED = "SUSPENDED"       # Suspended pending investigation


# ---------------------------------------------------------------------------
# Model ID validation
# ---------------------------------------------------------------------------

_MODEL_ID_PATTERN = re.compile(r"^MR-\d{4}-\d{3}$")


def validate_model_id(model_id: str) -> None:
    """
    Enforce MR-YYYY-NNN format for all model registrations.

    Raises:
        ValueError: If the ID does not match the required format.
    """
    if not _MODEL_ID_PATTERN.match(model_id):
        raise ValueError(
            f"Invalid model_id '{model_id}'. "
            f"Required format: MR-YYYY-NNN (e.g. MR-2026-001). "
            f"PRA SS1/23 Section 2.1: all models must be registered with a "
            f"standardised identifier."
        )


# ---------------------------------------------------------------------------
# ModelCard
# ---------------------------------------------------------------------------

@dataclass
class ModelCard:
    """
    PRA SS1/23-compliant model card for a single AI/ML model.

    Each field maps to a PRA SS1/23 inventory requirement.
    The model card is the canonical record for PRA supervisory enquiries
    (available within 5 business days per SS1/23 Section 5.1).

    EU AI Act compliance:
    - eu_ai_act_classification determines conformity assessment obligations
    - HIGH_RISK models require technical documentation (Article 11)
    - HIGH_RISK models require human oversight mechanism (Article 14)
    """
    # --- Identity ---
    model_id: str                           # Format: MR-YYYY-NNN
    model_name: str
    version: str                            # Semantic versioning: MAJOR.MINOR.PATCH
    purpose: str                            # Business use case description
    model_type: str                         # e.g. LLM, XGBoost, Logistic Regression

    # --- Inputs / Outputs ---
    inputs: List[str]                       # Data inputs (feature names / data sources)
    outputs: List[str]                      # Model outputs (predictions, scores, text)
    limitations: List[str]                  # Known limitations and failure modes

    # --- Risk and Classification ---
    risk_rating: RiskRating
    eu_ai_act_classification: EUAIActClassification

    # --- Ownership ---
    owner: str                              # First-line model owner (business)

    # --- Fields with defaults below ---
    pra_ss1_23_compliant: bool = False      # Set True only after validation sign-off
    developer: str = ""                     # Team responsible for development
    model_risk_contact: str = ""            # Second-line MRM contact

    # --- Agentic AI governance (PRA Oct 2025 / Feb 2026 roundtable requirements) ---
    is_agentic: bool = False                # True for ReAct/multi-hop/agentic AI systems
    agentic_scope: str = ""                 # Description of autonomous actions the model can take
    hitl_threshold: str = ""               # Human-in-the-loop trigger (e.g. "facilities >= £500k")
    board_risk_appetite_ref: str = ""      # Reference to Board-approved risk appetite statement

    # --- EU AI Act omnibus (political agreement 7 May 2026) ---
    ai_act_omnibus_revised_deadline: Optional[datetime.date] = None
    # HIGH_RISK standalone systems: revised deadline is 2 December 2027
    # Set to date(2027, 12, 2) for Annex III systems; None for LIMITED/MINIMAL

    # --- Validation ---
    validation_status: ValidationStatus = ValidationStatus.NOT_VALIDATED
    last_validated: Optional[datetime.date] = None
    next_validation_due: Optional[datetime.date] = None
    validation_frequency_months: int = 12  # Annual by default; 6m for HIGH/CRITICAL

    # --- Operational ---
    status: ModelStatus = ModelStatus.IN_DEVELOPMENT
    deployed_date: Optional[datetime.date] = None
    regulatory_approval_ref: Optional[str] = None   # e.g. "CC-2026-042"
    monitoring_plan: str = ""

    # --- Audit ---
    registered_at: datetime.datetime = field(
        default_factory=datetime.datetime.utcnow
    )
    last_updated: datetime.datetime = field(
        default_factory=datetime.datetime.utcnow
    )
    change_log: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        validate_model_id(self.model_id)
        if not self.model_name.strip():
            raise ValueError("model_name cannot be blank.")
        if not self.purpose.strip():
            raise ValueError("purpose cannot be blank.")

    def is_overdue_for_validation(self) -> bool:
        """Returns True if next_validation_due is in the past."""
        if self.next_validation_due is None:
            return False
        return self.next_validation_due < datetime.date.today()

    def requires_eu_ai_act_conformity_assessment(self) -> bool:
        """Returns True if this model requires EU AI Act conformity assessment."""
        return self.eu_ai_act_classification == EUAIActClassification.HIGH_RISK

    def effective_ai_act_deadline(self) -> Optional[datetime.date]:
        """
        Return the effective EU AI Act conformity assessment deadline.

        Reflects the EU AI Act Omnibus (7 May 2026) which postponed Annex III
        standalone high-risk obligations from 2 August 2026 to 2 December 2027.
        Returns ai_act_omnibus_revised_deadline if set, else None for non-HIGH_RISK.
        Prudent internal target remains 2 August 2026 regardless of omnibus.
        """
        if not self.requires_eu_ai_act_conformity_assessment():
            return None
        if self.ai_act_omnibus_revised_deadline:
            return self.ai_act_omnibus_revised_deadline
        return datetime.date(2027, 12, 2)  # Omnibus default for Annex III

    def requires_enhanced_agentic_validation(self) -> bool:
        """
        Returns True if this model requires enhanced agentic AI validation.

        PRA October 2025 AI/ML roundtable finding: traditional validation
        approaches (input→output mapping) are insufficient for agentic systems.
        Models flagged as agentic must have a documented agentic validation plan
        and Board-approved risk appetite before production deployment.
        """
        return self.is_agentic

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "version": self.version,
            "purpose": self.purpose,
            "model_type": self.model_type,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "limitations": self.limitations,
            "risk_rating": self.risk_rating.value,
            "eu_ai_act_classification": self.eu_ai_act_classification.value,
            "pra_ss1_23_compliant": self.pra_ss1_23_compliant,
            "owner": self.owner,
            "validation_status": self.validation_status.value,
            "last_validated": self.last_validated.isoformat() if self.last_validated else None,
            "next_validation_due": self.next_validation_due.isoformat() if self.next_validation_due else None,
            "status": self.status.value,
            "monitoring_plan": self.monitoring_plan,
            "is_agentic": self.is_agentic,
            "agentic_scope": self.agentic_scope,
            "hitl_threshold": self.hitl_threshold,
            "board_risk_appetite_ref": self.board_risk_appetite_ref,
            "ai_act_effective_deadline": (
                self.effective_ai_act_deadline().isoformat()
                if self.effective_ai_act_deadline() else None
            ),
        }


# ---------------------------------------------------------------------------
# ModelInventory
# ---------------------------------------------------------------------------

class ModelInventory:
    """
    AWB's central model inventory (PRA SS1/23 Section 2.1).

    Acts as the single source of truth for all AI/ML systems deployed or
    under development at AWB. Maintained by the Model Risk function (second line).

    Governance:
    - First line (model owners): register new models; update status
    - Second line (model risk): validate; assign risk ratings; monitor
    - Third line (internal audit): annual inventory completeness review
    - Board/ERCC: receive quarterly model risk summary
    """

    def __init__(self) -> None:
        self._models: Dict[str, ModelCard] = {}

    def register(self, model: ModelCard) -> None:
        """
        Register a new model in the inventory.

        Args:
            model: Validated ModelCard instance.

        Raises:
            ValueError: If model_id already exists.
        """
        if model.model_id in self._models:
            raise ValueError(
                f"Model '{model.model_id}' is already registered. "
                f"Use update_status() to modify an existing registration."
            )
        self._models[model.model_id] = model

    def update_status(
        self,
        model_id: str,
        validation_status: Optional[ValidationStatus] = None,
        pra_ss1_23_compliant: Optional[bool] = None,
        status: Optional[ModelStatus] = None,
        change_note: str = "",
    ) -> ModelCard:
        """
        Update the status of a registered model.

        Args:
            model_id: The model's registration ID.
            validation_status: New validation status.
            pra_ss1_23_compliant: Updated compliance flag.
            status: New operational status.
            change_note: Reason for the change (appended to change_log).

        Returns:
            Updated ModelCard.

        Raises:
            KeyError: If model_id is not found.
        """
        if model_id not in self._models:
            raise KeyError(f"Model '{model_id}' not found in inventory.")

        model = self._models[model_id]

        if validation_status is not None:
            model.validation_status = validation_status
        if pra_ss1_23_compliant is not None:
            model.pra_ss1_23_compliant = pra_ss1_23_compliant
        if status is not None:
            model.status = status
        if change_note:
            model.change_log.append(
                f"{datetime.datetime.utcnow().isoformat()} — {change_note}"
            )
        model.last_updated = datetime.datetime.utcnow()

        return model

    def get(self, model_id: str) -> ModelCard:
        """Retrieve a model by ID."""
        if model_id not in self._models:
            raise KeyError(f"Model '{model_id}' not found in inventory.")
        return self._models[model_id]

    def get_by_risk_rating(self, rating: RiskRating) -> List[ModelCard]:
        """Return all models with a given risk rating."""
        return [m for m in self._models.values() if m.risk_rating == rating]

    def get_overdue_validations(self) -> List[ModelCard]:
        """
        Return all models whose validation is overdue.

        PRA SS1/23 Section 3.1: medium/high-risk models must be validated
        at least annually. Overdue models must be escalated to the CRO.
        """
        return [m for m in self._models.values() if m.is_overdue_for_validation()]

    def get_by_status(self, status: ModelStatus) -> List[ModelCard]:
        """Return all models with a given operational status."""
        return [m for m in self._models.values() if m.status == status]

    def get_high_risk_eu_ai_act(self) -> List[ModelCard]:
        """Return models requiring EU AI Act conformity assessment."""
        return [
            m for m in self._models.values()
            if m.eu_ai_act_classification == EUAIActClassification.HIGH_RISK
        ]

    def get_agentic_models(self) -> List[ModelCard]:
        """
        Return all agentic AI models requiring enhanced validation.

        PRA October 2025 AI/ML roundtable: traditional MRM validation cannot
        scale for agentic systems. Each returned model must have an agentic
        validation plan and Board-approved risk appetite documented before
        production deployment (PRA SS1/23 Section 3.1, updated Feb 2026 guidance).
        """
        return [m for m in self._models.values() if m.is_agentic]

    def get_models_missing_board_appetite(self) -> List[ModelCard]:
        """
        Return agentic models without a Board risk appetite reference.

        PRA Oct 2025 finding: several firms deploying AI/ML without Board-approved
        risk appetite boundaries. This is a direct SS1/23 governance gap.
        """
        return [
            m for m in self.get_agentic_models()
            if not m.board_risk_appetite_ref.strip()
        ]

    def get_non_compliant(self) -> List[ModelCard]:
        """Return models not yet marked PRA SS1/23 compliant."""
        return [m for m in self._models.values() if not m.pra_ss1_23_compliant]

    def export_to_dict(self) -> List[Dict[str, Any]]:
        """Export full inventory as a list of dicts for reporting."""
        return [m.to_dict() for m in self._models.values()]

    def summary(self) -> Dict[str, Any]:
        """Return a high-level inventory summary for Board/ERCC reporting."""
        all_models = list(self._models.values())
        return {
            "total_models": len(all_models),
            "by_risk_rating": {r.value: len(self.get_by_risk_rating(r)) for r in RiskRating},
            "by_status": {s.value: len(self.get_by_status(s)) for s in ModelStatus},
            "overdue_validations": len(self.get_overdue_validations()),
            "eu_ai_act_high_risk": len(self.get_high_risk_eu_ai_act()),
            "pra_non_compliant": len(self.get_non_compliant()),
            # Agentic AI governance (PRA Oct 2025 / Feb 2026 roundtable)
            "agentic_models": len(self.get_agentic_models()),
            "agentic_missing_board_appetite": len(self.get_models_missing_board_appetite()),
            # EU AI Act omnibus: effective deadline is now 2 December 2027
            "eu_ai_act_omnibus_deadline": "2027-12-02",
            "as_at": datetime.datetime.utcnow().isoformat(),
        }

    def __len__(self) -> int:
        return len(self._models)

    def __contains__(self, model_id: str) -> bool:
        return model_id in self._models


# ---------------------------------------------------------------------------
# Pre-populated AWB model registry (Chapters 1–5)
# ---------------------------------------------------------------------------

def build_awb_model_inventory() -> ModelInventory:
    """
    Build AWB's pre-populated model inventory for Chapters 1–5 (AWB-AI-2025 programme).

    These are the models referenced throughout the book's primary thread:
    MR-2026-035 through MR-2026-039 (AWB-AI-2025 programme).
    """
    inventory = ModelInventory()

    # Chapter 1 — AI Customer Service Chatbot
    inventory.register(ModelCard(
        model_id="MR-2026-030",
        model_name="AWB Customer Service Intent Classifier",
        version="1.2.0",
        purpose=(
            "Classify inbound customer queries by intent to route to the appropriate "
            "service channel. Powers the AWB Digital Assistant on web and mobile."
        ),
        model_type="Gemini 3.5 Flash (LLM — intent classification)",
        inputs=["Customer message text", "Session context", "Channel identifier"],
        outputs=["Intent label", "Confidence score", "Routing action"],
        limitations=[
            "May misclassify complex multi-intent queries",
            "Performance may degrade on non-standard English or regional dialects",
            "Does not process audio or image inputs",
        ],
        risk_rating=RiskRating.MEDIUM,
        eu_ai_act_classification=EUAIActClassification.LIMITED,
        owner="Head of Digital Banking",
        developer="AWB Digital Products Team",
        model_risk_contact="Model Risk — Bristol",
        validation_status=ValidationStatus.VALIDATED,
        last_validated=datetime.date(2026, 1, 15),
        next_validation_due=datetime.date(2027, 1, 15),
        status=ModelStatus.PRODUCTION,
        deployed_date=datetime.date(2025, 6, 1),
        pra_ss1_23_compliant=True,
        monitoring_plan=(
            "Monthly accuracy monitoring; weekly volume and latency tracking; "
            "FCA Consumer Duty outcome review quarterly."
        ),
        change_log=["2026-01-15 — Initial validation completed by Model Risk."],
    ))

    # Chapter 2 — Document Analysis LLM
    inventory.register(ModelCard(
        model_id="MR-2026-035",
        model_name="AWB Credit Document Analyser",
        version="1.0.0",
        purpose=(
            "Extract structured financial data from unstructured credit documents "
            "(accounts, management information, loan applications) using LLM reasoning."
        ),
        model_type="Gemini 3.1 Pro (LLM — document extraction)",
        inputs=["PDF documents", "Scanned financial statements", "Credit applications"],
        outputs=["Extracted financial metrics", "Entity names", "Validation flags"],
        limitations=[
            "Accuracy dependent on document quality and formatting",
            "Does not handle handwritten documents",
            "Requires human review for outputs used in credit decisions ≥ £500,000",
        ],
        risk_rating=RiskRating.HIGH,
        eu_ai_act_classification=EUAIActClassification.HIGH_RISK,
        owner="Head of Credit Risk",
        developer="AWB AI Platform Team",
        model_risk_contact="Model Risk — Bristol",
        validation_status=ValidationStatus.VALIDATED,
        last_validated=datetime.date(2026, 2, 1),
        next_validation_due=datetime.date(2026, 8, 1),
        status=ModelStatus.PRODUCTION,
        deployed_date=datetime.date(2026, 2, 15),
        pra_ss1_23_compliant=True,
        # EU AI Act omnibus: Annex III deadline extended to 2 December 2027
        ai_act_omnibus_revised_deadline=datetime.date(2027, 12, 2),
        monitoring_plan=(
            "Bi-annual validation; monthly extraction accuracy monitoring "
            "against human-reviewed sample; PRA SS1/23 Section 4.1 performance reporting."
        ),
    ))

    # Chapter 2 — SME Financial Statement Analyser
    inventory.register(ModelCard(
        model_id="MR-2026-036",
        model_name="AWB SME Financial Statement Analyser",
        version="1.0.0",
        purpose=(
            "Extract key financial metrics from SME annual accounts and "
            "management accounts to accelerate credit underwriting. "
            "Processes PDF statements up to 150 pages in under 90 seconds."
        ),
        model_type="Gemini 3.5 Flash (LLM — structured extraction)",
        inputs=[
            "SME annual accounts (PDF)",
            "Management accounts (PDF)",
            "Companies House filings",
        ],
        outputs=[
            "Turnover, EBITDA, net debt, leverage ratio",
            "Extraction confidence score",
            "Validation flags for out-of-range values",
        ],
        limitations=[
            "Accuracy dependent on document legibility and formatting",
            "Does not interpret qualitative notes to accounts",
            "Requires human review for figures used in credit decisions",
        ],
        risk_rating=RiskRating.MEDIUM,
        eu_ai_act_classification=EUAIActClassification.HIGH_RISK,
        owner="Head of Credit Risk",
        developer="AWB AI Platform Team",
        model_risk_contact="Model Risk — Bristol",
        validation_status=ValidationStatus.VALIDATED,
        last_validated=datetime.date(2026, 2, 10),
        next_validation_due=datetime.date(2026, 8, 10),
        status=ModelStatus.PRODUCTION,
        deployed_date=datetime.date(2026, 2, 20),
        pra_ss1_23_compliant=True,
        # EU AI Act omnibus: Annex III deadline extended to 2 December 2027
        ai_act_omnibus_revised_deadline=datetime.date(2027, 12, 2),
        monitoring_plan=(
            "Monthly extraction accuracy sample review; "
            "bi-annual full validation; PRA SS1/23 Section 4.1 reporting."
        ),
    ))


    # Chapter 3 — AWB Credit Decision Agent
    inventory.register(ModelCard(
        model_id="MR-2026-037",
        model_name="AWB Credit Decision Agent",
        version="1.0.0",
        purpose=(
            "Orchestrate multi-step credit assessment using AI agent: document analysis, "
            "policy rule checking, covenant assessment, and credit memo drafting."
        ),
        model_type="Gemini 3.1 Pro (ReAct agent) + Gemini 3.5 Flash (memo drafting)",
        inputs=[
            "Credit application documents",
            "T24 exposure data",
            "Applicant financial statements",
        ],
        outputs=[
            "Credit policy assessment",
            "Risk rating (1–10)",
            "Credit memorandum",
            "APPROVE/REFER/DECLINE recommendation",
        ],
        limitations=[
            "Human oversight mandatory for facilities >= £500,000 (EU AI Act Article 14)",
            "Cannot assess qualitative management quality or character risk",
            "Dependent on completeness of T24 exposure data",
            "Agentic validation plan required — traditional input/output testing insufficient "
            "(PRA October 2025 AI/ML roundtable finding)",
        ],
        risk_rating=RiskRating.HIGH,
        eu_ai_act_classification=EUAIActClassification.HIGH_RISK,
        owner="Head of Credit Risk",
        developer="AWB AI Platform Team",
        model_risk_contact="Model Risk — Bristol",
        validation_status=ValidationStatus.VALIDATED,
        last_validated=datetime.date(2026, 3, 1),
        next_validation_due=datetime.date(2026, 9, 1),
        status=ModelStatus.PRODUCTION,
        deployed_date=datetime.date(2026, 3, 10),
        pra_ss1_23_compliant=True,
        regulatory_approval_ref="CC-2026-042",
        # Agentic AI governance flags (PRA Oct 2025 / Feb 2026 roundtable)
        is_agentic=True,
        agentic_scope=(
            "LangGraph ReAct loop: autonomous document ingestion, policy checking, "
            "covenant assessment, and memo drafting across 4 LLM nodes. "
            "HITL gate at £500k threshold per EU AI Act Article 14."
        ),
        hitl_threshold="Facilities >= £500,000 — mandatory credit officer review",
        board_risk_appetite_ref="AWB Board AI Risk Appetite Statement v1.1, Feb 2026, "
                                "Section 3.2 — Credit AI autonomous decision limits",
        # EU AI Act omnibus: revised deadline 2 December 2027
        ai_act_omnibus_revised_deadline=datetime.date(2027, 12, 2),
        monitoring_plan=(
            "Monthly recommendation accuracy review; human override rate tracking; "
            "EU AI Act Article 14 compliance log; quarterly Credit Committee review; "
            "agentic hop-chain audit log reviewed by Model Risk monthly."
        ),
    ))

    # Chapter 4 — AWB Regulatory Knowledge Assistant (MR-2026-038)
    inventory.register(ModelCard(
        model_id="MR-2026-038",
        model_name="AWB Regulatory Knowledge Assistant",
        version="2.0.0",
        purpose=(
            "Answer compliance team queries about PRA SS1/23, FCA Consumer Duty, "
            "EU AI Act, DORA, and Basel III/IV, grounded in actual regulatory text. "
            "v2.0: three-layer memory, contextual retrieval, agentic multi-hop RAG."
        ),
        model_type=(
            "Gemini 3.5 Flash (RAG + agentic multi-hop) — "
            "ChromaDB vector store, Google text-embedding-004"
        ),
        inputs=["Compliance team queries", "Regulatory document corpus", "Session context"],
        outputs=["Grounded regulatory answers", "Source citations", "Confidence scores",
                 "ReAct hop chain audit log"],
        limitations=[
            "Answers limited to documents in AWB's regulatory library",
            "Not a substitute for qualified legal advice",
            "Knowledge base must be updated when regulations change",
            "Agentic multi-hop path requires PRA agentic validation plan "
            "(PRA October 2025 AI/ML roundtable finding)",
        ],
        risk_rating=RiskRating.MEDIUM,
        eu_ai_act_classification=EUAIActClassification.LIMITED,
        owner="Head of Compliance",
        developer="AWB AI Platform Team",
        model_risk_contact="Model Risk — Bristol",
        validation_status=ValidationStatus.VALIDATED,
        last_validated=datetime.date(2026, 3, 5),
        next_validation_due=datetime.date(2027, 3, 5),
        status=ModelStatus.PRODUCTION,
        deployed_date=datetime.date(2026, 3, 15),
        pra_ss1_23_compliant=True,
        # Agentic flags — v2.0 AgenticRAGEngine uses ReAct loop for complex queries
        is_agentic=True,
        agentic_scope=(
            "AgenticRAGEngine ReAct loop: up to 6 retrieval hops for complex "
            "multi-regulatory queries. Simple queries routed to passive engine. "
            "ESCALATE_HUMAN action triggers compliance officer review."
        ),
        hitl_threshold="ESCALATE_HUMAN action or confidence score < 0.75",
        board_risk_appetite_ref="AWB Board AI Risk Appetite Statement v1.1, Feb 2026, "
                                "Section 4.1 — RAG and knowledge system autonomy limits",
        monitoring_plan=(
            "Quarterly citation accuracy review; monthly hallucination rate tracking; "
            "document freshness review when regulatory updates are published; "
            "monthly hop-chain audit review for agentic queries; "
            "FCA AI Input Zone evidence pack updated quarterly (from May 2026)."
        ),
    ))

    # Chapter 5 — AI Governance Platform (MR-2026-039)
    inventory.register(ModelCard(
        model_id="MR-2026-039",
        model_name="AWB AI Governance Platform",
        version="1.0.0",
        purpose=(
            "Governance tooling for PRA SS1/23 compliance: model inventory, "
            "validation reporting, monitoring, incident management, fairness testing."
        ),
        model_type="Rules-based governance framework (no ML component)",
        inputs=["Model performance metrics", "Validation test results", "Incident reports"],
        outputs=[
            "Model inventory reports",
            "Validation status",
            "Drift alerts",
            "Incident records",
            "Fairness reports",
        ],
        limitations=[
            "Framework quality dependent on completeness of model owner inputs",
            "Automated fairness testing requires representative demographic data",
        ],
        risk_rating=RiskRating.LOW,
        eu_ai_act_classification=EUAIActClassification.MINIMAL,
        owner="Chief Risk Officer",
        developer="AWB Model Risk Team",
        model_risk_contact="Model Risk — Bristol",
        validation_status=ValidationStatus.VALIDATED,
        last_validated=datetime.date(2026, 3, 10),
        next_validation_due=datetime.date(2027, 3, 10),
        status=ModelStatus.PRODUCTION,
        deployed_date=datetime.date(2026, 3, 20),
        pra_ss1_23_compliant=True,
        monitoring_plan="Annual framework review by Internal Audit (third line).",
    ))

    # Existing credit scoring model (pre-AI programme)
    inventory.register(ModelCard(
        model_id="MR-2022-015",
        model_name="AWB SME Credit Scoring Model",
        version="3.1.0",
        purpose=(
            "Score SME loan applications using financial ratios and bureau data "
            "to support credit underwriting decisions."
        ),
        model_type="Logistic Regression + XGBoost ensemble",
        inputs=[
            "Financial ratios (leverage, ICR, profitability)",
            "Credit bureau score",
            "Industry sector",
            "Years trading",
        ],
        outputs=["Credit score (0–1000)", "Risk band (A–E)", "Rejection flag"],
        limitations=[
            "Trained on 2019–2022 data; may underperform in novel economic conditions",
            "Does not incorporate management quality or qualitative factors",
            "Tested on UK SME population; less reliable for cross-border entities",
        ],
        risk_rating=RiskRating.CRITICAL,
        eu_ai_act_classification=EUAIActClassification.HIGH_RISK,
        owner="Head of Credit Risk",
        developer="AWB Quantitative Analytics",
        model_risk_contact="Model Risk — Bristol",
        validation_status=ValidationStatus.VALIDATED,
        last_validated=datetime.date(2025, 9, 1),
        next_validation_due=datetime.date(2026, 3, 1),   # Overdue as of June 2026
        validation_frequency_months=6,
        status=ModelStatus.PRODUCTION,
        deployed_date=datetime.date(2022, 8, 15),
        pra_ss1_23_compliant=True,
        regulatory_approval_ref="CC-2022-089",
        # EU AI Act omnibus: Annex III deadline extended to 2 December 2027
        ai_act_omnibus_revised_deadline=datetime.date(2027, 12, 2),
        monitoring_plan=(
            "Monthly GINI coefficient and PSI monitoring; semi-annual validation; "
            "annual IRB capital model review; PRA SS1/23 quarterly performance reporting."
        ),
    ))

    return inventory


# Module-level default inventory
AWB_INVENTORY = build_awb_model_inventory()
