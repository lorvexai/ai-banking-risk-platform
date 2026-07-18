"""
credit_agent/credit_memo_generator.py
AWB Automated Credit Decision Workflow — Credit Memo Generator
Chapter 3: Agentic AI for Financial Risk

Generates a validated, structured CreditMemo Pydantic model from agent findings.
The memo is stored as audit evidence per PRA SS1/23 Section 5.3.

Regulatory context:
- PRA SS1/23 Section 5.3: AI model decisions must be documented with full
  input/output traceability and retained for a minimum of 7 years.
- EU AI Act 2024 Article 14: Human oversight must be documented in writing
  for high-risk AI decisions (credit scoring is explicitly listed in Annex III).
- FCA Consumer Duty PS22/9: Decision rationale must be explainable to the
  customer in plain language.
- UK GDPR Article 22: Right to explanation for automated decisions.

Model registration: MR-2026-037
"""

from __future__ import annotations

import datetime
import json
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class CreditDecision(str, Enum):
    """Permitted credit decisions from the automated workflow."""
    APPROVE = "APPROVE"      # Meets all policy requirements
    DECLINE = "DECLINE"      # Critical policy breach; cannot be overridden
    REFER = "REFER"          # Requires Credit Committee review


class RegulatoryFlag(str, Enum):
    """Regulatory flags that must be documented in the credit memo."""
    EU_AI_ACT_HITL = "EU_AI_ACT_HITL"         # Human-in-the-loop mandatory
    PRA_MODEL_RISK = "PRA_MODEL_RISK"           # PRA SS1/23 model registration
    FCA_CONSUMER_DUTY = "FCA_CONSUMER_DUTY"     # FCA PS22/9 outcome tracking
    GDPR_AUTO_DECISION = "GDPR_AUTO_DECISION"   # UK GDPR Article 22


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class PolicyBreachSummary(BaseModel):
    """Condensed policy breach for inclusion in the credit memo."""
    rule_name: str
    actual_value: float
    threshold: float
    severity: str
    description: str

    class Config:
        frozen = True


class CovenantSummary(BaseModel):
    """Individual covenant recommended for the facility."""
    covenant_name: str
    metric: str
    threshold: float
    direction: str  # MAXIMUM or MINIMUM
    testing_frequency: str

    class Config:
        frozen = True


class AuditTrail(BaseModel):
    """
    Audit trail entry for PRA SS1/23 compliance.
    Retained for 7 years per UK statutory minimum.
    """
    entry_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    event_type: str
    actor: str  # "AGENT" or "HUMAN:{user_id}"
    details: Dict[str, Any]
    model_registration: str = "MR-2026-037"

    class Config:
        frozen = True

    def to_dict(self) -> dict:
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "actor": self.actor,
            "details": self.details,
            "model_registration": self.model_registration,
        }


# ---------------------------------------------------------------------------
# Primary CreditMemo model
# ---------------------------------------------------------------------------

class CreditMemo(BaseModel):
    """
    Structured credit memorandum produced by AWB's Automated Credit Decision
    Workflow (Chapter 3).

    This Pydantic model is the canonical output of the agentic workflow and
    serves as:
    1. The input to the Credit Committee decision process.
    2. Audit evidence for PRA SS1/23 Section 5.3.
    3. The basis for customer notification under FCA Consumer Duty.

    All fields are validated on construction; any invalid memo raises
    ValidationError before it can be stored or transmitted.
    """

    # --- Identity ---
    memo_id: str = Field(
        default_factory=lambda: f"CM-{datetime.date.today().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}",
        description="Unique memo identifier; format: CM-YYYYMMDD-XXXXXXXX",
    )
    created_at: datetime.datetime = Field(
        default_factory=datetime.datetime.utcnow,
        description="UTC timestamp of memo creation",
    )
    model_version: str = Field(
        default="1.0.0",
        description="Version of the credit decision model (PRA SS1/23 changelog)",
    )
    model_registration: str = Field(
        default="MR-2026-037",
        description="PRA SS1/23 model registration reference",
    )

    # --- Applicant ---
    applicant_name: str = Field(..., description="Legal name of the borrower entity")
    applicant_id: Optional[str] = Field(None, description="AWB T24 customer identifier")
    facility_amount_gbp: float = Field(..., gt=0, description="Proposed facility amount in GBP")
    facility_type: str = Field(..., description="TERM_LOAN, REVOLVING_CREDIT, etc.")
    facility_purpose: Optional[str] = Field(None, description="Stated purpose of the facility")

    # --- Decision ---
    recommendation: CreditDecision = Field(
        ..., description="AI agent recommendation: APPROVE, DECLINE, or REFER"
    )
    risk_rating: int = Field(
        ..., ge=1, le=10,
        description="Risk rating 1–10 (1 = lowest risk, 10 = highest risk)",
    )
    rationale: str = Field(
        ..., min_length=50,
        description="Plain-English rationale for the recommendation (FCA Consumer Duty)",
    )

    # --- Analysis ---
    key_risks: List[str] = Field(
        default_factory=list,
        description="List of key risks identified during analysis",
    )
    mitigants: List[str] = Field(
        default_factory=list,
        description="Risk mitigants and structural protections",
    )
    policy_breaches: List[PolicyBreachSummary] = Field(
        default_factory=list,
        description="Policy rule breaches identified",
    )
    recommended_covenants: List[CovenantSummary] = Field(
        default_factory=list,
        description="Financial covenants recommended for the facility",
    )

    # --- Conditions and Next Steps ---
    conditions: List[str] = Field(
        default_factory=list,
        description="Conditions precedent to facility drawdown",
    )
    next_steps: List[str] = Field(
        default_factory=list,
        description="Actions required post-decision",
    )

    # --- Regulatory ---
    regulatory_flags: List[RegulatoryFlag] = Field(
        default_factory=list,
        description="Regulatory obligations triggered by this decision",
    )
    human_review_required: bool = Field(
        ...,
        description="True if EU AI Act Article 14 human oversight is mandatory",
    )
    human_review_completed: bool = Field(
        default=False,
        description="Set to True only after Senior Credit Officer sign-off",
    )
    human_reviewer_id: Optional[str] = Field(
        None,
        description="Employee ID of the reviewing Senior Credit Officer",
    )
    human_review_timestamp: Optional[datetime.datetime] = Field(
        None,
        description="UTC timestamp of human review completion",
    )

    # --- Audit ---
    audit_trail: List[AuditTrail] = Field(
        default_factory=list,
        description="Ordered list of audit events (PRA SS1/23 Section 5.3)",
    )
    retention_until: datetime.date = Field(
        default_factory=lambda: (
            datetime.date.today() + datetime.timedelta(days=365 * 7)
        ),
        description="Retention date: 7 years from creation (UK statutory minimum)",
    )

    # --- Validators ---

    @field_validator("facility_amount_gbp")
    @classmethod
    def validate_facility_amount(cls, v: float) -> float:
        if v > 500_000_000:
            raise ValueError(
                "Facilities exceeding £500M are outside automated workflow scope; "
                "refer to Structured Finance team."
            )
        return v

    @field_validator("applicant_name")
    @classmethod
    def validate_applicant_name(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("applicant_name cannot be blank.")
        return v.strip()

    @model_validator(mode="after")
    def enforce_hitl_for_large_facilities(self) -> "CreditMemo":
        """
        EU AI Act 2024 Article 14: Human-in-the-loop is mandatory for
        credit decisions involving facilities of £500,000 or more.
        This validator enforces that human_review_required is True for
        such facilities and that RegulatoryFlag.EU_AI_ACT_HITL is present.
        """
        if self.facility_amount_gbp >= 500_000:
            if not self.human_review_required:
                raise ValueError(
                    "EU AI Act 2024 Article 14: human_review_required must be True "
                    "for facilities ≥ £500,000."
                )
            if RegulatoryFlag.EU_AI_ACT_HITL not in self.regulatory_flags:
                # Add flag programmatically rather than raising — more user-friendly
                self.regulatory_flags.append(RegulatoryFlag.EU_AI_ACT_HITL)
        return self

    @model_validator(mode="after")
    def validate_human_review_completion(self) -> "CreditMemo":
        """
        If human review has been marked complete, both reviewer ID and
        timestamp must be present (prevents incomplete sign-off records).
        """
        if self.human_review_completed:
            if not self.human_reviewer_id:
                raise ValueError(
                    "human_reviewer_id must be provided when human_review_completed is True."
                )
            if not self.human_review_timestamp:
                raise ValueError(
                    "human_review_timestamp must be provided when human_review_completed is True."
                )
        return self

    def complete_human_review(self, reviewer_id: str) -> "CreditMemo":
        """
        Record completion of human oversight review.

        Args:
            reviewer_id: Employee ID of the reviewing Senior Credit Officer.

        Returns:
            Updated CreditMemo (note: Pydantic v2 models are mutable by default).
        """
        self.human_reviewer_id = reviewer_id
        self.human_review_completed = True
        self.human_review_timestamp = datetime.datetime.utcnow()

        self.audit_trail.append(AuditTrail(
            event_type="HUMAN_REVIEW_COMPLETED",
            actor=f"HUMAN:{reviewer_id}",
            details={
                "memo_id": self.memo_id,
                "recommendation": self.recommendation.value,
                "facility_amount_gbp": self.facility_amount_gbp,
                "regulatory_basis": "EU AI Act 2024 Article 14",
            },
        ))

        return self

    def to_dict(self) -> dict:
        """Serialise to dict for storage and transmission."""
        data = self.model_dump()
        # Convert enums to strings
        data["recommendation"] = self.recommendation.value
        data["regulatory_flags"] = [f.value for f in self.regulatory_flags]
        data["created_at"] = self.created_at.isoformat()
        data["retention_until"] = self.retention_until.isoformat()
        data["audit_trail"] = [a.to_dict() for a in self.audit_trail]
        if self.human_review_timestamp:
            data["human_review_timestamp"] = self.human_review_timestamp.isoformat()
        return data

    def to_json(self, indent: int = 2) -> str:
        """Serialise to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_credit_memo_from_agent_output(
    agent_findings: Dict[str, Any],
    credit_application: Dict[str, Any],
) -> CreditMemo:
    """
    Build a validated CreditMemo from raw agent findings and credit application.

    This factory function is called by the agent after all tool outputs have
    been collected, and before the human oversight checkpoint.

    Args:
        agent_findings: Dict containing tool outputs keyed by tool name.
        credit_application: The original credit application data.

    Returns:
        Validated CreditMemo instance.

    Raises:
        pydantic.ValidationError: If any field fails validation.
    """
    policy = agent_findings.get("check_credit_policy", {})
    ratios = agent_findings.get("calculate_ratios", {})
    covenants = agent_findings.get("assess_covenants", {})
    exposure = agent_findings.get("fetch_t24_exposure", {})
    memo_draft = agent_findings.get("draft_credit_memo", {})

    facility_amount = credit_application.get("facility_amount_gbp", 0.0)

    # Build policy breach summaries
    policy_breach_summaries = [
        PolicyBreachSummary(
            rule_name=b["rule_name"],
            actual_value=b["actual_value"],
            threshold=b["threshold"],
            severity=b["severity"],
            description=b["description"],
        )
        for b in policy.get("breaches", [])
    ]

    # Build covenant summaries
    covenant_summaries = [
        CovenantSummary(
            covenant_name=c["covenant_name"],
            metric=c["metric"],
            threshold=c["threshold"],
            direction=c["direction"],
            testing_frequency=covenants.get("testing_frequency", "Quarterly"),
        )
        for c in covenants.get("recommended_covenants", [])
    ]

    # Determine regulatory flags
    regulatory_flags = [
        RegulatoryFlag.PRA_MODEL_RISK,
        RegulatoryFlag.FCA_CONSUMER_DUTY,
        RegulatoryFlag.GDPR_AUTO_DECISION,
    ]
    human_review_required = facility_amount >= 500_000
    if human_review_required:
        regulatory_flags.append(RegulatoryFlag.EU_AI_ACT_HITL)

    # Determine recommendation
    recommendation_str = memo_draft.get(
        "recommendation",
        policy.get("recommendation", "REFER"),
    )
    try:
        recommendation = CreditDecision(recommendation_str)
    except ValueError:
        recommendation = CreditDecision.REFER

    # Risk rating: derive from policy breaches if not explicitly set
    risk_rating = memo_draft.get("risk_rating", 5)
    blocking = policy.get("blocking_breach_count", 0)
    if blocking >= 2:
        risk_rating = max(risk_rating, 8)
    elif blocking == 1:
        risk_rating = max(risk_rating, 6)

    # Initial audit trail entry
    initial_audit = AuditTrail(
        event_type="CREDIT_MEMO_GENERATED",
        actor="AGENT",
        details={
            "applicant_name": credit_application.get("applicant_name", "Unknown"),
            "facility_amount_gbp": facility_amount,
            "policy_breach_count": policy.get("breach_count", 0),
            "recommendation": recommendation.value,
        },
    )

    return CreditMemo(
        applicant_name=credit_application.get("applicant_name", "Unknown Applicant"),
        applicant_id=credit_application.get("customer_id"),
        facility_amount_gbp=facility_amount,
        facility_type=credit_application.get("facility_type", "TERM_LOAN"),
        facility_purpose=credit_application.get("facility_purpose"),
        recommendation=recommendation,
        risk_rating=risk_rating,
        rationale=memo_draft.get(
            "rationale",
            f"Automated assessment of {credit_application.get('applicant_name', 'applicant')} "
            f"for a £{facility_amount:,.0f} {credit_application.get('facility_type', 'facility')}. "
            f"Policy assessment: {policy.get('recommendation', 'REFER')}. "
            f"Risk rating: {risk_rating}/10."
        ),
        key_risks=memo_draft.get("key_risks", ["Risk assessment pending human review."]),
        mitigants=memo_draft.get("mitigants", []),
        policy_breaches=policy_breach_summaries,
        recommended_covenants=covenant_summaries,
        conditions=memo_draft.get("conditions", []),
        next_steps=memo_draft.get("next_steps", []),
        regulatory_flags=regulatory_flags,
        human_review_required=human_review_required,
        audit_trail=[initial_audit],
    )
