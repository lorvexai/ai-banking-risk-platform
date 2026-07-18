"""
awb_commons.models — Shared Pydantic schemas for Chapter 8.
Avon & Wessex Bank plc (AWB) — AWB-AI-2025 programme.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class FraudSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class OpLossCategory(str, Enum):
    """Basel III operational risk event categories."""
    INTERNAL_FRAUD = "INTERNAL_FRAUD"
    EXTERNAL_FRAUD = "EXTERNAL_FRAUD"
    EMPLOYMENT_PRACTICES = "EMPLOYMENT_PRACTICES"
    CLIENTS_PRODUCTS = "CLIENTS_PRODUCTS"
    PHYSICAL_ASSETS = "PHYSICAL_ASSETS"
    BUSINESS_DISRUPTION = "BUSINESS_DISRUPTION"
    EXECUTION_DELIVERY = "EXECUTION_DELIVERY"


class FraudAlert(BaseModel):
    """Payment fraud alert raised by the detection system."""

    alert_id: UUID = Field(default_factory=uuid4)
    transaction_id: str
    account_id: str
    amount_gbp: float
    risk_score: float = Field(ge=0.0, le=1.0)
    severity: FraudSeverity
    triggered_rules: list[str]
    model_version: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    requires_human_review: bool = False

    class Config:
        use_enum_values = True


class OpLossEvent(BaseModel):
    """Operational loss event extracted by NLP pipeline."""

    event_id: UUID = Field(default_factory=uuid4)
    source_document_id: str
    event_category: OpLossCategory
    loss_amount_gbp: Optional[float] = None
    recovery_amount_gbp: float = 0.0
    net_loss_gbp: Optional[float] = None
    confidence_score: float = Field(ge=0.0, le=1.0)
    event_date: Optional[datetime] = None
    description: str
    sma_eligible: bool = True
    mr_reference: str = "MR-2026-050"
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def calculate_net_loss(self) -> float:
        """Compute net loss after recoveries."""
        if self.loss_amount_gbp is None:
            return 0.0
        return max(0.0, self.loss_amount_gbp - self.recovery_amount_gbp)

    class Config:
        use_enum_values = True


class FraudRiskScore(BaseModel):
    """Credit application fraud risk assessment."""

    application_id: str
    applicant_id: str
    risk_score: float = Field(ge=0.0, le=1.0)
    fraud_indicators: list[str]
    model_explanation: str
    recommended_action: str  # APPROVE / REVIEW / DECLINE
    confidence: float = Field(ge=0.0, le=1.0)
    mr_reference: str = "MR-2026-051"
    assessed_at: datetime = Field(default_factory=datetime.utcnow)


class ModelPrediction(BaseModel):
    """Generic model prediction wrapper for PRA SS1/23 audit."""

    prediction_id: UUID = Field(default_factory=uuid4)
    model_id: str
    input_hash: str
    prediction: dict
    confidence: float
    model_version: str
    latency_ms: int
    predicted_at: datetime = Field(default_factory=datetime.utcnow)
