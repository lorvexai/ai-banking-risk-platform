"""awb_commons/models.py — Chapter 10 Pydantic schemas."""
from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, Field


class RiskRating(str, Enum):
    LOW    = "LOW"
    MEDIUM = "MEDIUM"
    HIGH   = "HIGH"


class ModelStatus(str, Enum):
    DEVELOPMENT  = "DEVELOPMENT"
    VALIDATION   = "VALIDATION"
    APPROVED     = "APPROVED"
    ACTIVE       = "ACTIVE"
    UNDER_REVIEW = "UNDER_REVIEW"
    RETIRED      = "RETIRED"


class EUAIActClass(str, Enum):
    HIGH_RISK  = "HIGH_RISK"
    LIMITED    = "LIMITED"
    MINIMAL    = "MINIMAL"
    NOT_IN_SCOPE = "NOT_IN_SCOPE"


class ModelRecord(BaseModel):
    """PRA SS1/23 model registry entry."""
    mr_reference: str
    model_name: str
    chapter: int
    ss1_23_risk: RiskRating
    eu_ai_act: EUAIActClass
    status: ModelStatus
    owner: str
    validator: Optional[str] = None
    validated_at: Optional[datetime] = None
    next_revalidation: Optional[datetime] = None
    psi_threshold: float = 0.10
    gini_threshold: Optional[float] = None
    registered_at: datetime = Field(default_factory=datetime.utcnow)


class ValidationResult(BaseModel):
    """Outcome of a model validation exercise."""
    validation_id: UUID = Field(default_factory=uuid4)
    mr_reference: str
    validator_id: str
    gini_coefficient: Optional[float] = None
    psi: Optional[float] = None
    ks_statistic: Optional[float] = None
    accuracy: Optional[float] = None
    auc_roc: Optional[float] = None
    outcome: str  # PASS / CONDITIONAL_PASS / FAIL
    findings: list[str] = Field(default_factory=list)
    validated_at: datetime = Field(default_factory=datetime.utcnow)


class ABTestResult(BaseModel):
    """A/B test comparison between model versions."""
    test_id: UUID = Field(default_factory=uuid4)
    mr_reference: str
    control_version: str
    treatment_version: str
    sample_size_control: int
    sample_size_treatment: int
    metric_name: str
    control_value: float
    treatment_value: float
    lift_pct: float
    p_value: float
    statistically_significant: bool
    recommendation: str
    test_start: datetime
    test_end: Optional[datetime] = None


class LLMMonitoringSnapshot(BaseModel):
    """Monthly LLM performance monitoring record."""
    snapshot_id: UUID = Field(default_factory=uuid4)
    mr_reference: str
    snapshot_month: str
    faithfulness_score: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    avg_cost_per_query_gbp: float
    p50_latency_ms: int
    p95_latency_ms: int
    hallucination_rate_pct: float
    total_queries: int
    alerts_triggered: list[str] = Field(default_factory=list)
    snapshot_date: datetime = Field(default_factory=datetime.utcnow)
