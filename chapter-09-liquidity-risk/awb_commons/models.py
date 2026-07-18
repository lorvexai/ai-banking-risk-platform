"""awb_commons/models.py — Chapter 9 Pydantic schemas."""
from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, Field


class LiquidityMetric(str, Enum):
    LCR = "LCR"
    NSFR = "NSFR"
    CLAR = "CLAR"


class StressScenario(str, Enum):
    BASE = "BASE"
    IDIOSYNCRATIC = "IDIOSYNCRATIC"
    MARKET_WIDE = "MARKET_WIDE"
    COMBINED = "COMBINED"
    PRA_CST_SEVERE = "PRA_CST_SEVERE"


class CashFlowForecast(BaseModel):
    forecast_id: UUID = Field(default_factory=uuid4)
    forecast_date: datetime
    horizon_days: int
    net_position_gbp: float
    confidence_lower_gbp: float
    confidence_upper_gbp: float
    model_version: str = "lstm-v2.1-2025"
    mr_reference: str = "MR-2026-052"
    created_at: datetime = Field(default_factory=datetime.utcnow)


class LCRCalculation(BaseModel):
    calculation_id: UUID = Field(default_factory=uuid4)
    calculation_date: datetime
    hqla_gbp: float
    net_outflows_gbp: float
    lcr_pct: float
    regulatory_minimum_pct: float = 100.0
    compliant: bool
    scenario: StressScenario = StressScenario.BASE
    regulatory_reference: str = "CRR3 Art. 411-428"
    mr_reference: str = "MR-2026-053"

    def is_above_buffer(self, buffer_pct: float = 110.0) -> bool:
        return self.lcr_pct >= buffer_pct


class NSFRCalculation(BaseModel):
    calculation_id: UUID = Field(default_factory=uuid4)
    calculation_date: datetime
    available_stable_funding_gbp: float
    required_stable_funding_gbp: float
    nsfr_pct: float
    regulatory_minimum_pct: float = 100.0
    compliant: bool
    regulatory_reference: str = "CRR3 Art. 428a-428au"

    def calculate_nsfr(self) -> float:
        if self.required_stable_funding_gbp == 0:
            return 0.0
        return (self.available_stable_funding_gbp
                / self.required_stable_funding_gbp) * 100.0


class IntradayAlert(BaseModel):
    alert_id: UUID = Field(default_factory=uuid4)
    alert_time: datetime
    peak_usage_gbp: float
    available_buffer_gbp: float
    utilisation_pct: float
    requires_action: bool
    recommended_action: str
    mr_reference: str = "MR-2026-054"
    created_at: datetime = Field(default_factory=datetime.utcnow)
