"""
AWB CRO/CFO Executive Dashboard — Chapter 16
==============================================
Real-time capital, liquidity, credit risk, AML,
and model performance metrics for the CRO and CFO.

Metrics refreshed every 15 minutes from:
  - ERDW (Enterprise Risk Data Warehouse)
  - AI service APIs (PSI, RAGAS scores)

Regulatory coverage:
  - CET1, Tier 1, Total Capital (CRR3)
  - Leverage Ratio (CRR3 Art. 429)
  - LCR, NSFR, CLAR (CRR3)
  - ECL by portfolio segment (IFRS 9)
  - AML alert volume and SAR rate (POCA 2002)
  - PSI for 13 supervised models (PRA SS1/23)
  - RAGAS for LLM systems (PRA SS1/23)

Start:
    uvicorn dashboard:app --port 8084
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import FastAPI
from pydantic import BaseModel

log = logging.getLogger(__name__)

app = FastAPI(
    title="AWB CRO/CFO Dashboard API",
    version="1.0.0",
)

# ── Metric models ──────────────────────────────────

class CapitalRatios(BaseModel):
    """CRR3 capital adequacy ratios."""
    cet1_ratio: float
    tier1_ratio: float
    total_capital_ratio: float
    leverage_ratio: float          # CRR3 Art. 429
    cet1_minimum: float = 4.5
    t1_minimum: float = 6.0
    total_minimum: float = 8.0
    leverage_minimum: float = 3.0
    last_refreshed: str


class LiquidityMetrics(BaseModel):
    """LCR / NSFR / CLAR metrics."""
    lcr: float                     # >=100%
    nsfr: float                    # >=100%
    clar: float                    # UK supplement
    lcr_minimum: float = 100.0
    nsfr_minimum: float = 100.0
    last_refreshed: str


class ECLMetrics(BaseModel):
    """IFRS 9 Expected Credit Loss."""
    total_ecl_gbp: float
    stage1_ecl_gbp: float
    stage2_ecl_gbp: float
    stage3_ecl_gbp: float
    coverage_ratio: float
    last_refreshed: str


class AMLMetrics(BaseModel):
    """AML alert volume and quality."""
    alerts_mtd: int
    sars_filed_mtd: int
    sar_rate: float
    avg_quality_score: float       # 1-5 scale
    false_positive_rate: float
    last_refreshed: str


class ModelPerformance(BaseModel):
    """PSI scores for supervised models."""
    psi_scores: Dict[str, float]   # mr_id -> PSI
    ragas_scores: Dict[str, float] # mr_id -> RAGAS
    models_in_alert: list[str]     # PSI > 0.25
    last_refreshed: str


# ── Data refresh (stub — replace with ERDW calls) ──

def _now_iso() -> str:
    return datetime.now(
        tz=timezone.utc
    ).isoformat()


def get_capital_ratios() -> CapitalRatios:
    """
    Fetch capital ratios from ERDW.

    Production: SELECT cet1_ratio, tier1_ratio,
    total_capital_ratio, leverage_ratio
    FROM erdw.capital_ratios_latest
    WHERE reporting_date = CURRENT_DATE;
    """
    return CapitalRatios(
        cet1_ratio=14.2,
        tier1_ratio=15.8,
        total_capital_ratio=18.1,
        leverage_ratio=4.7,
        last_refreshed=_now_iso(),
    )


def get_liquidity() -> LiquidityMetrics:
    """Fetch LCR/NSFR/CLAR from ERDW."""
    return LiquidityMetrics(
        lcr=138.5,
        nsfr=112.3,
        clar=115.0,
        last_refreshed=_now_iso(),
    )


def get_ecl() -> ECLMetrics:
    """Fetch ECL breakdown from ERDW."""
    return ECLMetrics(
        total_ecl_gbp=187_400_000,
        stage1_ecl_gbp=12_200_000,
        stage2_ecl_gbp=54_800_000,
        stage3_ecl_gbp=120_400_000,
        coverage_ratio=0.67,
        last_refreshed=_now_iso(),
    )


def get_aml_metrics() -> AMLMetrics:
    """Fetch AML metrics from AML service."""
    return AMLMetrics(
        alerts_mtd=847,
        sars_filed_mtd=28,
        sar_rate=0.033,
        avg_quality_score=4.2,
        false_positive_rate=0.20,
        last_refreshed=_now_iso(),
    )


def get_model_performance() -> ModelPerformance:
    """
    Fetch PSI + RAGAS from MLflow registry.

    PSI alert threshold: > 0.25 (major shift).
    """
    psi = {
        "MR-2026-035": 0.08,
        "MR-2026-036": 0.11,
        "MR-2026-037": 0.07,
        "MR-2026-040": 0.09,
        "MR-2026-041": 0.13,
        "MR-2026-042": 0.06,
        "MR-2026-043": 0.10,
        "MR-2026-044": 0.08,
        "MR-2026-045": 0.12,
        "MR-2026-046": 0.09,
        "MR-2026-047": 0.07,
        "MR-2026-048": 0.11,
        "MR-2026-049": 0.08,
    }
    ragas = {
        "MR-2026-038": 0.87,
        "MR-2026-039": 0.91,
    }
    in_alert = [
        mr for mr, score in psi.items()
        if score > 0.25
    ]
    return ModelPerformance(
        psi_scores=psi,
        ragas_scores=ragas,
        models_in_alert=in_alert,
        last_refreshed=_now_iso(),
    )


# ── Endpoints ──────────────────────────────────────

@app.get("/health")
async def health() -> Dict[str, str]:
    return {
        "status": "healthy",
        "service": "cro-dashboard",
    }


@app.get(
    "/capital-ratios",
    response_model=CapitalRatios,
)
async def capital_ratios() -> CapitalRatios:
    """CRR3 capital and leverage ratios."""
    return get_capital_ratios()


@app.get(
    "/liquidity",
    response_model=LiquidityMetrics,
)
async def liquidity() -> LiquidityMetrics:
    """LCR / NSFR / CLAR liquidity metrics."""
    return get_liquidity()


@app.get("/ecl", response_model=ECLMetrics)
async def ecl() -> ECLMetrics:
    """IFRS 9 ECL by stage."""
    return get_ecl()


@app.get("/aml", response_model=AMLMetrics)
async def aml() -> AMLMetrics:
    """AML alert volume and quality metrics."""
    return get_aml_metrics()


@app.get(
    "/model-performance",
    response_model=ModelPerformance,
)
async def model_performance() -> ModelPerformance:
    """PSI and RAGAS scores for all models."""
    return get_model_performance()


@app.get("/summary")
async def summary() -> Dict[str, Any]:
    """Combined CRO/CFO summary view."""
    cap = get_capital_ratios()
    liq = get_liquidity()
    ecl_ = get_ecl()
    aml_ = get_aml_metrics()
    mp = get_model_performance()
    return {
        "capital": cap.model_dump(),
        "liquidity": liq.model_dump(),
        "ecl": ecl_.model_dump(),
        "aml": aml_.model_dump(),
        "model_performance": mp.model_dump(),
        "platform": {
            "systems_deployed": 23,
            "programme": "AWB-AI-2025",
            "investment_gbp": 3_200_000,
            "year1_roi_gbp": 6_700_000,
        },
    }
