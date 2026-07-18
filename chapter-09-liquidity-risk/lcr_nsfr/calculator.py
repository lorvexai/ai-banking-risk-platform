"""
lcr_nsfr/calculator.py — AWB LCR and NSFR Calculator.
Model ID: MR-2026-053 | PRA SS1/23 Risk: LOW
CRR3 Art. 411-428 (LCR) and Art. 428a-428au (NSFR).
Avon & Wessex Bank plc (AWB) — AWB-AI-2025 programme.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import datetime
from awb_commons.models import (
    LCRCalculation, NSFRCalculation, StressScenario
)

logger = logging.getLogger(__name__)

# CRR3 Art. 422 — retail deposit run-off rates
RETAIL_RUN_OFF_STABLE    = 0.05   # stable (insured)
RETAIL_RUN_OFF_LESS_STABLE = 0.10  # less stable

# CRR3 Art. 425 — inflow cap
INFLOW_CAP_PCT = 0.75  # max 75% of outflows

# CRR3 Art. 428 — HQLA haircuts
HQLA_LEVEL_1_HAIRCUT  = 0.00
HQLA_LEVEL_2A_HAIRCUT = 0.15
HQLA_LEVEL_2B_HAIRCUT = 0.50
HQLA_LEVEL_2_CAP      = 0.40  # max 40% of HQLA


@dataclass
class HQLAPortfolio:
    """High Quality Liquid Assets portfolio breakdown."""
    level_1_central_bank_gbp: float = 0.0
    level_1_gov_bonds_gbp:    float = 0.0
    level_2a_covered_bonds_gbp: float = 0.0
    level_2b_corp_bonds_gbp:  float = 0.0


@dataclass
class StressOutflows:
    """30-day stress outflow assumptions (CRR3 Art. 422-424)."""
    retail_stable_gbp:        float = 0.0
    retail_less_stable_gbp:   float = 0.0
    wholesale_operational_gbp: float = 0.0
    wholesale_non_op_gbp:     float = 0.0
    committed_facilities_gbp: float = 0.0
    derivatives_collateral_gbp: float = 0.0


@dataclass
class StressInflows:
    """30-day stress inflow assumptions (CRR3 Art. 425)."""
    maturing_loans_gbp:     float = 0.0
    committed_inflows_gbp:  float = 0.0
    other_inflows_gbp:      float = 0.0


@dataclass
class NSFRInputs:
    """NSFR inputs: available vs required stable funding."""
    tier1_capital_gbp:          float = 0.0
    tier2_capital_gbp:          float = 0.0
    stable_retail_deposits_gbp: float = 0.0
    less_stable_deposits_gbp:   float = 0.0
    wholesale_funding_1y_gbp:   float = 0.0
    loans_lt_1y_gbp:            float = 0.0
    loans_gt_1y_gbp:            float = 0.0
    hqla_unencumbered_gbp:      float = 0.0
    other_assets_gbp:           float = 0.0


class LCRCalculator:
    """
    CRR3 LCR calculator: HQLA / net 30-day stress outflows.

    Regulatory minimum: 100% (CRR3 Art. 412).
    AWB internal buffer: 110% (ILAA requirement).
    PRA reports LCR monthly via FSA047/048 returns.
    """

    def calculate(
        self,
        hqla: HQLAPortfolio,
        outflows: StressOutflows,
        inflows: StressInflows,
        scenario: StressScenario = StressScenario.BASE,
        calculation_date: datetime | None = None,
    ) -> LCRCalculation:
        """
        Compute LCR with full HQLA haircuts and inflow cap.

        Args:
            hqla: HQLA portfolio by tier.
            outflows: 30-day stress outflow assumptions.
            inflows: 30-day stress inflow assumptions.
            scenario: Stress scenario for run-off rates.
            calculation_date: Valuation date.

        Returns:
            LCRCalculation with ratio and compliance flag.
        """
        adjusted_hqla = self._adjusted_hqla(hqla)
        gross_outflows = self._gross_outflows(
            outflows, scenario
        )
        gross_inflows = self._gross_inflows(inflows)
        capped_inflows = min(
            gross_inflows, gross_outflows * INFLOW_CAP_PCT
        )
        net_outflows = max(
            gross_outflows - capped_inflows, 1.0
        )
        lcr_pct = (adjusted_hqla / net_outflows) * 100.0

        logger.info(
            "LCR: HQLA=£%.1fB outflows=£%.1fB "
            "LCR=%.1f%% scenario=%s",
            adjusted_hqla / 1e9,
            net_outflows / 1e9,
            lcr_pct,
            scenario,
        )
        return LCRCalculation(
            calculation_date=calculation_date or datetime.utcnow(),
            hqla_gbp=adjusted_hqla,
            net_outflows_gbp=net_outflows,
            lcr_pct=round(lcr_pct, 2),
            compliant=(lcr_pct >= 100.0),
            scenario=scenario,
        )

    # ── Private helpers ───────────────────────────────────────────

    def _adjusted_hqla(self, hqla: HQLAPortfolio) -> float:
        level_1 = (
            hqla.level_1_central_bank_gbp
            + hqla.level_1_gov_bonds_gbp
        )
        level_2a = (
            hqla.level_2a_covered_bonds_gbp
            * (1 - HQLA_LEVEL_2A_HAIRCUT)
        )
        level_2b = (
            hqla.level_2b_corp_bonds_gbp
            * (1 - HQLA_LEVEL_2B_HAIRCUT)
        )
        total_hqla = level_1 + level_2a + level_2b
        # Level 2 cap: max 40% of total HQLA
        level_2_sum = level_2a + level_2b
        max_level_2 = total_hqla * HQLA_LEVEL_2_CAP
        if level_2_sum > max_level_2:
            total_hqla = level_1 + max_level_2
        return total_hqla

    def _gross_outflows(
        self,
        out: StressOutflows,
        scenario: StressScenario,
    ) -> float:
        stress_multiplier = {
            StressScenario.BASE:          1.0,
            StressScenario.IDIOSYNCRATIC: 1.15,
            StressScenario.MARKET_WIDE:   1.25,
            StressScenario.COMBINED:      1.40,
            StressScenario.PRA_CST_SEVERE:1.55,
        }.get(scenario, 1.0)
        base = (
            out.retail_stable_gbp * RETAIL_RUN_OFF_STABLE
            + out.retail_less_stable_gbp
            * RETAIL_RUN_OFF_LESS_STABLE
            + out.wholesale_operational_gbp * 0.25
            + out.wholesale_non_op_gbp * 1.00
            + out.committed_facilities_gbp * 0.10
            + out.derivatives_collateral_gbp * 0.20
        )
        return base * stress_multiplier

    def _gross_inflows(self, inf: StressInflows) -> float:
        return (
            inf.maturing_loans_gbp * 0.50
            + inf.committed_inflows_gbp * 0.50
            + inf.other_inflows_gbp * 0.00
        )


class NSFRCalculator:
    """
    CRR3 NSFR calculator: ASF / RSF.
    Regulatory minimum: 100% (CRR3 Art. 428b).
    """

    # ASF factors (CRR3 Art. 428d-428h)
    ASF_FACTORS = {
        'tier1_capital': 1.00,
        'tier2_capital': 1.00,
        'stable_retail': 0.95,
        'less_stable_retail': 0.90,
        'wholesale_gt_1y': 1.00,
    }
    # RSF factors (CRR3 Art. 428p-428ae)
    RSF_FACTORS = {
        'loans_lt_1y': 0.50,
        'loans_gt_1y': 0.65,
        'hqla_unencumbered': 0.05,
        'other_assets': 0.85,
    }

    def calculate(
        self,
        inputs: NSFRInputs,
        calculation_date: datetime | None = None,
    ) -> NSFRCalculation:
        """Compute NSFR from stable funding inputs."""
        asf = self._available_stable_funding(inputs)
        rsf = self._required_stable_funding(inputs)
        nsfr_pct = (asf / max(rsf, 1.0)) * 100.0
        logger.info(
            "NSFR: ASF=£%.1fB RSF=£%.1fB NSFR=%.1f%%",
            asf / 1e9, rsf / 1e9, nsfr_pct,
        )
        return NSFRCalculation(
            calculation_date=calculation_date or datetime.utcnow(),
            available_stable_funding_gbp=asf,
            required_stable_funding_gbp=rsf,
            nsfr_pct=round(nsfr_pct, 2),
            compliant=(nsfr_pct >= 100.0),
        )

    def _available_stable_funding(
        self, inp: NSFRInputs
    ) -> float:
        return (
            inp.tier1_capital_gbp * 1.00
            + inp.tier2_capital_gbp * 1.00
            + inp.stable_retail_deposits_gbp * 0.95
            + inp.less_stable_deposits_gbp * 0.90
            + inp.wholesale_funding_1y_gbp * 1.00
        )

    def _required_stable_funding(
        self, inp: NSFRInputs
    ) -> float:
        return (
            inp.loans_lt_1y_gbp * 0.50
            + inp.loans_gt_1y_gbp * 0.65
            + inp.hqla_unencumbered_gbp * 0.05
            + inp.other_assets_gbp * 0.85
        )
