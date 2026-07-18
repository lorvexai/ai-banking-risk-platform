"""PRA Concurrent Stress Test (CST) — scenario loader.

Replaces CCAR/DFAST (US-only) with UK-applicable:
  - PRA CST: annual exercise, adverse + severe scenarios
  - BoE CBES: Climate Biennial Exploratory Scenario (2026)
"""
from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict
import logging

log = logging.getLogger(__name__)


@dataclass
class PRAStressScenario:
    """PRA CST scenario parameters."""
    name: str           # "adverse" or "severe"
    rwa_growth_pct: Decimal
    lcr_outflow_multiplier: Decimal
    credit_loss_rate_pct: Decimal
    gdp_shock_pct: Decimal


@dataclass
class CBESScenario:
    """BoE CBES climate scenario (2026 exercise)."""
    name: str           # "early_action", "late_action", "no_action"
    stranded_assets_pct: Decimal   # % of portfolio stranded
    transition_risk_rwa_uplift: Decimal
    physical_risk_rwa_uplift: Decimal


class PRAStressScenarioLoader:
    """Load PRA CST adverse and severe stress parameters.

    AWB submits stressed capital projections covering a 3-year
    horizon per PRA CST methodology. This loader provides the
    prescribed scenario parameters for COREP stress returns.

    PRA CST replaces CCAR/DFAST (US Federal Reserve) for UK banks.
    """

    SCENARIOS: Dict[str, PRAStressScenario] = {
        "base": PRAStressScenario(
            name="base",
            rwa_growth_pct=Decimal("0.00"),
            lcr_outflow_multiplier=Decimal("1.00"),
            credit_loss_rate_pct=Decimal("0.80"),
            gdp_shock_pct=Decimal("0.00"),
        ),
        "adverse": PRAStressScenario(
            name="adverse",
            rwa_growth_pct=Decimal("0.25"),
            lcr_outflow_multiplier=Decimal("1.15"),
            credit_loss_rate_pct=Decimal("2.10"),
            gdp_shock_pct=Decimal("-3.50"),
        ),
        "severe": PRAStressScenario(
            name="severe",
            rwa_growth_pct=Decimal("0.45"),
            lcr_outflow_multiplier=Decimal("1.35"),
            credit_loss_rate_pct=Decimal("4.80"),
            gdp_shock_pct=Decimal("-7.20"),
        ),
    }

    def get_scenario(
        self, scenario_name: str
    ) -> PRAStressScenario:
        """Load a named PRA CST scenario.

        Args:
            scenario_name: "base", "adverse", or "severe".

        Returns:
            PRAStressScenario with parameterisation.

        Raises:
            KeyError: If scenario name not recognised.
        """
        if scenario_name not in self.SCENARIOS:
            raise KeyError(
                f"Unknown PRA CST scenario: {scenario_name}. "
                f"Valid: {list(self.SCENARIOS.keys())}"
            )
        scenario = self.SCENARIOS[scenario_name]
        log.info(
            "PRA CST scenario loaded: %s "
            "(RWA +%s%% GDP %s%%)",
            scenario.name,
            scenario.rwa_growth_pct * 100,
            scenario.gdp_shock_pct,
        )
        return scenario


class CBESScenarioLoader:
    """Load BoE CBES climate scenario parameters (2026 edition).

    CBES is exploratory (not capital-setting). Results inform
    supervisory dialogue on AWB's climate risk management.
    Three pathways per BoE CBES framework:
      - Early action: orderly transition, lower physical risk
      - Late action: disorderly transition, moderate physical risk
      - No additional action: high physical risk
    """

    SCENARIOS: Dict[str, CBESScenario] = {
        "early_action": CBESScenario(
            name="early_action",
            stranded_assets_pct=Decimal("0.08"),
            transition_risk_rwa_uplift=Decimal("0.12"),
            physical_risk_rwa_uplift=Decimal("0.05"),
        ),
        "late_action": CBESScenario(
            name="late_action",
            stranded_assets_pct=Decimal("0.15"),
            transition_risk_rwa_uplift=Decimal("0.25"),
            physical_risk_rwa_uplift=Decimal("0.12"),
        ),
        "no_action": CBESScenario(
            name="no_action",
            stranded_assets_pct=Decimal("0.22"),
            transition_risk_rwa_uplift=Decimal("0.08"),
            physical_risk_rwa_uplift=Decimal("0.35"),
        ),
    }

    def get_scenario(
        self, scenario_name: str
    ) -> CBESScenario:
        """Load a BoE CBES climate scenario.

        Args:
            scenario_name: One of the three CBES pathways.

        Returns:
            CBESScenario with RWA uplift parameters.
        """
        if scenario_name not in self.SCENARIOS:
            raise KeyError(
                f"Unknown CBES scenario: {scenario_name}"
            )
        scenario = self.SCENARIOS[scenario_name]
        log.info(
            "CBES scenario: %s (stranded=%s%% RWA+%s%%)",
            scenario.name,
            scenario.stranded_assets_pct * 100,
            scenario.transition_risk_rwa_uplift * 100,
        )
        return scenario
