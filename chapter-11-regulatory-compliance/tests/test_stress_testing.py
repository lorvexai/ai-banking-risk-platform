"""Tests for PRA CST and BoE CBES stress scenario loaders."""
import pytest
from decimal import Decimal

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from stress_testing.pra_cst_loader import (
    PRAStressScenarioLoader, CBESScenarioLoader,
)


class TestPRAStressScenarioLoader:
    @pytest.fixture
    def loader(self): return PRAStressScenarioLoader()

    def test_base_scenario_no_stress(self, loader):
        """Base scenario: zero RWA growth, no GDP shock."""
        scenario = loader.get_scenario("base")
        assert scenario.rwa_growth_pct == Decimal("0.00")
        assert scenario.gdp_shock_pct == Decimal("0.00")

    def test_adverse_scenario_parameters(self, loader):
        """Adverse: moderate stress parameters."""
        s = loader.get_scenario("adverse")
        assert s.rwa_growth_pct > 0
        assert s.gdp_shock_pct < 0
        assert s.credit_loss_rate_pct > Decimal("1.0")

    def test_severe_scenario_more_stressed(self, loader):
        """Severe scenario is more stressed than adverse."""
        adv = loader.get_scenario("adverse")
        sev = loader.get_scenario("severe")
        assert sev.rwa_growth_pct > adv.rwa_growth_pct
        assert sev.gdp_shock_pct < adv.gdp_shock_pct
        assert sev.credit_loss_rate_pct > adv.credit_loss_rate_pct

    def test_no_ccar_dfast_scenarios(self, loader):
        """CCAR/DFAST are US-only — not present in PRA CST."""
        with pytest.raises(KeyError):
            loader.get_scenario("ccar_adverse")
        with pytest.raises(KeyError):
            loader.get_scenario("dfast_severely_adverse")

    def test_unknown_scenario_raises(self, loader):
        """Unknown scenario name raises KeyError."""
        with pytest.raises(KeyError):
            loader.get_scenario("made_up_scenario")

    def test_three_pra_cst_scenarios_available(self, loader):
        """PRA CST provides base, adverse, severe scenarios."""
        for name in ["base", "adverse", "severe"]:
            s = loader.get_scenario(name)
            assert s.name == name


class TestCBESScenarioLoader:
    @pytest.fixture
    def loader(self): return CBESScenarioLoader()

    def test_early_action_lowest_physical_risk(self, loader):
        """Early action: lowest physical risk uplift."""
        s = loader.get_scenario("early_action")
        assert s.physical_risk_rwa_uplift < Decimal("0.15")

    def test_no_action_highest_physical_risk(self, loader):
        """No additional action: highest physical risk."""
        no = loader.get_scenario("no_action")
        ea = loader.get_scenario("early_action")
        assert no.physical_risk_rwa_uplift > ea.physical_risk_rwa_uplift

    def test_three_cbes_pathways(self, loader):
        """BoE CBES provides three climate pathways."""
        for name in ["early_action", "late_action", "no_action"]:
            s = loader.get_scenario(name)
            assert s.name == name

    def test_cbes_exploratory_not_capital_setting(self, loader):
        """CBES is exploratory — strandeds are illustrative."""
        s = loader.get_scenario("late_action")
        assert Decimal("0") < s.stranded_assets_pct < Decimal("1")

    def test_unknown_cbes_scenario_raises(self, loader):
        """Unknown CBES scenario raises KeyError."""
        with pytest.raises(KeyError):
            loader.get_scenario("unknown_pathway")
