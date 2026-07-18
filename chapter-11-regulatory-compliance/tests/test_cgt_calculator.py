"""Tests for HMRC CGT Section 104 Pool Calculator."""
import pytest
from datetime import date
from decimal import Decimal

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hmrc_tax.section104_calculator import (
    Section104PoolCalculator, ISAReportingEngine,
    CGT_ANNUAL_ALLOWANCE, ISA_ANNUAL_LIMIT,
)


class TestSection104PoolCalculator:
    """TCGA 1992 s104 pool — average cost method."""

    @pytest.fixture
    def calc(self):
        return Section104PoolCalculator("Barclays PLC")

    def test_single_acquisition_pool_tracking(self, calc):
        """Pool tracks total quantity and cost."""
        calc.add_acquisition(
            date(2023, 3, 1),
            Decimal("1000"),
            Decimal("2500.00"),
        )
        assert calc._pool.total_quantity == Decimal("1000")
        assert calc._pool.total_qualifying_expenditure == (
            Decimal("2500.00")
        )

    def test_average_cost_per_unit(self, calc):
        """Average cost = total cost / total quantity."""
        calc.add_acquisition(
            date(2023, 3, 1), Decimal("1000"), Decimal("2500")
        )
        calc.add_acquisition(
            date(2024, 1, 15), Decimal("500"), Decimal("1500")
        )
        # avg = 4000 / 1500 = 2.6667
        expected_avg = Decimal("4000") / Decimal("1500")
        assert abs(
            calc._pool.average_cost_per_unit - expected_avg
        ) < Decimal("0.001")

    def test_disposal_gain_calculation(self, calc):
        """Disposal gain = proceeds - pool allowable cost."""
        calc.add_acquisition(
            date(2023, 3, 1), Decimal("1000"), Decimal("2000")
        )
        disposal = calc.add_disposal(
            date(2025, 11, 15),
            Decimal("500"),
            Decimal("1800"),
        )
        # Allowable cost = 500 × (2000/1000) = £1000
        assert disposal.allowable_cost_gbp == Decimal("1000.00")
        assert disposal.gain_or_loss_gbp == Decimal("800.00")
        assert disposal.is_gain

    def test_disposal_loss_detection(self, calc):
        """Disposal at below pool average records a loss."""
        calc.add_acquisition(
            date(2022, 6, 1), Decimal("1000"), Decimal("5000")
        )
        disposal = calc.add_disposal(
            date(2025, 10, 1),
            Decimal("500"),
            Decimal("1800"),
        )
        assert not disposal.is_gain

    def test_pool_reduces_after_disposal(self, calc):
        """Pool quantity decreases after disposal."""
        calc.add_acquisition(
            date(2023, 1, 1), Decimal("2000"), Decimal("6000")
        )
        calc.add_disposal(
            date(2025, 6, 1), Decimal("500"), Decimal("2000")
        )
        assert calc._pool.total_quantity == Decimal("1500")

    def test_disposal_exceeds_pool_raises(self, calc):
        """Cannot dispose more than pool holds."""
        calc.add_acquisition(
            date(2023, 1, 1), Decimal("100"), Decimal("500")
        )
        with pytest.raises(ValueError, match="Cannot dispose"):
            calc.add_disposal(
                date(2025, 1, 1), Decimal("200"), Decimal("1000")
            )

    def test_zero_quantity_acquisition_raises(self, calc):
        """Zero quantity acquisition must raise."""
        with pytest.raises(ValueError):
            calc.add_acquisition(
                date(2023, 1, 1), Decimal("0"), Decimal("500")
            )

    def test_net_gain_aggregation(self, calc):
        """Total net gain aggregates gains and losses."""
        calc.add_acquisition(
            date(2022, 1, 1), Decimal("1000"), Decimal("2000")
        )
        calc.add_disposal(
            date(2025, 3, 1), Decimal("500"), Decimal("1500")
        )  # Gain £500
        calc.add_disposal(
            date(2025, 9, 1), Decimal("500"), Decimal("800")
        )  # Loss £200
        assert calc.total_net_gain == Decimal("300.00")

    def test_cgt_annual_allowance_2025_26(self):
        """CGT annual allowance 2025-26: £3,000."""
        assert CGT_ANNUAL_ALLOWANCE == Decimal("3000")


class TestISAReportingEngine:
    """ISA subscription monitoring and HMRC ISATR XML."""

    @pytest.fixture
    def engine(self):
        return ISAReportingEngine("2025-26")

    def test_subscription_within_limit(self, engine):
        """Subscription within £20,000 limit returns True."""
        result = engine.record_subscription(
            "CLIENT-001", Decimal("15000")
        )
        assert result is True

    def test_subscription_breaches_limit(self, engine):
        """Over-subscription returns False and logs error."""
        engine.record_subscription("CLIENT-002", Decimal("18000"))
        result = engine.record_subscription(
            "CLIENT-002", Decimal("3000")
        )  # Total £21,000 > £20,000
        assert result is False

    def test_isa_annual_limit_2025_26(self):
        """ISA annual limit 2025-26: £20,000."""
        assert ISA_ANNUAL_LIMIT == Decimal("20000")

    def test_isatr_xml_generated(self, engine):
        """ISATR XML contains client ID and subscription."""
        engine.record_subscription("CLI-999", Decimal("10000"))
        xml = engine.generate_isatr_xml("CLI-999")
        assert "CLI-999" in xml
        assert "10000" in xml
        assert "COMPLIANT" in xml

    def test_isatr_xml_breach_status(self, engine):
        """ISATR XML shows BREACH for over-subscribed client."""
        engine.record_subscription("CLI-888", Decimal("25000"))
        xml = engine.generate_isatr_xml("CLI-888")
        assert "BREACH" in xml
