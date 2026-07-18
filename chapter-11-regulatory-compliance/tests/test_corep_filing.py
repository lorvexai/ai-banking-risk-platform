"""Tests for COREP Filing Engine and XBRL generation."""
import pytest
from datetime import date
from decimal import Decimal

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from basel_reporting.corep_filing_engine import (
    COREPFilingEngine, SAExposureData, IRBExposureData,
    SA_RISK_WEIGHTS,
)
from awb_commons.models import RWAResult, XBRLInstance
from mjrrp.xbrl_filer import EBAXBRLFiler

Q4_2025 = date(2025, 12, 31)


class TestCOREPFilingEngine:
    @pytest.fixture
    def engine(self):
        return COREPFilingEngine(Q4_2025)

    @pytest.fixture
    def sa_exposures(self):
        return [
            SAExposureData("retail", Decimal("8e9"), SA_RISK_WEIGHTS["retail"]),
            SAExposureData("corporates", Decimal("4e9"), SA_RISK_WEIGHTS["corporates"]),
            SAExposureData("residential_mortgage", Decimal("6e9"), SA_RISK_WEIGHTS["residential_mortgage"]),
        ]

    @pytest.fixture
    def irb_exposures(self):
        return [
            IRBExposureData("<0.5%", Decimal("5e9"), Decimal("0.45"), Decimal("2e9")),
            IRBExposureData("0.5-1%", Decimal("4e9"), Decimal("0.45"), Decimal("3e9")),
            IRBExposureData("1-2%", Decimal("2.2e9"), Decimal("0.45"), Decimal("2.2e9")),
            IRBExposureData(">5%", Decimal("0.8e9"), Decimal("0.45"), Decimal("4e9")),
        ]

    def test_c0200_generates_xbrl(self, engine, sa_exposures):
        """C 02.00 generation returns XBRLInstance."""
        instance = engine.generate_c0200(sa_exposures)
        assert instance.return_code == "C 02.00"

    def test_c0800_generates_xbrl(self, engine, irb_exposures):
        """C 08.00 generation returns XBRLInstance."""
        instance = engine.generate_c0800(irb_exposures)
        assert instance.return_code == "C 08.00"

    def test_sa_rwa_calculation(self):
        """SA RWA = Exposure × Risk Weight."""
        exp = SAExposureData(
            "retail", Decimal("1_000_000_000"),
            SA_RISK_WEIGHTS["retail"],
        )
        assert exp.rwa_gbp == Decimal("750_000_000")  # 75% RW

    def test_residential_mortgage_rw_35pct(self):
        """Residential mortgage risk weight = 35% (CRR3 Art.125)."""
        assert SA_RISK_WEIGHTS["residential_mortgage"] == Decimal("0.35")

    def test_sme_supporting_factor(self):
        """SME supporting factor: 76.5% per CRR3."""
        assert SA_RISK_WEIGHTS["sme"] == Decimal("0.765")

    def test_cross_return_consistency_passes(self, engine):
        """Consistent returns pass cross-validation."""
        rwa = RWAResult(
            quarter_end=Q4_2025,
            credit_risk_sa_gbp=Decimal("8.4e9"),
            credit_risk_irb_gbp=Decimal("11.2e9"),
            market_risk_frtb_gbp=Decimal("0.8e9"),
            operational_risk_sma_gbp=Decimal("1.4e9"),
        )
        # Exact match passes
        result = engine.validate_cross_return_consistency(
            Decimal("8.4e9"), Decimal("11.2e9"), rwa
        )
        assert result is True

    def test_cross_return_inconsistency_raises(self, engine):
        """Inconsistent returns raise ValueError (PRA Tier 1)."""
        rwa = RWAResult(
            quarter_end=Q4_2025,
            credit_risk_sa_gbp=Decimal("8.4e9"),
            credit_risk_irb_gbp=Decimal("11.2e9"),
            market_risk_frtb_gbp=Decimal("0.8e9"),
            operational_risk_sma_gbp=Decimal("1.4e9"),
        )
        with pytest.raises(ValueError, match="Tier 1 error"):
            engine.validate_cross_return_consistency(
                Decimal("8.0e9"),   # 2.4% discrepancy
                Decimal("11.2e9"), rwa,
            )


class TestEBAXBRLFiler:
    @pytest.fixture
    def filer(self):
        return EBAXBRLFiler(entity_id="AWB-UK-001", dry_run=True)

    def test_xbrl_instance_generation(self, filer):
        """XBRL instance document generated with correct return code."""
        instance = filer.generate_xbrl_instance_document(
            "C 47.00", Q4_2025,
            {"eba_c_47.00_tier1": "4100000000"}
        )
        assert instance.return_code == "C 47.00"
        assert "AWB-UK-001" in instance.xml_content
        assert "4100000000" in instance.xml_content

    def test_xbrl_taxonomy_version(self, filer):
        """EBA Taxonomy version = 4.0 (effective Q1 2025)."""
        instance = filer.generate_xbrl_instance_document(
            "C 72.00", Q4_2025, {}
        )
        assert instance.taxonomy_version == "4.0"

    def test_xbrl_validation_wellformed(self, filer):
        """Well-formed XML passes validation."""
        instance = filer.generate_xbrl_instance_document(
            "C 80.00", Q4_2025, {"nsfr": "118"}
        )
        validated = filer.validate_against_eba_taxonomy(instance)
        assert validated.is_valid

    def test_dry_run_submit(self, filer):
        """Dry run submission sets filing reference without real call."""
        instance = filer.generate_xbrl_instance_document(
            "C 47.00", Q4_2025, {"lr": "8.9"}
        )
        filer.validate_against_eba_taxonomy(instance)
        corep = filer.submit_to_pra_gabriel(instance, Q4_2025)
        assert corep.validation_passed
        assert "DRY" in corep.filing_reference

    def test_invalid_xbrl_blocks_submission(self, filer):
        """Invalid XBRL cannot be submitted to Gabriel."""
        bad = XBRLInstance(
            return_code="C 02.00",
            validation_errors=["Schema violation"]
        )
        with pytest.raises(ValueError, match="validation error"):
            filer.submit_to_pra_gabriel(bad, Q4_2025)
