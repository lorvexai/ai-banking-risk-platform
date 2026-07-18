"""Tests: KYC Credit Borrower Screening MR-2026-063."""
import pytest
from datetime import date
from decimal import Decimal
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from kyc_credit import (
    CompaniesHouseClient, UBOTracer, KYCCreditScreener,
    UBO_THRESHOLD_PCT, MAX_OWNERSHIP_LAYERS, EDD_LARGE_EXPOSURE_GBP,
    FATF_HIGH_RISK,
)
from awb_commons.models import KYCStatus, UBORecord

CLEAN = [{"name":"John S","ownership_pct":60.0,"control_type":"shares",
          "is_pep":False,"nationality":"GB"}]
PEP   = [{"name":"Min X","ownership_pct":51.0,"control_type":"shares",
          "is_pep":True,"nationality":"GB"}]
HR    = [{"name":"Ivan V","ownership_pct":100.0,"control_type":"shares",
          "is_pep":False,"nationality":"RU"}]
LOW   = [{"name":"Minor","ownership_pct":10.0,"control_type":"shares",
          "is_pep":False,"nationality":"GB"}]


class TestCompaniesHouseClient:
    @pytest.fixture
    def ch(self): return CompaniesHouseClient()

    def test_company_data_returned(self, ch):
        d = ch.get_company_data("12345678")
        assert d["company_number"] == "12345678"
        assert "company_status" in d

    def test_active_no_flags(self, ch):
        d = {"company_status":"active","date_of_creation":"2020-01-01"}
        flags = ch.check_red_flags(d)
        assert "company_status_dissolved" not in flags

    def test_dissolved_red_flag(self, ch):
        d = {"company_status":"dissolved","date_of_creation":"2010-01-01"}
        assert "company_status_dissolved" in ch.check_red_flags(d)

    def test_dormant_red_flag(self, ch):
        d = {"company_status":"dormant","date_of_creation":"2015-01-01"}
        assert "company_status_dormant" in ch.check_red_flags(d)

    def test_psc_returns_list(self, ch):
        assert isinstance(ch.get_psc_register("12345678"), list)

    def test_uk_api_not_sec_edgar(self):
        """Companies House (UK) — NOT SEC EDGAR (US-only)."""
        assert "companieshouse.gov.uk" in CompaniesHouseClient.BASE_URL

    def test_no_sec_edgar_api_calls(self):
        """Confirm no SEC EDGAR (US) API calls in code."""
        import kyc_credit
        source = open(kyc_credit.__file__).read()
        assert "sec.gov" not in source.lower()
        # EDGAR may appear in docstrings explaining it is NOT applicable
        assert "companieshouse.gov.uk" in source  # UK API used


class TestUBOTracer:
    @pytest.fixture
    def t(self): return UBOTracer()

    def test_above_threshold(self, t):
        ubos = t.trace_ubos("E1","CH1", CLEAN)
        assert len(ubos) == 1
        assert ubos[0].ubo_name == "John S"

    def test_below_threshold_excluded(self, t):
        ubos = t.trace_ubos("E2","CH2", LOW)
        assert len(ubos) == 0

    def test_threshold_25pct(self):
        """MLR 2017 Reg. 28(3)(b): >25% ownership threshold."""
        assert UBO_THRESHOLD_PCT == 25.0

    def test_max_layers_4(self):
        """JMLSG Part II Banking: up to 4 ownership layers."""
        assert MAX_OWNERSHIP_LAYERS == 4

    def test_pep_flagged(self, t):
        ubos = t.trace_ubos("E3","CH3", PEP)
        assert ubos[0].is_pep is True

    def test_high_risk_jurisdiction(self, t):
        ubos = t.trace_ubos("E4", None, HR)
        assert ubos[0].high_risk_jurisdiction is True

    def test_psc_verified_with_ch(self, t):
        ubos = t.trace_ubos("E5","CH5", CLEAN)
        assert ubos[0].psc_register_verified is True

    def test_no_ch_not_verified(self, t):
        ubos = t.trace_ubos("E6", None, CLEAN)
        assert ubos[0].psc_register_verified is False

    def test_deep_chain_complex(self, t):
        ubos = t.trace_ubos("E7","CH7", CLEAN)
        assert t.is_complex_structure(ubos, 4) is True

    def test_shallow_not_complex(self, t):
        ubos = t.trace_ubos("E8","CH8", CLEAN)
        assert t.is_complex_structure(ubos, 1) is False

    def test_no_ubos_complex(self, t):
        assert t.is_complex_structure([], 1) is True

    def test_pep_requires_edd(self):
        u = UBORecord(entity_id="E",ubo_name="T",ownership_pct=60,
                      control_type="shares",is_pep=True)
        assert u.requires_edd is True

    def test_clean_no_edd(self):
        u = UBORecord(entity_id="E",ubo_name="T",ownership_pct=60,
                      control_type="shares")
        assert u.requires_edd is False

    def test_ubo_threshold_constant(self):
        u = UBORecord(entity_id="E",ubo_name="T",ownership_pct=60,
                      control_type="shares")
        assert u.UBO_THRESHOLD_PCT == 25.0


class TestKYCCreditScreener:
    @pytest.fixture
    def s(self): return KYCCreditScreener()

    def test_clean_cdd_pass(self, s):
        r = s.screen_entity("E1","ACME","CH1","GB",CLEAN)
        assert r.status == KYCStatus.CDD_PASS
        assert not r.blocks_credit_decision

    def test_pep_ubo_edd(self, s):
        r = s.screen_entity("E2","PEP Corp","CH2","GB",PEP)
        assert r.status == KYCStatus.EDD_REQUIRED
        assert r.blocks_credit_decision
        assert "ubo_is_pep" in r.edd_reasons

    def test_high_risk_entity_country(self, s):
        r = s.screen_entity("E3","RU Corp",None,"RU",HR)
        assert r.edd_required and r.blocks_credit_decision

    def test_complex_structure_edd(self, s):
        r = s.screen_entity("E4","Complex","CH4","GB",CLEAN,
                            entity_chain_depth=4)
        assert r.edd_required

    def test_large_exposure_edd(self, s):
        r = s.screen_entity("E5","Big Borrow","CH5","GB",CLEAN,
                            proposed_exposure_gbp=Decimal("2000000"))
        assert r.edd_required
        assert any("large_exposure" in reason for reason in r.edd_reasons)

    def test_edd_threshold_1m(self):
        """MLR 2017 Reg. 33: EDD for large exposures."""
        assert EDD_LARGE_EXPOSURE_GBP == Decimal("1000000")

    def test_credit_gate_cleared(self, s):
        r = s.screen_entity("E6","Good","CH6","GB",CLEAN)
        assert s.get_credit_gate_decision(r) == "CLEARED"

    def test_credit_gate_edd_required(self, s):
        r = s.screen_entity("E7","PEP Inc","CH7","GB",PEP)
        assert s.get_credit_gate_decision(r) == "EDD_REQUIRED"

    def test_sar_filed_returns_blocked_only(self, s):
        """POCA 2002 s.333A: credit agent NEVER knows SAR was filed."""
        r = s.screen_entity("E8","Good Ltd","CH8","GB",CLEAN)
        # SAR filed but agent only sees BLOCKED
        gate = s.get_credit_gate_decision(r, sar_filed=True)
        assert gate == "BLOCKED"
        assert gate != "SAR_FILED"  # s.333A: SAR existence never disclosed

    def test_cleared_overridden_by_sar(self, s):
        """Even CDD_PASS is blocked if SAR filed (s.333A)."""
        r = s.screen_entity("E9","Good Co","CH9","GB",CLEAN)
        assert r.status == KYCStatus.CDD_PASS  # KYC clear
        gate = s.get_credit_gate_decision(r, sar_filed=True)
        assert gate == "BLOCKED"  # But SAR overrides

    def test_model_id_mr_2026_052(self, s):
        r = s.screen_entity("E10","Test","CH10","GB",CLEAN)
        assert r.model_id == "MR-2026-063"

    def test_assessed_date_today(self, s):
        r = s.screen_entity("E11","Test","CH11","GB",CLEAN)
        assert r.assessed_date == date.today()

    def test_ubos_populated(self, s):
        r = s.screen_entity("E12","Multi","CH12","GB",CLEAN)
        assert len(r.ubos) == 1

    def test_ch2_mrd_2026_035_reuse(self):
        """MR-2026-063 receives document output from Ch 2 MR-2026-035."""
        screener = KYCCreditScreener(model_id="MR-2026-063")
        assert screener._model_id == "MR-2026-063"

    def test_ch3_mrd_2026_037_gate(self):
        """MR-2026-063 provides KYC gate in Ch 3 LangGraph pipeline."""
        s = KYCCreditScreener()
        r = s.screen_entity("E13","Test","CH13","GB",CLEAN)
        gate = s.get_credit_gate_decision(r)
        # Gate output feeds MR-2026-037 Credit Decision Agent
        assert gate in ("CLEARED", "EDD_REQUIRED", "BLOCKED")

    def test_eu_ai_act_annex_iii_5b(self):
        """EU AI Act HIGH-RISK Annex III §5b: KYC affects credit assessment."""
        # MR-2026-063 is HIGH-RISK §5b — affects creditworthiness determination
        import kyc_credit
        source = open(kyc_credit.__file__).read()
        assert "MR-2026-063" in source
        assert "§5b" in source or "5b" in source

    def test_no_bsa_reference(self):
        """No BSA / FinCEN references — UK POCA 2002 primary."""
        import kyc_credit
        source = open(kyc_credit.__file__).read()
        assert "Bank Secrecy Act" not in source
        assert "FinCEN" not in source
        assert "PATRIOT" not in source

    def test_companies_house_not_sec_edgar(self):
        """Companies House (UK) — NOT SEC EDGAR (US-only)."""
        s = KYCCreditScreener()
        assert hasattr(s._ch_client, 'BASE_URL')
        assert "companieshouse" in s._ch_client.BASE_URL
