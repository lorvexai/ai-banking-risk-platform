"""Tests: Digital Identity Platform MR-2026-062 — POCA/MLR 2017."""
import pytest
from datetime import date, timedelta
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from digital_identity import (
    KYCDocumentVerifier, LivenessDetector, PEPSanctionsScreener,
    KYCEngine, LIVENESS_AUTO_PASS, LIVENESS_MANUAL_REVIEW,
    SANCTIONS_AUTO_BLOCK, SANCTIONS_REVIEW, FATF_HIGH_RISK_JURISDICTIONS,
)
from awb_commons.models import KYCStatus, KYCDocumentExtract, PEPSanctionsResult


class TestKYCDocumentVerifier:
    @pytest.fixture
    def v(self): return KYCDocumentVerifier()

    def test_passport_verified(self, v):
        e = v.verify_document(b"img", "passport")
        assert e.document_type == "passport"
        assert e.verification_status == "VERIFIED"
        assert e.confidence >= 0.90

    def test_driving_licence(self, v):
        e = v.verify_document(b"img", "driving_licence")
        assert e.document_type == "driving_licence"

    def test_utility_bill(self, v):
        e = v.verify_document(b"img", "utility_bill")
        assert e.document_type == "utility_bill"

    def test_unsupported_raises(self, v):
        with pytest.raises(ValueError, match="Unsupported"):
            v.verify_document(b"img", "bank_statement")

    def test_model_id_mr_2026_050(self, v):
        e = v.verify_document(b"img", "passport")
        assert e.model_id == "MR-2026-062"

    def test_expiry_in_future(self, v):
        e = v.verify_document(b"img", "passport")
        assert e.expiry_date > date.today()

    def test_mrz_valid_for_passport(self, v):
        e = v.verify_document(b"img", "passport")
        assert e.mrz_valid is True

    def test_mrz_not_for_utility_bill(self, v):
        e = v.verify_document(b"img", "utility_bill")
        assert e.mrz_valid is False

    def test_validate_mrz_correct_length(self, v):
        l1 = "P" + "<" * 43
        l2 = "A" * 44
        result = v.validate_mrz(l1, l2)
        assert isinstance(result, bool)

    def test_validate_mrz_wrong_length(self, v):
        assert v.validate_mrz("SHORT", "ALSO_SHORT") is False

    def test_kycdocumentextract_schema(self, v):
        e = v.verify_document(b"img", "passport")
        assert isinstance(e, KYCDocumentExtract)
        assert hasattr(e, 'mrz_valid')


class TestLivenessDetector:
    @pytest.fixture
    def d(self): return LivenessDetector()

    def test_auto_pass_score(self, d):
        score, decision = d.check_liveness(b"video", b"photo")
        assert score >= LIVENESS_AUTO_PASS
        assert decision == "AUTO_PASS"

    def test_manual_review_range(self, d):
        assert d._classify_score(0.89) == "MANUAL_REVIEW"
        assert d._classify_score(0.86) == "MANUAL_REVIEW"

    def test_declined_below_085(self, d):
        assert d._classify_score(0.80) == "DECLINED"
        assert d._classify_score(0.50) == "DECLINED"

    def test_threshold_constants(self):
        assert LIVENESS_AUTO_PASS == 0.92
        assert LIVENESS_MANUAL_REVIEW == 0.85

    def test_custom_thresholds(self):
        d = LivenessDetector(auto_pass_threshold=0.95, review_threshold=0.88)
        assert d._auto_pass == 0.95

    def test_returns_score_and_decision(self, d):
        score, decision = d.check_liveness(b"v", b"p")
        assert isinstance(score, float)
        assert decision in ("AUTO_PASS", "MANUAL_REVIEW", "DECLINED")


class TestPEPSanctionsScreener:
    @pytest.fixture
    def s(self): return PEPSanctionsScreener()

    def test_clean_gb(self, s):
        r = s.screen_individual("C1", "John Smith", date(1980,1,1), "GB")
        assert not r.is_pep and not r.sanctions_hit and not r.requires_edd

    def test_russia_edd(self, s):
        r = s.screen_individual("C2", "Ivan P", date(1975,3,15), "RU")
        assert r.requires_edd

    def test_north_korea_edd(self, s):
        r = s.screen_individual("C3", "Kim T", date(1985,4,1), "KP")
        assert r.requires_edd

    def test_iran_edd(self, s):
        r = s.screen_individual("C4", "Ali R", date(1980,1,1), "IR")
        assert r.requires_edd

    def test_ofsi_in_source(self, s):
        r = s.screen_individual("C5", "Test", date(1990,1,1))
        assert "OFSI" in (r.screening_source_version or "")

    def test_screened_at_set(self, s):
        r = s.screen_individual("C6", "Name", date(1990,1,1))
        assert r.screened_at is not None

    def test_match_score_exact(self, s):
        assert s.calculate_match_score("John Smith", "John Smith") == 1.0

    def test_match_score_different(self, s):
        assert s.calculate_match_score("John Smith", "Mary Jones") < 1.0

    def test_classify_auto_block(self, s):
        assert s.classify_match(0.97) == "AUTO_BLOCK"

    def test_classify_review(self, s):
        assert s.classify_match(0.88) == "COMPLIANCE_REVIEW"

    def test_classify_clear(self, s):
        assert s.classify_match(0.70) == "CLEAR"

    def test_pep_lookback_12_months(self):
        r = PEPSanctionsResult(customer_id="C", name_screened="T")
        assert r.pep_look_back_months == 12

    def test_fatf_list_contents(self):
        assert "KP" in FATF_HIGH_RISK_JURISDICTIONS
        assert "RU" in FATF_HIGH_RISK_JURISDICTIONS
        assert "GB" not in FATF_HIGH_RISK_JURISDICTIONS


class TestKYCEngine:
    @pytest.fixture
    def engine(self): return KYCEngine()

    @pytest.fixture
    def good_doc(self):
        return KYCDocumentExtract(
            document_type="passport", full_name="Jane Brown",
            date_of_birth=date(1990,3,15), document_number="PP123",
            expiry_date=date.today()+timedelta(days=1825),
            issuing_country="GB", mrz_valid=True,
            confidence=0.97, verification_status="VERIFIED",
        )

    @pytest.fixture
    def clean_pep(self):
        return PEPSanctionsResult(customer_id="C1", name_screened="Jane Brown")

    def test_clean_cdd_pass(self, engine, good_doc, clean_pep):
        d = engine.assess_customer("C1", good_doc, 0.96, "AUTO_PASS", clean_pep)
        assert d.status == KYCStatus.CDD_PASS
        assert not d.review_required

    def test_sanctions_auto_block(self, engine, good_doc):
        s = PEPSanctionsResult(customer_id="CBAD", name_screened="Bad Actor",
                               sanctions_hit=True, match_score=0.97,
                               sanctions_lists_matched=["OFSI"])
        d = engine.assess_customer("CBAD", good_doc, 0.96, "AUTO_PASS", s)
        assert d.status == KYCStatus.SANCTIONS_HIT

    def test_liveness_declined(self, engine, good_doc, clean_pep):
        d = engine.assess_customer("C2", good_doc, 0.70, "DECLINED", clean_pep)
        assert d.status == KYCStatus.DECLINED

    def test_expired_doc_declined(self, engine, clean_pep):
        expired = KYCDocumentExtract(
            document_type="passport", full_name="Old",
            date_of_birth=date(1960,1,1), document_number="EXP",
            expiry_date=date(2020,1,1), issuing_country="GB",
            mrz_valid=True, confidence=0.95, verification_status="EXPIRED",
        )
        d = engine.assess_customer("C3", expired, 0.96, "AUTO_PASS", clean_pep)
        assert d.status == KYCStatus.DECLINED

    def test_pep_triggers_edd(self, engine, good_doc):
        pep = PEPSanctionsResult(customer_id="CPEP", name_screened="Min X",
                                 is_pep=True, requires_edd=True)
        d = engine.assess_customer("CPEP", good_doc, 0.96, "AUTO_PASS", pep)
        assert d.status == KYCStatus.EDD_REQUIRED
        assert "pep_identified" in (d.edd_trigger or "")

    def test_high_risk_jx_edd(self, engine, good_doc):
        hr = PEPSanctionsResult(customer_id="CHR", name_screened="T", requires_edd=True)
        d = engine.assess_customer("CHR", good_doc, 0.96, "AUTO_PASS", hr)
        assert d.status == KYCStatus.EDD_REQUIRED

    def test_manual_liveness_edd(self, engine, good_doc, clean_pep):
        d = engine.assess_customer("C4", good_doc, 0.88, "MANUAL_REVIEW", clean_pep)
        assert d.status == KYCStatus.EDD_REQUIRED
        assert "liveness_manual_review" in (d.edd_trigger or "")

    def test_biometric_template_deleted(self, engine, good_doc, clean_pep):
        d = engine.assess_customer("C5", good_doc, 0.96, "AUTO_PASS", clean_pep)
        assert d.biometric_template_deleted is True

    def test_model_id_correct(self, engine, good_doc, clean_pep):
        d = engine.assess_customer("C6", good_doc, 0.96, "AUTO_PASS", clean_pep)
        assert d.decided_by == "MR-2026-062"
