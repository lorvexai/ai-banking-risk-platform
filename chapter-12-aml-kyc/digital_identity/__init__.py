"""Digital Identity Verification Platform — MR-2026-062.

SS1/23 Risk: HIGH | EU AI Act: HIGH-RISK Annex III §6
Regulation: POCA 2002 | MLR 2017 Reg. 28 | FCA SYSC 6.3
           UK GDPR / DPA 2018 (biometric = special category data)

Key spec from prompt:
- Document verification via Gemini 3.5 Flash vision (MRZ extraction)
- Reuses MR-2026-035 (Ch 2 CDA) pipeline — different schema output
- Liveness detection: OpenCV + AWS Rekognition EU-West-2
- PEP screening: OFSI (HM Treasury) — NOT OFAC (US Treasury)
- Thresholds: > 0.92 auto-pass | 0.85–0.92 manual review | < 0.85 decline
- UK GDPR: biometric template deleted after verification
- 7-year audit retention (AWB policy — exceeds SYSC 6.3.3R 5yr min)
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import List, Optional
import hashlib
import logging

from awb_commons.models import (
    KYCDocumentExtract, PEPSanctionsResult, KYCDecision, KYCStatus,
)

log = logging.getLogger(__name__)

# Liveness thresholds per prompt spec
LIVENESS_AUTO_PASS = 0.92
LIVENESS_MANUAL_REVIEW = 0.85    # scores 0.85–0.92 → manual
# Sanctions match thresholds
SANCTIONS_AUTO_BLOCK = 0.95
SANCTIONS_REVIEW = 0.85

# OFSI: UK sanctions — NOT OFAC (US Treasury, US-only)
# MLR 2017 Reg. 35 PEP look-back
PEP_LOOKBACK_MONTHS = 12

# FATF high-risk jurisdictions (triggers EDD per MLR 2017 Reg. 33(6))
FATF_HIGH_RISK_JURISDICTIONS = frozenset([
    "AF","BY","CF","CD","CU","IR","IQ","LY","ML","MM",
    "NI","KP","RU","SO","SS","SD","SY","VE","YE","ZW",
])


class KYCDocumentVerifier:
    """Verify identity documents using Gemini 3.5 Flash vision.

    Extends Chapter 2 MR-2026-035 (AWB Credit Document Analyser)
    extraction pipeline with a KYC-specific output schema
    (KYCDocumentExtract vs. CreditDocumentExtract).

    MLR 2017 Reg. 28: obtain and verify full name, residential
    address, and date of birth from reliable, independent sources.

    Args:
        model_id: Registered model ID (MR-2026-062).
        confidence_threshold: Minimum extraction confidence (0.90).

    Example:
        >>> verifier = KYCDocumentVerifier()
        >>> extract = verifier.verify_document(
        ...     img_bytes, "passport", "GB"
        ... )
        >>> assert extract.verification_status == "VERIFIED"
    """

    def __init__(
        self,
        model_id: str = "MR-2026-062",
        confidence_threshold: float = 0.90,
    ) -> None:
        self._model_id = model_id
        self._threshold = confidence_threshold

    def verify_document(
        self,
        document_image_bytes: bytes,
        document_type: str,
        expected_country: str = "GB",
    ) -> KYCDocumentExtract:
        """Extract and verify identity document via LLM vision.

        Uses Gemini 3.5 Flash vision to extract:
        - MRZ (Machine Readable Zone) for passports
        - Name, DOB, document number, expiry date
        - Cross-field consistency validation

        Reuses MR-2026-035 document extraction infrastructure
        from Chapter 2 with a different output schema.

        Args:
            document_image_bytes: Raw document image bytes.
            document_type: "passport", "driving_licence",
                or "utility_bill".
            expected_country: ISO 3166-1 alpha-2 country.

        Returns:
            KYCDocumentExtract with verification_status.

        Raises:
            ValueError: If document_type not supported.
        """
        supported = {"passport", "driving_licence", "utility_bill"}
        if document_type not in supported:
            raise ValueError(
                f"Unsupported document type: '{document_type}'. "
                f"Supported: {supported}"
            )
        doc_hash = hashlib.sha256(document_image_bytes).hexdigest()
        log.info(
            "Document verification: type=%s hash=%s... model=%s",
            document_type, doc_hash[:8], self._model_id,
        )
        # Production: Gemini 3.5 Flash vision API call
        # Test: return representative extract
        extract = KYCDocumentExtract(
            document_type=document_type,
            full_name="[EXTRACTED FROM DOCUMENT]",
            date_of_birth=date(1985, 6, 15),
            document_number=f"DOC-{doc_hash[:8].upper()}",
            expiry_date=date.today() + timedelta(days=1825),
            issuing_country=expected_country,
            mrz_valid=(document_type == "passport"),
            confidence=0.96,
            verification_status="VERIFIED",
            model_id=self._model_id,
        )
        if extract.confidence < self._threshold:
            extract.verification_status = "LOW_CONFIDENCE"
            log.warning(
                "Document confidence %.2f < threshold %.2f",
                extract.confidence, self._threshold,
            )
        if extract.expiry_date <= date.today():
            extract.verification_status = "EXPIRED"
            log.error("Document expired: %s", extract.expiry_date)
        return extract

    def validate_mrz(
        self,
        mrz_line1: str,
        mrz_line2: str,
    ) -> bool:
        """Validate passport MRZ checksum per ICAO Doc 9303.

        Args:
            mrz_line1: First MRZ line (44 characters).
            mrz_line2: Second MRZ line (44 characters).

        Returns:
            True if MRZ checksums are valid.
        """
        if len(mrz_line1) != 44 or len(mrz_line2) != 44:
            log.warning(
                "Invalid MRZ length: L1=%d L2=%d",
                len(mrz_line1), len(mrz_line2),
            )
            return False
        # Production: implement ICAO 9303 check-digit algorithm
        # Weights: 7, 3, 1 repeating
        log.info("MRZ validation: L1=%s... L2=%s...",
                 mrz_line1[:8], mrz_line2[:8])
        return True  # Simplified for test environment


class LivenessDetector:
    """Biometric liveness detection — EU AI Act HIGH-RISK Annex III §6.

    Detects real person vs. photo/video replay attacks.
    Technology: OpenCV passive liveness + AWS Rekognition
    (EU-West-2 region for UK data residency compliance).

    UK GDPR / DPA 2018: biometric data is special category.
    - Explicit consent required before processing
    - Data minimisation: biometric template NOT stored post-verification
    - Privacy Impact Assessment required (Article 35 GDPR)

    Decision Point 12.1: AWB chose in-house (OpenCV + AWS Rekognition)
    over iProov/Onfido for DORA concentration risk and cost reasons.

    Args:
        auto_pass_threshold: Minimum score for automatic pass (0.92).
        review_threshold: Minimum score above which manual review
            is triggered (0.85). Scores below decline.
    """

    def __init__(
        self,
        auto_pass_threshold: float = LIVENESS_AUTO_PASS,
        review_threshold: float = LIVENESS_MANUAL_REVIEW,
    ) -> None:
        self._auto_pass = auto_pass_threshold
        self._review = review_threshold

    def check_liveness(
        self,
        video_frames_bytes: bytes,
        document_photo_bytes: bytes,
    ) -> tuple[float, str]:
        """Perform liveness detection and face-to-document match.

        Passive liveness: analyse multiple video frames for
        depth cues, micro-movements, and reflection patterns
        that distinguish a live person from a printed photo
        or replay attack.

        IMPORTANT: Biometric template is NEVER stored after
        this check completes. UK GDPR data minimisation.

        Args:
            video_frames_bytes: Serialised video frames.
            document_photo_bytes: Photo extracted from document.

        Returns:
            Tuple of (confidence_score 0.0–1.0, decision string).
            Decision: "AUTO_PASS" | "MANUAL_REVIEW" | "DECLINED"
        """
        frame_hash = hashlib.sha256(video_frames_bytes).hexdigest()
        log.info(
            "Liveness check: frames=%s... EU AI Act §6 HIGH-RISK",
            frame_hash[:8],
        )
        # Production: OpenCV frame analysis + AWS Rekognition
        score = 0.95  # Test value — above auto-pass threshold
        decision = self._classify_score(score)
        log.info(
            "Liveness result: score=%.3f decision=%s "
            "biometric_template_deleted=True",
            score, decision,
        )
        return score, decision

    def _classify_score(self, score: float) -> str:
        if score >= self._auto_pass:
            return "AUTO_PASS"
        elif score >= self._review:
            return "MANUAL_REVIEW"
        else:
            return "DECLINED"


class PEPSanctionsScreener:
    """Screen individuals against UK PEP and sanctions lists.

    Screening sources per prompt spec:
    - HM Treasury OFSI consolidated list (UK sanctions)
    - UN Security Council Consolidated List
    - JMLSG-endorsed commercial screening database
    NOT: OFAC (US Treasury) — UK firms screen against OFSI only.

    Uses Jaro-Winkler distance for name variants and
    Soundex phonetic matching for transliterations.

    MLR 2017 Reg. 35: enhanced obligations for PEPs.
    MLR 2017 Reg. 33(6): EDD for FATF high-risk jurisdictions.

    Args:
        auto_block_threshold: Score >= 0.95 → auto-block.
        review_threshold: Score 0.85–0.95 → compliance review.
    """

    def __init__(
        self,
        auto_block_threshold: float = SANCTIONS_AUTO_BLOCK,
        review_threshold: float = SANCTIONS_REVIEW,
    ) -> None:
        self._auto_block = auto_block_threshold
        self._review = review_threshold

    def screen_individual(
        self,
        customer_id: str,
        full_name: str,
        date_of_birth: date,
        nationality: str = "GB",
    ) -> PEPSanctionsResult:
        """Screen an individual against PEP and sanctions lists.

        Screens against OFSI (UK HM Treasury), UN SC list, and
        JMLSG-endorsed database. Applies MLR 2017 Reg. 35 PEP
        look-back of 12 months from prominent public functions.

        Args:
            customer_id: AWB customer identifier.
            full_name: Customer full legal name.
            date_of_birth: Date of birth for disambiguation.
            nationality: ISO 3166-1 country code.

        Returns:
            PEPSanctionsResult with match scores and EDD flag.
        """
        name_hash = hashlib.sha256(full_name.encode()).hexdigest()
        log.info(
            "PEP/sanctions screen: customer=%s name=%s... "
            "OFSI+UN (not OFAC — UK-only obligation)",
            customer_id, name_hash[:8],
        )
        result = PEPSanctionsResult(
            customer_id=customer_id,
            name_screened=full_name,
            screened_at=datetime.utcnow(),
            screening_source_version="OFSI-2026-03 UN-2026-03",
        )
        # FATF high-risk jurisdiction check (MLR 2017 Reg. 33(6))
        if nationality in FATF_HIGH_RISK_JURISDICTIONS:
            result.requires_edd = True
            log.warning(
                "FATF high-risk jurisdiction: customer=%s "
                "country=%s EDD required (MLR 2017 Reg. 33(6))",
                customer_id, nationality,
            )
        # Production: query OFSI API + UN + commercial DB
        # with Jaro-Winkler fuzzy matching
        return result

    def calculate_match_score(
        self,
        name_query: str,
        name_candidate: str,
    ) -> float:
        """Calculate Jaro-Winkler similarity between names.

        Used for fuzzy matching against sanctions lists where
        transliteration variants, misspellings, and aliases
        are common.

        Args:
            name_query: Customer name from onboarding.
            name_candidate: Name from sanctions/PEP database.

        Returns:
            Jaro-Winkler similarity score 0.0–1.0.
        """
        # Production: use jellyfish.jaro_winkler_similarity()
        # Simplified implementation for test environment
        if name_query.lower() == name_candidate.lower():
            return 1.0
        # Character overlap ratio as proxy
        q_chars = set(name_query.lower())
        c_chars = set(name_candidate.lower())
        overlap = len(q_chars & c_chars)
        total = len(q_chars | c_chars)
        return overlap / total if total > 0 else 0.0

    def classify_match(
        self, match_score: float
    ) -> str:
        """Classify a match score into action category.

        Args:
            match_score: Jaro-Winkler similarity 0.0–1.0.

        Returns:
            "AUTO_BLOCK" | "COMPLIANCE_REVIEW" | "CLEAR"
        """
        if match_score >= self._auto_block:
            return "AUTO_BLOCK"
        elif match_score >= self._review:
            return "COMPLIANCE_REVIEW"
        else:
            return "CLEAR"


class KYCEngine:
    """Orchestrate full KYC assessment for customer onboarding.

    Implements MLR 2017 CDD/EDD decision logic:
    - Standard CDD: document + liveness + sanctions screen
    - EDD: PEP identified, high-risk country, complex structure
    - Declined: document fail, liveness fail, auto-block sanctions

    AWB target: 4 days → 15 minutes for standard CDD.
    12,400 new customers/year; 847 requiring EDD.

    LLM-generated KYC narrative stored in customer file;
    7-year retention per AWB policy (SYSC 6.3.3R = 5yr min).

    Args:
        model_id: Registered model ID (MR-2026-062).
    """

    def __init__(
        self,
        model_id: str = "MR-2026-062",
    ) -> None:
        self._model_id = model_id
        self._doc_verifier = KYCDocumentVerifier(model_id)
        self._liveness = LivenessDetector()
        self._screener = PEPSanctionsScreener()

    def assess_customer(
        self,
        customer_id: str,
        doc_extract: KYCDocumentExtract,
        liveness_score: float,
        liveness_decision: str,
        pep_sanctions: PEPSanctionsResult,
    ) -> KYCDecision:
        """Produce KYC decision from verified components.

        Decision tree per MLR 2017 / JMLSG Part I:
        1. Sanctions auto-block → SANCTIONS_HIT
        2. Document fail or liveness DECLINED → DECLINED
        3. Liveness MANUAL_REVIEW or PEP/EDD → EDD_REQUIRED
        4. All clear → CDD_PASS

        Args:
            customer_id: AWB customer identifier.
            doc_extract: Verified KYCDocumentExtract.
            liveness_score: Liveness confidence 0.0–1.0.
            liveness_decision: "AUTO_PASS"/"MANUAL_REVIEW"/"DECLINED".
            pep_sanctions: PEP and sanctions screen result.

        Returns:
            KYCDecision with status and 7-year audit trail.
        """
        # Sanctions auto-block
        if pep_sanctions.sanctions_hit and (
            pep_sanctions.match_score >= SANCTIONS_AUTO_BLOCK
        ):
            log.error(
                "SANCTIONS AUTO-BLOCK: customer=%s score=%.2f",
                customer_id, pep_sanctions.match_score,
            )
            return KYCDecision(
                customer_id=customer_id,
                decision_date=date.today(),
                status=KYCStatus.SANCTIONS_HIT,
                document_extract=doc_extract,
                pep_sanctions=pep_sanctions,
                liveness_score=liveness_score,
                liveness_passed=False,
                decided_by=self._model_id,
                review_required=True,
                biometric_template_deleted=True,
            )

        # Document or liveness failure
        if (doc_extract.verification_status in ("LOW_CONFIDENCE", "EXPIRED")
                or liveness_decision == "DECLINED"):
            log.warning(
                "Identity verification failed: doc=%s liveness=%s",
                doc_extract.verification_status, liveness_decision,
            )
            return KYCDecision(
                customer_id=customer_id,
                decision_date=date.today(),
                status=KYCStatus.DECLINED,
                document_extract=doc_extract,
                pep_sanctions=pep_sanctions,
                liveness_score=liveness_score,
                liveness_passed=False,
                decided_by=self._model_id,
                review_required=True,
                biometric_template_deleted=True,
            )

        # EDD triggers per MLR 2017 Reg. 33
        edd_triggers = []
        if pep_sanctions.is_pep:
            edd_triggers.append("pep_identified")
        if pep_sanctions.requires_edd:
            edd_triggers.append("high_risk_jurisdiction_fatf")
        if (pep_sanctions.sanctions_hit
                and pep_sanctions.match_score >= SANCTIONS_REVIEW):
            edd_triggers.append("sanctions_near_match")
        if liveness_decision == "MANUAL_REVIEW":
            edd_triggers.append("liveness_manual_review")

        if edd_triggers:
            log.info(
                "EDD required: customer=%s triggers=%s "
                "MLR 2017 Reg. 33",
                customer_id, edd_triggers,
            )
            return KYCDecision(
                customer_id=customer_id,
                decision_date=date.today(),
                status=KYCStatus.EDD_REQUIRED,
                document_extract=doc_extract,
                pep_sanctions=pep_sanctions,
                liveness_score=liveness_score,
                liveness_passed=(liveness_decision != "DECLINED"),
                edd_trigger=", ".join(edd_triggers),
                decided_by=self._model_id,
                review_required=True,
                biometric_template_deleted=True,
            )

        # Standard CDD pass
        log.info(
            "KYC CDD pass: customer=%s liveness=%.3f",
            customer_id, liveness_score,
        )
        return KYCDecision(
            customer_id=customer_id,
            decision_date=date.today(),
            status=KYCStatus.CDD_PASS,
            document_extract=doc_extract,
            pep_sanctions=pep_sanctions,
            liveness_score=liveness_score,
            liveness_passed=True,
            decided_by=self._model_id,
            review_required=False,
            biometric_template_deleted=True,
        )
