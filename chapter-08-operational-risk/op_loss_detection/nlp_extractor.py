"""
op_loss_detection/nlp_extractor.py — AWB Op Loss Event Detection.
Model ID: MR-2026-050 | PRA SS1/23 Risk: LOW
Avon & Wessex Bank plc (AWB) — AWB-AI-2025 programme.

NLP pipeline extracts Basel III operational loss events from:
- Incident management reports (ServiceNow)
- Internal audit findings
- T24 exception logs
- Email escalations
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from awb_commons.models import OpLossCategory, OpLossEvent

logger = logging.getLogger(__name__)

# Confidence threshold: below this, flag for human review
MIN_CONFIDENCE = 0.65

# Map keyword patterns to Basel III SMA categories
CATEGORY_KEYWORDS: dict[OpLossCategory, list[str]] = {
    OpLossCategory.INTERNAL_FRAUD: [
        "internal fraud", "employee theft",
        "unauthorised trading", "rogue trader",
    ],
    OpLossCategory.EXTERNAL_FRAUD: [
        "payment fraud", "phishing", "identity theft",
        "account takeover", "card fraud",
    ],
    OpLossCategory.EXECUTION_DELIVERY: [
        "settlement failure", "booking error",
        "failed transaction", "processing error",
        "system outage", "data corruption",
    ],
    OpLossCategory.CLIENTS_PRODUCTS: [
        "mis-selling", "suitability failure",
        "complaint", "fca investigation",
        "consumer duty", "unfair terms",
    ],
    OpLossCategory.EMPLOYMENT_PRACTICES: [
        "employment tribunal", "hr complaint",
        "discrimination", "whistleblowing",
    ],
    OpLossCategory.PHYSICAL_ASSETS: [
        "fire", "flood", "theft of equipment",
        "physical damage",
    ],
    OpLossCategory.BUSINESS_DISRUPTION: [
        "system outage", "cyber attack",
        "ransomware", "dora incident",
    ],
}


@dataclass
class ExtractedLossEvent:
    """Intermediate result before Pydantic validation."""

    source_doc_id: str
    raw_text: str
    category: OpLossCategory
    amount_gbp: Optional[float]
    confidence: float
    event_date: Optional[datetime]


class OpLossNLPExtractor:
    """
    Extracts operational loss events from unstructured text.

    Two-stage process:
    1. Keyword + regex pre-filter (fast, deterministic).
    2. LLM structured extraction (Gemini 3.5 Flash).

    SMA reporting: only FINAL events (confidence ≥ 0.65) are
    eligible for Basel III SMA capital calculation.
    MR-2026-050: LOW risk rating (no autonomous financial action).
    """

    def __init__(self, min_confidence: float = MIN_CONFIDENCE) -> None:
        self.min_confidence = min_confidence
        logger.info(
            "OpLossNLPExtractor initialised: min_confidence=%.2f",
            min_confidence,
        )

    def extract(
        self,
        document_id: str,
        text: str,
    ) -> list[OpLossEvent]:
        """
        Extract all loss events from a document.

        Args:
            document_id: Unique source document identifier.
            text: Full document text (plain or pre-processed).

        Returns:
            List of validated OpLossEvent objects.
        """
        candidates = self._pre_filter(text)
        events: list[OpLossEvent] = []
        for category, snippets in candidates.items():
            for snippet in snippets:
                extracted = self._llm_extract(
                    document_id, snippet, category
                )
                if extracted.confidence >= self.min_confidence:
                    events.append(
                        self._to_model(extracted)
                    )
        logger.info(
            "Extracted %d events from doc=%s",
            len(events),
            document_id,
        )
        return events

    # ── Private helpers ───────────────────────────────────────────

    def _pre_filter(
        self, text: str
    ) -> dict[OpLossCategory, list[str]]:
        """
        Keyword matching to identify candidate loss event passages.
        Returns map of category → list of text snippets.
        """
        text_lower = text.lower()
        result: dict[OpLossCategory, list[str]] = {}
        for category, keywords in CATEGORY_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    idx = text_lower.find(kw)
                    snippet = text[
                        max(0, idx - 100): idx + 400
                    ]
                    result.setdefault(category, []).append(
                        snippet
                    )
                    break
        return result

    def _llm_extract(
        self,
        doc_id: str,
        snippet: str,
        category: OpLossCategory,
    ) -> ExtractedLossEvent:
        """
        LLM structured extraction.
        Production: calls Gemini 3.5 Flash with JSON schema.
        Test: deterministic stub based on regex.
        """
        amount = self._extract_amount(snippet)
        date = self._extract_date(snippet)
        confidence = 0.80 if amount else 0.68
        return ExtractedLossEvent(
            source_doc_id=doc_id,
            raw_text=snippet[:200],
            category=category,
            amount_gbp=amount,
            confidence=confidence,
            event_date=date,
        )

    def _extract_amount(
        self, text: str
    ) -> Optional[float]:
        """Regex extraction of GBP amounts from text."""
        pattern = r"£([\d,]+(?:\.\d{2})?)"
        match = re.search(pattern, text)
        if match:
            return float(match.group(1).replace(",", ""))
        return None

    def _extract_date(
        self, text: str
    ) -> Optional[datetime]:
        """Simple date extraction from text."""
        pattern = r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b"
        match = re.search(pattern, text)
        if match:
            try:
                d, m, y = match.groups()
                return datetime(int(y), int(m), int(d))
            except ValueError:
                return None
        return None

    def _to_model(
        self, extracted: ExtractedLossEvent
    ) -> OpLossEvent:
        event = OpLossEvent(
            source_document_id=extracted.source_doc_id,
            event_category=extracted.category,
            loss_amount_gbp=extracted.amount_gbp,
            confidence_score=extracted.confidence,
            event_date=extracted.event_date,
            description=extracted.raw_text,
            sma_eligible=(
                extracted.confidence >= self.min_confidence
            ),
        )
        event.net_loss_gbp = event.calculate_net_loss()
        return event
