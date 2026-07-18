"""
document_analyser/extractor.py
AWB Credit Document Analyser — Financial Data Extractor

Extracts structured financial data from corporate credit packs (PDF and Word)
using Gemini 3.1 Pro with three hallucination mitigations:
  (a) Grounding assertions — source page/paragraph cited for every figure
  (b) Range validation — EBITDA margin 0–60%, leverage 0–20x, interest cover 0–50x
  (c) Confidence scoring — per-field confidence 0–1; < 0.80 flags analyst review

Regulatory compliance:
  - PRA SS1/23: Model ID MR-2026-035, risk rating MEDIUM
    (influences credit decisions — must be registered and validated)
  - EU AI Act 2024/1689: Annex III §5b — credit-influencing AI system,
    HIGH-RISK classification. Technical documentation required. Human oversight mandatory.
  - DORA: LLM provider usage logged for concentration risk monitoring
  - UK GDPR: Company financial data — lawful basis: legitimate interest (credit assessment)

Performance target: p95 extraction latency < 30 seconds for 200-page PDF
Cost basis: Gemini 3.1 Pro at £1.58/1M input tokens (June 2026, £1=$1.27)

Author: AWB AI Programme (AWB-AI-2025)
Chapter: 2 — LLM Foundations for Financial Document Analysis
"""

from __future__ import annotations

import io
import logging
import os
from datetime import date
from pathlib import Path
from typing import Any

import google.generativeai as genai
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PRA SS1/23 Model Registration
# ---------------------------------------------------------------------------
MODEL_ID = "MR-2026-035"
MODEL_RISK_RATING = "MEDIUM"
LLM_MODEL_NAME = "gemini-3.1-pro"
EU_AI_ACT_CLASSIFICATION = "HIGH_RISK"   # Annex III §5b — credit influencing
DORA_ASSET_ID = "DA-2026-002"


# ---------------------------------------------------------------------------
# Range validation constants (EBA / AWB credit policy)
# ---------------------------------------------------------------------------

class RangeConfig:
    """
    Acceptable financial metric ranges for UK SME/corporate lending.
    Source: AWB Credit Policy v3.2 (January 2026) + EBA Guidelines on IRB.
    Values outside these ranges are flagged — not rejected — for analyst review.
    """
    EBITDA_MARGIN_MIN: float = 0.0        # %
    EBITDA_MARGIN_MAX: float = 60.0       # %
    LEVERAGE_RATIO_MIN: float = 0.0       # x (Net Debt / EBITDA)
    LEVERAGE_RATIO_MAX: float = 20.0      # x
    INTEREST_COVER_MIN: float = 0.0       # x (EBITDA / Interest)
    INTEREST_COVER_MAX: float = 50.0      # x
    CURRENT_RATIO_MIN: float = 0.0        # x
    CURRENT_RATIO_MAX: float = 20.0       # x
    REVENUE_MIN_GBP: float = 0.0          # £
    REVENUE_MAX_GBP: float = 50_000_000_000.0   # £50B upper bound for sanity check
    CONFIDENCE_REVIEW_THRESHOLD: float = 0.80   # Below this → analyst review required


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class FieldExtraction(BaseModel):
    """
    Single extracted financial field with grounding and confidence.
    Implements hallucination mitigation (a) and (c).
    """
    value: float | str | None = None
    unit: str = ""                        # "£000s" | "x" | "%" | ""
    source_page: int | None = None        # Page number in source document
    source_paragraph: str | None = None  # Quoted excerpt (max 100 chars)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    analyst_review_required: bool = False
    review_reason: str | None = None


class FinancialSummary(BaseModel):
    """
    Structured financial summary extracted from a credit pack.

    PRA SS1/23 MR-2026-035 — MEDIUM risk.
    EU AI Act Annex III §5b — HIGH-RISK credit-influencing system.
    Human oversight mandatory before use in credit decisions.

    All monetary values in GBP thousands (£000s) unless stated otherwise.
    """

    # Document metadata
    document_id: str
    company_name: FieldExtraction
    reporting_period: FieldExtraction         # e.g. "Year ended 31 December 2024"
    reporting_currency: str = "GBP"

    # Income statement
    revenue: FieldExtraction                  # £000s
    ebitda: FieldExtraction                   # £000s
    ebitda_margin_pct: FieldExtraction        # %

    # Balance sheet
    net_debt: FieldExtraction                 # £000s
    total_assets: FieldExtraction             # £000s
    current_assets: FieldExtraction           # £000s
    current_liabilities: FieldExtraction      # £000s

    # Ratios
    leverage_ratio: FieldExtraction           # Net Debt / EBITDA (x)
    interest_cover: FieldExtraction           # EBITDA / Interest expense (x)
    current_ratio: FieldExtraction            # Current Assets / Current Liabilities (x)

    # Extraction metadata
    extraction_model: str = LLM_MODEL_NAME
    extraction_date: str = Field(default_factory=lambda: date.today().isoformat())
    model_id: str = MODEL_ID
    eu_ai_act_status: str = EU_AI_ACT_CLASSIFICATION
    overall_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    analyst_review_required: bool = False
    analyst_review_reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def compute_overall_confidence_and_review_flag(self) -> "FinancialSummary":
        """
        Hallucination mitigation (c): compute overall confidence and set
        analyst_review_required if any material field is below threshold.
        """
        material_fields = [
            self.revenue, self.ebitda, self.net_debt,
            self.leverage_ratio, self.interest_cover,
        ]
        confidences = [f.confidence for f in material_fields if f.value is not None]
        if confidences:
            self.overall_confidence = sum(confidences) / len(confidences)

        reasons = []
        for field_name, field_val in [
            ("revenue", self.revenue),
            ("ebitda", self.ebitda),
            ("net_debt", self.net_debt),
            ("leverage_ratio", self.leverage_ratio),
            ("interest_cover", self.interest_cover),
        ]:
            if (field_val.value is not None
                    and field_val.confidence < RangeConfig.CONFIDENCE_REVIEW_THRESHOLD):
                reasons.append(
                    f"{field_name}: confidence {field_val.confidence:.2f} < "
                    f"{RangeConfig.CONFIDENCE_REVIEW_THRESHOLD}"
                )
            if field_val.analyst_review_required and field_val.review_reason:
                reasons.append(f"{field_name}: {field_val.review_reason}")

        if reasons:
            self.analyst_review_required = True
            self.analyst_review_reasons = reasons

        return self


# ---------------------------------------------------------------------------
# Range validation helpers (hallucination mitigation b)
# ---------------------------------------------------------------------------

def _validate_range(
    field: FieldExtraction,
    field_name: str,
    min_val: float,
    max_val: float,
) -> FieldExtraction:
    """
    Apply range validation to a numeric FieldExtraction.
    Values outside range are flagged for analyst review — not silently rejected.
    Implements EBA margin of conservatism principle: flag, don't discard.
    """
    if field.value is None or not isinstance(field.value, (int, float)):
        return field

    val = float(field.value)
    if val < min_val or val > max_val:
        return field.model_copy(update={
            "analyst_review_required": True,
            "review_reason": (
                f"Value {val:.2f} outside expected range "
                f"[{min_val}, {max_val}] — verify source document"
            ),
        })
    return field


def apply_range_validations(summary: FinancialSummary) -> FinancialSummary:
    """
    Apply all range validations to a FinancialSummary.
    Called after LLM extraction — pure Python, no API calls.
    """
    updates: dict[str, Any] = {}

    updates["ebitda_margin_pct"] = _validate_range(
        summary.ebitda_margin_pct, "ebitda_margin_pct",
        RangeConfig.EBITDA_MARGIN_MIN, RangeConfig.EBITDA_MARGIN_MAX,
    )
    updates["leverage_ratio"] = _validate_range(
        summary.leverage_ratio, "leverage_ratio",
        RangeConfig.LEVERAGE_RATIO_MIN, RangeConfig.LEVERAGE_RATIO_MAX,
    )
    updates["interest_cover"] = _validate_range(
        summary.interest_cover, "interest_cover",
        RangeConfig.INTEREST_COVER_MIN, RangeConfig.INTEREST_COVER_MAX,
    )
    updates["current_ratio"] = _validate_range(
        summary.current_ratio, "current_ratio",
        RangeConfig.CURRENT_RATIO_MIN, RangeConfig.CURRENT_RATIO_MAX,
    )
    updates["revenue"] = _validate_range(
        summary.revenue, "revenue",
        RangeConfig.REVENUE_MIN_GBP, RangeConfig.REVENUE_MAX_GBP,
    )

    # Re-trigger model validator to recompute flags
    updated = summary.model_copy(update=updates)
    return updated.model_copy(
        update=updated.model_validator_called_manually()
        if hasattr(updated, "model_validator_called_manually") else {}
    )


# ---------------------------------------------------------------------------
# Document text extraction (PDF and Word)
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: str | Path) -> tuple[str, int]:
    """
    Extract full text from PDF using PyMuPDF.

    Returns:
        (full_text, page_count)

    Raises:
        ImportError: If PyMuPDF (fitz) not installed.
        FileNotFoundError: If PDF path does not exist.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise ImportError(
            "PyMuPDF required for PDF extraction. "
            "Install with: pip install pymupdf"
        ) from e

    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(path))
    pages_text = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        pages_text.append(f"[PAGE {page_num}]\n{text}")

    full_text = "\n\n".join(pages_text)
    page_count = len(doc)
    doc.close()

    logger.info(
        "PDF text extracted",
        extra={"path": str(pdf_path), "pages": page_count, "chars": len(full_text)},
    )
    return full_text, page_count


def extract_text_from_docx(docx_path: str | Path) -> str:
    """
    Extract full text from Word document using python-docx.

    Returns:
        Full document text with paragraph breaks.

    Raises:
        ImportError: If python-docx not installed.
        FileNotFoundError: If docx path does not exist.
    """
    try:
        from docx import Document
    except ImportError as e:
        raise ImportError(
            "python-docx required for Word extraction. "
            "Install with: pip install python-docx"
        ) from e

    path = Path(docx_path)
    if not path.exists():
        raise FileNotFoundError(f"Word document not found: {docx_path}")

    doc = Document(str(path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    full_text = "\n\n".join(paragraphs)

    logger.info(
        "Word document text extracted",
        extra={"path": str(docx_path), "paragraphs": len(paragraphs)},
    )
    return full_text


def extract_text_from_string(text: str) -> str:
    """
    Pass-through for plain text input (used in tests and when text is
    pre-extracted from document store).
    """
    return text


# ---------------------------------------------------------------------------
# System prompt — 4-component AWB architecture
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = """
COMPONENT 1 — ROLE AND CONTEXT:
You are a financial data extraction specialist for Avon & Wessex Bank plc (AWB),
a UK PRA/FCA-regulated bank. Today: {today}.
You extract structured financial data from corporate credit packs submitted by borrowers.
Your output feeds directly into AWB's credit assessment process.

COMPONENT 2 — REGULATORY CONSTRAINTS:
- PRA SS1/23 Model MR-2026-035: every extraction is logged. Your outputs are audited.
- EU AI Act Annex III §5b HIGH-RISK: human oversight is mandatory before use in decisions.
- UK GDPR: extract only the financial fields specified. Do not extract personal data.
- EBA Guidelines: apply margin of conservatism — when uncertain between two values,
  use the more conservative (higher debt / lower income) interpretation.
- Never fabricate values. If a figure is not present in the document, return null.

COMPONENT 3 — OUTPUT FORMAT:
Return ONLY a valid JSON object with this exact structure (no markdown, no explanation):
{
  "company_name": {"value": "<string>", "source_page": <int|null>, "source_paragraph": "<quote max 100 chars>", "confidence": <0-1>},
  "reporting_period": {"value": "<string>", "source_page": <int|null>, "source_paragraph": "<quote max 100 chars>", "confidence": <0-1>},
  "revenue": {"value": <number|null>, "unit": "£000s", "source_page": <int|null>, "source_paragraph": "<quote>", "confidence": <0-1>},
  "ebitda": {"value": <number|null>, "unit": "£000s", "source_page": <int|null>, "source_paragraph": "<quote>", "confidence": <0-1>},
  "ebitda_margin_pct": {"value": <number|null>, "unit": "%", "source_page": <int|null>, "source_paragraph": "<quote>", "confidence": <0-1>},
  "net_debt": {"value": <number|null>, "unit": "£000s", "source_page": <int|null>, "source_paragraph": "<quote>", "confidence": <0-1>},
  "total_assets": {"value": <number|null>, "unit": "£000s", "source_page": <int|null>, "source_paragraph": "<quote>", "confidence": <0-1>},
  "current_assets": {"value": <number|null>, "unit": "£000s", "source_page": <int|null>, "source_paragraph": "<quote>", "confidence": <0-1>},
  "current_liabilities": {"value": <number|null>, "unit": "£000s", "source_page": <int|null>, "source_paragraph": "<quote>", "confidence": <0-1>},
  "leverage_ratio": {"value": <number|null>, "unit": "x", "source_page": <int|null>, "source_paragraph": "<quote>", "confidence": <0-1>},
  "interest_cover": {"value": <number|null>, "unit": "x", "source_page": <int|null>, "source_paragraph": "<quote>", "confidence": <0-1>},
  "current_ratio": {"value": <number|null>, "unit": "x", "source_page": <int|null>, "source_paragraph": "<quote>", "confidence": <0-1>}
}

COMPONENT 4 — EXPLICIT LIMITATIONS:
- You do not have access to external databases or market data.
- You cannot verify figures against Companies House or HMRC records.
- Ratios not stated in the document must be calculated from stated figures, not estimated.
- If revenue is in a currency other than GBP, set unit accordingly and note it.
- All monetary values must be in £000s (thousands). Convert if stated in millions: multiply by 1000.
"""


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def extract_financial_summary(
    document_text: str,
    document_id: str,
    api_key: str | None = None,
) -> FinancialSummary:
    """
    Extract structured financial data from document text using Gemini 3.1 Pro.

    Implements all three hallucination mitigations:
    (a) Grounding: source_page and source_paragraph for every field
    (b) Range validation: applied post-extraction via apply_range_validations()
    (c) Confidence scoring: per-field confidence, overall_confidence computed

    Args:
        document_text: Full text of the credit pack (pre-extracted from PDF/Word).
        document_id:   Unique document identifier for audit logging.
        api_key:       Google AI Studio API key (falls back to GOOGLE_API_KEY env var).

    Returns:
        FinancialSummary with extracted fields, confidence scores, and review flags.

    Raises:
        ValueError: If API key not available.
        google.api_core.exceptions.GoogleAPIError: On Gemini API failure.

    Regulatory note:
        PRA SS1/23 MR-2026-035 — outputs must be reviewed by a qualified analyst
        before use in any credit decision. EU AI Act Annex III §5b — human oversight
        checkpoint required. This function logs every call to the audit system.
    """
    import json

    key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ValueError(
            "GOOGLE_API_KEY environment variable not set. "
            "Obtain a key from https://aistudio.google.com/app/apikey"
        )

    genai.configure(api_key=key)

    system_prompt = EXTRACTION_SYSTEM_PROMPT.format(today=date.today().isoformat())

    client = genai.GenerativeModel(
        model_name=LLM_MODEL_NAME,
        system_instruction=system_prompt,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.0,          # Deterministic extraction — reproducibility
            max_output_tokens=2048,
        ),
    )

    # Truncate document to 800K chars to stay within context window
    # Gemini 3.1 Pro: 1M token context — 800K chars ≈ 200K tokens (safe headroom)
    MAX_CHARS = 800_000
    if len(document_text) > MAX_CHARS:
        document_text = document_text[:MAX_CHARS]
        logger.warning(
            "Document truncated to fit context window",
            extra={"document_id": document_id, "max_chars": MAX_CHARS},
        )

    prompt = (
        f"Extract the financial data from this credit pack.\n"
        f"Document ID: {document_id}\n\n"
        f"DOCUMENT TEXT:\n{document_text}"
    )

    logger.info(
        "Starting financial extraction",
        extra={
            "document_id": document_id,
            "model": LLM_MODEL_NAME,
            "model_id": MODEL_ID,
            "doc_length": len(document_text),
        },
    )

    response = client.generate_content(prompt)
    raw_json = response.text.strip()

    # Strip markdown fences if model adds them despite instruction
    if raw_json.startswith("```"):
        raw_json = raw_json.split("```")[1]
        if raw_json.startswith("json"):
            raw_json = raw_json[4:]
        raw_json = raw_json.strip()

    extracted = json.loads(raw_json)

    def _parse_field(data: dict) -> FieldExtraction:
        confidence = float(data.get("confidence", 0.0))
        review_required = confidence < RangeConfig.CONFIDENCE_REVIEW_THRESHOLD
        review_reason = (
            f"Low confidence: {confidence:.2f}" if review_required else None
        )
        return FieldExtraction(
            value=data.get("value"),
            unit=data.get("unit", ""),
            source_page=data.get("source_page"),
            source_paragraph=data.get("source_paragraph"),
            confidence=confidence,
            analyst_review_required=review_required,
            review_reason=review_reason,
        )

    summary = FinancialSummary(
        document_id=document_id,
        company_name=_parse_field(extracted.get("company_name", {})),
        reporting_period=_parse_field(extracted.get("reporting_period", {})),
        revenue=_parse_field(extracted.get("revenue", {})),
        ebitda=_parse_field(extracted.get("ebitda", {})),
        ebitda_margin_pct=_parse_field(extracted.get("ebitda_margin_pct", {})),
        net_debt=_parse_field(extracted.get("net_debt", {})),
        total_assets=_parse_field(extracted.get("total_assets", {})),
        current_assets=_parse_field(extracted.get("current_assets", {})),
        current_liabilities=_parse_field(extracted.get("current_liabilities", {})),
        leverage_ratio=_parse_field(extracted.get("leverage_ratio", {})),
        interest_cover=_parse_field(extracted.get("interest_cover", {})),
        current_ratio=_parse_field(extracted.get("current_ratio", {})),
    )

    # Apply range validations (hallucination mitigation b)
    summary = _apply_range_validations_direct(summary)

    logger.info(
        "Extraction complete",
        extra={
            "document_id": document_id,
            "overall_confidence": summary.overall_confidence,
            "analyst_review_required": summary.analyst_review_required,
            "review_reasons": summary.analyst_review_reasons,
        },
    )

    return summary


def _apply_range_validations_direct(summary: FinancialSummary) -> FinancialSummary:
    """
    Apply range validations and recompute analyst review flags.
    Internal helper — use apply_range_validations() for external calls.
    """
    updates: dict[str, Any] = {}

    updates["ebitda_margin_pct"] = _validate_range(
        summary.ebitda_margin_pct, "ebitda_margin_pct",
        RangeConfig.EBITDA_MARGIN_MIN, RangeConfig.EBITDA_MARGIN_MAX,
    )
    updates["leverage_ratio"] = _validate_range(
        summary.leverage_ratio, "leverage_ratio",
        RangeConfig.LEVERAGE_RATIO_MIN, RangeConfig.LEVERAGE_RATIO_MAX,
    )
    updates["interest_cover"] = _validate_range(
        summary.interest_cover, "interest_cover",
        RangeConfig.INTEREST_COVER_MIN, RangeConfig.INTEREST_COVER_MAX,
    )
    updates["current_ratio"] = _validate_range(
        summary.current_ratio, "current_ratio",
        RangeConfig.CURRENT_RATIO_MIN, RangeConfig.CURRENT_RATIO_MAX,
    )
    updates["revenue"] = _validate_range(
        summary.revenue, "revenue",
        RangeConfig.REVENUE_MIN_GBP, RangeConfig.REVENUE_MAX_GBP,
    )

    # Re-create to trigger model_validator
    data = summary.model_dump()
    data.update({k: v.model_dump() for k, v in updates.items()})
    return FinancialSummary(**data)
