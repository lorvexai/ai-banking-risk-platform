"""
chatbot/response_generator.py
AWB AI Customer Service Platform — Response Generator

Generates customer-facing responses using Gemini 3.5 Flash, grounded
in product catalogue data or account summaries. All responses pass
through the FCA compliance filter before delivery.

Regulatory compliance:
  - FCA Consumer Duty PS22/9: responses must be fair, clear, not misleading
  - UK GDPR: no personal data echoed back beyond necessary minimum
  - FCA SYSC: regulated activity boundary enforced (no financial advice)

Author: AWB AI Programme (AWB-AI-2025)
Chapter: 1 — The AI Transformation of Risk and Compliance in Banking
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import date

import google.generativeai as genai
from pydantic import BaseModel

from chatbot.classifier import CustomerIntent, IntentResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models for grounding context
# ---------------------------------------------------------------------------

@dataclass
class AccountSummary:
    """Minimal account data from Temenos T24. Never cache — always fresh."""
    customer_id: str
    account_number_masked: str   # e.g. "****4321" — last 4 digits only
    product_type: str            # "Current Account" | "Business Loan" etc.
    available_balance_gbp: float
    currency: str = "GBP"


@dataclass
class ProductInfo:
    """Static product catalogue entry. Refreshed from AWB CMS daily."""
    product_name: str
    product_type: str
    key_features: list[str] = field(default_factory=list)
    eligibility_criteria: list[str] = field(default_factory=list)
    current_rate_pct: float | None = None          # AER/APR where applicable
    rate_description: str | None = None             # e.g. "2.50% AER variable"
    apply_url: str = "https://awb.co.uk/apply"


class DraftResponse(BaseModel):
    """Structured response before compliance filter."""
    text: str
    citations: list[str] = []          # Source references (product catalogue, T24)
    confidence: float = 1.0
    contains_rate_information: bool = False
    contains_account_data: bool = False


# ---------------------------------------------------------------------------
# Sample product catalogue (production: loaded from AWB CMS / Redis)
# ---------------------------------------------------------------------------

PRODUCT_CATALOGUE: dict[str, ProductInfo] = {
    "isa": ProductInfo(
        product_name="AWB Cash ISA",
        product_type="savings",
        key_features=[
            "Tax-free interest up to £20,000 annual ISA allowance",
            "Instant access — no notice period",
            "FSCS protected up to £85,000",
        ],
        eligibility_criteria=[
            "UK resident aged 18 or over",
            "Valid National Insurance number required",
        ],
        current_rate_pct=4.25,
        rate_description="4.25% AER variable",
        apply_url="https://awb.co.uk/apply/isa",
    ),
    "sme_loan": ProductInfo(
        product_name="AWB SME Business Loan",
        product_type="lending",
        key_features=[
            "Loans from £25,000 to £5,000,000",
            "Fixed and variable rate options",
            "Up to 7-year term",
            "Security may be required",
        ],
        eligibility_criteria=[
            "UK-registered business",
            "Minimum 2 years trading history",
            "Turnover review required for loans over £500,000",
        ],
        rate_description="Rates from 6.5% APR (subject to credit assessment)",
        apply_url="https://awb.co.uk/apply/business-loan",
    ),
    "mortgage": ProductInfo(
        product_name="AWB Residential Mortgage",
        product_type="lending",
        key_features=[
            "Fixed rate 2-year, 5-year and 10-year products",
            "Up to 95% LTV",
            "Overpayment facility (up to 10% per year penalty-free)",
        ],
        eligibility_criteria=[
            "UK resident",
            "Minimum income £25,000 (single applicant)",
            "Subject to full credit assessment and property valuation",
        ],
        rate_description="Fixed rates from 4.79% (5-year)",
        apply_url="https://awb.co.uk/apply/mortgage",
    ),
}

# ---------------------------------------------------------------------------
# Response generation system prompt
# ---------------------------------------------------------------------------

RESPONSE_SYSTEM_PROMPT = """
You are a helpful customer service assistant for Avon & Wessex Bank plc (AWB),
a UK PRA/FCA-regulated bank. Today: {today}.

TONE: Professional, warm, concise. Use plain English. Avoid jargon.
Maximum response length: 3 sentences for routine enquiries.

FCA CONSUMER DUTY RULES (non-negotiable):
1. Be fair, clear, and not misleading in every response.
2. Never provide financial advice — describe products factually, never recommend.
3. Always make it clear that a human adviser is available if needed.
4. Never state rates as definitive without noting they are variable/subject to change.
5. If a customer asks which product is "best for them" — direct them to an adviser.

UK GDPR: Only reference the account information provided in the context.
Do not speculate about account details not explicitly provided.

Format: plain text, no markdown, no bullet points (channel may not render them).
"""


# ---------------------------------------------------------------------------
# Generator function
# ---------------------------------------------------------------------------

def generate_response(
    intent_result: IntentResult,
    account_data: AccountSummary | None = None,
    product_info: ProductInfo | None = None,
    session_context: str = "",
    api_key: str | None = None,
) -> DraftResponse:
    """
    Generate a customer-facing response grounded in account or product data.

    Args:
        intent_result:   Classified intent from classifier.py.
        account_data:    T24 account summary (for balance/account enquiries).
        product_info:    Product catalogue entry (for product/rate enquiries).
        session_context: Last 10 turns of conversation history.
        api_key:         Google AI Studio API key.

    Returns:
        DraftResponse — text plus metadata for compliance filter.
    """
    key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")

    genai.configure(api_key=key)

    # Build grounding context
    grounding_parts: list[str] = []
    citations: list[str] = []
    contains_rate = False
    contains_account = False

    if account_data:
        grounding_parts.append(
            f"Account data (from T24 core banking):\n"
            f"  Account: {account_data.account_number_masked}\n"
            f"  Product: {account_data.product_type}\n"
            f"  Available balance: £{account_data.available_balance_gbp:,.2f}\n"
        )
        citations.append("AWB T24 core banking (real-time)")
        contains_account = True

    if product_info:
        features_text = "\n".join(f"  - {f}" for f in product_info.key_features)
        eligibility_text = "\n".join(f"  - {e}" for e in product_info.eligibility_criteria)
        grounding_parts.append(
            f"Product information (from AWB product catalogue):\n"
            f"  Product: {product_info.product_name}\n"
            f"  Key features:\n{features_text}\n"
            f"  Eligibility:\n{eligibility_text}\n"
            + (f"  Rate: {product_info.rate_description}\n" if product_info.rate_description else "")
            + f"  Apply: {product_info.apply_url}\n"
        )
        citations.append(f"AWB Product Catalogue — {product_info.product_name}")
        if product_info.rate_description:
            contains_rate = True

    grounding_context = "\n\n".join(grounding_parts) if grounding_parts else "No account or product data available."

    system_prompt = RESPONSE_SYSTEM_PROMPT.format(today=date.today().isoformat())

    client = genai.GenerativeModel(
        model_name="gemini-3.5-flash",
        system_instruction=system_prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.3,         # Slightly creative for natural language
            max_output_tokens=512,
        ),
    )

    prompt = (
        f"Customer intent: {intent_result.intent.value}\n"
        f"Entities extracted: {intent_result.entities}\n\n"
        f"Context data:\n{grounding_context}\n\n"
        f"Session history:\n{session_context}\n\n"
        f"Generate a helpful, FCA-compliant response for this customer."
    )

    response = client.generate_content(prompt)

    draft = DraftResponse(
        text=response.text.strip(),
        citations=citations,
        confidence=intent_result.confidence,
        contains_rate_information=contains_rate,
        contains_account_data=contains_account,
    )

    logger.info(
        "Response generated",
        extra={
            "intent": intent_result.intent.value,
            "contains_rate": contains_rate,
            "contains_account_data": contains_account,
            "response_length": len(draft.text),
        },
    )

    return draft
