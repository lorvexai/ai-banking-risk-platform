"""
chatbot/classifier.py
AWB AI Customer Service Platform — Intent Classifier

Classifies incoming customer messages into structured intent categories
using Gemini 3.5 Flash with Pydantic structured output.

Regulatory compliance:
  - FCA Consumer Duty PS22/9: every classification logged with full audit trail
  - PRA SS1/23: NOT a registered model (no credit/risk decision influence)
  - DORA ICT Asset: CS-2026-001

Performance target: p95 latency < 800ms
Author: AWB AI Programme (AWB-AI-2025)
Chapter: 1 — The AI Transformation of Risk and Compliance in Banking
"""

import logging
import os
from datetime import date
from enum import Enum

import google.generativeai as genai
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Intent taxonomy
# ---------------------------------------------------------------------------

class CustomerIntent(str, Enum):
    BALANCE_ENQUIRY   = "balance_enquiry"
    PRODUCT_ENQUIRY   = "product_enquiry"
    RATE_ENQUIRY      = "rate_enquiry"
    PAYMENT_SUPPORT   = "payment_support"
    COMPLAINT         = "complaint"         # Always escalate — FCA Consumer Duty
    ACCOUNT_CHANGE    = "account_change"    # Always escalate — regulated activity
    OUT_OF_SCOPE      = "out_of_scope"


# Hard-coded escalation intents — safety rule, NOT a model decision
# Implements FCA Consumer Duty PS22/9: customers always have access to human agent
ALWAYS_ESCALATE: frozenset[CustomerIntent] = frozenset({
    CustomerIntent.COMPLAINT,
    CustomerIntent.ACCOUNT_CHANGE,
})

# Confidence threshold below which escalation is triggered
ESCALATION_CONFIDENCE_THRESHOLD: float = 0.75


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------

class IntentResult(BaseModel):
    """
    Structured intent classification output.

    Regulatory: FCA Consumer Duty PS22/9 — every classification logged.
    Performance: p95 latency < 800ms (Gemini 3.5 Flash API call).
    Audit: intent + confidence + escalation_reason stored in PostgreSQL 7yr.
    """

    intent: CustomerIntent
    confidence: float = Field(ge=0.0, le=1.0, description="Model confidence 0-1")
    entities: dict[str, str] = Field(
        default_factory=dict,
        description="Extracted entities e.g. {'product': 'ISA', 'amount': '5000'}"
    )
    requires_escalation: bool = Field(
        description="True for complaint, account_change, low confidence, or distress signals"
    )
    escalation_reason: str | None = Field(
        default=None,
        description="Reason code: 'always_escalate_intent' | 'low_confidence' | 'vulnerable_customer' | None"
    )


# ---------------------------------------------------------------------------
# System prompt — 4-component AWB architecture
# Component 1: Role + Context
# Component 2: Regulatory constraints (FCA Consumer Duty)
# Component 3: Output format requirements
# Component 4: Explicit limitations
# ---------------------------------------------------------------------------

INTENT_SYSTEM_PROMPT = """
You are a customer service classifier for Avon & Wessex Bank plc (AWB),
a UK PRA/FCA-regulated bank. Today: {today}.

AWB Products: current accounts, savings accounts, ISAs, SME business loans,
residential mortgages, overdrafts, foreign currency accounts.

REGULATORY CONSTRAINTS (FCA Consumer Duty PS22/9):
- Always treat customers fairly.
- If a customer appears distressed, upset, or the query involves a complaint,
  set requires_escalation=true and escalation_reason='vulnerable_customer'.
- If the query involves account changes, set requires_escalation=true
  and escalation_reason='always_escalate_intent'.
- NEVER provide financial advice — classify only and let regulated advisers advise.
- NEVER recommend specific products — describe factually if asked.

OUTPUT FORMAT:
- Return a single JSON object matching the IntentResult schema.
- confidence: your certainty that you have correctly identified the primary intent.
- entities: extract relevant named entities (product names, amounts, dates, account types).
- If confidence < 0.75, set requires_escalation=true and escalation_reason='low_confidence'.

LIMITATIONS:
- You do not have access to real-time account data — do not claim to.
- You cannot execute transactions — classify and route only.
- If the message contains content unrelated to AWB banking services,
  use intent=out_of_scope.
"""


# ---------------------------------------------------------------------------
# Classifier function
# ---------------------------------------------------------------------------

def classify_intent(
    message: str,
    session_context: str = "",
    api_key: str | None = None,
) -> IntentResult:
    """
    Classify a customer message into a structured IntentResult.

    Args:
        message:         Raw customer message text.
        session_context: Serialised conversation history (last 10 turns).
        api_key:         Google AI Studio API key. Falls back to
                         GOOGLE_API_KEY env var.

    Returns:
        IntentResult with intent, confidence, entities, and escalation flags.

    Raises:
        ValueError: If the API key is not available.
        google.api_core.exceptions.GoogleAPIError: On Gemini API failure.
    """
    key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ValueError(
            "GOOGLE_API_KEY environment variable not set. "
            "Obtain a key from https://aistudio.google.com/app/apikey"
        )

    genai.configure(api_key=key)

    system_prompt = INTENT_SYSTEM_PROMPT.format(today=date.today().isoformat())

    client = genai.GenerativeModel(
        model_name="gemini-3.5-flash",
        system_instruction=system_prompt,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=IntentResult,
            temperature=0.05,   # Near-deterministic for classification
            max_output_tokens=256,
        ),
    )

    prompt = (
        f"Session context:\n{session_context}\n\n"
        f"Customer message:\n{message}"
        if session_context
        else f"Customer message:\n{message}"
    )

    response = client.generate_content(prompt)
    result = IntentResult.model_validate_json(response.text)

    # Apply hard-coded FCA Consumer Duty escalation rules
    # These override model decisions — safety rules are never model decisions
    if result.intent in ALWAYS_ESCALATE and not result.requires_escalation:
        result = result.model_copy(update={
            "requires_escalation": True,
            "escalation_reason": "always_escalate_intent",
        })

    logger.info(
        "Intent classified",
        extra={
            "intent": result.intent.value,
            "confidence": result.confidence,
            "requires_escalation": result.requires_escalation,
            "escalation_reason": result.escalation_reason,
            "entities": result.entities,
        },
    )

    return result
