"""
chatbot/pipeline.py
AWB AI Customer Service Platform — Main Orchestration Pipeline

Orchestrates the full customer interaction pipeline:
  1. Session context retrieval (Redis / in-memory fallback)
  2. Intent classification (Gemini 3.5 Flash — classifier.py)
  3. Grounding data fetch (T24 mock / product catalogue)
  4. Response generation (Gemini 3.5 Flash — response_generator.py)
  5. FCA compliance filter (compliance_filter.py)
  6. Session update
  7. Audit log write (PostgreSQL/SQLite — audit_log.py)

Architecture reference: Figure 1.4a — AI Customer Service Sequence Diagram
Regulatory: FCA Consumer Duty PS22/9, DORA ICT Asset CS-2026-001
Author: AWB AI Programme (AWB-AI-2025)
Chapter: 1 — The AI Transformation of Risk and Compliance in Banking
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from chatbot.audit_log import build_interaction_log, log_interaction
from chatbot.classifier import (
    CustomerIntent,
    IntentResult,
    classify_intent,
)
from chatbot.compliance_filter import ComplianceResult, compliance_check
from chatbot.response_generator import (
    AccountSummary,
    DraftResponse,
    PRODUCT_CATALOGUE,
    generate_response,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session management (in-memory for local dev; Redis in production)
# ---------------------------------------------------------------------------

_session_store: dict[str, list[dict]] = {}   # session_id -> list of turns


def _get_session_context(session_id: str, max_turns: int = 10) -> str:
    """Retrieve last N turns as formatted string. Production: Redis GET."""
    turns = _session_store.get(session_id, [])[-max_turns:]
    if not turns:
        return ""
    lines = []
    for turn in turns:
        lines.append(f"Customer: {turn['message']}")
        if turn.get("response"):
            lines.append(f"Assistant: {turn['response']}")
    return "\n".join(lines)


def _update_session(session_id: str, message: str, response: str) -> None:
    """Append turn to session history. Production: Redis RPUSH with TTL 30min."""
    if session_id not in _session_store:
        _session_store[session_id] = []
    _session_store[session_id].append({
        "message": message,
        "response": response,
    })


# ---------------------------------------------------------------------------
# Mock T24 data fetch (production: Temenos T24 REST API)
# ---------------------------------------------------------------------------

def _fetch_account_data(customer_id: str, intent: CustomerIntent) -> AccountSummary | None:
    """
    Fetch account data from T24 core banking system.
    Production: GET /accounts/{customer_id}/balance via T24 REST API.
    Local development: returns mock data.
    """
    if intent != CustomerIntent.BALANCE_ENQUIRY:
        return None

    # Mock T24 response — production uses real T24 REST API
    return AccountSummary(
        customer_id=customer_id,
        account_number_masked="****4321",
        product_type="Current Account",
        available_balance_gbp=2_847.65,
    )


# ---------------------------------------------------------------------------
# Pipeline response dataclass
# ---------------------------------------------------------------------------

@dataclass
class PipelineResponse:
    """Complete response from the chatbot pipeline."""
    session_id: str
    interaction_id: str
    response_text: str
    requires_escalation: bool
    escalation_reason: str | None
    intent: str
    confidence: float
    compliance_approved: bool
    compliance_flags: list[str] = field(default_factory=list)
    latency_ms: int = 0


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def process_customer_message(
    session_id: str,
    message: str,
    customer_id: str | None = None,
    customer_segment: str = "retail",
    channel: str = "web",
    api_key: str | None = None,
) -> PipelineResponse:
    """
    Process one customer message through the full AWB chatbot pipeline.

    Args:
        session_id:        Unique session identifier (UUID).
        message:           Raw customer message text.
        customer_id:       Authenticated customer ID from API Gateway (or None).
        customer_segment:  "retail" | "sme" | "private".
        channel:           "web" | "app" | "ivr".
        api_key:           Google AI Studio API key (falls back to env var).

    Returns:
        PipelineResponse with response text, escalation status, and audit ID.

    Pipeline steps follow Figure 1.4a sequence diagram exactly.
    """
    start_time = time.monotonic()

    # ----------------------------------------------------------------
    # Step 1: Session context retrieval
    # ----------------------------------------------------------------
    session_context = _get_session_context(session_id)
    logger.debug("Session context retrieved", extra={"session_id": session_id})

    # ----------------------------------------------------------------
    # Step 2: Intent classification
    # ----------------------------------------------------------------
    intent_result: IntentResult = classify_intent(
        message=message,
        session_context=session_context,
        api_key=api_key,
    )
    logger.info(
        "Intent classified",
        extra={"intent": intent_result.intent.value,
               "confidence": intent_result.confidence},
    )

    # ----------------------------------------------------------------
    # Step 3: Complaint / account_change hard escalation
    # (FCA Consumer Duty — human agent always available)
    # ----------------------------------------------------------------
    if intent_result.requires_escalation:
        escalation_response = (
            "I can see you need some assistance that's best handled by one of our "
            "advisers. I've flagged your query and a member of our team will contact "
            "you shortly. Alternatively, you can call us directly on 0800 123 4567, "
            "or visit any AWB branch."
        )
        # Log and return immediately for escalated intents
        latency_ms = int((time.monotonic() - start_time) * 1000)
        _update_session(session_id, message, escalation_response)

        # Minimal compliance result for escalation path
        from chatbot.compliance_filter import ComplianceResult
        escalation_compliance = ComplianceResult(
            approved=True,
            modified_text=escalation_response,
            escalation_required=True,
            audit_notes="ESCALATION_PATH: compliance filter bypassed",
        )

        log_record = build_interaction_log(
            session_id=session_id,
            customer_id=customer_id,
            customer_segment=customer_segment,
            channel=channel,
            message_text=message,
            intent_result=intent_result,
            response_text=escalation_response,
            compliance_result=escalation_compliance,
            escalated_to_agent=True,
            latency_ms=latency_ms,
        )
        interaction_id = log_interaction(log_record)

        return PipelineResponse(
            session_id=session_id,
            interaction_id=interaction_id,
            response_text=escalation_response,
            requires_escalation=True,
            escalation_reason=intent_result.escalation_reason,
            intent=intent_result.intent.value,
            confidence=intent_result.confidence,
            compliance_approved=True,
            latency_ms=latency_ms,
        )

    # ----------------------------------------------------------------
    # Step 4: Grounding data fetch
    # ----------------------------------------------------------------
    account_data = _fetch_account_data(
        customer_id=customer_id or "UNAUTHENTICATED",
        intent=intent_result.intent,
    )

    # Match product info from catalogue based on extracted entities
    product_info = None
    product_key = intent_result.entities.get("product", "").lower()
    for key, info in PRODUCT_CATALOGUE.items():
        if key in product_key or product_key in info.product_name.lower():
            product_info = info
            break

    # ----------------------------------------------------------------
    # Step 5: Response generation
    # ----------------------------------------------------------------
    draft: DraftResponse = generate_response(
        intent_result=intent_result,
        account_data=account_data,
        product_info=product_info,
        session_context=session_context,
        api_key=api_key,
    )

    # ----------------------------------------------------------------
    # Step 6: FCA compliance filter
    # ----------------------------------------------------------------
    compliance_result: ComplianceResult = compliance_check(
        draft=draft,
        customer_segment=customer_segment,
    )

    final_response_text = compliance_result.modified_text
    escalated = compliance_result.escalation_required

    # ----------------------------------------------------------------
    # Step 7: Session update
    # ----------------------------------------------------------------
    _update_session(session_id, message, final_response_text)

    # ----------------------------------------------------------------
    # Step 8: Audit log write (FCA Consumer Duty — every interaction logged)
    # ----------------------------------------------------------------
    latency_ms = int((time.monotonic() - start_time) * 1000)

    log_record = build_interaction_log(
        session_id=session_id,
        customer_id=customer_id,
        customer_segment=customer_segment,
        channel=channel,
        message_text=message,
        intent_result=intent_result,
        response_text=final_response_text,
        compliance_result=compliance_result,
        escalated_to_agent=escalated,
        latency_ms=latency_ms,
    )
    interaction_id = log_interaction(log_record)

    logger.info(
        "Pipeline complete",
        extra={
            "interaction_id": interaction_id,
            "latency_ms": latency_ms,
            "intent": intent_result.intent.value,
            "escalated": escalated,
            "compliance_approved": compliance_result.approved,
        },
    )

    return PipelineResponse(
        session_id=session_id,
        interaction_id=interaction_id,
        response_text=final_response_text,
        requires_escalation=escalated,
        escalation_reason=intent_result.escalation_reason if escalated else None,
        intent=intent_result.intent.value,
        confidence=intent_result.confidence,
        compliance_approved=compliance_result.approved,
        compliance_flags=[f.value for f in compliance_result.flags],
        latency_ms=latency_ms,
    )
