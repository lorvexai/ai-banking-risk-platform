"""
chatbot/compliance_filter.py
AWB AI Customer Service Platform — FCA Consumer Duty Compliance Filter

Screens all AI-generated responses before delivery to customers.
Implements FCA Consumer Duty PS22/9 obligations:
  - Fair, clear, and not misleading
  - No financial advice (regulated activity boundary)
  - Vulnerable customer identification
  - Rate information must include variability disclaimer

This filter is a hard-coded rules engine, NOT an AI model.
Rules must not be overridden by model confidence or user context.

Regulatory reference:
  FCA Consumer Duty — Policy Statement PS22/9, effective 31 July 2023
  COBS 4.2: Fair, clear and not misleading communications
  FCA PRIN 12: The Consumer Principle

Author: AWB AI Programme (AWB-AI-2025)
Chapter: 1 — The AI Transformation of Risk and Compliance in Banking
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum

from chatbot.response_generator import DraftResponse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Compliance result schema
# ---------------------------------------------------------------------------

class ComplianceFlag(str, Enum):
    FINANCIAL_ADVICE_RISK    = "financial_advice_risk"
    MISLEADING_RATE          = "misleading_rate"
    MISSING_RATE_DISCLAIMER  = "missing_rate_disclaimer"
    DEFINITIVE_RECOMMENDATION = "definitive_recommendation"
    PROHIBITED_LANGUAGE       = "prohibited_language"


@dataclass
class ComplianceResult:
    """
    Result of compliance screening.

    approved:      False means the response must not be delivered as-is.
    flags:         Issues identified (informational even when approved).
    modified_text: Cleaned text (if filter modified content); else original text.
    escalation_required: Override escalation even if classifier did not flag.
    """
    approved: bool
    flags: list[ComplianceFlag] = field(default_factory=list)
    modified_text: str = ""
    escalation_required: bool = False
    audit_notes: str = ""


# ---------------------------------------------------------------------------
# Prohibited patterns
# Patterns that represent regulated activity or misleading language
# ---------------------------------------------------------------------------

# Phrases that constitute financial advice (COBS 9A / FSMA s.19)
FINANCIAL_ADVICE_PATTERNS: list[re.Pattern] = [
    re.compile(r"\byou should (?:invest|buy|choose|take out|apply for)\b", re.I),
    re.compile(r"\bI (?:recommend|suggest|advise)\b", re.I),
    re.compile(r"\bbest (?:product|account|rate|option) for you\b", re.I),
    re.compile(r"\bperfect for your (?:needs|situation|circumstances)\b", re.I),
    re.compile(r"\bmy advice (?:would be|is)\b", re.I),
]

# Phrases that assert rate certainty without variability notice
MISLEADING_RATE_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bguaranteed (?:rate|return|interest)\b", re.I),
    re.compile(r"\bfixed forever\b", re.I),
    re.compile(r"\bwill always (?:pay|earn|receive)\b", re.I),
]

# Rate disclaimer required whenever rate information is included
RATE_DISCLAIMER = (
    " Rates are variable and subject to change. "
    "Current rates are correct as of today's date."
)

# Human agent availability reminder (FCA Consumer Duty — always available)
HUMAN_AGENT_REMINDER = (
    " If you'd like to speak with one of our advisers, "
    "please call 0800 123 4567 or visit any AWB branch."
)


# ---------------------------------------------------------------------------
# Compliance filter function
# ---------------------------------------------------------------------------

def compliance_check(
    draft: DraftResponse,
    customer_segment: str = "retail",
    is_vulnerable_flag: bool = False,
) -> ComplianceResult:
    """
    Screen a draft response for FCA Consumer Duty compliance.

    Rules applied (in order — ALL rules run regardless of earlier findings):

    1. Financial advice detection — block and escalate
    2. Misleading rate language — block and escalate
    3. Rate disclaimer — append if missing and response contains rate data
    4. Definitive recommendation language — soften
    5. Human agent reminder — append to all approved responses

    Args:
        draft:              DraftResponse from response_generator.py.
        customer_segment:   "retail" | "sme" | "private" — affects tone rules.
        is_vulnerable_flag: True if session context flagged vulnerable customer.

    Returns:
        ComplianceResult — approved/rejected with flags and modified text.
    """
    text = draft.text
    flags: list[ComplianceFlag] = []
    escalation_required = False
    audit_notes_parts: list[str] = []

    # ------------------------------------------------------------------
    # Rule 1: Financial advice detection
    # FCA COBS 9A — providing personal recommendations is a regulated activity
    # ------------------------------------------------------------------
    for pattern in FINANCIAL_ADVICE_PATTERNS:
        if pattern.search(text):
            flags.append(ComplianceFlag.FINANCIAL_ADVICE_RISK)
            escalation_required = True
            audit_notes_parts.append(
                f"FINANCIAL_ADVICE_RISK: pattern '{pattern.pattern}' matched"
            )
            logger.warning(
                "Financial advice language detected — escalation required",
                extra={"pattern": pattern.pattern, "text_excerpt": text[:100]},
            )
            break   # One flag sufficient — stop scanning advice patterns

    # ------------------------------------------------------------------
    # Rule 2: Misleading rate language
    # FCA PRIN 7 — fair and not misleading
    # ------------------------------------------------------------------
    for pattern in MISLEADING_RATE_PATTERNS:
        if pattern.search(text):
            flags.append(ComplianceFlag.MISLEADING_RATE)
            escalation_required = True
            audit_notes_parts.append(
                f"MISLEADING_RATE: pattern '{pattern.pattern}' matched"
            )
            logger.warning(
                "Misleading rate language detected",
                extra={"pattern": pattern.pattern},
            )
            break

    # ------------------------------------------------------------------
    # Rule 3: Rate disclaimer
    # If response contains rate information, disclaimer is mandatory
    # ------------------------------------------------------------------
    rate_disclaimer_keywords = re.compile(r"\b(?:%|AER|APR|interest rate|per cent)\b", re.I)
    if draft.contains_rate_information or rate_disclaimer_keywords.search(text):
        if RATE_DISCLAIMER.strip().lower() not in text.lower():
            flags.append(ComplianceFlag.MISSING_RATE_DISCLAIMER)
            text = text.rstrip() + RATE_DISCLAIMER
            audit_notes_parts.append("MISSING_RATE_DISCLAIMER: disclaimer appended")
            logger.info("Rate disclaimer appended to response")

    # ------------------------------------------------------------------
    # Rule 4: Definitive recommendation softening
    # Products must be described factually, not recommended
    # ------------------------------------------------------------------
    recommendation_patterns = [
        (re.compile(r"\bthis is (?:the )?(?:best|right|ideal) (?:account|product|option)\b", re.I),
         "this may be worth considering"),
        (re.compile(r"\byou'll love\b", re.I), "you might find"),
    ]
    for pattern, replacement in recommendation_patterns:
        if pattern.search(text):
            flags.append(ComplianceFlag.DEFINITIVE_RECOMMENDATION)
            text = pattern.sub(replacement, text)
            audit_notes_parts.append(
                f"DEFINITIVE_RECOMMENDATION: softened pattern '{pattern.pattern}'"
            )

    # ------------------------------------------------------------------
    # Rule 5: Human agent reminder
    # FCA Consumer Duty — human support always available
    # Append to all non-escalated responses
    # ------------------------------------------------------------------
    if not escalation_required:
        if HUMAN_AGENT_REMINDER.strip().lower() not in text.lower():
            text = text.rstrip() + HUMAN_AGENT_REMINDER

    # ------------------------------------------------------------------
    # Vulnerable customer handling
    # If session flagged vulnerable customer, add signposting
    # ------------------------------------------------------------------
    if is_vulnerable_flag and not escalation_required:
        escalation_required = True
        audit_notes_parts.append("VULNERABLE_CUSTOMER: escalation override applied")
        logger.info("Vulnerable customer flag — escalating regardless of intent")

    # ------------------------------------------------------------------
    # Determine approval
    # Block responses with financial advice or misleading rates
    # ------------------------------------------------------------------
    blocking_flags = {
        ComplianceFlag.FINANCIAL_ADVICE_RISK,
        ComplianceFlag.MISLEADING_RATE,
    }
    approved = not any(f in blocking_flags for f in flags)

    # If not approved, replace with safe fallback
    if not approved:
        text = (
            "I'm sorry, I'm not able to help with that specific query through our chat service. "
            "One of our advisers will be happy to assist you. "
            "Please call 0800 123 4567 or visit any AWB branch."
        )
        logger.warning(
            "Response blocked by compliance filter — fallback delivered",
            extra={"flags": [f.value for f in flags]},
        )

    result = ComplianceResult(
        approved=approved,
        flags=flags,
        modified_text=text,
        escalation_required=escalation_required,
        audit_notes="; ".join(audit_notes_parts) if audit_notes_parts else "PASS",
    )

    logger.info(
        "Compliance check complete",
        extra={
            "approved": approved,
            "flags": [f.value for f in flags],
            "escalation_required": escalation_required,
        },
    )

    return result
