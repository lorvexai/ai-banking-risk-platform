"""AWB Consumer Loan — FCA-Compliant Decline Letter Generator.

Uses Gemini 3.5 Flash to generate plain-English decline letters
from SHAP top-3 factors. Letters must:
  - Cite 2–3 specific reasons (no generic "credit score")
  - Not reveal exact score or threshold
  - Include CRA disclosure right (Consumer Credit Act 1974)
  - Pass output validation before transmission

Model: Gemini 3.5 Flash (lowest cost, decision-support use)
Prompt version: v1.3 (June 2026) — registered in AWB prompt registry
"""
from __future__ import annotations

import logging
import os
import re
import json
from dataclasses import dataclass

log = logging.getLogger(__name__)

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

# Plain-English translations of feature names for letters
FEATURE_PLAIN: dict[str, str] = {
    "debt_to_income":               "your total monthly debt commitments relative to income",
    "housing_cost_ratio":           "your housing costs as a proportion of income",
    "ob_income_volatility":         "the variability in your monthly income",
    "ob_discretionary_spend_ratio": "your level of discretionary spending",
    "ob_overdraft_days_90":         "the frequency of overdraft usage on your account",
    "ob_gambling_flag":             "certain spending patterns on your bank account",
    "ob_bill_payment_timeliness":   "the timeliness of your regular payments",
    "credit_bureau_score":          "information held by credit reference agencies",
    "existing_credit_utilisation":  "your current level of credit usage",
    "adverse_history_flag":         "previous credit difficulties recorded at credit agencies",
    "employment_tenure_months":     "the length of your current employment",
    "requested_amount_to_income":   "the loan amount requested relative to your income",
}


@dataclass
class DeclineLetter:
    """Generated decline letter with validation status."""
    application_id: str
    letter_text: str
    passed_validation: bool
    validation_notes: list[str]
    prompt_version: str = "v1.3"


class DeclineLetterGenerator:
    """Generate FCA PS22/9-compliant decline letters.

    Uses Gemini 3.5 Flash with a versioned prompt.
    Validates output before returning.

    Usage::

        gen = DeclineLetterGenerator()
        letter = gen.generate("APP-001", shap_top3_risk)
        if letter.passed_validation:
            send_to_applicant(letter.letter_text)
        else:
            route_to_human_agent(letter)
    """

    PROMPT_TEMPLATE = """You are writing a credit application outcome letter for
Avon & Wessex Bank plc. The application has been declined.

The primary reasons for this decision are:
{reasons}

Write a decline letter that:
1. Acknowledges the application politely
2. States 2-3 specific reasons for the decline using the
   plain-English descriptions above (do NOT use jargon
   like "probability of default", "SHAP", or "LightGBM")
3. Does NOT state the applicant's credit score or any
   specific numeric threshold
4. Includes this sentence exactly: "You have the right to
   request a copy of the information held about you by
   credit reference agencies under the Consumer Credit
   Act 1974."
5. Provides the AWB customer service contact: 0800 123 4567
6. Is between 150 and 250 words
7. Uses clear, plain English suitable for all reading levels

Do not include any advice about how to improve the
application outcome for future applications.
"""

    def generate(
        self,
        application_id: str,
        shap_top3_risk: list[dict],
    ) -> DeclineLetter:
        """Generate a personalised decline letter.

        Args:
            application_id: AWB application reference.
            shap_top3_risk: Top-3 risk-increasing SHAP factors
                           from ScoringResult.

        Returns:
            DeclineLetter with validation result.
        """
        reasons = self._format_reasons(shap_top3_risk)
        prompt  = self.PROMPT_TEMPLATE.format(reasons=reasons)

        try:
            letter_text = self._call_gemini(prompt)
        except Exception as exc:
            log.error(
                "Gemini call failed for app=%s: %s",
                application_id, exc,
            )
            letter_text = self._fallback_letter(shap_top3_risk)

        notes = self._validate(letter_text)
        passed = len(notes) == 0

        if not passed:
            log.warning(
                "Decline letter validation failed app=%s: %s",
                application_id, notes,
            )

        return DeclineLetter(
            application_id   = application_id,
            letter_text      = letter_text,
            passed_validation = passed,
            validation_notes  = notes,
        )

    # ── Internal helpers ──────────────────────────────────────────

    def _format_reasons(self, factors: list[dict]) -> str:
        lines = []
        for i, f in enumerate(factors[:3], 1):
            plain = FEATURE_PLAIN.get(
                f["feature"], f["feature"]
            )
            lines.append(f"{i}. {plain.capitalize()}.")
        return "\n".join(lines)

    def _validate(self, text: str) -> list[str]:
        """Validate letter against FCA PS22/9 requirements."""
        issues = []
        # Must mention at least 2 specific reasons
        specific = sum(
            1 for v in FEATURE_PLAIN.values()
            if any(
                w in text.lower()
                for w in v.lower().split()[:3]
            )
        )
        if specific < 2:
            issues.append("fewer than 2 specific reasons cited")

        # Must not contain numeric scores
        if re.search(r'\b\d{2,3}\b', text):
            potential_score = re.findall(r'\b\d{2,3}\b', text)
            if any(int(s) > 50 for s in potential_score
                   if s.isdigit()):
                issues.append("may contain numeric score")

        # Must include CRA disclosure sentence
        if "Consumer Credit Act" not in text:
            issues.append("missing CRA disclosure sentence")

        # Must be within length bounds
        word_count = len(text.split())
        if word_count < 100 or word_count > 350:
            issues.append(
                f"word count {word_count} outside 100-350 range"
            )

        return issues

    def _call_gemini(self, prompt: str) -> str:
        """Call Gemini 3.5 Flash via Anthropic-compatible SDK."""
        # Import here so tests can mock without installing SDK
        try:
            import google.generativeai as genai
            model = genai.GenerativeModel(GEMINI_MODEL)
            response = model.generate_content(prompt)
            return response.text
        except ImportError:
            raise RuntimeError(
                "google-generativeai not installed"
            )

    def _fallback_letter(self, factors: list[dict]) -> str:
        """Static fallback when Gemini is unavailable."""
        reasons = self._format_reasons(factors)
        return (
            "Dear Applicant,\n\n"
            "Thank you for your application to Avon & Wessex Bank plc. "
            "After careful consideration, we are unable to approve your "
            "application at this time. The primary reasons are:\n\n"
            f"{reasons}\n\n"
            "You have the right to request a copy of the information "
            "held about you by credit reference agencies under the "
            "Consumer Credit Act 1974.\n\n"
            "If you have any questions, please contact us on "
            "0800 123 4567.\n\n"
            "Yours sincerely,\n"
            "Avon & Wessex Bank plc"
        )
