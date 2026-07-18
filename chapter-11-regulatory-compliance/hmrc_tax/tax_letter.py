"""LLM-generated client CGT tax letter (MR-2026-047).

FCA Consumer Duty PS22/9 compliant: clear, fair,
personalised. Requires adviser review before sending.
"""
from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal
import logging

log = logging.getLogger(__name__)
MODEL = "gemini-3.5-flash"


@dataclass
class TaxLetter:
    """AI-assisted client tax summary letter."""
    client_id: str
    tax_year: str
    total_gains: Decimal
    tax_liability: Decimal
    letter_body: str
    is_ai_assisted: bool = True
    requires_adviser_review: bool = True
    adviser_reviewed: bool = False


class HMRCTaxLetterGenerator:
    """Generate personalised CGT letters via Gemini.

    All output is labelled AI-ASSISTED and requires
    adviser review per EU AI Act Art. 14 oversight
    and FCA Consumer Duty PS22/9.
    """

    PROMPT_TEMPLATE = """You are a UK tax adviser at AWB
Wealth Management. Write a clear, concise client letter
explaining their Capital Gains Tax for {tax_year}.

Client gains: £{total_gains:,.2f}
CGT allowance: £{cgt_allowance:,.2f}
Taxable gain: £{taxable_gain:,.2f}
Estimated tax: £{tax_liability:,.2f}
Payment deadline: 31 January {payment_year}

Requirements:
- Plain English (FCA COBS 4 — clear and fair)
- Personalised to this client's situation
- Include: what they owe, how to pay, key dates
- Maximum 200 words
"""

    def __init__(self, api_key: str) -> None:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self._model = genai.GenerativeModel(MODEL)
        except ImportError:
            self._model = None  # for testing
            log.warning("google-generativeai not installed")

    def generate_client_summary(
        self,
        client_id: str,
        tax_year: str,
        total_gains: Decimal,
        tax_liability: Decimal,
        cgt_allowance: Decimal = Decimal("3000"),
    ) -> TaxLetter:
        """Generate AI-assisted client tax letter.

        Args:
            client_id: AWB client reference.
            tax_year: e.g. '2025-2026'.
            total_gains: Total capital gains (£).
            tax_liability: Estimated CGT owed (£).
            cgt_allowance: HMRC annual allowance (£).

        Returns:
            TaxLetter requiring adviser review before
            sending (EU AI Act Art. 14 human oversight).
        """
        taxable = max(
            Decimal("0"), total_gains - cgt_allowance
        )
        payment_year = int(tax_year.split("-")[1])
        prompt = self.PROMPT_TEMPLATE.format(
            tax_year=tax_year,
            total_gains=total_gains,
            cgt_allowance=cgt_allowance,
            taxable_gain=taxable,
            tax_liability=tax_liability,
            payment_year=payment_year,
        )
        if self._model:
            resp = self._model.generate_content(prompt)
            raw_body = resp.text
        else:
            raw_body = f"[Draft letter for {client_id}]"

        body = (
            "AI-ASSISTED DRAFT — ADVISER REVIEW "
            "REQUIRED BEFORE SENDING\n\n"
            + raw_body
        )
        log.info(
            "Tax letter drafted for client %s",
            client_id,
        )
        return TaxLetter(
            client_id=client_id,
            tax_year=tax_year,
            total_gains=total_gains,
            tax_liability=tax_liability,
            letter_body=body,
        )
