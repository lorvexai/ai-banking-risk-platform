"""HMRC Tax Letter Generator — AI-assisted client summaries.

Model ID: MR-2026-047 | Risk: LOW
LLM: Gemini 3.5 Flash | FCA PS22/9 compliant
EU AI Act Art. 14: adviser review mandatory — never auto-send.
"""
from __future__ import annotations
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import List

import google.generativeai as genai

from awb_commons.models import CGTDisposal, TaxLetter

log = logging.getLogger(__name__)

CGT_ALLOWANCE_2025_26 = Decimal("3000")
PAYMENT_DEADLINE = "31 January 2027"


class HMRCTaxLetterGenerator:
    """Generate FCA PS22/9 compliant client tax letters.

    Uses Gemini 3.5 Flash to produce personalised plain-English
    summaries of CGT liability. All output requires adviser review
    before sending (EU AI Act Art. 14 human oversight principle).

    Args:
        model_id: Registered model ID (MR-2026-047).

    Raises:
        ValueError: If API key not configured.
    """

    def __init__(
        self,
        model_id: str = "MR-2026-047",
    ) -> None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable required"
            )
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(
            "gemini-3.5-flash"
        )
        self._model_id = model_id
        log.info(
            "HMRCTaxLetterGenerator initialised: %s",
            model_id,
        )

    def generate_client_summary(
        self,
        client_id: str,
        client_name: str,
        tax_year: str,
        disposals: List[CGTDisposal],
        total_gain_gbp: Decimal,
    ) -> TaxLetter:
        """Generate personalised CGT summary letter.

        Produces a 200-300 word plain-English explanation of the
        client's CGT position. Always marked AI-ASSISTED DRAFT;
        requires_adviser_review is always True.

        Args:
            client_id: AWB client identifier.
            client_name: Client's full name for personalisation.
            tax_year: e.g., "2025-26".
            disposals: List of CGTDisposal records.
            total_gain_gbp: Net gain after offsetting losses.

        Returns:
            TaxLetter with requires_adviser_review=True.
        """
        taxable = max(
            Decimal("0"),
            total_gain_gbp - CGT_ALLOWANCE_2025_26,
        )
        prompt = self._build_prompt(
            client_name, tax_year, disposals,
            total_gain_gbp, taxable,
        )
        response = self._model.generate_content(prompt)
        letter_content = (
            f"AI-ASSISTED DRAFT — REQUIRES ADVISER REVIEW\n\n"
            f"{response.text}\n\n"
            f"This letter was prepared using AI assistance "
            f"(Gemini 3.5 Flash, MR-2026-047) and must be "
            f"reviewed and approved by a qualified tax adviser "
            f"before sending. Not for client distribution in "
            f"this form."
        )
        log.info(
            "Tax letter generated: client=%s gain=£%s "
            "taxable=£%s review_required=True",
            client_id, total_gain_gbp, taxable,
        )
        return TaxLetter(
            client_id=client_id,
            tax_year=tax_year,
            letter_content=letter_content,
            total_gain_gbp=total_gain_gbp,
            tax_liability_gbp=taxable,
            requires_adviser_review=True,
            model_id=self._model_id,
        )

    def _build_prompt(
        self,
        name: str,
        tax_year: str,
        disposals: List[CGTDisposal],
        total_gain: Decimal,
        taxable: Decimal,
    ) -> str:
        disposal_summary = "\n".join(
            f"- {d.asset_name}: proceeds £{d.disposal_proceeds_gbp:.2f}, "
            f"cost £{d.allowable_cost_gbp:.2f}, "
            f"{'gain' if d.is_gain else 'loss'} £{abs(d.gain_or_loss_gbp):.2f}"
            for d in disposals[:10]
        )
        return f"""Write a plain-English letter (200-300 words) to {name}
explaining their CGT position for {tax_year}:

Disposals:
{disposal_summary}

Total net gain: £{total_gain:.2f}
Annual CGT allowance (2025-26): £{CGT_ALLOWANCE_2025_26:.2f}
Taxable gain after allowance: £{taxable:.2f}
Payment deadline: {PAYMENT_DEADLINE}

Requirements:
- Clear, fair, not misleading (FCA PS22/9 / COBS 4)
- Plain English — avoid jargon
- Explain how to pay if tax is owed
- State this is a summary only; adviser review recommended
- British English throughout"""
