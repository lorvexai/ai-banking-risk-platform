"""
chapter_07/var_monitoring/var_attribution_reporter.py
AWB VaR P&L Attribution Reporter — Gemini 3.5 Flash
Generates PRA-reportable breach explanations in ~90s
MR-2026-046 | EU AI Act Art. 14 — Human attestation required
awb_commons
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

import httpx

log = logging.getLogger(__name__)

ANTHROPIC_API_URL = (
    "https://api.anthropic.com/v1/messages"
)
MODEL_ID = "claude-haiku-4-5-20251001"   # fast, cost-effective

ATTRIBUTION_SYSTEM_PROMPT = """You are the AWB Market Risk
reporting assistant. Your role is to generate clear,
professional, PRA-reportable P&L attribution reports
when a VaR exception occurs. Always:
- State explicitly this report is AI-ASSISTED and
  requires Head of Market Risk attestation.
- Cite FRTB Art. 325x or CRR3 Art. 325bf as applicable.
- Keep the report to maximum 400 words.
- Use formal regulatory language appropriate for
  PRA submission.
- Never invent position data not provided to you.
"""


@dataclass
class AttributionReport:
    """VaR exception P&L attribution report."""

    date: str
    model_id: str
    var_99_gbp: float
    actual_loss_gbp: float
    top_contributors: Dict[str, float]
    narrative: str
    is_ai_assisted: bool = True
    requires_attestation: bool = True
    generated_at: str = ""

    def __post_init__(self) -> None:
        if not self.generated_at:
            self.generated_at = (
                datetime.utcnow().isoformat()
            )


class VaRAttributionReporter:
    """
    Generates PRA-reportable P&L attribution reports
    using an LLM within 90 seconds of a VaR exception.

    GUARDRAIL: All reports are clearly labelled
    AI-ASSISTED. Head of Market Risk must attest
    before PRA submission (EU AI Act Art. 14).

    Model: MR-2026-046 reporting component
    Cost: ~£0.14 per report at Claude Haiku 4.5 rates

    Args:
        max_tokens: Maximum output tokens (default 600)
    """

    def __init__(
        self, max_tokens: int = 600
    ) -> None:
        self.max_tokens = max_tokens

    def generate_breach_report(
        self,
        date: str,
        var_99_gbp: float,
        actual_loss_gbp: float,
        component_var: Dict[str, float],
        position_summary: Optional[str] = None,
    ) -> AttributionReport:
        """
        Generate P&L attribution for a VaR exception.

        Args:
            date: Exception date (YYYY-MM-DD)
            var_99_gbp: 99% VaR estimate (GBP)
            actual_loss_gbp: Actual loss (GBP, positive)
            component_var: {risk_factor: component_var}
            position_summary: Optional desk description
        Returns:
            AttributionReport with PRA-ready narrative
        Raises:
            RuntimeError: If API call fails
        """
        top_5 = dict(
            sorted(
                component_var.items(),
                key=lambda x: abs(x[1]),
                reverse=True,
            )[:5]
        )
        user_prompt = self._build_prompt(
            date, var_99_gbp, actual_loss_gbp,
            top_5, position_summary
        )
        log.info(
            "MR-2026-046: Generating P&L attribution "
            "for %s exception £%.0f vs VaR £%.0f",
            date, actual_loss_gbp, var_99_gbp,
        )
        try:
            narrative = self._call_llm(user_prompt)
        except Exception as exc:
            log.error(
                "Attribution LLM call failed: %s", exc
            )
            raise RuntimeError(
                f"Attribution report failed: {exc}"
            ) from exc

        report = AttributionReport(
            date=date,
            model_id="MR-2026-046",
            var_99_gbp=var_99_gbp,
            actual_loss_gbp=actual_loss_gbp,
            top_contributors=top_5,
            narrative=narrative,
        )
        log.info(
            "Attribution report generated: %s chars, "
            "ATTESTATION REQUIRED before PRA submission",
            len(narrative),
        )
        return report

    def _build_prompt(
        self,
        date: str,
        var_99: float,
        loss: float,
        top_5: Dict[str, float],
        position_summary: Optional[str],
    ) -> str:
        """Build structured prompt for LLM."""
        contributors_str = "\n".join(
            f"  - {k}: £{v:,.0f}"
            for k, v in top_5.items()
        )
        prompt = (
            f"VaR Exception Report Required — {date}\n\n"
            f"AWB Model: MR-2026-046 "
            f"(Real-Time VaR Engine)\n"
            f"99% 1-Day VaR: £{var_99:,.0f}\n"
            f"Actual Loss: £{loss:,.0f}\n"
            f"Exception Magnitude: "
            f"£{loss - var_99:,.0f} above VaR\n\n"
            f"Top 5 Risk Factor Contributors:\n"
            f"{contributors_str}\n"
        )
        if position_summary:
            prompt += (
                f"\nPosition Context:\n{position_summary}\n"
            )
        prompt += (
            "\nGenerate a formal PRA-reportable P&L "
            "attribution report covering:\n"
            "1. Which positions drove the loss\n"
            "2. Which risk factors moved adversely\n"
            "3. Whether the loss was within model "
            "expectations given market moves\n"
            "4. Recommended hedging or risk-reduction "
            "actions\n"
            "5. FRTB Art. 325x / CRR3 Art. 325bf "
            "regulatory reference\n\n"
            "IMPORTANT: Begin with: "
            "'AI-ASSISTED REPORT — ATTESTATION REQUIRED"
            " (EU AI Act Art. 14)'"
        )
        return prompt

    def _call_llm(self, prompt: str) -> str:
        """Call LLM API and return generated text."""
        payload = {
            "model": MODEL_ID,
            "max_tokens": self.max_tokens,
            "system": ATTRIBUTION_SYSTEM_PROMPT,
            "messages": [
                {"role": "user", "content": prompt}
            ],
        }
        resp = httpx.post(
            ANTHROPIC_API_URL,
            json=payload,
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()
        text_blocks = [
            b["text"]
            for b in data.get("content", [])
            if b.get("type") == "text"
        ]
        return "\n".join(text_blocks)
