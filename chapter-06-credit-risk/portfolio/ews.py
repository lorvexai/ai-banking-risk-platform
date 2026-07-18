"""AWB Portfolio Monitoring — Early Warning Signal System.

Model ID:    MR-2026-042 (LLM news scanner component)
Risk rating: LOW (PRA SS1/23 — decision-support only)
             Rules core is NOT a registered model (deterministic)

12 triggers across 4 categories:
  Financial deterioration (4)
  Behavioural signals (4)
  External signals (3)
  LLM-augmented news scan (1) — MR-2026-042

All Red/Amber alerts reviewed by credit analyst before
any credit action is taken. EWS does NOT trigger
automated credit limit changes or facility reviews.

Daily scoring across AWB's 12,400 corporate facilities.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from awb_commons.schemas import EWSResult, RAGStatus
from awb_commons.audit import AuditLogger

log = logging.getLogger(__name__)

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

# Red threshold: >= 3 triggers OR any HIGH trigger
# Amber threshold: 1-2 triggers
HIGH_TRIGGERS = frozenset({
    "covenant_breach",
    "interest_cover_floor",
    "director_disqualification",
    "waiver_request",
})

_audit = AuditLogger("MR-2026-042")


@dataclass
class FacilityData:
    """T24 and external data snapshot for EWS scoring."""
    facility_id: str
    # Financial deterioration
    revenue_yoy_change: float         # -0.20 = 20% decline
    ebitda_margin_qoq_change: float   # basis points
    covenant_breach: bool
    interest_cover: float             # x times
    # Behavioural
    days_past_due: int                # max in period
    utilisation_rate: float           # 0.0–1.0
    missed_reporting: bool
    waiver_request_received: bool
    # External
    sector_pmi: float                 # Markit PMI
    ch_filing_anomaly: bool           # Companies House flag
    director_disqualification: bool
    # Optional — Companies House / news
    company_name: Optional[str] = None
    company_number: Optional[str] = None


class EarlyWarningSystem:
    """AWB 12-trigger EWS — rules-based core + LLM news scan.

    Rules core satisfies PRA SS1/23 explainability:
    each trigger is deterministic, documented, and auditable.

    LLM news scanner (MR-2026-042) is a supplementary signal
    registered separately as LOW risk decision-support.

    Usage::

        ews = EarlyWarningSystem()
        result = ews.score("F-1234", data)
        # result.status ∈ {RED, AMBER, GREEN}
        # result.triggers_fired → list of named triggers
    """

    # Trigger thresholds — reviewed quarterly by credit committee
    REVENUE_DECLINE_THRESHOLD   = -0.20   # >20% YoY decline
    EBITDA_MARGIN_BPS_THRESHOLD = -500    # >500bps QoQ decline
    INTEREST_COVER_FLOOR        = 1.5     # × contractual minimum
    DAYS_PAST_DUE_THRESHOLD     = 5       # days
    UTILISATION_SPIKE_THRESHOLD = 0.90    # 90% of facility limit
    SECTOR_PMI_THRESHOLD        = 45.0    # contractionary

    def score(
        self,
        facility_id: str,
        data: FacilityData,
        scan_news: bool = True,
    ) -> EWSResult:
        """Score a facility against all 12 EWS triggers.

        Args:
            facility_id: AWB facility reference.
            data: FacilityData snapshot for this facility.
            scan_news: If True, call Gemini news scanner
                      (MR-2026-042). Set False in tests or
                      for Green-rated facilities.

        Returns:
            EWSResult with RAG status and fired triggers.
        """
        triggered = self._eval_rules(data)

        news_flag = None
        if scan_news and data.company_name:
            news_flag = self._scan_news(
                facility_id,
                data.company_name,
                data.company_number,
            )
            if news_flag == "MATERIAL":
                triggered.append("adverse_news_material")

        score   = len(triggered) * 10 / 12
        status  = self._rag_status(triggered)

        _audit.log_ews(facility_id, status.value, triggered, score)
        log.info(
            "EWS fac=%s status=%s score=%.1f triggers=%s",
            facility_id, status.value, score, triggered,
        )
        return EWSResult(
            facility_id   = facility_id,
            status        = status,
            triggers_fired = triggered,
            ews_score     = round(score, 2),
            news_flag     = news_flag,
        )

    def score_portfolio(
        self,
        facilities: list[tuple[str, FacilityData]],
    ) -> list[EWSResult]:
        """Score a portfolio batch (daily scheduled run)."""
        results = []
        for fid, data in facilities:
            # Only scan news for Red/Amber-risk facilities
            # (score > 4 based on rules alone)
            pre_score = len(self._eval_rules(data))
            scan = pre_score >= 1 and data.company_name is not None
            results.append(self.score(fid, data, scan_news=scan))
        return results

    # ── Rule evaluation ───────────────────────────────────────────

    def _eval_rules(self, data: FacilityData) -> list[str]:
        """Evaluate the 11 deterministic rules."""
        fired = []

        # Financial deterioration (4 triggers)
        if data.revenue_yoy_change < self.REVENUE_DECLINE_THRESHOLD:
            fired.append("revenue_decline_20pct")
        if data.ebitda_margin_qoq_change < self.EBITDA_MARGIN_BPS_THRESHOLD:
            fired.append("ebitda_margin_decline_500bps")
        if data.covenant_breach:
            fired.append("covenant_breach")
        if data.interest_cover < self.INTEREST_COVER_FLOOR:
            fired.append("interest_cover_floor")

        # Behavioural signals (4 triggers)
        if data.days_past_due > self.DAYS_PAST_DUE_THRESHOLD:
            fired.append("days_past_due")
        if data.utilisation_rate > self.UTILISATION_SPIKE_THRESHOLD:
            fired.append("utilisation_spike")
        if data.missed_reporting:
            fired.append("missed_reporting")
        if data.waiver_request_received:
            fired.append("waiver_request")

        # External signals (3 triggers)
        if data.sector_pmi < self.SECTOR_PMI_THRESHOLD:
            fired.append("sector_pmi_contractionary")
        if data.ch_filing_anomaly:
            fired.append("ch_filing_anomaly")
        if data.director_disqualification:
            fired.append("director_disqualification")

        return fired

    def _rag_status(self, triggered: list[str]) -> RAGStatus:
        """Compute RAG status from fired triggers."""
        if any(t in HIGH_TRIGGERS for t in triggered):
            return RAGStatus.RED
        if len(triggered) >= 3:
            return RAGStatus.RED
        if len(triggered) >= 1:
            return RAGStatus.AMBER
        return RAGStatus.GREEN

    # ── LLM news scanner ─────────────────────────────────────────

    def _scan_news(
        self,
        facility_id: str,
        company_name: str,
        company_number: Optional[str],
    ) -> str:
        """Scan news and CH filings for adverse signals.

        Calls Gemini 3.5 Flash with structured JSON prompt.
        Returns "MATERIAL" | "NONE".

        Registered as MR-2026-042 (LOW risk, decision-support).
        Output is advisory only — all Material flags reviewed
        by a credit analyst before any action.
        """
        prompt = f"""You are screening corporate news for a UK bank's
credit monitoring system.

Company: {company_name}
Company number: {company_number or "N/A"}

Search recent news (last 7 days) and Companies House filings.
Look for: insolvency proceedings, director changes, profit warnings,
covenant disclosures, regulatory investigations, or significant
adverse events.

Respond ONLY with a JSON object (no preamble):
{{
  "severity": "MATERIAL" | "NONE",
  "summary": "<max 100 words if MATERIAL, empty string if NONE>",
  "sources": ["<url1>", "<url2>"],
  "confidence": "HIGH" | "MEDIUM" | "LOW"
}}"""

        try:
            raw = self._call_gemini(prompt)
            parsed = json.loads(raw.strip())
            severity = parsed.get("severity", "NONE")
            log.info(
                "EWS news scan fac=%s company=%s severity=%s",
                facility_id, company_name, severity,
            )
            return severity
        except (json.JSONDecodeError, Exception) as exc:
            log.warning(
                "EWS news scan failed fac=%s: %s",
                facility_id, exc,
            )
            return "NONE"

    def _call_gemini(self, prompt: str) -> str:
        """Thin wrapper around Gemini API."""
        try:
            import google.generativeai as genai
            model = genai.GenerativeModel(GEMINI_MODEL)
            response = model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"},
            )
            return response.text
        except ImportError:
            raise RuntimeError(
                "google-generativeai not installed. "
                "Use EWSResult with news_flag=None in tests."
            )
