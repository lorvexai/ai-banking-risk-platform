"""AWB Regulatory Change Impact Analyzer (Section 11.4A).

Model ID: MR-2026-069-REG | Risk: MEDIUM (PRA SS1/23)
EU AI Act: LIMITED scope (decision support, not a creditworthiness or
eligibility determination) | LLM: Gemini 3.1 Pro

Chapter 4's Regulatory Knowledge Assistant (MR-2026-038) detects a new
or amended PRA, FCA, or EBA publication and marks superseded documents
via SupersessionDetector.detect_and_mark(), which writes a
StateChangeRecord audit event (chapter-04-rag-systems/awb_commons/rag/
supersession_detector.py) for every document status change. What that
module does NOT do is tell AWB's regulatory reporting team what a change
means for AWB specifically. This module closes that gap (Section 11.4A.1).

Architecture (Section 11.4A.2):
  This analyzer consumes RegulatoryChangeEvent records — this repo's
  chapter-11-local mirror of chapter 4's StateChangeRecord schema,
  since each chapter in this platform is an independently installable
  package (see pyproject.toml in every chapter directory) and none
  import another chapter's code directly. In production, the two
  chapters share a StateChangeRecord audit event feed (a table, topic,
  or queue keyed the same way in both chapters); RegulatoryChangeEvent
  intentionally mirrors StateChangeRecord field-for-field so the two
  stay interoperable without a hard Python dependency between chapters.

  On receipt of an event where to_status == "FINAL" (a new or amended
  publication has gone live), Gemini 3.1 Pro is given the diff between
  the old and new document text and AWB's system-to-regulation
  CrossReferenceTable, a table mapping each of the 23-plus registered
  model IDs and each COREP/Pillar 3 template to the specific CRR3
  articles, PRA rulebook chapters, or EBA ITS provisions it implements.
  The model identifies which cross-reference rows are affected and
  produces a ChangeImpactReport: affected system IDs, affected report
  templates, affected controls, and a plain-English summary of what
  changed. Matching against the cross-reference table is deterministic;
  only the plain-English summary is generative, keeping the
  highest-stakes output free of hallucination risk.

Human gate (Section 11.4A.3): every report routes to the Head of
Regulatory Reporting for review before any system change is scoped.
The Analyzer recommends; it never triggers automated code changes.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

MODEL_ID = "MR-2026-069-REG"
GEMINI_ANALYSIS_MODEL = os.getenv("GEMINI_ANALYSIS_MODEL", "gemini-3.1-pro")


# ── Change event (mirrors chapter-04's StateChangeRecord) ─────────────────

class DocumentStatus(str, Enum):
    """Mirrors chapter-04-rag-systems/awb_commons/rag/
    supersession_detector.py DocumentStatus. Kept in sync manually since
    chapters do not import each other's packages (see module docstring).
    """
    DRAFT = "DRAFT"
    CONSULTATION = "CONSULTATION"
    FINAL = "FINAL"
    SUPERSEDED = "SUPERSEDED"


@dataclass
class RegulatoryChangeEvent:
    """Chapter-11-local mirror of chapter 4's StateChangeRecord.

    Field names and semantics match StateChangeRecord exactly so that a
    real deployment can map the shared audit event feed onto this type
    with no translation logic beyond a dataclass construction.
    """
    document_id: str
    from_status: DocumentStatus
    to_status: DocumentStatus
    changed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    changed_by: str = "awb.rag.supersession_detector"
    reason: str = ""
    triggered_by: Optional[str] = None


# ── Cross-reference table ──────────────────────────────────────────────────

class CitationType(str, Enum):
    CRR3_ARTICLE = "CRR3_ARTICLE"
    PRA_RULEBOOK = "PRA_RULEBOOK"
    EBA_ITS = "EBA_ITS"


@dataclass(frozen=True)
class CrossReferenceRow:
    """One row of AWB's system-to-regulation cross-reference table
    (Section 11.4A.2). Maintained by the regulatory reporting team;
    cited throughout this chapter's model registry."""
    citation_type: CitationType
    citation_ref: str            # e.g. "CRR3 Art. 92a"
    affected_system_id: str      # e.g. "MR-2026-072"
    affected_report_template: Optional[str] = None  # e.g. "COREP C 02.00"
    affected_control: Optional[str] = None


class CrossReferenceTable:
    """In-memory system-to-regulation cross-reference table.

    Production implementation reads from PostgreSQL (Section 11.4A.2).
    This stub ships AWB's Q1 2026 CRR3 output floor illustrative rows
    (Section 11.4A.3) plus a handful of others so the pipeline can be
    exercised in tests and demos without a live database.
    """

    _DEFAULT_ROWS: List[CrossReferenceRow] = [
        CrossReferenceRow(
            citation_type=CitationType.CRR3_ARTICLE,
            citation_ref="CRR3 Art. 92a",
            affected_system_id="MR-2026-072",
            affected_report_template="COREP C 02.00",
            affected_control="Basel Credit Risk Reporting output floor control",
        ),
        CrossReferenceRow(
            citation_type=CitationType.CRR3_ARTICLE,
            citation_ref="CRR3 Art. 92a",
            affected_system_id="MR-2026-071",
            affected_report_template="COREP C 02.00",
            affected_control="MJRRP RWA calculation engine output floor check",
        ),
        CrossReferenceRow(
            # Chapter 6 IRB PD model — Section 11.4A.3's worked example names
            # this as the third system affected by a CRR3 Art. 92a output
            # floor amendment, alongside the two rows above. Uses MR-2026-043
            # as registered in chapter-06-credit-risk/corporate_pd/model.py
            # and the book's Appendix Model Registry Index.
            citation_type=CitationType.CRR3_ARTICLE,
            citation_ref="CRR3 Art. 92a",
            affected_system_id="MR-2026-043",
            affected_report_template="COREP C 08.00",
            affected_control="Chapter 6 IRB PD model output floor exposure "
                              "(compute_sa_floor_rwa, Section 6.8A.2)",
        ),
        CrossReferenceRow(
            citation_type=CitationType.CRR3_ARTICLE,
            citation_ref="CRR3 Art. 429",
            affected_system_id="MR-2026-071",
            affected_report_template="COREP C 47.00",
            affected_control="Leverage ratio minimum breach control",
        ),
        CrossReferenceRow(
            citation_type=CitationType.CRR3_ARTICLE,
            citation_ref="CRR3 Arts 411-428",
            affected_system_id="MR-2026-071",
            affected_report_template="COREP C 72.00",
            affected_control="LCR minimum breach control",
        ),
    ]

    def __init__(self, rows: Optional[List[CrossReferenceRow]] = None) -> None:
        self._rows = rows if rows is not None else list(self._DEFAULT_ROWS)

    def find_affected(self, citation_ref: str) -> List[CrossReferenceRow]:
        """Return all cross-reference rows matching a citation, e.g.
        'CRR3 Art. 92a'. Deterministic lookup — no LLM involvement."""
        return [r for r in self._rows if r.citation_ref == citation_ref]

    def add_row(self, row: CrossReferenceRow) -> None:
        self._rows.append(row)


# ── Impact report ────────────────────────────────────────────────────────

@dataclass
class ChangeImpactReport:
    """Output of a single change-impact analysis run (Section 11.4A.2).
    Routes to the Head of Regulatory Reporting for review before any
    system change is scoped (Section 11.4A.3)."""
    document_id: str
    citation_ref: str
    affected_system_ids: List[str]
    affected_report_templates: List[str]
    affected_controls: List[str]
    summary_text: str
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    reviewed_by: Optional[str] = None
    review_notes: Optional[str] = None
    model_id: str = MODEL_ID

    @property
    def requires_review(self) -> bool:
        return self.reviewed_by is None


class RegulatoryChangeAnalyzer:
    """Maps a regulatory change event to affected systems, report
    templates, and controls (Section 11.4A.2).

    Usage::

        analyzer = RegulatoryChangeAnalyzer()
        event = RegulatoryChangeEvent(
            document_id="CRR3_output_floor_2026Q1",
            from_status=DocumentStatus.CONSULTATION,
            to_status=DocumentStatus.FINAL,
        )
        report = analyzer.analyze_change(
            event, citation_ref="CRR3 Art. 92a",
            old_text=old_provision_text, new_text=new_provision_text,
        )
        analyzer.route_for_review(report)  # Head of Regulatory Reporting
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        cross_reference: Optional[CrossReferenceTable] = None,
    ) -> None:
        self.model_id = model_id
        self.cross_reference = cross_reference or CrossReferenceTable()

    def analyze_change(
        self,
        event: RegulatoryChangeEvent,
        citation_ref: str,
        old_text: str,
        new_text: str,
    ) -> ChangeImpactReport:
        """Produce a ChangeImpactReport for a single regulatory change.

        Only processes events where `to_status` is FINAL — draft and
        consultation publications do not trigger impact analysis
        (Section 11.4A.2). Raises ValueError otherwise.
        """
        if event.to_status != DocumentStatus.FINAL:
            raise ValueError(
                f"Impact analysis only runs on FINAL publications, "
                f"got to_status={event.to_status}"
            )

        rows = self.cross_reference.find_affected(citation_ref)
        affected_system_ids = sorted({r.affected_system_id for r in rows})
        affected_templates = sorted(
            {r.affected_report_template for r in rows if r.affected_report_template}
        )
        affected_controls = sorted(
            {r.affected_control for r in rows if r.affected_control}
        )

        summary = self._summarise_change(citation_ref, old_text, new_text)

        log.info(
            "Change impact analysed: doc=%s citation=%s systems=%d templates=%d",
            event.document_id, citation_ref,
            len(affected_system_ids), len(affected_templates),
        )

        return ChangeImpactReport(
            document_id=event.document_id,
            citation_ref=citation_ref,
            affected_system_ids=affected_system_ids,
            affected_report_templates=affected_templates,
            affected_controls=affected_controls,
            summary_text=summary,
            model_id=self.model_id,
        )

    def route_for_review(
        self, report: ChangeImpactReport, reviewer: str = "Head of Regulatory Reporting"
    ) -> ChangeImpactReport:
        """Human gate (Section 11.4A.3): the Analyzer recommends, it
        does not trigger automated code changes. No downstream action
        is taken until this is called."""
        log.info(
            "Change impact report for %s routed to %s for review",
            report.document_id, reviewer,
        )
        report.review_notes = f"Routed to {reviewer}"
        return report

    def confirm_reviewed(
        self, report: ChangeImpactReport, reviewer: str, notes: Optional[str] = None
    ) -> ChangeImpactReport:
        """Record that the Head of Regulatory Reporting has reviewed the
        report (Section 11.4A.3)."""
        report.reviewed_by = reviewer
        report.review_notes = notes or report.review_notes
        log.info("Change impact report for %s reviewed by %s", report.document_id, reviewer)
        return report

    # ── Generative step (Section 11.4A.2) ───────────────────────────────

    def _summarise_change(self, citation_ref: str, old_text: str, new_text: str) -> str:
        """Plain-English summary of what changed, using Gemini 3.1 Pro.
        Falls back to a deterministic diff-length note when no live API
        key is available, so the pipeline is fully exercisable offline.
        """
        try:
            import google.generativeai as genai

            model = genai.GenerativeModel(GEMINI_ANALYSIS_MODEL)
            prompt = (
                f"Summarise in plain English what changed in {citation_ref} "
                f"between these two versions for a UK bank's regulatory "
                f"reporting team.\n\nOLD:\n{old_text}\n\nNEW:\n{new_text}"
            )
            return model.generate_content(prompt).text
        except Exception as exc:  # noqa: BLE001 — offline/template fallback
            log.info("Analysis LLM unavailable (%s); using template summary", exc)
            return (
                f"AI-ASSISTED DRAFT — REVIEW REQUIRED\n\n"
                f"{citation_ref} was amended (old text {len(old_text)} chars, "
                f"new text {len(new_text)} chars). Manual comparison required "
                f"to confirm the nature of the change.\n"
                f"[Generated by {MODEL_ID}]"
            )
