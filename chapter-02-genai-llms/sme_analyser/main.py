"""AWB SME Financial Statement Analyser (MR-2026-036) — entry point.

Thin orchestration layer over the shared document_analyser components:
routes SME statutory accounts (typically < 50 pages) to Gemini 3.5 Flash
per the Section 2.6.2 model selection matrix, and applies the SME-specific
extraction schema (turnover, EBITDA, director loans, going-concern notes).

Run: python -m sme_analyser.main <path-to-accounts.pdf>
"""
from __future__ import annotations

import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class SMEFinancialSummary:
    company_name: str
    turnover_gbp: float
    ebitda_gbp: float
    director_loans_gbp: float
    going_concern_flag: bool
    source_pages: dict[str, int]


def analyse(pdf_path: str) -> SMEFinancialSummary:
    """Extract an SMEFinancialSummary from a statutory accounts PDF.

    Production path: document_analyser.extractor with the SME prompt from
    document_analyser.prompt_patterns; every figure carries a citing page
    number that is verified before the summary is released (Section 2.4).
    """
    # Demonstration stub — replace with extractor.run(pdf_path, schema=SME).
    return SMEFinancialSummary(
        company_name="Mercia Instruments Ltd",
        turnover_gbp=48_200_000.0,
        ebitda_gbp=6_100_000.0,
        director_loans_gbp=250_000.0,
        going_concern_flag=False,
        source_pages={"turnover_gbp": 4, "ebitda_gbp": 5, "director_loans_gbp": 18},
    )


if __name__ == "__main__":
    summary = analyse(sys.argv[1] if len(sys.argv) > 1 else "sample.pdf")
    print(summary)
