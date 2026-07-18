"""
awb_commons/rag/hybrid_router.py
AWB Hybrid RAG — Query Classification and Routing
Chapter 4: Section 4.6 — Hybrid Structured and Unstructured RAG

Classifies regulatory queries into three types:
  - DOCUMENT: Answered from ChromaDB regulatory corpus only
  - DATA:     Answered from PostgreSQL AWB operational data only
  - HYBRID:   Requires both sources in a single synthesised response

Safety constraints (non-negotiable):
  - SQL tool is READ-ONLY; parameterised queries only
  - LLM never constructs dynamic SQL from user input
  - Schema (column names/types) exposed to LLM; data values never shown
    until they appear in the final synthesised answer

AWB examples:
  - DOCUMENT: "What is CRR3 Article 153 slotting treatment?"
  - DATA:     "What was our LCR ratio on 15 June 2026?"
  - HYBRID:   "Is our LCR compliant with CRR3 minimums?"

Regulatory context:
  PRA SS1/23 MR-2026-038: LOW risk; decision-support only
  DORA Art. 9: ICT asset RKA-2026-001 registered
  FCA PS22/9: Responses cite regulatory sources

Usage:
    router = HybridRouter()
    q_type = router.classify("Is our leverage ratio above 3%?")
    # Returns QueryType.HYBRID
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger("awb.rag.hybrid_router")


class QueryType(str, Enum):
    """Classification of regulatory query by retrieval path."""
    DOCUMENT = "document"   # ChromaDB corpus only
    DATA     = "data"       # PostgreSQL only
    HYBRID   = "hybrid"     # Both sources required


class ClassificationResult(BaseModel):
    """Result of query classification with reasoning."""
    query:       str
    query_type:  QueryType
    reasoning:   str = Field(
        default="",
        description="LLM reasoning for classification"
    )
    latency_ms:  float = Field(
        default=0.0,
        description="Classification latency in milliseconds"
    )

    model_config = {"frozen": True}


# Keywords that strongly indicate each query type
_DOCUMENT_SIGNALS = {
    "what is", "what does", "explain", "define",
    "requirement", "regulation", "article", "section",
    "guidance", "policy", "standard", "obligation",
    "capital treatment", "risk weight", "eligibility",
}

_DATA_SIGNALS = {
    "our current", "our today", "awb's",
    "last quarter", "last week", "yesterday",
    "last tuesday", "last monday", "last wednesday",
    "as of", "balance sheet",
    "what was our", "what is our",
    "our lcr", "our leverage", "our nsfr",
    "our ratio", "our position",
}

_HYBRID_SIGNALS = {
    "compliant", "breach", "above", "below",
    "compared to", "vs minimum", "vs requirement",
    "exceed", "meets", "satisfies", "adequate",
    "within limit", "threshold",
    "minimum requirements", "minimum today",
    "above the", "below the",
    "is our", "are we", "do we",
}


def _heuristic_classify(query: str) -> Optional[QueryType]:
    """
    Fast heuristic classification without LLM call.
    Returns None if heuristic is not confident enough.

    Covers ~70% of AWB query patterns at zero API cost.
    Remaining 30% fall through to LLM classification.
    """
    q_lower = query.lower()

    hybrid_score  = sum(1 for s in _HYBRID_SIGNALS  if s in q_lower)
    data_score    = sum(1 for s in _DATA_SIGNALS    if s in q_lower)
    doc_score     = sum(1 for s in _DOCUMENT_SIGNALS if s in q_lower)

    if hybrid_score >= 2:
        return QueryType.HYBRID
    if data_score >= 2 and doc_score == 0:
        return QueryType.DATA
    if doc_score >= 2 and data_score == 0:
        return QueryType.DOCUMENT
    return None   # Unclear — use LLM


class HybridRouter(BaseModel):
    """
    Route queries to the correct retrieval path.

    Uses a two-stage approach:
    1. Fast heuristic classification (no API call; handles ~70% of queries)
    2. LLM classification for ambiguous queries (Gemini 3.5 Flash)

    Performance targets (AWB production):
    - Heuristic path:  < 1ms
    - LLM path:       < 400ms P95

    Args:
        model_id: LLM model identifier for fallback classification.
        use_heuristic_first: If True, attempt keyword classification
            before calling the LLM. Recommended for production.
    """

    model_id:             str  = "gemini-3.5-flash"
    use_heuristic_first:  bool = True

    model_config = {"frozen": True}

    def classify(
        self,
        query: str,
        llm_client=None,
    ) -> ClassificationResult:
        """
        Classify query as DOCUMENT, DATA, or HYBRID.

        Args:
            query: The regulatory question to classify.
            llm_client: Optional LLM client for fallback.
                If None and heuristic fails, defaults to DOCUMENT.

        Returns:
            ClassificationResult with type and reasoning.

        Raises:
            ValueError: If query is empty.
        """
        if not query or not query.strip():
            raise ValueError("Query must not be empty")

        import time
        t0 = time.monotonic()

        # Stage 1: fast heuristic
        if self.use_heuristic_first:
            heuristic = _heuristic_classify(query)
            if heuristic is not None:
                ms = (time.monotonic() - t0) * 1000
                logger.debug(
                    "Heuristic classify: %s -> %s (%.1fms)",
                    query[:60], heuristic, ms
                )
                return ClassificationResult(
                    query=query,
                    query_type=heuristic,
                    reasoning="Heuristic keyword match",
                    latency_ms=ms,
                )

        # Stage 2: LLM classification
        if llm_client is None:
            logger.warning(
                "No LLM client; defaulting ambiguous query to DOCUMENT"
            )
            ms = (time.monotonic() - t0) * 1000
            return ClassificationResult(
                query=query,
                query_type=QueryType.DOCUMENT,
                reasoning="Default (no LLM client available)",
                latency_ms=ms,
            )

        q_type, reasoning = self._llm_classify(query, llm_client)
        ms = (time.monotonic() - t0) * 1000
        logger.info(
            "LLM classify: %s -> %s (%.1fms)",
            query[:60], q_type, ms
        )
        return ClassificationResult(
            query=query,
            query_type=q_type,
            reasoning=reasoning,
            latency_ms=ms,
        )

    def _llm_classify(
        self,
        query: str,
        llm_client,
    ) -> tuple[QueryType, str]:
        """
        Use LLM to classify ambiguous query.

        Returns (QueryType, reasoning_string).
        """
        prompt = (
            "You are classifying a regulatory banking query into one "
            "of three categories.\n\n"
            "DOCUMENT: Needs regulatory text only "
            "(rules, definitions, requirements)\n"
            "DATA: Needs live AWB operational data only "
            "(ratios, positions, balances)\n"
            "HYBRID: Needs both regulatory text AND live data "
            "(compliance checks, threshold comparisons)\n\n"
            "Respond with exactly: TYPE: <type>\\nREASON: <one line>\n\n"
            f"Query: {query}"
        )
        try:
            response = llm_client.generate(prompt)
            lines = response.strip().splitlines()
            type_line = next(
                (l for l in lines if l.startswith("TYPE:")), ""
            )
            reason_line = next(
                (l for l in lines if l.startswith("REASON:")), ""
            )
            type_str = type_line.replace("TYPE:", "").strip().lower()
            reasoning = reason_line.replace("REASON:", "").strip()
            return QueryType(type_str), reasoning
        except Exception as exc:
            logger.warning(
                "LLM classification failed (%s); defaulting to DOCUMENT",
                exc
            )
            return QueryType.DOCUMENT, f"LLM error: {exc}"
