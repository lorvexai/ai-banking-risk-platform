"""
rag_assistant/query_engine.py
AWB Regulatory Knowledge Assistant — RAG Query Engine
Chapter 4: Retrieval-Augmented Generation for Compliance

Implements the full RAG pipeline:
    Query → Embed → Retrieve → Generate → Ground → Return

Hallucination guard:
    If no retrieved context scores above MIN_RELEVANCE_SCORE (0.70), returns
    the standard uncertainty response rather than generating an answer.
    This prevents the assistant from inventing regulatory guidance.

System prompt architecture (4-component, Chapter 2 standard):
    1. Role + Context
    2. Regulatory constraints
    3. Output format
    4. Explicit limitations

Regulatory context:
- PRA SS1/23: LLM used in regulatory advisory context = registered model (MR-2026-038)
- FCA PS22/9: Responses must be explainable and avoid misleading customers
- UK GDPR: No personal data in regulatory queries; system handles policy only
"""

from __future__ import annotations

import datetime
import json
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from rag_assistant.vector_store import (
    RegulatoryVectorStore,
    SearchResult,
    MockEmbeddingProvider,
    MIN_RELEVANCE_SCORE,
)

logger = logging.getLogger("awb.rag.query_engine")

# PRA SS1/23 model registration
MODEL_REGISTRATION = "MR-2026-038"
LLM_MODEL = "gemini-3.5-flash"

# Standard uncertainty response (hallucination guard)
UNCERTAINTY_RESPONSE = (
    "I cannot find specific guidance on this topic in AWB's regulatory library. "
    "Please consult the relevant regulatory document directly, or contact AWB's "
    "Regulatory Affairs team for authoritative guidance."
)


# ---------------------------------------------------------------------------
# Pydantic output model
# ---------------------------------------------------------------------------

class Citation(BaseModel):
    """A single regulatory citation supporting a response."""
    document_reference: str = Field(..., description="Regulatory document reference (e.g. 'SS1/23')")
    document_name: str = Field(..., description="Full document name")
    section_number: Optional[str] = Field(None, description="Section number if identified")
    source_file: str = Field(..., description="Source filename")
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    excerpt: str = Field(..., description="Relevant excerpt from the source (≤ 300 chars)")

    class Config:
        frozen = True


class RegulatoryAnswer(BaseModel):
    """
    Structured answer from the AWB Regulatory Knowledge Assistant.

    Pydantic validation ensures:
    - At least one citation is always present (grounding requirement)
    - Confidence is within valid range
    - Caveats are provided for anything other than high-confidence answers

    Regulatory context:
    - PRA SS1/23: Model output logged with full traceability (MR-2026-038)
    - FCA PS22/9: Answer must be explainable and include source citations
    """
    query: str = Field(..., description="The original query")
    answer: str = Field(..., min_length=10, description="The generated answer")
    citations: List[Citation] = Field(
        ...,
        min_length=0,
        description="Source citations supporting the answer",
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Confidence score based on retrieval relevance",
    )
    caveats: List[str] = Field(
        default_factory=list,
        description="Important caveats or limitations on this answer",
    )
    is_uncertainty_response: bool = Field(
        default=False,
        description="True if the hallucination guard triggered (no relevant context found)",
    )
    model_registration: str = Field(
        default=MODEL_REGISTRATION,
        description="PRA SS1/23 model registration reference",
    )
    generated_at: str = Field(
        default_factory=lambda: datetime.datetime.utcnow().isoformat() + "Z",
    )

    @model_validator(mode="after")
    def citations_required_unless_uncertainty(self) -> "RegulatoryAnswer":
        """
        Non-uncertainty answers must have at least one citation.
        Grounding requirement: every factual statement must be traceable to source.
        Uses model_validator (after) so is_uncertainty_response is already set.
        """
        if not self.is_uncertainty_response and len(self.citations) == 0:
            raise ValueError(
                "Non-uncertainty answers must include at least one citation. "
                "Grounding requirement: all regulatory guidance must be traceable."
            )
        return self

    @model_validator(mode="after")
    def caveats_required_for_low_confidence(self) -> "RegulatoryAnswer":
        """Low-confidence answers must include at least one caveat."""
        if self.confidence < 0.75 and not self.is_uncertainty_response and len(self.caveats) == 0:
            raise ValueError(
                "Answers with confidence < 0.75 must include at least one caveat."
            )
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "citations": [c.model_dump() for c in self.citations],
            "confidence": round(self.confidence, 3),
            "caveats": self.caveats,
            "is_uncertainty_response": self.is_uncertainty_response,
            "model_registration": self.model_registration,
            "generated_at": self.generated_at,
        }


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

def build_system_prompt(context_chunks: List[SearchResult]) -> str:
    """
    Build the 4-component system prompt for the LLM.

    Chapter 2 standard 4-component architecture:
    1. Role + Context
    2. Regulatory constraints
    3. Output format
    4. Explicit limitations

    Args:
        context_chunks: Retrieved regulatory document chunks.

    Returns:
        System prompt string.
    """
    context_text = "\n\n---\n\n".join(
        f"[SOURCE: {r.document_name} | {r.metadata.get('document_reference', '')} | "
        f"Section: {r.section_number or 'N/A'} | Score: {r.relevance_score:.2f}]\n{r.text}"
        for r in context_chunks
    )

    return f"""## COMPONENT 1: ROLE AND CONTEXT
You are the AWB Regulatory Knowledge Assistant, an AI system deployed by Avon & Wessex
Regional Bank (AWB) to help the compliance team understand regulatory requirements.
You are registered under PRA SS1/23 model inventory as MR-2026-038.

Your knowledge is strictly limited to the regulatory documents retrieved from AWB's
regulatory library. You do not use general knowledge to answer regulatory questions.

## COMPONENT 2: REGULATORY CONSTRAINTS
- ALWAYS ground your answers in the retrieved context below.
- ALWAYS cite the source document, section, and document reference.
- NEVER speculate or extrapolate beyond what the retrieved documents state.
- NEVER present regulatory guidance as definitive legal advice.
- If context is insufficient, say so explicitly — do not invent regulatory text.
- Use British English spelling and PRA/FCA/EBA citation format throughout.
- All monetary figures should be in GBP (£) unless the source document specifies otherwise.

## COMPONENT 3: OUTPUT FORMAT
Provide a structured response with:
1. A clear, concise answer (2–4 paragraphs) written in plain language
2. Specific citations to source documents (document reference, section number)
3. Any important caveats or limitations
4. Do NOT reproduce lengthy verbatim extracts from source documents

## COMPONENT 4: EXPLICIT LIMITATIONS
- This assistant provides regulatory information, NOT legal advice.
- Always refer complex compliance questions to AWB's Regulatory Affairs team.
- Regulatory requirements change; verify against the latest published version.
- This system is subject to PRA SS1/23 model risk governance (MR-2026-038).

---

## RETRIEVED REGULATORY CONTEXT

{context_text}

---

Answer the user's question using ONLY the context above. If the context does not
contain sufficient information, state this clearly."""


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

class LLMGenerationClient:
    """
    Abstraction over Gemini 3.5 Flash for response generation.

    In production: uses Google Generative AI SDK.
    In testing: MockLLMGenerationClient returns deterministic responses.
    """

    def __init__(self, model: str = LLM_MODEL, api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key

    def generate(self, system_prompt: str, user_query: str) -> str:
        """
        Generate a response using Gemini 3.5 Flash.

        Production implementation:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(
                model_name=self.model,
                system_instruction=system_prompt,
            )
            response = model.generate_content(user_query)
            return response.text
        """
        raise NotImplementedError(
            "LLMGenerationClient.generate() requires a live Google API key. "
            "Use MockLLMGenerationClient in tests."
        )


class MockLLMGenerationClient(LLMGenerationClient):
    """
    Mock LLM client that returns a deterministic response for testing.
    Response references the first source document from the prompt.
    """

    def generate(self, system_prompt: str, user_query: str) -> str:
        # Extract first source reference from system prompt for realism
        import re
        source_match = re.search(r"\[SOURCE: ([^\|]+) \| ([^\|]+) \|", system_prompt)
        if source_match:
            doc_name = source_match.group(1).strip()
            doc_ref = source_match.group(2).strip()
            return (
                f"Based on the regulatory guidance in {doc_name} ({doc_ref}), "
                f"the relevant requirements are as follows. Firms must ensure "
                f"compliance with the applicable provisions as set out in the "
                f"retrieved context. The key obligations include maintaining "
                f"appropriate governance, documentation, and oversight mechanisms "
                f"as specified in the relevant regulatory text. Please refer to "
                f"the cited sections for the precise wording of the requirements."
            )
        return (
            "Based on the retrieved regulatory context, the relevant requirements "
            "are as set out in the applicable regulatory documents. Firms should "
            "ensure compliance with the specific provisions cited."
        )


# ---------------------------------------------------------------------------
# Query engine
# ---------------------------------------------------------------------------

class RegulatoryQueryEngine:
    """
    Full RAG pipeline for AWB's Regulatory Knowledge Assistant.

    Pipeline:
        1. Embed query (via vector store's embedding provider)
        2. Retrieve top-k relevant chunks (ChromaDB semantic search)
        3. Apply hallucination guard (check relevance scores)
        4. Build 4-component system prompt
        5. Generate answer (Gemini 3.5 Flash)
        6. Construct RegulatoryAnswer with citations
        7. Return validated Pydantic model

    Usage:
        engine = RegulatoryQueryEngine(vector_store, llm_client)
        answer = engine.query("What are PRA's model validation requirements?")
        print(answer.answer)
        for citation in answer.citations:
            print(citation.document_reference, citation.section_number)
    """

    def __init__(
        self,
        vector_store: RegulatoryVectorStore,
        llm_client: Optional[LLMGenerationClient] = None,
        min_relevance_score: float = MIN_RELEVANCE_SCORE,
        top_k: int = 5,
    ):
        self.vector_store = vector_store
        self.llm_client = llm_client or MockLLMGenerationClient()
        self.min_relevance_score = min_relevance_score
        self.top_k = top_k

    def query(
        self,
        user_query: str,
        regulator_filter: Optional[str] = None,
    ) -> RegulatoryAnswer:
        """
        Process a regulatory query end-to-end.

        Args:
            user_query: Natural language question about regulatory requirements.
            regulator_filter: Optional filter to restrict retrieval to a specific
                regulator (e.g. "PRA", "FCA", "EBA", "EC").

        Returns:
            RegulatoryAnswer with grounded response, citations, and confidence.
        """
        if not user_query.strip():
            raise ValueError("Query cannot be empty.")

        logger.info("RAG query: %r (filter: %s)", user_query[:100], regulator_filter)

        # Step 1 & 2: Retrieve relevant chunks
        retrieved = self.vector_store.search(
            query=user_query,
            top_k=self.top_k,
            regulator_filter=regulator_filter,
        )

        # Step 3: Hallucination guard
        relevant = [r for r in retrieved if r.relevance_score >= self.min_relevance_score]

        if not relevant:
            logger.info(
                "Hallucination guard triggered: no results above threshold %.2f",
                self.min_relevance_score,
            )
            return RegulatoryAnswer(
                query=user_query,
                answer=UNCERTAINTY_RESPONSE,
                citations=[],
                confidence=0.0,
                caveats=[
                    "No sufficiently relevant regulatory guidance was found for this query.",
                    "Please consult the regulatory document directly.",
                ],
                is_uncertainty_response=True,
            )

        # Step 4: Build system prompt with retrieved context
        system_prompt = build_system_prompt(relevant)

        # Step 5: Generate answer
        try:
            raw_answer = self.llm_client.generate(
                system_prompt=system_prompt,
                user_query=user_query,
            )
        except Exception as exc:
            logger.error("LLM generation failed: %s", exc)
            # Graceful degradation: return retrieved context as answer
            raw_answer = (
                "The language model is currently unavailable. "
                "The following regulatory context was retrieved:\n\n" +
                "\n\n".join(r.text[:200] + "..." for r in relevant[:2])
            )

        # Step 6: Build citations from retrieved chunks
        citations = []
        for result in relevant[:5]:  # Cap at 5 citations
            excerpt = result.text[:280] + "..." if len(result.text) > 280 else result.text
            citations.append(Citation(
                document_reference=result.metadata.get("document_reference", result.document_name),
                document_name=result.document_name,
                section_number=result.section_number or None,
                source_file=result.source_file,
                relevance_score=result.relevance_score,
                excerpt=excerpt,
            ))

        # Step 7: Compute confidence from top relevance score
        top_score = relevant[0].relevance_score if relevant else 0.0
        confidence = min(top_score, 1.0)

        # Add caveats for lower-confidence answers
        caveats = [
            "This response is based on extracted regulatory text and does not constitute legal advice.",
            "Always verify against the latest published version of the regulatory document.",
        ]
        if confidence < 0.80:
            caveats.append(
                f"Retrieval confidence is {confidence:.0%}. "
                "Consider rephrasing your query for more targeted results."
            )

        return RegulatoryAnswer(
            query=user_query,
            answer=raw_answer,
            citations=citations,
            confidence=confidence,
            caveats=caveats,
            is_uncertainty_response=False,
        )

    def query_with_logging(
        self,
        user_query: str,
        regulator_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Query with full audit logging (PRA SS1/23 Section 4.2).

        Returns the answer as a dict suitable for storage in the audit database.
        Audit records retained for 7 years per UK statutory minimum.
        """
        answer = self.query(user_query, regulator_filter=regulator_filter)
        log_entry = {
            **answer.to_dict(),
            "model_registration": MODEL_REGISTRATION,
            "model_name": self.llm_client.model,
        }
        logger.info(
            "RAG audit log | confidence=%.2f | citations=%d | uncertainty=%s",
            answer.confidence,
            len(answer.citations),
            answer.is_uncertainty_response,
        )
        return log_entry
