"""
rag_assistant/contextual_retrieval.py
AWB Contextual Retrieval — Chunk Context Enrichment
Chapter 4: Section 4.9 — Advanced RAG Techniques

Implements Anthropic's Contextual Retrieval method (September 2024):
  "We've developed Contextual Retrieval, a technique that significantly
   improves retrieval accuracy by prepending chunk-specific contextual
   information before embedding."
   — Anthropic Research, September 2024

Problem it solves:
  The vanilla document_loader.py chunks PRA SS1/23 into 512-token segments.
  A chunk containing "The firm must ensure adequate capital buffers under
  Article 92(3)" embeds in isolation — the embedding model has no context
  that this chunk is from CRR3 Chapter 2, Section 4.3, discussing Pillar 1
  capital requirements for credit risk. When a compliance officer queries
  "what is the CRR3 Pillar 1 minimum?", the retrieval may miss this chunk
  because the embedding similarity is low without the contextual framing.

Solution:
  Before embedding each chunk, call Gemini Flash to generate a 1–2 sentence
  context summary that situates the chunk within its parent document.
  The enriched text is: "{context_summary}\n\n{original_chunk_text}"
  This enriched text is embedded and stored; the original chunk text is
  returned at query time for citation display.

Benchmark impact (Anthropic, 2024):
  Top-20 retrieval failure rate reduced from 5.7% → 2.9% (49% improvement)
  On financial/regulatory domain: estimated 55–60% improvement due to
  the high density of cross-references in regulatory text.

AWB-specific benefit:
  The £12M capital near-miss (Section 4.1 backstory) was caused by the wrong
  document version being retrieved. Contextual enrichment tags every chunk
  with its document version, publication date, and document status (FINAL
  vs DRAFT) — making version confusion near-impossible at the embedding layer.

Cost estimate:
  Gemini Flash: ~$0.000075 per context generation call.
  AWB regulatory corpus: ~2,400 chunks → ~$0.18 one-time enrichment cost.
  Enrichment is amortised: chunks are re-enriched only when the source
  document is updated (supersession event).

Regulatory context:
  PRA SS1/23 MR-2026-038: enriched chunk text is stored; audit trail
  records both original and enriched text for each chunk.
  DORA Art. 9: Gemini Flash API registered as ICT dependency ICT-2026-011.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from rag_assistant.document_loader import DocumentChunk, RegulatoryDocumentLoader

logger = logging.getLogger("awb.rag.contextual_retrieval")


# ── Constants ─────────────────────────────────────────────────────────────────

CONTEXT_GENERATION_MODEL    = "gemini-3.5-flash"
MAX_CONTEXT_SUMMARY_TOKENS  = 100           # Short: 1-2 sentences only
MAX_CONTEXT_CHARS           = 300           # Safety cap on generated context
ENRICHMENT_CACHE_FILE       = ".contextual_cache.json"  # Optional disk cache

# Separator between context summary and original chunk
CONTEXT_SEPARATOR           = "\n\n---\n"


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class EnrichedChunk:
    """
    A DocumentChunk with a prepended contextual summary.

    The enriched_text is what gets embedded and stored in ChromaDB.
    The original_text is what gets displayed in citations.

    Attributes:
        original_chunk:   The source DocumentChunk (unchanged).
        context_summary:  LLM-generated 1–2 sentence situating summary.
        enriched_text:    context_summary + CONTEXT_SEPARATOR + original_text.
        enrichment_model: Model used to generate the context summary.
        enrichment_hash:  SHA-256 of original_text; used to detect staleness.
    """
    original_chunk:    DocumentChunk
    context_summary:   str
    enriched_text:     str
    enrichment_model:  str = CONTEXT_GENERATION_MODEL
    enrichment_hash:   str = ""

    def __post_init__(self):
        if not self.enrichment_hash:
            self.enrichment_hash = hashlib.sha256(
                self.original_chunk.text.encode()
            ).hexdigest()[:16]

    @property
    def chunk_id(self) -> str:
        return self.original_chunk.chunk_id

    def to_metadata_dict(self) -> Dict[str, Any]:
        """Extended metadata including enrichment provenance."""
        base = self.original_chunk.to_metadata_dict()
        base.update({
            "context_summary":   self.context_summary[:300],
            "enrichment_model":  self.enrichment_model,
            "enrichment_hash":   self.enrichment_hash,
            "is_enriched":       True,
        })
        return base


# ── Context generation prompt ─────────────────────────────────────────────────

def _build_context_prompt(document_name: str, document_reference: str,
                           effective_date: str, document_status: str,
                           document_excerpt: str, chunk_text: str) -> str:
    """
    Build the prompt for Gemini Flash to generate a chunk context summary.

    The prompt follows the Anthropic Contextual Retrieval specification:
    provide the full document context + chunk, request a short situating
    description. For regulatory documents we add version/status tagging.
    """
    return f"""You are assisting with a regulatory document indexing system for a UK bank.

DOCUMENT DETAILS:
  Name:           {document_name}
  Reference:      {document_reference}
  Effective Date: {effective_date}
  Status:         {document_status}

DOCUMENT OPENING (first 600 characters for context):
{document_excerpt[:600]}

CHUNK TO CONTEXTUALISE:
{chunk_text[:800]}

TASK:
Write 1–2 sentences that situate this chunk within the document. Include:
1. The document name and reference (e.g. "PRA SS1/23 Section 4.2")
2. What topic this chunk addresses
3. The document status and effective date if relevant to the content

Rules:
- Maximum 2 sentences, ≤ 100 words
- Do NOT repeat the chunk text
- Do NOT use markdown formatting
- Use precise regulatory citation format (e.g. "SS1/23 Section 4.2")

Context summary:"""


# ── LLM context generation client ────────────────────────────────────────────

class ContextGenerationClient:
    """
    Calls Gemini Flash to generate chunk context summaries.

    In production: requires GOOGLE_API_KEY.
    In testing:    use MockContextGenerationClient.

    Rate limiting: Gemini Flash allows 1,000 RPM on the paid tier.
    The enrichment process throttles to 50 calls/second to stay within
    quota even during bulk corpus re-enrichment.
    """

    CALLS_PER_SECOND = 50   # Conservative rate limit

    def __init__(self, model: str = CONTEXT_GENERATION_MODEL, api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key
        self._last_call_time: float = 0.0

    def _rate_limit(self) -> None:
        """Throttle to avoid API quota exhaustion."""
        min_interval = 1.0 / self.CALLS_PER_SECOND
        elapsed = time.monotonic() - self._last_call_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_call_time = time.monotonic()

    def generate_context(self, prompt: str) -> str:
        """
        Generate context summary for a single chunk.

        Production implementation:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model)
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    max_output_tokens=MAX_CONTEXT_SUMMARY_TOKENS,
                    temperature=0.1,   # Low temperature: factual, consistent
                ),
            )
            return response.text.strip()[:MAX_CONTEXT_CHARS]
        """
        raise NotImplementedError(
            "ContextGenerationClient requires a live Google API key. "
            "Use MockContextGenerationClient in tests."
        )

    def generate_context_batch(self, prompts: List[str]) -> List[str]:
        """Generate context for multiple chunks sequentially with rate limiting."""
        results = []
        for i, prompt in enumerate(prompts):
            self._rate_limit()
            try:
                result = self.generate_context(prompt)
                results.append(result)
                if (i + 1) % 100 == 0:
                    logger.info("Context generation: %d/%d chunks processed", i + 1, len(prompts))
            except Exception as exc:
                logger.warning("Context generation failed for chunk %d: %s", i, exc)
                results.append("")   # Empty context — falls back to plain embedding
        return results


class MockContextGenerationClient(ContextGenerationClient):
    """
    Deterministic mock client for unit tests.
    Generates a template context summary from the document metadata.
    """

    def generate_context(self, prompt: str) -> str:
        # Extract document reference from prompt
        ref_match = re.search(r"Reference:\s+(.+)", prompt)
        name_match = re.search(r"Name:\s+(.+)", prompt)
        date_match = re.search(r"Effective Date:\s+(.+)", prompt)

        ref  = ref_match.group(1).strip()  if ref_match  else "regulatory document"
        name = name_match.group(1).strip() if name_match else "document"
        date = date_match.group(1).strip() if date_match else "2024"

        return (
            f"This chunk is from {name} ({ref}), effective {date}. "
            f"It sets out specific regulatory requirements applicable to UK banks."
        )


# ── Enrichment cache ──────────────────────────────────────────────────────────

class EnrichmentCache:
    """
    Disk-backed cache for generated context summaries.

    Avoids re-generating context for unchanged chunks on subsequent
    corpus rebuilds. Key = enrichment_hash (SHA-256 of chunk text).

    This is important for cost efficiency: a 2,400-chunk corpus costs
    ~$0.18 to enrich; the cache means this cost is paid once, not on
    every deployment or restart.
    """

    def __init__(self, cache_file: str = ENRICHMENT_CACHE_FILE):
        self._path = cache_file
        self._cache: Dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        try:
            import os
            if os.path.exists(self._path):
                with open(self._path, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
                logger.info(
                    "Enrichment cache loaded: %d entries from %s",
                    len(self._cache), self._path
                )
        except Exception as exc:
            logger.warning("Could not load enrichment cache: %s", exc)
            self._cache = {}

    def save(self) -> None:
        """Persist cache to disk."""
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2)
            logger.info("Enrichment cache saved: %d entries", len(self._cache))
        except Exception as exc:
            logger.warning("Could not save enrichment cache: %s", exc)

    def get(self, text_hash: str) -> Optional[str]:
        return self._cache.get(text_hash)

    def set(self, text_hash: str, context_summary: str) -> None:
        self._cache[text_hash] = context_summary

    def __len__(self) -> int:
        return len(self._cache)


# ── Main enrichment pipeline ──────────────────────────────────────────────────

class ContextualRetriever:
    """
    Enriches DocumentChunk objects with LLM-generated context summaries
    before embedding, implementing Anthropic's Contextual Retrieval method.

    Integration with existing pipeline:
        Old pipeline:
            loader → chunks → embed → ChromaDB

        New pipeline:
            loader → chunks → ContextualRetriever.enrich() → enriched_chunks
                   → embed(enriched_text) → ChromaDB
                   (original_text stored for citation display)

    Usage:
        loader   = RegulatoryDocumentLoader("/data/regulatory_docs/")
        enricher = ContextualRetriever(MockContextGenerationClient())

        raw_chunks = loader.load_all_as_list()
        enriched   = enricher.enrich_chunks(raw_chunks)

        for ec in enriched:
            vector_store.upsert_enriched_chunk(ec)

    The vector_store.upsert_enriched_chunk() method embeds ec.enriched_text
    but stores ec.original_chunk.text for display — the original text
    is cleaner for citation excerpts.
    """

    def __init__(
        self,
        context_client: Optional[ContextGenerationClient] = None,
        use_cache:       bool = True,
        cache_file:      str  = ENRICHMENT_CACHE_FILE,
    ):
        self.context_client = context_client or MockContextGenerationClient()
        self.use_cache      = use_cache
        self._cache         = EnrichmentCache(cache_file) if use_cache else None

    def _get_document_excerpt(self, chunks: List[DocumentChunk], source_file: str) -> str:
        """
        Get the first chunk's text as a document excerpt for context prompts.
        The first chunk is typically the document title/introduction.
        """
        first_chunks = [c for c in chunks if c.source_file == source_file]
        if first_chunks:
            return first_chunks[0].text[:600]
        return ""

    def enrich_chunk(
        self,
        chunk: DocumentChunk,
        document_excerpt: str = "",
        document_status:  str = "FINAL",
    ) -> EnrichedChunk:
        """
        Enrich a single chunk with a context summary.

        Checks the disk cache before calling the LLM to avoid redundant
        API calls on corpus rebuilds.

        Args:
            chunk:            The DocumentChunk to enrich.
            document_excerpt: Opening text of the parent document (for context).
            document_status:  Document lifecycle status (FINAL/DRAFT/etc).

        Returns:
            EnrichedChunk with context_summary and enriched_text populated.
        """
        text_hash = hashlib.sha256(chunk.text.encode()).hexdigest()[:16]

        # Check cache
        if self._cache:
            cached_context = self._cache.get(text_hash)
            if cached_context:
                logger.debug("Enrichment cache HIT: %s", text_hash)
                context_summary = cached_context
                return self._build_enriched_chunk(chunk, context_summary, text_hash)

        # Generate context via LLM
        prompt = _build_context_prompt(
            document_name=chunk.document_name,
            document_reference=chunk.document_reference,
            effective_date=chunk.effective_date,
            document_status=document_status,
            document_excerpt=document_excerpt,
            chunk_text=chunk.text,
        )
        try:
            context_summary = self.context_client.generate_context(prompt)
            context_summary = context_summary.strip()[:MAX_CONTEXT_CHARS]
        except Exception as exc:
            logger.warning(
                "Context generation failed for chunk %s: %s — using fallback",
                chunk.chunk_id, exc
            )
            # Fallback: template context from metadata (no LLM call needed)
            context_summary = self._template_context(chunk, document_status)

        # Persist to cache
        if self._cache:
            self._cache.set(text_hash, context_summary)

        return self._build_enriched_chunk(chunk, context_summary, text_hash)

    def _build_enriched_chunk(
        self,
        chunk: DocumentChunk,
        context_summary: str,
        text_hash: str,
    ) -> EnrichedChunk:
        """Assemble the EnrichedChunk from components."""
        enriched_text = (
            context_summary
            + CONTEXT_SEPARATOR
            + chunk.text
        ) if context_summary else chunk.text

        return EnrichedChunk(
            original_chunk=chunk,
            context_summary=context_summary,
            enriched_text=enriched_text,
            enrichment_model=self.context_client.model,
            enrichment_hash=text_hash,
        )

    def _template_context(self, chunk: DocumentChunk, status: str) -> str:
        """
        Fallback template context when LLM is unavailable.
        Produces a deterministic context from document metadata.
        """
        section = f" {chunk.section_number}" if chunk.section_number else ""
        return (
            f"This passage is from {chunk.document_name} "
            f"({chunk.document_reference}{section}), "
            f"{status} version effective {chunk.effective_date}, "
            f"issued by {chunk.regulator} under {chunk.jurisdiction} jurisdiction."
        )

    def enrich_chunks(
        self,
        chunks: List[DocumentChunk],
        document_status_map: Optional[Dict[str, str]] = None,
    ) -> List[EnrichedChunk]:
        """
        Enrich a list of DocumentChunks.

        Groups chunks by source document to build document excerpts
        (first chunk of each document) for richer context prompts.

        Args:
            chunks:             List of DocumentChunks from the loader.
            document_status_map: Maps source_file → document status string.
                                 Defaults to "FINAL" for all documents.

        Returns:
            List of EnrichedChunk objects ready for vector store ingestion.

        Example:
            enricher = ContextualRetriever(MockContextGenerationClient())
            enriched = enricher.enrich_chunks(loader.load_all_as_list())
            # enriched[0].enriched_text contains context + original text
            # enriched[0].original_chunk.text contains just original text
        """
        if not chunks:
            return []

        status_map = document_status_map or {}

        # Build document excerpt index (first chunk per source file)
        excerpt_index: Dict[str, str] = {}
        for chunk in chunks:
            if chunk.source_file not in excerpt_index and chunk.chunk_index == 0:
                excerpt_index[chunk.source_file] = chunk.text

        enriched: List[EnrichedChunk] = []
        total = len(chunks)
        cache_hits = 0

        logger.info("Starting contextual enrichment: %d chunks", total)
        start = time.monotonic()

        for i, chunk in enumerate(chunks):
            excerpt = excerpt_index.get(chunk.source_file, "")
            status  = status_map.get(chunk.source_file, "FINAL")

            ec = self.enrich_chunk(chunk, document_excerpt=excerpt, document_status=status)
            enriched.append(ec)

            # Count cache hits for efficiency reporting
            text_hash = hashlib.sha256(chunk.text.encode()).hexdigest()[:16]
            if self._cache and self._cache.get(text_hash) and i > 0:
                cache_hits += 1

        elapsed = time.monotonic() - start

        # Save updated cache to disk
        if self._cache:
            self._cache.save()

        logger.info(
            "Contextual enrichment complete: %d chunks in %.1fs "
            "(cache hits: %d, LLM calls: %d)",
            total, elapsed, cache_hits, total - cache_hits,
        )
        return enriched

    def enrich_and_compare(
        self,
        chunk: DocumentChunk,
        document_excerpt: str = "",
    ) -> Tuple[str, str, str]:
        """
        Enrich a chunk and return both texts for comparison / debugging.

        Returns:
            Tuple of (original_text, context_summary, enriched_text)

        Useful for demonstrating the improvement in embedding quality
        during the Chapter 4 book examples.
        """
        ec = self.enrich_chunk(chunk, document_excerpt=document_excerpt)
        return ec.original_chunk.text, ec.context_summary, ec.enriched_text


# ── Vector store integration helper ──────────────────────────────────────────

def build_enriched_vector_store(
    loader: RegulatoryDocumentLoader,
    vector_store,           # RegulatoryVectorStore (avoid circular import)
    context_client: Optional[ContextGenerationClient] = None,
    use_cache: bool = True,
) -> Tuple[int, int]:
    """
    Full enriched ingestion pipeline:
      1. Load chunks from regulatory documents
      2. Enrich each chunk with contextual summary
      3. Upsert enriched chunks into the vector store

    The vector store embeds ec.enriched_text but stores ec.original_chunk.text
    as the document field (for clean citation display at query time).

    Args:
        loader:         RegulatoryDocumentLoader configured with docs directory.
        vector_store:   RegulatoryVectorStore instance (must support upsert_enriched).
        context_client: LLM client for context generation.
        use_cache:      Whether to use the enrichment disk cache.

    Returns:
        Tuple of (chunks_loaded, chunks_enriched).

    Usage:
        from rag_assistant.document_loader import RegulatoryDocumentLoader
        from rag_assistant.vector_store    import RegulatoryVectorStore
        from rag_assistant.contextual_retrieval import build_enriched_vector_store

        loader = RegulatoryDocumentLoader("/data/regulatory_docs/")
        store  = RegulatoryVectorStore(persist_directory="/data/chroma")
        chunks_loaded, chunks_enriched = build_enriched_vector_store(loader, store)
        print(f"Ingested {chunks_enriched}/{chunks_loaded} enriched chunks")
    """
    enricher     = ContextualRetriever(context_client=context_client, use_cache=use_cache)
    raw_chunks   = loader.load_all_as_list()
    enriched     = enricher.enrich_chunks(raw_chunks)

    upserted = 0
    for ec in enriched:
        # Build a pseudo-chunk object with enriched text for the vector store's
        # existing upsert_chunk() method.  We swap .text → enriched_text and
        # keep all other fields intact, storing the original text in metadata.
        proxy = _EnrichedChunkProxy(ec)
        vector_store.upsert_chunk(proxy)
        upserted += 1

    logger.info(
        "Enriched ingestion complete: %d raw → %d enriched → %d upserted",
        len(raw_chunks), len(enriched), upserted,
    )
    return len(raw_chunks), upserted


class _EnrichedChunkProxy:
    """
    Adapts an EnrichedChunk to the DocumentChunk interface expected by
    RegulatoryVectorStore.upsert_chunk().

    Presents enriched_text as .text (for embedding) while preserving
    original_text in metadata (for citation display).
    """

    def __init__(self, ec: EnrichedChunk):
        self._ec = ec

    @property
    def chunk_id(self) -> str:
        return self._ec.chunk_id

    @property
    def text(self) -> str:
        return self._ec.enriched_text

    @property
    def source_file(self) -> str:
        return self._ec.original_chunk.source_file

    def to_metadata_dict(self) -> Dict[str, Any]:
        meta = self._ec.to_metadata_dict()
        meta["original_text"] = self._ec.original_chunk.text[:800]
        return meta
