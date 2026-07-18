"""
AWB Regulatory Knowledge Assistant
Chapter 4: Retrieval-Augmented Generation for Compliance

Avon & Wessex Bank plc — RAG-powered regulatory knowledge base.
Model registration: MR-2026-038 (PRA SS1/23)

Module inventory (v2.0.0):
  document_loader.py       — Token-aware chunking, regulatory document registry
  vector_store.py          — ChromaDB semantic search, cosine similarity
  query_engine.py          — Full RAG pipeline with hallucination guard + reranking
  rag_memory.py            — Three-layer memory (Redis cache, session, PostgreSQL)
  contextual_retrieval.py  — Anthropic contextual enrichment before embedding
  agentic_rag.py           — Multi-hop ReAct retrieval for complex queries
"""

from rag_assistant.document_loader import (
    RegulatoryDocumentLoader,
    DocumentChunk,
    LoadedDocument,
    chunk_text,
    REGULATORY_DOCUMENT_REGISTRY,
)
from rag_assistant.vector_store import (
    RegulatoryVectorStore,
    SearchResult,
    EmbeddingProvider,
    MockEmbeddingProvider,
)
from rag_assistant.query_engine import (
    RegulatoryQueryEngine,
    RegulatoryAnswer,
    Citation,
    MockLLMGenerationClient,
)

# ── v2.0: Memory layer ────────────────────────────────────────────────────────
from rag_assistant.rag_memory import (
    RAGMemory,
    RedisQueryCache,
    SessionMemory,
    UserPreferenceMemory,
    SessionTurn,
    UserPreference,
    QueryAuditRecord,
    get_rag_memory,
)

# ── v2.0: Contextual retrieval ────────────────────────────────────────────────
from rag_assistant.contextual_retrieval import (
    ContextualRetriever,
    EnrichedChunk,
    ContextGenerationClient,
    MockContextGenerationClient,
    EnrichmentCache,
    build_enriched_vector_store,
)

# ── v2.0: Agentic RAG ─────────────────────────────────────────────────────────
from rag_assistant.agentic_rag import (
    AgenticRAGEngine,
    AgenticRAGTools,
    AgenticRAGResult,
    RetrievalHop,
    RetrievalAction,
    HopStatus,
)

__all__ = [
    # Document loading
    "RegulatoryDocumentLoader",
    "DocumentChunk",
    "LoadedDocument",
    "chunk_text",
    "REGULATORY_DOCUMENT_REGISTRY",
    # Vector store
    "RegulatoryVectorStore",
    "SearchResult",
    "EmbeddingProvider",
    "MockEmbeddingProvider",
    # Query engine
    "RegulatoryQueryEngine",
    "RegulatoryAnswer",
    "Citation",
    "MockLLMGenerationClient",
    # Memory layer (v2.0)
    "RAGMemory",
    "RedisQueryCache",
    "SessionMemory",
    "UserPreferenceMemory",
    "SessionTurn",
    "UserPreference",
    "QueryAuditRecord",
    "get_rag_memory",
    # Contextual retrieval (v2.0)
    "ContextualRetriever",
    "EnrichedChunk",
    "ContextGenerationClient",
    "MockContextGenerationClient",
    "EnrichmentCache",
    "build_enriched_vector_store",
    # Agentic RAG (v2.0)
    "AgenticRAGEngine",
    "AgenticRAGTools",
    "AgenticRAGResult",
    "RetrievalHop",
    "RetrievalAction",
    "HopStatus",
]

__version__ = "2.0.0"
__model_registration__ = "MR-2026-038"
