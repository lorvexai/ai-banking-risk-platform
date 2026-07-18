"""
rag_assistant/vector_store.py
AWB Regulatory Knowledge Assistant — Vector Store
Chapter 4: Retrieval-Augmented Generation for Compliance

Manages ChromaDB collections for regulatory document embeddings.
Supports semantic search with metadata filtering by regulator.

ChromaDB collection: "awb_regulatory_knowledge"
Embedding model: Google text-embedding-004 (768 dimensions)
Top-k retrieval: 5 results (configurable)

DORA Article 15: Third-party AI dependencies must be documented.
This module uses Google's embedding API; AWB must register this as an
external dependency in the ICT asset inventory.

Production note: Switch persist_directory from None (in-memory) to
a path on AWB's NAS share for persistent storage between restarts.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("awb.rag.vector_store")

# Default constants
DEFAULT_COLLECTION_NAME = "awb_regulatory_knowledge"
DEFAULT_TOP_K = 5
MIN_RELEVANCE_SCORE = 0.70       # Hallucination guard threshold (cosine similarity)
EMBEDDING_MODEL = "models/text-embedding-004"

# Supported regulators for metadata filtering
VALID_REGULATORS = {"PRA", "FCA", "EC", "EBA", "UNKNOWN"}


# ---------------------------------------------------------------------------
# Embedding provider abstraction
# ---------------------------------------------------------------------------

class EmbeddingProvider:
    """
    Abstraction over Google's text-embedding-004 model.

    In production: requires GOOGLE_API_KEY environment variable.
    In testing: use MockEmbeddingProvider to avoid live API calls.

    DORA compliance: if Google's API is unavailable, the provider raises
    EmbeddingProviderError so the query engine can apply fallback logic.
    """

    def __init__(self, model: str = EMBEDDING_MODEL, api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key

    def embed(self, text: str) -> List[float]:
        """
        Embed a single text string.

        Args:
            text: Text to embed (will be truncated to model's context limit).

        Returns:
            List of floats (768 dimensions for text-embedding-004).

        Raises:
            RuntimeError: If the embedding API call fails.
        """
        # Production implementation:
        # import google.generativeai as genai
        # genai.configure(api_key=self.api_key or os.environ["GOOGLE_API_KEY"])
        # result = genai.embed_content(
        #     model=self.model,
        #     content=text,
        #     task_type="retrieval_document",
        # )
        # return result["embedding"]
        raise NotImplementedError(
            "EmbeddingProvider.embed() requires a live Google API key. "
            "Use MockEmbeddingProvider in tests."
        )

    def embed_query(self, query: str) -> List[float]:
        """Embed a query string (uses task_type='retrieval_query' for better results)."""
        # Production: genai.embed_content(task_type="retrieval_query", ...)
        raise NotImplementedError(
            "EmbeddingProvider.embed_query() requires a live Google API key."
        )

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts in a single API call."""
        raise NotImplementedError(
            "EmbeddingProvider.embed_batch() requires a live Google API key."
        )


class MockEmbeddingProvider(EmbeddingProvider):
    """
    Deterministic mock embedding provider for unit tests.

    Produces consistent 768-dimensional embeddings without API calls.
    Similarity is computed on character-hash basis — not semantically
    meaningful, but sufficient for testing retrieval logic.
    """

    EMBEDDING_DIM = 768

    def embed(self, text: str) -> List[float]:
        """Produce a deterministic pseudo-embedding from text hash."""
        import hashlib
        digest = hashlib.sha256(text.encode()).digest()
        # Expand 32-byte digest to 768 floats using cycling
        floats = []
        for i in range(self.EMBEDDING_DIM):
            byte_val = digest[i % 32]
            floats.append((byte_val / 255.0) - 0.5)  # Normalise to [-0.5, 0.5]
        return floats

    def embed_query(self, query: str) -> List[float]:
        return self.embed(query)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]


# ---------------------------------------------------------------------------
# Search result
# ---------------------------------------------------------------------------

class SearchResult:
    """A single semantic search result with text, metadata, and relevance score."""

    def __init__(
        self,
        chunk_id: str,
        text: str,
        metadata: Dict[str, Any],
        relevance_score: float,
    ):
        self.chunk_id = chunk_id
        self.text = text
        self.metadata = metadata
        self.relevance_score = relevance_score

    @property
    def document_name(self) -> str:
        return self.metadata.get("document_name", "Unknown Document")

    @property
    def regulator(self) -> str:
        return self.metadata.get("regulator", "UNKNOWN")

    @property
    def section_number(self) -> str:
        return self.metadata.get("section_number", "")

    @property
    def source_file(self) -> str:
        return self.metadata.get("source_file", "")

    def to_citation(self) -> str:
        """Format as a regulatory citation string."""
        ref = self.metadata.get("document_reference", self.document_name)
        section = f", {self.section_number}" if self.section_number else ""
        return f"{ref}{section} (relevance: {self.relevance_score:.2f})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text[:200] + "..." if len(self.text) > 200 else self.text,
            "metadata": self.metadata,
            "relevance_score": round(self.relevance_score, 4),
        }

    def __repr__(self) -> str:
        return (
            f"SearchResult(doc={self.document_name!r}, "
            f"score={self.relevance_score:.3f}, "
            f"section={self.section_number!r})"
        )


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

class RegulatoryVectorStore:
    """
    ChromaDB-backed vector store for AWB's regulatory knowledge base.

    Supports:
    - Upsert of DocumentChunk objects (idempotent)
    - Semantic search with optional metadata filtering by regulator
    - Relevance score filtering (hallucination guard: threshold 0.70)

    Usage (production — persistent):
        store = RegulatoryVectorStore(persist_directory="/data/chroma")
        store.build_from_loader(loader)
        results = store.search("what are PRA model validation requirements?")

    Usage (testing — in-memory):
        store = RegulatoryVectorStore(
            persist_directory=None,
            embedding_provider=MockEmbeddingProvider(),
        )
    """

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        persist_directory: Optional[str] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        top_k: int = DEFAULT_TOP_K,
        min_relevance_score: float = MIN_RELEVANCE_SCORE,
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_provider = embedding_provider or EmbeddingProvider()
        self.top_k = top_k
        self.min_relevance_score = min_relevance_score
        self._collection = None
        self._client = None
        self._initialised = False

    def _get_client(self):
        """Lazily initialise ChromaDB client."""
        if self._client is None:
            import chromadb
            if self.persist_directory:
                self._client = chromadb.PersistentClient(path=self.persist_directory)
            else:
                self._client = chromadb.EphemeralClient()
        return self._client

    def _get_collection(self):
        """Lazily initialise ChromaDB collection."""
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    @property
    def collection(self):
        return self._get_collection()

    def upsert_chunk(self, chunk) -> None:
        """
        Upsert a single DocumentChunk into the vector store.

        Upsert is idempotent: re-ingesting the same document does not
        create duplicates (ChromaDB upsert semantics).

        Args:
            chunk: DocumentChunk object (from document_loader).
        """
        if not chunk.text.strip():
            logger.warning("Skipping empty chunk: %s", chunk.chunk_id)
            return

        embedding = self.embedding_provider.embed(chunk.text)
        metadata = chunk.to_metadata_dict()

        self._get_collection().upsert(
            ids=[chunk.chunk_id],
            embeddings=[embedding],
            documents=[chunk.text],
            metadatas=[metadata],
        )
        logger.debug("Upserted chunk: %s (%s)", chunk.chunk_id, chunk.source_file)

    def upsert_chunks(self, chunks: List) -> int:
        """
        Upsert a list of DocumentChunk objects.

        Args:
            chunks: List of DocumentChunk objects.

        Returns:
            Number of chunks successfully upserted.
        """
        if not chunks:
            return 0

        # Batch for efficiency
        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for chunk in chunks:
            if not chunk.text.strip():
                continue
            embedding = self.embedding_provider.embed(chunk.text)
            ids.append(chunk.chunk_id)
            embeddings.append(embedding)
            documents.append(chunk.text)
            metadatas.append(chunk.to_metadata_dict())

        if ids:
            self._get_collection().upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            logger.info("Upserted %d chunks into collection '%s'", len(ids), self.collection_name)

        return len(ids)

    def build_from_loader(self, loader) -> int:
        """
        Ingest all documents from a RegulatoryDocumentLoader.

        Args:
            loader: RegulatoryDocumentLoader instance.

        Returns:
            Total chunks ingested.
        """
        chunks = loader.load_all_as_list()
        return self.upsert_chunks(chunks)

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        regulator_filter: Optional[str] = None,
        jurisdiction_filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Perform semantic search over the regulatory knowledge base.

        Args:
            query: Natural language query (e.g. "PRA model validation requirements").
            top_k: Number of results to return (defaults to self.top_k).
            regulator_filter: Filter by regulator code (e.g. "PRA", "FCA", "EBA").
            jurisdiction_filter: Filter by jurisdiction ("UK" or "EU").

        Returns:
            List of SearchResult objects, sorted by relevance score descending.
            Only results with relevance_score >= min_relevance_score are returned
            (hallucination guard).
        """
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if regulator_filter and regulator_filter not in VALID_REGULATORS:
            raise ValueError(
                f"Invalid regulator_filter '{regulator_filter}'. "
                f"Valid options: {VALID_REGULATORS}"
            )

        k = top_k or self.top_k
        query_embedding = self.embedding_provider.embed_query(query)

        # Build ChromaDB where clause for metadata filtering
        where_clause = None
        if regulator_filter and jurisdiction_filter:
            where_clause = {
                "$and": [
                    {"regulator": {"$eq": regulator_filter}},
                    {"jurisdiction": {"$eq": jurisdiction_filter}},
                ]
            }
        elif regulator_filter:
            where_clause = {"regulator": {"$eq": regulator_filter}}
        elif jurisdiction_filter:
            where_clause = {"jurisdiction": {"$eq": jurisdiction_filter}}

        query_kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where_clause:
            query_kwargs["where"] = where_clause

        try:
            raw = self._get_collection().query(**query_kwargs)
        except Exception as exc:
            logger.error("ChromaDB query failed: %s", exc)
            raise RuntimeError(f"Vector store query failed: {exc}") from exc

        results: List[SearchResult] = []
        ids = raw.get("ids", [[]])[0]
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        for chunk_id, doc, meta, distance in zip(ids, documents, metadatas, distances):
            # ChromaDB cosine distance ∈ [0, 2]; convert to similarity ∈ [-1, 1]
            # Similarity = 1 - (distance / 2)  → maps [0,2] to [1,0]
            similarity = 1.0 - (distance / 2.0)
            results.append(SearchResult(
                chunk_id=chunk_id,
                text=doc,
                metadata=meta,
                relevance_score=round(similarity, 4),
            ))

        # Sort by relevance descending
        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results

    def search_above_threshold(
        self,
        query: str,
        top_k: Optional[int] = None,
        regulator_filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Search and return only results above the minimum relevance threshold.

        This is the entry point for the hallucination guard:
        if this returns an empty list, the query engine should return the
        uncertainty response rather than generating a potentially hallucinated answer.
        """
        results = self.search(query, top_k=top_k, regulator_filter=regulator_filter)
        return [r for r in results if r.relevance_score >= self.min_relevance_score]

    def count(self) -> int:
        """Return the number of chunks in the collection."""
        return self._get_collection().count()

    def delete_collection(self) -> None:
        """Delete the collection (for testing teardown)."""
        try:
            self._get_client().delete_collection(self.collection_name)
            self._collection = None
        except Exception:
            pass  # Collection may not exist

    def get_chunk_by_id(self, chunk_id: str) -> Optional[SearchResult]:
        """Retrieve a specific chunk by its ID."""
        try:
            raw = self._get_collection().get(
                ids=[chunk_id],
                include=["documents", "metadatas"],
            )
            if raw["ids"]:
                return SearchResult(
                    chunk_id=raw["ids"][0],
                    text=raw["documents"][0],
                    metadata=raw["metadatas"][0],
                    relevance_score=1.0,
                )
        except Exception:
            pass
        return None
