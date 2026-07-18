"""
rag_assistant/document_loader.py
AWB Regulatory Knowledge Assistant — Document Loader
Chapter 4: Retrieval-Augmented Generation for Compliance

Loads regulatory documents from text files, applies token-aware chunking,
and produces metadata-enriched document chunks ready for vector store ingestion.

Chunking strategy:
- Target chunk size: 512 tokens (≈ 2,048 characters at ~4 chars/token)
- Overlap: 64 tokens (≈ 256 characters) — preserves cross-chunk context
- Chunk boundaries respect paragraph breaks where possible

Regulatory documents supported:
- PRA SS1/23 (Model Risk Management)
- FCA PS22/9 (Consumer Duty)
- EU AI Act 2024 — Annex III and Articles 9–14
- DORA — ICT Risk Management (Articles 5–16)

British English throughout; PRA/FCA/EBA citation format.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHUNK_TOKEN_SIZE = 512          # Target tokens per chunk
CHUNK_OVERLAP_TOKENS = 64       # Overlap tokens between adjacent chunks
CHARS_PER_TOKEN_APPROX = 4      # Approximate characters per token (GPT-4 style)

CHUNK_SIZE_CHARS = CHUNK_TOKEN_SIZE * CHARS_PER_TOKEN_APPROX       # 2048 chars
CHUNK_OVERLAP_CHARS = CHUNK_OVERLAP_TOKENS * CHARS_PER_TOKEN_APPROX  # 256 chars

# Registry of known regulatory documents with metadata
REGULATORY_DOCUMENT_REGISTRY: Dict[str, Dict] = {
    "PRA_SS1_23_extract.txt": {
        "document_name": "PRA SS1/23 — Model Risk Management Principles for Banks",
        "regulator": "PRA",
        "document_reference": "SS1/23",
        "effective_date": "2023-05-17",
        "jurisdiction": "UK",
        "document_type": "Supervisory Statement",
    },
    "FCA_PS22_9_extract.txt": {
        "document_name": "FCA PS22/9 — A New Consumer Duty",
        "regulator": "FCA",
        "document_reference": "PS22/9",
        "effective_date": "2023-07-31",
        "jurisdiction": "UK",
        "document_type": "Policy Statement",
    },
    "EU_AI_Act_Annex_III_extract.txt": {
        "document_name": "EU AI Act 2024 — Annex III and Articles 9–14",
        "regulator": "EC",
        "document_reference": "EU AI Act 2024",
        "effective_date": "2024-08-02",
        "jurisdiction": "EU",
        "document_type": "Regulation",
    },
    "DORA_ICT_extract.txt": {
        "document_name": "DORA — ICT Risk Management (Articles 5–16)",
        "regulator": "EBA",
        "document_reference": "DORA 2022/2554",
        "effective_date": "2025-01-17",
        "jurisdiction": "EU",
        "document_type": "Regulation",
    },
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DocumentChunk:
    """
    A single chunk of regulatory text with metadata.

    chunk_id is a stable deterministic UUID derived from document_name + chunk_index,
    ensuring idempotent upserts into the vector store.
    """
    chunk_id: str
    text: str
    document_name: str
    regulator: str
    document_reference: str
    effective_date: str
    jurisdiction: str
    document_type: str
    source_file: str
    chunk_index: int
    total_chunks: int
    section_number: Optional[str] = None
    approximate_tokens: int = 0

    def to_metadata_dict(self) -> Dict:
        """Serialise metadata for ChromaDB storage."""
        return {
            "document_name": self.document_name,
            "regulator": self.regulator,
            "document_reference": self.document_reference,
            "effective_date": self.effective_date,
            "jurisdiction": self.jurisdiction,
            "document_type": self.document_type,
            "source_file": self.source_file,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "section_number": self.section_number or "",
            "approximate_tokens": self.approximate_tokens,
        }


@dataclass
class LoadedDocument:
    """A fully loaded regulatory document with all its chunks."""
    source_file: str
    document_name: str
    regulator: str
    total_chunks: int
    chunks: List[DocumentChunk] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Section detection
# ---------------------------------------------------------------------------

# Patterns that indicate the start of a new section
_SECTION_PATTERNS = [
    re.compile(r"^(SECTION\s+\d+[\.\d]*[:.])", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^(\d+\.\d+[\.\d]*\s+[A-Z][A-Za-z\s]+)", re.MULTILINE),
    re.compile(r"^(ARTICLE\s+\d+[\.\d]*[:.])", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^(ANNEX\s+[IVX]+[:.])", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^([A-Z][A-Z\s]{5,})\s*$", re.MULTILINE),  # ALL-CAPS headings
]


def _detect_section(text: str) -> Optional[str]:
    """
    Attempt to detect the section number/heading for a text chunk.
    Returns the first match found, or None.
    """
    for pattern in _SECTION_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).strip()[:100]  # Truncate to 100 chars
    return None


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _split_into_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs on double newlines."""
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def _approx_tokens(text: str) -> int:
    """Approximate token count using character-based heuristic."""
    return max(1, len(text) // CHARS_PER_TOKEN_APPROX)


def chunk_text(text: str) -> List[str]:
    """
    Split text into overlapping chunks of approximately CHUNK_SIZE_CHARS characters,
    respecting paragraph boundaries where possible.

    Strategy:
    1. Split on paragraphs first (natural document structure)
    2. Accumulate paragraphs until chunk size is reached
    3. Apply overlap by including the last CHUNK_OVERLAP_CHARS of the previous chunk

    Args:
        text: The full text to chunk.

    Returns:
        List of text chunk strings.
    """
    paragraphs = _split_into_paragraphs(text)
    chunks: List[str] = []
    current_chunk_parts: List[str] = []
    current_size = 0
    overlap_tail = ""

    for para in paragraphs:
        para_size = len(para)

        # If this single paragraph exceeds chunk size, split it by sentences
        if para_size > CHUNK_SIZE_CHARS:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sentence in sentences:
                if current_size + len(sentence) > CHUNK_SIZE_CHARS and current_chunk_parts:
                    chunk_text_str = overlap_tail + "\n\n".join(current_chunk_parts)
                    chunks.append(chunk_text_str.strip())
                    # Set overlap from end of completed chunk
                    overlap_tail = chunk_text_str[-CHUNK_OVERLAP_CHARS:] + "\n\n"
                    current_chunk_parts = [sentence]
                    current_size = len(sentence)
                else:
                    current_chunk_parts.append(sentence)
                    current_size += len(sentence)
        elif current_size + para_size > CHUNK_SIZE_CHARS and current_chunk_parts:
            # Flush current chunk
            chunk_text_str = overlap_tail + "\n\n".join(current_chunk_parts)
            chunks.append(chunk_text_str.strip())
            # Set overlap
            overlap_tail = chunk_text_str[-CHUNK_OVERLAP_CHARS:] + "\n\n"
            current_chunk_parts = [para]
            current_size = para_size
        else:
            current_chunk_parts.append(para)
            current_size += para_size

    # Flush remaining content
    if current_chunk_parts:
        chunk_text_str = overlap_tail + "\n\n".join(current_chunk_parts)
        chunks.append(chunk_text_str.strip())

    return [c for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# Document loader
# ---------------------------------------------------------------------------

class RegulatoryDocumentLoader:
    """
    Loads regulatory documents from a directory, applies chunking, and
    produces DocumentChunk objects for vector store ingestion.

    Usage:
        loader = RegulatoryDocumentLoader("/path/to/regulatory_documents/")
        for chunk in loader.load_all():
            vector_store.upsert(chunk)
    """

    def __init__(self, documents_dir: str):
        self.documents_dir = Path(documents_dir)
        if not self.documents_dir.exists():
            raise FileNotFoundError(
                f"Regulatory documents directory not found: {self.documents_dir}"
            )

    def _make_chunk_id(self, source_file: str, chunk_index: int) -> str:
        """
        Generate a stable, deterministic chunk ID.
        Using a predictable format rather than random UUID ensures idempotent
        upserts — re-running the loader does not create duplicates.
        """
        raw = f"{source_file}::chunk::{chunk_index}"
        # Use UUID5 (name-based SHA-1) for determinism
        return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))

    def load_file(self, file_path: Path) -> LoadedDocument:
        """
        Load and chunk a single regulatory document file.

        Args:
            file_path: Path to the .txt regulatory document.

        Returns:
            LoadedDocument containing all chunks.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file produces no usable chunks.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Document file not found: {file_path}")

        source_file = file_path.name
        text = file_path.read_text(encoding="utf-8")

        # Look up metadata from registry; fall back to defaults
        meta = REGULATORY_DOCUMENT_REGISTRY.get(source_file, {
            "document_name": source_file.replace("_", " ").replace(".txt", ""),
            "regulator": "UNKNOWN",
            "document_reference": source_file,
            "effective_date": "2024-01-01",
            "jurisdiction": "UK",
            "document_type": "Regulatory Document",
        })

        raw_chunks = chunk_text(text)
        if not raw_chunks:
            raise ValueError(f"No chunks produced from file: {file_path}")

        total_chunks = len(raw_chunks)
        doc_chunks: List[DocumentChunk] = []

        for idx, chunk_str in enumerate(raw_chunks):
            chunk = DocumentChunk(
                chunk_id=self._make_chunk_id(source_file, idx),
                text=chunk_str,
                document_name=meta["document_name"],
                regulator=meta["regulator"],
                document_reference=meta["document_reference"],
                effective_date=meta["effective_date"],
                jurisdiction=meta["jurisdiction"],
                document_type=meta["document_type"],
                source_file=source_file,
                chunk_index=idx,
                total_chunks=total_chunks,
                section_number=_detect_section(chunk_str),
                approximate_tokens=_approx_tokens(chunk_str),
            )
            doc_chunks.append(chunk)

        return LoadedDocument(
            source_file=source_file,
            document_name=meta["document_name"],
            regulator=meta["regulator"],
            total_chunks=total_chunks,
            chunks=doc_chunks,
        )

    def load_all(self) -> Iterator[DocumentChunk]:
        """
        Yield DocumentChunk objects for all recognised regulatory documents
        in the documents directory.

        Only files listed in REGULATORY_DOCUMENT_REGISTRY are loaded.
        Unrecognised .txt files are skipped with a warning.

        Yields:
            DocumentChunk objects.
        """
        loaded_count = 0
        for filename in REGULATORY_DOCUMENT_REGISTRY:
            file_path = self.documents_dir / filename
            if not file_path.exists():
                continue  # File not present in this deployment; skip silently
            doc = self.load_file(file_path)
            for chunk in doc.chunks:
                yield chunk
            loaded_count += 1

        if loaded_count == 0:
            raise RuntimeError(
                f"No recognised regulatory documents found in {self.documents_dir}. "
                f"Expected files: {list(REGULATORY_DOCUMENT_REGISTRY.keys())}"
            )

    def load_all_as_list(self) -> List[DocumentChunk]:
        """Convenience method: load all chunks into a list."""
        return list(self.load_all())

    def get_document_summary(self) -> List[Dict]:
        """Return a summary of all loaded documents (for logging/reporting)."""
        summary = []
        for filename, meta in REGULATORY_DOCUMENT_REGISTRY.items():
            file_path = self.documents_dir / filename
            if file_path.exists():
                text = file_path.read_text(encoding="utf-8")
                chunks = chunk_text(text)
                summary.append({
                    "filename": filename,
                    "document_name": meta["document_name"],
                    "regulator": meta["regulator"],
                    "chunk_count": len(chunks),
                    "total_characters": len(text),
                    "approximate_tokens": _approx_tokens(text),
                })
        return summary
