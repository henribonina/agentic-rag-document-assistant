"""Metadata-aware text chunking for semantic retrieval."""

from dataclasses import dataclass, field
from typing import Any, Iterable

from src.document_loader import LoadedDocument


DEFAULT_CHUNK_SIZE = 1_000
DEFAULT_CHUNK_OVERLAP = 150


@dataclass(frozen=True)
class TextChunk:
    """A retrieval unit created from an ingested document."""

    chunk_id: str
    text: str
    source: str
    index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def character_count(self) -> int:
        return len(self.text)


def _choose_boundary(text: str, start: int, ideal_end: int) -> int:
    """Prefer a natural boundary near the requested chunk end."""
    if ideal_end >= len(text):
        return len(text)

    minimum_end = start + max((ideal_end - start) // 2, 1)
    for separator in ("\n\n", "\n", ". ", " "):
        boundary = text.rfind(separator, minimum_end, ideal_end)
        if boundary != -1:
            return boundary + len(separator)
    return ideal_end


def split_document(
    document: LoadedDocument,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> tuple[TextChunk, ...]:
    """Split one document into overlapping, metadata-rich chunks."""
    if chunk_size < 100:
        raise ValueError("chunk_size must be at least 100 characters.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    text = document.text.strip()
    if not text:
        return ()

    chunks = []
    start = 0
    index = 0
    source_key = "".join(
        character.lower() if character.isalnum() else "-"
        for character in document.source
    ).strip("-") or "document"

    while start < len(text):
        ideal_end = min(start + chunk_size, len(text))
        end = _choose_boundary(text, start, ideal_end)
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunk_id = f"{source_key}-{index:04d}"
            metadata = {
                **document.metadata,
                "chunk_id": chunk_id,
                "chunk_index": index,
                "start_character": start,
                "end_character": end,
                "chunk_character_count": len(chunk_text),
            }
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    source=document.source,
                    index=index,
                    metadata=metadata,
                )
            )
            index += 1

        if end >= len(text):
            break
        next_start = max(end - chunk_overlap, start + 1)
        start = next_start

    return tuple(chunks)


def split_documents(
    documents: Iterable[LoadedDocument],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> tuple[TextChunk, ...]:
    """Split a collection of documents while preserving their order."""
    chunks = []
    for document in documents:
        chunks.extend(split_document(document, chunk_size, chunk_overlap))
    return tuple(chunks)
