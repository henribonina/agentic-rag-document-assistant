"""Semantic retrieval and source-label helpers."""

from dataclasses import dataclass
import re
from typing import Any, Protocol

from src.vector_store import SearchResult


PAGE_PATTERN = re.compile(r"^## Page (\d+)", re.MULTILINE)
SHEET_PATTERN = re.compile(r"^## Sheet: ([^\n]+)", re.MULTILINE)


class SearchableVectorStore(Protocol):
    """Minimal vector-store interface required by the retriever."""

    def search(self, query: str, top_k: int = 4) -> tuple[SearchResult, ...]: ...


@dataclass(frozen=True)
class RetrievedPassage:
    """Ranked passage prepared for presentation in the interface."""

    rank: int
    chunk_id: str
    text: str
    source: str
    location: str
    relevance: float
    metadata: dict[str, Any]


def _location_label(text: str, metadata: dict[str, Any]) -> str:
    """Build a human-readable source location when one is available."""
    page_match = PAGE_PATTERN.search(text)
    if page_match:
        return f"Page {page_match.group(1)}"

    sheet_match = SHEET_PATTERN.search(text)
    if sheet_match:
        return f"Sheet {sheet_match.group(1).strip()}"

    chunk_index = metadata.get("chunk_index")
    if isinstance(chunk_index, int):
        return f"Chunk {chunk_index + 1}"
    return "Document passage"


def _relevance_from_cosine_distance(distance: float) -> float:
    """Convert Chroma cosine distance into a bounded display score."""
    return max(0.0, min(1.0, 1.0 - distance))


def retrieve_passages(
    vector_store: SearchableVectorStore,
    query: str,
    top_k: int = 4,
) -> tuple[RetrievedPassage, ...]:
    """Retrieve and rank the passages most relevant to a user question."""
    normalized_query = " ".join(query.split())
    if len(normalized_query) < 5:
        raise ValueError("Enter a question containing at least 5 characters.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    results = vector_store.search(normalized_query, top_k=top_k)
    passages = []
    for rank, result in enumerate(results, start=1):
        metadata = dict(result.metadata)
        source = str(metadata.get("source") or "Uploaded document")
        passages.append(
            RetrievedPassage(
                rank=rank,
                chunk_id=result.chunk_id,
                text=result.text,
                source=source,
                location=_location_label(result.text, metadata),
                relevance=_relevance_from_cosine_distance(result.distance),
                metadata=metadata,
            )
        )
    return tuple(passages)
