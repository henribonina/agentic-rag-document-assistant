"""Embedding providers and Chroma vector storage."""

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Protocol, Sequence

from src.text_splitter import TextChunk


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


class EmbeddingProvider(Protocol):
    """Common interface for local and hosted embedding providers."""

    name: str
    dimension: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class LocalHashEmbeddings:
    """Deterministic offline embeddings for development and automated tests."""

    name = "Local deterministic"

    def __init__(self, dimension: int = 384) -> None:
        if dimension < 64:
            raise ValueError("Embedding dimension must be at least 64.")
        self.dimension = dimension

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = TOKEN_PATTERN.findall(text.lower())
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            position = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[position] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return vector

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class OpenAIEmbeddingsProvider:
    """OpenAI embeddings adapter loaded only when explicitly selected."""

    name = "OpenAI text-embedding-3-small"
    dimension = 1_536

    def __init__(self, api_key: str, model: str = "text-embedding-3-small") -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not configured in the local .env file.")
        from langchain_openai import OpenAIEmbeddings

        self._client = OpenAIEmbeddings(api_key=api_key, model=model)
        self.name = f"OpenAI {model}"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._client.embed_documents(list(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._client.embed_query(text)


@dataclass(frozen=True)
class VectorIndexResult:
    """Summary returned after chunks are embedded and indexed."""

    collection_name: str
    chunk_count: int
    embedding_provider: str
    embedding_dimension: int


@dataclass(frozen=True)
class SearchResult:
    """One semantic-search match returned from Chroma."""

    chunk_id: str
    text: str
    metadata: dict[str, Any]
    distance: float


def _safe_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Convert metadata values into scalar types accepted by Chroma."""
    safe = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            safe[key] = value
        elif value is None:
            safe[key] = ""
        else:
            safe[key] = json.dumps(value, sort_keys=True, default=str)
    return safe


class ChromaVectorStore:
    """Thin wrapper around an in-memory Chroma collection."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        collection_name: str = "agentic_rag_documents",
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError(
                "Chroma is not installed. Run: pip install -r requirements.txt"
            ) from exc

        self.embedding_provider = embedding_provider
        self.collection_name = collection_name
        self._client = chromadb.Client()
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def index_chunks(self, chunks: Sequence[TextChunk]) -> VectorIndexResult:
        """Embed and upsert chunks into the Chroma collection."""
        if not chunks:
            raise ValueError("At least one text chunk is required for indexing.")

        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_provider.embed_documents(texts)
        if len(embeddings) != len(chunks):
            raise ValueError("The embedding provider returned an unexpected result count.")

        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=texts,
            metadatas=[_safe_metadata(chunk.metadata) for chunk in chunks],
            embeddings=embeddings,
        )
        dimension = len(embeddings[0]) if embeddings else 0
        return VectorIndexResult(
            collection_name=self.collection_name,
            chunk_count=self._collection.count(),
            embedding_provider=self.embedding_provider.name,
            embedding_dimension=dimension,
        )

    def search(self, query: str, top_k: int = 4) -> tuple[SearchResult, ...]:
        """Return the nearest chunks for a natural-language query."""
        if not query.strip():
            raise ValueError("A non-empty search query is required.")
        collection_size = self._collection.count()
        if collection_size == 0:
            return ()
        query_embedding = self.embedding_provider.embed_query(query)
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(max(1, top_k), collection_size),
            include=["documents", "metadatas", "distances"],
        )
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        return tuple(
            SearchResult(
                chunk_id=chunk_id,
                text=text,
                metadata=metadata or {},
                distance=float(distance),
            )
            for chunk_id, text, metadata, distance in zip(
                ids, documents, metadatas, distances
            )
        )


def create_embedding_provider(mode: str, api_key: str = "") -> EmbeddingProvider:
    """Create the configured embedding provider."""
    normalized = mode.strip().lower()
    if normalized == "openai":
        return OpenAIEmbeddingsProvider(api_key=api_key)
    if normalized == "local":
        return LocalHashEmbeddings()
    raise ValueError(f"Unknown embedding mode: {mode}.")
