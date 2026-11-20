from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray

from conversational_toolkit.chunking.base import Chunk


class ChunkRecord(Chunk):
    """A chunk as stored in a vector store."""

    id: str
    embedding: list[float]


class ChunkMatch(ChunkRecord):
    """A retrieved chunk."""

    score: float
    """Relevance; higher is better (cosine similarity for vector stores)."""


class VectorStore(ABC):
    @abstractmethod
    async def insert_chunks(self, chunks: list[Chunk], embedding: NDArray[np.float64]) -> None:
        """Store 'chunks' with their embeddings, one row of 'embedding' per chunk."""

    @abstractmethod
    async def get_chunks_by_embedding(
        self, embedding: NDArray[np.float64], top_k: int, filters: dict[str, Any] | None = None
    ) -> list[ChunkMatch]:
        """Return the 'top_k' chunks most similar to 'embedding', optionally filtered by metadata."""

    @abstractmethod
    async def get_chunks_by_filter(self, filters: dict[str, Any] | None = None) -> list[ChunkRecord]:
        """Return all chunks whose metadata matches 'filters' (all chunks if None)."""
