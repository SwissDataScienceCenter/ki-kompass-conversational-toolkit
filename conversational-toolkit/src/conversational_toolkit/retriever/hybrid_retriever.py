import asyncio
from typing import Any

from conversational_toolkit.retriever.base import Retriever
from conversational_toolkit.utils.retriever import reciprocal_rank_fusion
from conversational_toolkit.vectorstores.base import ChunkMatch


class HybridRetriever(Retriever[ChunkMatch]):
    """Queries all sub-retrievers in parallel and fuses their rankings with RRF (damping constant 'rrf_k')."""

    def __init__(self, retrievers: list[Retriever[Any]], top_k: int, rrf_k: int = 60) -> None:
        super().__init__(top_k)
        self.retrievers = retrievers
        self.rrf_k = rrf_k

    async def retrieve(self, query: str) -> list[ChunkMatch]:
        rankings = await asyncio.gather(*(r.retrieve(query) for r in self.retrievers))
        return reciprocal_rank_fusion(rankings, k=self.rrf_k)[: self.top_k]
