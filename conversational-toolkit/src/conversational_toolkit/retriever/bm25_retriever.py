import re

import numpy as np
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from conversational_toolkit.retriever.base import Retriever
from conversational_toolkit.vectorstores.base import ChunkMatch, ChunkRecord, VectorStore


class BM25Retriever(Retriever[ChunkMatch]):
    """BM25 Okapi over all chunks of a vector store, indexed on the first query (later inserts are not seen)."""

    def __init__(self, vector_store: VectorStore, top_k: int) -> None:
        super().__init__(top_k)
        self.vs = vector_store
        self.corpus: list[ChunkRecord] = []
        self._bm25: BM25Okapi | None = None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    async def retrieve(self, query: str) -> list[ChunkMatch]:
        if self._bm25 is None:
            self.corpus = await self.vs.get_chunks_by_filter()
            self._bm25 = BM25Okapi([self._tokenize(chunk.content) for chunk in self.corpus])

        scores = self._bm25.get_scores(self._tokenize(query))
        top = np.argsort(-scores, kind="stable")[: self.top_k]
        return [ChunkMatch(**self.corpus[i].model_dump(), score=float(scores[i])) for i in top]
