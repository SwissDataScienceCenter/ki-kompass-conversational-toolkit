import asyncio

from conversational_toolkit.retriever.base import Retriever
from conversational_toolkit.vectorstores.base import ChunkMatch, VectorStore


class ContextWindowRetriever(Retriever[ChunkMatch]):
    """Appends the 'window_size' preceding and following chunks of the same file to each retrieved chunk.

    Neighbours are found via the 'source_file' and 'chunk_index' metadata that 'PDFChunker' sets; chunks without
    them are returned unchanged.
    """

    def __init__(
        self, retriever: Retriever[ChunkMatch], vector_store: VectorStore, window_size: int = 1, top_k: int = 5
    ):
        super().__init__(top_k)
        self.retriever = retriever
        self.vector_store = vector_store
        self.window_size = window_size

    async def retrieve(self, query: str) -> list[ChunkMatch]:
        base_chunks = await self.retriever.retrieve(query)
        return list(await asyncio.gather(*(self._expand(c) for c in base_chunks[: self.top_k])))

    async def _expand(self, chunk: ChunkMatch) -> ChunkMatch:
        idx = chunk.metadata.get("chunk_index")
        source = chunk.metadata.get("source_file")
        if idx is None or source is None:
            return chunk

        neighbours = await self.vector_store.get_chunks_by_filter(
            {
                "$and": [
                    {"source_file": {"$eq": source}},
                    {"chunk_index": {"$gte": int(idx) - self.window_size}},
                    {"chunk_index": {"$lte": int(idx) + self.window_size}},
                ]
            }
        )
        window = sorted(neighbours, key=lambda c: int(c.metadata["chunk_index"]))
        if len(window) <= 1:
            return chunk
        return chunk.model_copy(
            update={
                "content": "\n\n".join(c.content for c in window),
                "metadata": {**chunk.metadata, "context_window_size": self.window_size},
            }
        )
