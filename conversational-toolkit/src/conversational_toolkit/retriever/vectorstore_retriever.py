import asyncio

from conversational_toolkit.embeddings.base import EmbeddingsModel
from conversational_toolkit.retriever.base import Retriever
from conversational_toolkit.vectorstores.base import ChunkMatch, VectorStore


class VectorStoreRetriever(Retriever[ChunkMatch]):
    def __init__(self, embedding_model: EmbeddingsModel, vector_store: VectorStore, top_k: int):
        super().__init__(top_k)
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    async def retrieve(self, query: str) -> list[ChunkMatch]:
        embeddings = await self.embedding_model.get_embeddings(query)
        return await self.vector_store.get_chunks_by_embedding(embeddings[0], self.top_k)


class CompositeVectorStoreRetriever(Retriever[ChunkMatch]):
    """Queries several (embedding model, vector store) pairs in parallel and concatenates their results."""

    def __init__(self, embedding_models: list[EmbeddingsModel], vector_stores: list[VectorStore], top_k: list[int]):
        super().__init__(top_k=sum(top_k))
        self.retrievers = [
            VectorStoreRetriever(model, store, k)
            for model, store, k in zip(embedding_models, vector_stores, top_k, strict=True)
        ]

    async def retrieve(self, query: str) -> list[ChunkMatch]:
        results = await asyncio.gather(*(r.retrieve(query) for r in self.retrievers))
        return [chunk for result in results for chunk in result]
