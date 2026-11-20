from typing import Any

import numpy as np
from numpy.typing import NDArray
from pgvector.sqlalchemy import Vector  # type: ignore[import-untyped, import-not-found]
from sqlalchemy import Column, MetaData, Select, String, Table, insert, select, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from conversational_toolkit.chunking.base import Chunk
from conversational_toolkit.utils.database import generate_uid
from conversational_toolkit.vectorstores.base import ChunkMatch, ChunkRecord, VectorStore


class PGVectorStore(VectorStore):
    """pgvector table of chunks. 'filters' are metadata key/value pairs that must all match (JSONB containment).

    Requires the 'postgres' extra. Call 'enable_vector_extension' and 'create_table' once before use.
    """

    def __init__(self, engine: AsyncEngine, table_name: str, embeddings_size: int):
        self.engine = engine
        self.make_session = async_sessionmaker(bind=engine)
        self.metadata = MetaData()
        self.table = Table(
            table_name,
            self.metadata,
            Column("id", String, primary_key=True),
            Column("title", String),
            Column("content", String),
            Column("mime_type", String),
            Column("embedding", Vector(embeddings_size)),
            Column("chunk_metadata", JSONB, nullable=True),
        )

    async def enable_vector_extension(self) -> None:
        async with self.engine.begin() as connection:
            await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    async def create_table(self) -> None:
        async with self.engine.begin() as connection:
            await connection.run_sync(self.metadata.create_all)

    def _record(self, row: Any) -> dict[str, Any]:
        return {
            "id": row.id,
            "title": row.title,
            "content": row.content,
            "mime_type": row.mime_type,
            "metadata": row.chunk_metadata or {},
            "embedding": np.asarray(row.embedding).tolist(),
        }

    def _filtered(self, query: Select, filters: dict[str, Any] | None) -> Select:
        return query.where(self.table.c.chunk_metadata.contains(filters)) if filters else query

    async def insert_chunks(self, chunks: list[Chunk], embedding: NDArray[np.float64]) -> None:
        rows = [
            {
                "id": generate_uid(),
                "title": chunk.title,
                "content": chunk.content,
                "embedding": emb,
                "mime_type": chunk.mime_type,
                "chunk_metadata": chunk.metadata,
            }
            for chunk, emb in zip(chunks, embedding, strict=True)
        ]
        async with self.make_session() as session, session.begin():
            await session.execute(insert(self.table), rows)

    async def get_chunks_by_embedding(
        self, embedding: NDArray[np.float64], top_k: int, filters: dict[str, Any] | None = None
    ) -> list[ChunkMatch]:
        score = (1 - self.table.c.embedding.cosine_distance(embedding)).label("score")
        query = self._filtered(select(self.table, score), filters).order_by(score.desc()).limit(top_k)
        async with self.make_session() as session:
            rows = await session.execute(query)
        return [ChunkMatch(**self._record(row), score=row.score) for row in rows]

    async def get_chunks_by_filter(self, filters: dict[str, Any] | None = None) -> list[ChunkRecord]:
        async with self.make_session() as session:
            rows = await session.execute(self._filtered(select(self.table), filters))
        return [ChunkRecord(**self._record(row)) for row in rows]
