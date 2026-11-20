import json
from typing import Any

import chromadb
import numpy as np
from numpy.typing import NDArray

from conversational_toolkit.chunking.base import Chunk
from conversational_toolkit.utils.database import generate_uid
from conversational_toolkit.vectorstores.base import ChunkMatch, ChunkRecord, VectorStore


def _encode_metadata(value: Any) -> Any:
    return json.dumps(value) if isinstance(value, list | dict) else value


def _decode_metadata(value: Any) -> Any:
    if isinstance(value, str) and value[:1] in ("[", "{"):
        try:
            return json.loads(value)
        except ValueError:
            pass
    return value


class ChromaDBVectorStore(VectorStore):
    """Persistent ChromaDB collection. New collections use cosine distance; list/dict metadata is stored as JSON."""

    def __init__(self, db_path: str, collection_name: str = "default_collection"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name, configuration={"hnsw": {"space": "cosine"}}
        )
        hnsw = self.collection.configuration.get("hnsw") or {}
        self.space: str = hnsw.get("space", "l2")

    def reset(self) -> None:
        """Delete all chunks and recreate the collection with cosine distance."""
        name = self.collection.name
        self.client.delete_collection(name)
        self.collection = self.client.create_collection(name=name, configuration={"hnsw": {"space": "cosine"}})
        self.space = "cosine"

    def _similarity(self, distance: float) -> float:
        # Chroma's l2 is the squared distance, which is 2 - 2 * cosine for normalised embeddings.
        return 1 - distance / 2 if self.space == "l2" else 1 - distance

    def _record(self, chunk_id: str, document: str, metadata: Any) -> dict[str, Any]:
        meta = {k: _decode_metadata(v) for k, v in (metadata or {}).items()}
        return {
            "id": chunk_id,
            "title": str(meta.get("title", "")),
            "mime_type": str(meta.get("mime_type", "")),
            "content": document,
            "metadata": meta,
            "embedding": [],
        }

    async def insert_chunks(self, chunks: list[Chunk], embedding: NDArray[np.float64]) -> None:
        self.collection.add(
            ids=[generate_uid() for _ in chunks],
            embeddings=embedding.tolist(),  # type: ignore[arg-type]
            metadatas=[
                {k: _encode_metadata(v) for k, v in {"title": c.title, "mime_type": c.mime_type, **c.metadata}.items()}
                for c in chunks
            ],
            documents=[c.content for c in chunks],
        )

    async def get_chunks_by_embedding(
        self, embedding: NDArray[np.float64], top_k: int, filters: dict[str, Any] | None = None
    ) -> list[ChunkMatch]:
        results = self.collection.query(query_embeddings=embedding.tolist(), n_results=top_k, where=filters)  # type: ignore[arg-type]
        documents, metadatas, distances = results["documents"], results["metadatas"], results["distances"]
        assert documents is not None and metadatas is not None and distances is not None  # included by default
        return [
            ChunkMatch(**self._record(i, doc, meta), score=self._similarity(dist))
            for i, doc, meta, dist in zip(results["ids"][0], documents[0], metadatas[0], distances[0], strict=True)
        ]

    async def get_chunks_by_filter(self, filters: dict[str, Any] | None = None) -> list[ChunkRecord]:
        """Return all chunks matching a Chroma 'where' filter, e.g. '{"source_file": {"$eq": "a.pdf"}}'."""
        results = self.collection.get(where=filters or None)  # type: ignore[arg-type]
        documents, metadatas = results["documents"], results["metadatas"]
        assert documents is not None and metadatas is not None  # included by default
        return [
            ChunkRecord(**self._record(i, doc, meta))
            for i, doc, meta in zip(results["ids"], documents, metadatas, strict=True)
        ]
