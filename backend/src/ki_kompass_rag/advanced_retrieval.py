"""Retrieval strategies beyond single-query semantic search: BM25, hybrid (RRF) and metadata filtering."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from loguru import logger

from conversational_toolkit.embeddings.base import EmbeddingsModel
from conversational_toolkit.llms.base import LLM, LLMMessage, MessageContent, Roles
from conversational_toolkit.retriever.bm25_retriever import BM25Retriever
from conversational_toolkit.retriever.hybrid_retriever import HybridRetriever
from conversational_toolkit.retriever.vectorstore_retriever import VectorStoreRetriever
from conversational_toolkit.utils.retriever import build_query_with_chunks
from conversational_toolkit.vectorstores.base import ChunkMatch, VectorStore
from conversational_toolkit.vectorstores.chromadb import ChromaDBVectorStore

from ki_kompass_rag.baseline_rag import SYSTEM_PROMPT


@dataclass
class RetrievalResult:
    strategy: str
    chunks: Sequence[ChunkMatch]
    filters: dict[str, Any] | None = None

    def __str__(self) -> str:
        return "\n".join(
            [f"[{self.strategy}]"]
            + [
                f"  {c.metadata.get('source_file', '?'):<50} | {c.title!r:.50}"
                for c in self.chunks[:5]
            ]
        )


async def retrieve_baseline(
    query: str,
    embedding_model: EmbeddingsModel,
    vector_store: ChromaDBVectorStore,
    top_k: int = 5,
) -> RetrievalResult:
    """Single-query semantic retrieval."""
    retriever = VectorStoreRetriever(embedding_model, vector_store, top_k=top_k)
    return RetrievalResult("baseline", await retriever.retrieve(query))


async def retrieve_bm25(
    query: str, vector_store: VectorStore, top_k: int = 5
) -> RetrievalResult:
    """BM25 keyword retrieval over all chunks in vector_store."""
    retriever = BM25Retriever(vector_store=vector_store, top_k=top_k)
    return RetrievalResult("bm25", await retriever.retrieve(query))


async def retrieve_hybrid(
    query: str,
    embedding_model: EmbeddingsModel,
    vector_store: ChromaDBVectorStore,
    top_k: int = 5,
    rrf_k: int = 60,
) -> RetrievalResult:
    """Semantic + BM25 fused with Reciprocal Rank Fusion."""
    hybrid = HybridRetriever(
        retrievers=[
            VectorStoreRetriever(embedding_model, vector_store, top_k=top_k),
            BM25Retriever(vector_store=vector_store, top_k=top_k),
        ],
        top_k=top_k,
        rrf_k=rrf_k,
    )
    return RetrievalResult("hybrid", await hybrid.retrieve(query))


async def retrieve_with_metadata_filter(
    query: str,
    embedding_model: EmbeddingsModel,
    vector_store: ChromaDBVectorStore,
    filters: dict[str, Any],
    top_k: int = 5,
) -> RetrievalResult:
    """Semantic search restricted by a ChromaDB 'where' filter, e.g. {"source_file": {"$eq": "EPD_tesa.pdf"}}."""
    embedding = await embedding_model.get_embeddings(query)
    chunks = await vector_store.get_chunks_by_embedding(
        embedding[0], top_k=top_k, filters=filters
    )
    return RetrievalResult("metadata_filter", chunks, filters=filters)


async def compare_retrieval_strategies(
    query: str,
    embedding_model: EmbeddingsModel,
    vector_store: ChromaDBVectorStore,
    top_k: int = 5,
    metadata_filters: dict[str, Any] | None = None,
) -> dict[str, RetrievalResult]:
    """Run baseline, BM25, hybrid and, if metadata_filters is given, filtered retrieval."""
    logger.info(f"Comparing retrieval strategies for: {query!r}")
    results = {
        "baseline": await retrieve_baseline(
            query, embedding_model, vector_store, top_k
        ),
        "bm25": await retrieve_bm25(query, vector_store, top_k),
        "hybrid": await retrieve_hybrid(query, embedding_model, vector_store, top_k),
    }
    if metadata_filters is not None:
        results["metadata_filter"] = await retrieve_with_metadata_filter(
            query, embedding_model, vector_store, metadata_filters, top_k
        )
    return results


def print_strategy_comparison(
    results: dict[str, RetrievalResult],
    relevant_keywords: list[str],
    top_n: int = 3,
) -> None:
    """Print the top_n sources per strategy; ✓ marks chunks containing a relevant keyword."""
    keywords = [kw.lower() for kw in relevant_keywords]
    print(f"\nRelevant keywords: {relevant_keywords}")
    for name, r in results.items():
        print(f"\n{name}" + (f"  (filter: {r.filters})" if r.filters else ""))
        for chunk in r.chunks[:top_n]:
            src = chunk.metadata.get("source_file", "?")
            title = chunk.title or "(no title)"
            text = (src + title + chunk.content).lower()
            marker = "✓" if any(kw in text for kw in keywords) else "·"
            print(f"  {marker}  {src:<48}  {title[:38]!r}")


async def rag_answer_from_chunks(
    llm: LLM, chunks: Sequence[ChunkMatch], query: str
) -> str:
    """Answer query from pre-retrieved chunks with SYSTEM_PROMPT (no retrieval)."""
    messages = [
        LLMMessage(
            role=Roles.SYSTEM, content=[MessageContent(type="text", text=SYSTEM_PROMPT)]
        ),
        LLMMessage(
            role=Roles.USER,
            content=[
                MessageContent(
                    type="text", text=build_query_with_chunks(query, list(chunks))
                )
            ],
        ),
    ]
    answer = await llm.generate(messages)
    return "".join(c.text for c in answer.content if c.text)
