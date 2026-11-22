"""Baseline RAG: load_chunks -> build_vector_store -> inspect_retrieval -> build_agent -> ask.

Run: BACKEND=openai|ollama [MODEL=..] [EMBEDDING_MODEL=..] [QUERY=..] [RESET_VS=1] python -m ki_kompass_rag.baseline_rag
"""

import asyncio
import os
from collections import Counter
from pathlib import Path

from loguru import logger

from conversational_toolkit.agents.base import Agent, AgentAnswer, QueryWithContext
from conversational_toolkit.agents.rag import RAG
from conversational_toolkit.chunking.base import Chunk
from conversational_toolkit.chunking.excel_chunker import ExcelChunker
from conversational_toolkit.chunking.markdown_chunker import MarkdownChunker
from conversational_toolkit.chunking.pdf_chunker import PDFChunker
from conversational_toolkit.embeddings.base import EmbeddingsModel
from conversational_toolkit.embeddings.openai import OpenAIEmbeddings
from conversational_toolkit.llms.base import LLM, LLMMessage
from conversational_toolkit.llms.ollama import OllamaLLM
from conversational_toolkit.llms.openai import OpenAILLM
from conversational_toolkit.retriever.vectorstore_retriever import VectorStoreRetriever
from conversational_toolkit.vectorstores.base import ChunkMatch
from conversational_toolkit.vectorstores.chromadb import ChromaDBVectorStore

ROOT = Path(__file__).parents[3]
DATA_DIR = ROOT / "data"
DB_DIR = Path(os.getenv("DB_DIR", ROOT / "backend" / "db"))
VS_PATH = DB_DIR / "data_vs.db"

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
# An OpenAI "text-embedding-*" model, or any sentence-transformers model name (runs locally).
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
RETRIEVER_TOP_K = 5
SEED = 42
MAX_FILE_SIZE_MB = 20

SYSTEM_PROMPT = (
    "You are a helpful AI assistant specialised in sustainability and product compliance for PrimePack AG.\n\n"
    "You will receive document excerpts relevant to the user's question. "
    "Produce the best possible answer using only the information in those excerpts.\n\n"
    "Rules:\n"
    "- Use the provided excerpts as your only source of truth. Do not rely on outside knowledge.\n"
    "- Use all relevant excerpts when forming your answer.\n"
    "- If the answer cannot be found in the excerpts, clearly say that you do not know.\n"
    "- Always cite the source document by file name for any claim you make.\n"
    "- If excerpts contain conflicting information, report both values and flag the conflict.\n"
    "- Distinguish between third-party verified claims (EPDs) and self-declared supplier claims."
)

# The PDFs in data/ have a text layer, so OCR only adds time.
_PDF_CHUNKER = PDFChunker(do_ocr=False)
_CHUNKERS: dict[str, PDFChunker | ExcelChunker | MarkdownChunker] = {
    ".pdf": _PDF_CHUNKER,
    ".xlsx": ExcelChunker(),
    ".xls": ExcelChunker(),
    ".md": MarkdownChunker(),
    ".txt": MarkdownChunker(),
}


def build_llm(
    backend: str,
    model_name: str | None = None,
    temperature: float = 0.3,
    response_format=None,
) -> LLM:
    """Backends: openai (gpt-4o-mini), ollama (mistral-nemo:12b)."""
    match backend.lower().strip():
        case "openai":
            name = model_name or "gpt-4o-mini"
            llm: LLM = OpenAILLM(
                model_name=name,
                temperature=temperature,
                seed=SEED,
                response_format=response_format,
            )
        case "ollama":
            name = model_name or "mistral-nemo:12b"
            llm = OllamaLLM(
                model_name=name,
                temperature=temperature,
                seed=SEED,
                response_format="json" if response_format else None,
            )
        case _:
            raise ValueError(
                f"Unsupported backend {backend!r}. Choose 'openai' or 'ollama'."
            )
    logger.info(f"LLM backend: {backend} ({name})")
    return llm


def load_chunks(max_files: int | None = None, with_images: bool = False) -> list[Chunk]:
    """Chunk the supported files in DATA_DIR (EVALUATION* excluded); text only unless with_images."""
    files = []
    for f in sorted(DATA_DIR.iterdir()):
        if not f.is_file() or f.name.startswith("EVALUATION"):
            continue
        if f.suffix.lower() not in _CHUNKERS:
            logger.warning(f"Skipping unsupported file: {f.name}")
        elif f.stat().st_size > MAX_FILE_SIZE_MB * 2**20:
            logger.warning(f"Skipping {f.name}: larger than {MAX_FILE_SIZE_MB} MB")
        else:
            files.append(f)
    files = files[:max_files]
    logger.info(f"Chunking {len(files)} files from {DATA_DIR}")

    chunks: list[Chunk] = []
    for f in files:
        chunker = _CHUNKERS[f.suffix.lower()]
        try:
            if isinstance(chunker, PDFChunker):
                file_chunks = chunker.make_chunks(str(f), write_images=with_images)
            else:
                file_chunks = chunker.make_chunks(str(f))
        except Exception as exc:
            logger.warning(f"Skipping {f.name}: {exc}")
            continue
        for chunk in file_chunks:
            # "source" and "title" are what the frontend displays.
            chunk.metadata |= {
                "source_file": f.name,
                "source": f.name,
                "title": chunk.title,
            }
        chunks.extend(file_chunks)

    logger.info(f"Done, {len(chunks)} chunks total")
    return chunks


def inspect_chunks(chunks: list[Chunk], sample_size: int = 5) -> None:
    """Log chunk counts per file and a few sample chunks."""
    counts = Counter(c.metadata.get("source_file", "unknown") for c in chunks)
    logger.info(f"Total chunks: {len(chunks)}; Source files: {len(counts)}")
    for fname, n in sorted(counts.items()):
        logger.info(f"{fname}: {n} chunks")
    for chunk in chunks[:sample_size]:
        logger.info(f"[{chunk.metadata.get('source_file', '?')}] {chunk.title!r}")
        logger.info(f"{chunk.content[:200].strip()!r}")


def build_embeddings(model_name: str = EMBEDDING_MODEL) -> EmbeddingsModel:
    if model_name.startswith("text-embedding-"):
        return OpenAIEmbeddings(model_name)
    # Deferred: loads torch.
    from conversational_toolkit.embeddings.sentence_transformer import (
        SentenceTransformerEmbeddings,
    )

    return SentenceTransformerEmbeddings(model_name)


def store_embedding_model(vector_store: ChromaDBVectorStore) -> str:
    """Model the store was built with; stores from before this was recorded used the default."""
    metadata = vector_store.collection.metadata or {}
    return str(metadata.get("embedding_model", DEFAULT_EMBEDDING_MODEL))


async def build_vector_store(
    chunks: list[Chunk],
    embedding_model: EmbeddingsModel,
    db_path: Path = VS_PATH,
    reset: bool = False,
    batch_size: int = 100,
) -> ChromaDBVectorStore:
    """Embed chunks into a ChromaDB store at db_path; skipped if the store is non-empty and not reset."""
    vector_store = ChromaDBVectorStore(db_path=str(db_path))

    if reset:
        vector_store.reset()
    elif (count := vector_store.collection.count()) > 0:
        logger.info(
            f"Vector store already contains {count} chunks, skipping embedding."
        )
        return vector_store

    if model_name := getattr(embedding_model, "model_name", None):
        vector_store.collection.modify(metadata={"embedding_model": model_name})
    n_batches = -(-len(chunks) // batch_size)
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        if all(c.mime_type.startswith("text") for c in batch):
            embeddings = await embedding_model.get_embeddings(
                [c.content for c in batch]
            )
        else:
            # Multimodal models (CLIP, Qwen3-VL) take the Chunk objects.
            embeddings = await embedding_model.get_embeddings(batch)  # type: ignore[arg-type]
        await vector_store.insert_chunks(chunks=batch, embedding=embeddings)
        logger.info(f"Embedded batch {i // batch_size + 1}/{n_batches}")

    logger.info(f"Vector store written to {db_path}")
    return vector_store


async def load_or_build_vector_store(
    reset: bool = False,
) -> tuple[ChromaDBVectorStore, EmbeddingsModel]:
    """Open the notebooks' store at VS_PATH; rebuild it if empty, reset, or built with another embedding model."""
    embedding_model = build_embeddings()
    vector_store = ChromaDBVectorStore(db_path=str(VS_PATH))
    # Stores from older versions use L2 and 1024-dim embeddings; rebuild them.
    if (
        reset
        or vector_store.collection.count() == 0
        or vector_store.space != "cosine"
        or store_embedding_model(vector_store) != EMBEDDING_MODEL
    ):
        vector_store = await build_vector_store(
            load_chunks(), embedding_model, db_path=VS_PATH, reset=True
        )
    logger.info(f"Vector store {VS_PATH}: {vector_store.collection.count()} chunks")
    return vector_store, embedding_model


async def inspect_retrieval(
    query: str,
    vector_store: ChromaDBVectorStore,
    embedding_model: EmbeddingsModel,
    top_k: int = RETRIEVER_TOP_K,
) -> list[ChunkMatch]:
    """Retrieve top_k chunks for query and print them."""
    retriever = VectorStoreRetriever(embedding_model, vector_store, top_k=top_k)
    results = await retriever.retrieve(query)
    print(f"Top-{top_k} chunks for {query!r}:")
    for i, r in enumerate(results, 1):
        print(
            f"  [{i}] score={r.score:.4f}  file={r.metadata.get('source_file', '?')!r}  title={r.title!r}"
        )
        print(f"       {r.content[:1000].strip()!r}")
    return results


def build_agent(
    vector_store: ChromaDBVectorStore,
    embedding_model: EmbeddingsModel,
    llm: LLM,
    top_k: int,
    system_prompt: str,
    number_query_expansion: int = 0,
    enable_hyde: bool = False,
) -> RAG:
    """RAG agent over vector_store; number_query_expansion > 0 adds LLM sub-queries fused with RRF."""
    return RAG(
        llm=llm,
        utility_llm=llm,
        system_prompt=system_prompt,
        retrievers=[VectorStoreRetriever(embedding_model, vector_store, top_k=top_k)],
        number_query_expansion=number_query_expansion,
        enable_hyde=enable_hyde,
    )


async def ask(
    agent: Agent,
    query: str,
    history: list[LLMMessage] | None = None,
) -> AgentAnswer:
    """Answer query with agent (RAG or ToolAgent), print the answer and its source files."""
    answer = await agent.answer(QueryWithContext(query=query, history=history or []))
    print("".join(c.text for c in answer.content if c.text))
    files = dict.fromkeys(
        s.metadata.get("source_file", s.title) for s in answer.sources
    )
    print(f"Sources: {', '.join(files) or 'none'}")
    return answer


async def run_pipeline(
    backend: str,
    model_name: str | None = None,
    query: str = "What sustainability certifications do the pallets have?",
    reset_vs: bool = False,
) -> AgentAnswer:
    """Build or reuse the store at VS_PATH, inspect retrieval, and answer query."""
    vector_store, embedding_model = await load_or_build_vector_store(reset=reset_vs)
    await inspect_retrieval(query, vector_store, embedding_model)
    llm = build_llm(backend, model_name=model_name)
    agent = build_agent(
        vector_store,
        embedding_model,
        llm,
        top_k=RETRIEVER_TOP_K,
        system_prompt=SYSTEM_PROMPT,
    )
    return await ask(agent, query)


if __name__ == "__main__":
    if not (_backend := os.getenv("BACKEND")):
        raise SystemExit("Set BACKEND to one of: openai, ollama")
    asyncio.run(
        run_pipeline(
            backend=_backend,
            model_name=os.getenv("MODEL"),
            query=os.getenv("QUERY", "What materials is the Lara pallet made out of?"),
            reset_vs=os.getenv("RESET_VS") == "1",
        )
    )
