# Backend RAG Pipeline

This package (`ki_kompass_rag`) contains the notebooks and the supporting Python modules that run the RAG pipeline for the PrimePack AG sustainability use case.

---

## Package structure

```
backend/
├── db/                                      # Vector stores and conversation data (created at runtime)
│   ├── vs_text/                             # ChromaDB collection for text chunks (OpenAI embeddings)
│   ├── vs_image/                            # ChromaDB collection for image chunks (Qwen3-VL embeddings)
│   └── data_vs.db/                          # ChromaDB collection for the baseline RAG pipeline
├── notebooks/                               # Workshop notebooks
│   ├── english/                             # T00_setup_build_db ... T14_rag_subagent
│   └── german/                              # German versions (same file names)
└── src/ki_kompass_rag/
    ├── baseline_rag.py             # Five-step RAG pipeline (chunking, embedding, retrieval, generation)
    ├── ingestion.py                # Parser comparison, chunking utilities, token analysis
    ├── evaluation.py               # Shared EVALUATION_QUERIES with ground-truth answers
    ├── advanced_retrieval.py       # BM25, hybrid, metadata-filter retrieval
    └── main.py                              # FastAPI server entry point (controller + frontend)
```

---

## T00: Building the vector stores (`T00_setup_build_db.ipynb`)

A one-off setup notebook that populates both ChromaDB collections from the raw corpus in `data/`.

| Step | What it does |
|------|--------------|
| Load all chunks | `load_chunks()` parses every file in `data/` and splits them into chunks |
| Build text store | Embeds text chunks with OpenAI `text-embedding-3-small` and persists to `db/vs_text/` |
| Build image store | Embeds image chunks with Qwen3-VL and persists to `db/vs_image/` |

Both stores are written to `backend/db/` by default (override with the `DB_DIR` environment variable).

---

## Feature tracks

### T01: Baseline RAG Pipeline (`T01_baseline_rag.ipynb`)

Introduces the five-stage RAG loop and demonstrates it end-to-end against the PrimePack AG corpus.

| Step | Function               | What it does                                                                |
|------|------------------------|-----------------------------------------------------------------------------|
| 1    | `load_chunks()`        | Load all documents from `data/` and split into chunks (skips files > 20 MB) |
| 2    | `build_vector_store()` | Embed chunks and persist to ChromaDB                                        |
| 3    | `inspect_retrieval()`  | Run a semantic search and print ranked results                              |
| 4    | `build_agent()`        | Assemble the RAG agent from the retriever and an LLM backend                |
| 5    | `ask()`                | Send a query and stream the grounded answer                                 |

### T02: Document Ingestion (`T02_ingestion_chunking.ipynb`)

Deep-dive into document parsing and chunking; separate from the baseline so the ingestion workflow can be explored independently.

- PDF parser comparison: markitdown vs. docling
- Chunking strategies: fixed-size, header-based, paragraph-aware
- Token limit: embedding model truncates at 256 tokens
- Retrieval impact of chunking strategy choices

### T04: Evaluation (`T04_evaluation.ipynb`)

Systematic measurement of RAG pipeline quality using RAGAS.

- Part 1 (reference-free): Faithfulness and AnswerRelevancy
- Part 2 (ground truth): ContextPrecision and ContextRecall
- Shared queries defined in `evaluation.py`

### T05: Dataset Creation (`T05_synthetic_dataset.ipynb`)

Scaling the evaluation dataset beyond manual queries.

- Manual "golden query" reference set
- Synthetic Q&A generation from vector store chunks via LLM

### T06: Structured Outputs (`T06_structured_outputs.ipynb`)

Making claim quality machine-readable.

- Three prompting approaches: baseline prose, entity-grounding, JSON mode
- Pydantic validation layer for output reliability
- Evidence levels: VERIFIED / CLAIMED / MISSING / MIXED

### T07: Advanced Retrieval (`T07_advanced_retrieval.ipynb`)

Four retrieval strategies compared side-by-side.

- Baseline semantic search
- BM25 (exact keyword matching)
- Hybrid (semantic + BM25 via Reciprocal Rank Fusion)
- Metadata filtering (scope search to known documents)

### T08–T09: Conversation and Query Techniques

- `T08`: Multi-turn conversation management with history
- `T09`: Query expansion and HyDE for better semantic retrieval

### T10–T14: Tools and Agents

Multi-step agent patterns for complex queries.

- `T10`: LLM tool use fundamentals
- `T11`: LLM agents
- `T12`: RAG as a tool
- `T13`: RAG agent
- `T14`: RAG subagent for parallel multi-document queries

---

## Running the backend server

`main.py` starts a FastAPI server that wires together the RAG agent and in-memory conversation databases:

```bash
# From the project root
cd backend
BACKEND=openai python -m ki_kompass_rag.main

# Or with uvicorn for hot reload during development
uvicorn ki_kompass_rag.main:app --reload --port 8080
```

The server starts at `http://localhost:8080`. The frontend (in `frontend/`) connects to it automatically.

---

## Running the pipeline from the command line

```bash
# Default (OpenAI)
python -m ki_kompass_rag.baseline_rag

# Ollama backend
BACKEND=ollama python -m ki_kompass_rag.baseline_rag

# Custom query
QUERY="Which tape products have a verified EPD?" \
python -m ki_kompass_rag.baseline_rag

# Force rebuild of the vector store
RESET_VS=1 python -m ki_kompass_rag.baseline_rag

# Override model
MODEL=gpt-4o BACKEND=openai python -m ki_kompass_rag.baseline_rag
```

The vector store is written to `backend/db/data_vs.db/` by default. Set `DB_DIR=/path/to/db` to use a different location. On subsequent runs, re-embedding is skipped if the store already exists (`RESET_VS=1` forces a rebuild).

---

## LLM backends

| Backend  | Environment variable | Default model      |
|----------|----------------------|--------------------|
| `openai` | `OPENAI_API_KEY`     | `gpt-4o-mini`      |
| `ollama` | —                    | `mistral-nemo:12b` |

Set `BACKEND=<name>` and optionally `MODEL=<model-name>` as environment variables before running any feature module.
