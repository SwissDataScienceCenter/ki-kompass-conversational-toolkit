# ki_kompass_rag

The PrimePack AG application of `conversational_toolkit`: the pipeline modules the notebooks import, the script that builds the chat app's vector stores, and the FastAPI chat backend. Setup and the notebook overview are in the [root README](../README.md).

## Layout

```
backend/
├── notebooks/english/        T01 to T14
├── notebooks/german/         T01 to T14, German text
├── db/                       created at runtime, git-ignored
│   ├── data_vs.db/           notebook and CLI store (text, OpenAI embeddings)
│   ├── vs_text/              chat app text store (OpenAI embeddings)
│   ├── vs_image/             chat app image store (Qwen3-VL embeddings)
│   └── *.json                chat app conversations, messages, reactions, sources, users
└── src/ki_kompass_rag/
    ├── __init__.py           loads <repo>/.env.local
    ├── baseline_rag.py       paths, build_llm, load_chunks, vector store, RAG agent, ask, CLI
    ├── build_app_db.py       builds vs_text and vs_image for the chat app
    ├── main.py               FastAPI chat backend
    ├── ingestion.py          chunking strategies and chunk-size statistics (T02)
    ├── evaluation.py         EVALUATION_QUERIES with ground-truth answers (T04)
    ├── advanced_retrieval.py BM25, hybrid and metadata-filter retrieval (T07)
    ├── tool_agents.py        example tools and RAG-as-a-tool wrappers (T10 to T14)
    └── utils/json.py         parse_llm_json_stream for partial JSON from a stream
```

## baseline_rag

| Name | Purpose |
|---|---|
| `ROOT`, `DATA_DIR`, `DB_DIR`, `VS_PATH` | Repository root, `data/`, `backend/db/` (override with `DB_DIR`), `DB_DIR/data_vs.db` |
| `build_llm(backend, model_name=None, ...)` | LLM for `openai` or `ollama` |
| `load_chunks(max_files=None, with_images=False)` | Chunk every supported file in `data/` except `EVALUATION_*`; skips files over 20 MB |
| `build_vector_store(chunks, embedding_model, db_path, reset)` | Embed chunks into a ChromaDB store; does nothing if the store is non-empty and `reset` is false |
| `load_or_build_vector_store(reset=False)` | Open `VS_PATH`, building it from all documents if it is empty |
| `inspect_retrieval(query, vector_store, embedding_model)` | Print the top-k chunks with scores |
| `build_agent(vector_store, embedding_model, llm, top_k, system_prompt, ...)` | `RAG` agent, with optional query expansion and HyDE |
| `ask(agent, query, history=None)` | Answer a query, print the answer and source files, return the `AgentAnswer` |
| `run_pipeline(backend, model_name=None, query=..., reset_vs=False)` | All of the above in one call |

Command line, using the shared store at `VS_PATH`:

```bash
BACKEND=openai uv run python -m ki_kompass_rag.baseline_rag
BACKEND=openai QUERY="Which tape products have a verified EPD?" uv run python -m ki_kompass_rag.baseline_rag
BACKEND=openai RESET_VS=1 uv run python -m ki_kompass_rag.baseline_rag   # rebuild the store
```

`BACKEND` is required here; there is no default.

## Chat app

```bash
uv run python -m ki_kompass_rag.build_app_db                # or: make build-db
uv run python -m ki_kompass_rag.build_app_db --skip-images  # text store only
uv run python -m ki_kompass_rag.main                        # or: make run
uv run uvicorn ki_kompass_rag.main:app --reload --port 8080 # with auto-reload
```

`build_app_db` rebuilds `DB_DIR/vs_text` (text chunks, OpenAI embeddings) and `DB_DIR/vs_image` (image chunks extracted from the PDFs, Qwen3-VL on CPU) from scratch.

`main` serves the toolkit API on `0.0.0.0:8080`. The agent is a `RAG` over a hybrid retriever (`vs_text` semantic plus BM25, fused with RRF) and an image retriever over `vs_image`. It answers in a JSON schema containing the answer, the IDs of the sources it used and follow-up questions; only the cited sources are returned. Qwen3-VL is loaded at startup even if `vs_image` is empty. `BACKEND` defaults to `openai`. Conversations are stored as JSON files in `DB_DIR` and logs go to `<repo>/logs/api.log`. The frontend reaches the backend through `SERVER_URL` in `frontend/.env`.

## LLM backends

| `BACKEND` | Default model | Requires |
|---|---|---|
| `openai` | `gpt-4o-mini` | `OPENAI_API_KEY` |
| `ollama` | `mistral-nemo:12b` | `ollama serve` on `localhost:11434` and the pulled model |

The `baseline_rag` command line accepts `MODEL` to override the default; the chat app always uses the default. Embeddings use OpenAI `text-embedding-3-small` with every backend, so `OPENAI_API_KEY` is always required.
