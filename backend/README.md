# ki_kompass_rag

[Deutsche Version](README.de.md)

The PrimePack AG application of `conversational_toolkit`: the pipeline modules the notebooks import, the script that builds the chat app's vector stores, and the FastAPI chat backend. Setup and the notebook overview are in the [root README](../README.md).

## Adapting to your own documents

This package is where the toolkit is applied to one use case. To use it for your own documents and questions, change it here; `conversational_toolkit` stays as it is.

| What | Where |
|---|---|
| Documents | Replace the files in `data/`; PDF, Excel, Markdown and text files are supported |
| File types and chunking | `_CHUNKERS` in `baseline_rag.py`; strategies to compare in `ingestion.py` (T02) |
| System prompts | `SYSTEM_PROMPT` in `baseline_rag.py` (notebooks, command line) and in `main.py` (chat app) |
| Answer format of the chat app | `ANSWER_SCHEMA` in `main.py` |
| Test questions | `EVALUATION_QUERIES` in `evaluation.py` (T04, T05) |
| Tools | `tool_agents.py` (T10 to T14) |
| Models | `BACKEND`, `MODEL` and `EMBEDDING_MODEL`, see [Models](#models) |
| Chat app name and texts | `frontend/src/config.ts` and `frontend/src/lib/lang/` |

After changing the documents or the chunking, rebuild the stores: `load_or_build_vector_store(reset=True)` for the notebooks and `make build-db` for the chat app.

## Layout

```
backend/
├── notebooks/english/        T01 to T14
├── notebooks/german/         T01 to T14, German text
├── db/                       created at runtime, git-ignored
│   ├── data_vs.db/           notebook and CLI store (text, EMBEDDING_MODEL)
│   ├── vs_text/              chat app text store (EMBEDDING_MODEL)
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

## Command line

```bash
BACKEND=openai uv run python -m ki_kompass_rag.baseline_rag
BACKEND=openai QUERY="Which tape products have a verified EPD?" uv run python -m ki_kompass_rag.baseline_rag
```

`RESET_VS=1` rebuilds the notebook store.

## Chat app

`make build-db` builds `vs_text` and `vs_image`; `uv run python -m ki_kompass_rag.build_app_db --skip-images` builds the text store only. `make run` serves the API on port 8080. The agent combines hybrid retrieval over `vs_text` (semantic search plus BM25) with image retrieval over `vs_image` and answers in `ANSWER_SCHEMA`. Conversations are stored as JSON files in `DB_DIR`, logs in `<repo>/logs/api.log`.

## Models

- `BACKEND`: `openai` (default model `gpt-4o-mini`) or `ollama`. Required for the command line; the chat app defaults to `openai`.
- `MODEL`: overrides the LLM. Required with `ollama`: any chat model from the [Ollama library](https://ollama.com/library), pulled and served on `localhost:11434`.
- `EMBEDDING_MODEL`: default `text-embedding-3-small`. Names starting with `text-embedding-` use OpenAI; any other name loads a local [sentence-transformers](https://huggingface.co/models?library=sentence-transformers) model. Each store records its model: the notebook store is rebuilt when the model changes, and the chat app asks for `make build-db`. Image embeddings always use Qwen3-VL.
