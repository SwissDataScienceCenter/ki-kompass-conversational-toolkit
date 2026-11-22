# ki_kompass_rag

[English version](README.md)

Die Anwendung von `conversational_toolkit` auf die PrimePack AG: die Pipeline-Module, die die Notebooks importieren, das Skript, das die Vector Stores der Chat-App baut, und das FastAPI-Backend der Chat-App. Setup und Übersicht der Notebooks stehen im [README im Hauptverzeichnis](../README.de.md).

## Aufbau

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

## baseline_rag

| Name | Zweck |
|---|---|
| `ROOT`, `DATA_DIR`, `DB_DIR`, `VS_PATH` | Hauptverzeichnis des Repos, `data/`, `backend/db/` (mit `DB_DIR` änderbar), `DB_DIR/data_vs.db` |
| `build_llm(backend, model_name=None, ...)` | LLM für `openai` oder `ollama` |
| `load_chunks(max_files=None, with_images=False)` | Chunkt jede unterstützte Datei in `data/` ausser `EVALUATION_*`; überspringt Dateien über 20 MB |
| `build_vector_store(chunks, embedding_model, db_path, reset)` | Embeddet Chunks in einen ChromaDB-Store; tut nichts, wenn der Store nicht leer und `reset` false ist |
| `load_or_build_vector_store(reset=False)` | Öffnet `VS_PATH` und baut den Store aus allen Dokumenten, wenn er leer ist |
| `inspect_retrieval(query, vector_store, embedding_model)` | Gibt die Top-k-Chunks mit Scores aus |
| `build_agent(vector_store, embedding_model, llm, top_k, system_prompt, ...)` | `RAG`-Agent, optional mit Query Expansion und HyDE |
| `ask(agent, query, history=None)` | Beantwortet eine Frage, gibt Antwort und Quelldateien aus und liefert die `AgentAnswer` zurück |
| `run_pipeline(backend, model_name=None, query=..., reset_vs=False)` | Alles oben in einem Aufruf |

Kommandozeile, mit dem gemeinsamen Store unter `VS_PATH`:

```bash
BACKEND=openai uv run python -m ki_kompass_rag.baseline_rag
BACKEND=openai QUERY="Which tape products have a verified EPD?" uv run python -m ki_kompass_rag.baseline_rag
BACKEND=openai RESET_VS=1 uv run python -m ki_kompass_rag.baseline_rag   # rebuild the store
```

`BACKEND` muss hier gesetzt sein; es gibt keinen Standardwert.

## Chat-App

```bash
uv run python -m ki_kompass_rag.build_app_db                # or: make build-db
uv run python -m ki_kompass_rag.build_app_db --skip-images  # text store only
uv run python -m ki_kompass_rag.main                        # or: make run
uv run uvicorn ki_kompass_rag.main:app --reload --port 8080 # with auto-reload
```

`build_app_db` baut `DB_DIR/vs_text` (Text-Chunks, `EMBEDDING_MODEL`) und `DB_DIR/vs_image` (aus den PDFs extrahierte Bild-Chunks, Qwen3-VL auf der CPU) von Grund auf neu.

`main` stellt die API des Toolkits auf `0.0.0.0:8080` bereit. Der Agent ist ein `RAG` über einem hybriden Retriever (semantische Suche in `vs_text` plus BM25, vereint mit RRF) und einem Bild-Retriever über `vs_image`. Er antwortet in einem JSON-Schema mit der Antwort, den IDs der verwendeten Quellen und Folgefragen; zurückgegeben werden nur die zitierten Quellen. Qwen3-VL wird beim Start geladen, auch wenn `vs_image` leer ist. `BACKEND` ist standardmässig `openai`. Konversationen werden als JSON-Dateien in `DB_DIR` gespeichert, Logs gehen nach `<repo>/logs/api.log`. Das Frontend erreicht das Backend über `SERVER_URL` in `frontend/.env`.

## LLM-Backends

| `BACKEND` | Standardmodell | Voraussetzung |
|---|---|---|
| `openai` | `gpt-4o-mini` | `OPENAI_API_KEY` |
| `ollama` | `mistral-nemo:12b` | `ollama serve` auf `localhost:11434` und das heruntergeladene Modell |

`MODEL` ersetzt das Standardmodell in der Chat-App und in der Kommandozeile von `baseline_rag`, z. B. `BACKEND=ollama MODEL=<name> make run` für ein beliebiges Chat-Modell aus der [Ollama-Bibliothek](https://ollama.com/library).

## Embedding-Modelle

`EMBEDDING_MODEL` wählt das Embedding-Modell für den Store der Notebooks, die Kommandozeile und die Chat-App. Ein Name, der mit `text-embedding-` beginnt, verwendet die OpenAI-API (Standard `text-embedding-3-small`); jeder andere Name lädt ein lokales [sentence-transformers](https://huggingface.co/models?library=sentence-transformers)-Modell, das beim ersten Aufruf heruntergeladen wird.

Jeder Store speichert, mit welchem Modell er gebaut wurde. `load_or_build_vector_store()` baut `data_vs.db` neu, wenn sich das Modell ändert. Die Chat-App startet erst, wenn `make build-db` `vs_text` mit dem neuen Modell neu gebaut hat. Bild-Embeddings verwenden immer Qwen3-VL.
