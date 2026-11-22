# ki_kompass_rag

[English version](README.md)

Die Anwendung von `conversational_toolkit` auf die PrimePack AG: die Pipeline-Module, die die Notebooks importieren, das Skript, das die Vector Stores der Chat-App baut, und das FastAPI-Backend der Chat-App. Setup und Übersicht der Notebooks stehen im [README im Hauptverzeichnis](../README.de.md).

## Eigene Dokumente verwenden

In diesem Paket wird das Toolkit auf einen Anwendungsfall angewendet. Wer es für eigene Dokumente und Fragen nutzen will, passt es hier an; `conversational_toolkit` bleibt unverändert.

| Was | Wo |
|---|---|
| Dokumente | Die Dateien in `data/` ersetzen; unterstützt werden PDF-, Excel-, Markdown- und Textdateien |
| Dateitypen und Chunking | `_CHUNKERS` in `baseline_rag.py`; Strategien zum Vergleichen in `ingestion.py` (T02) |
| System-Prompts | `SYSTEM_PROMPT` in `baseline_rag.py` (Notebooks, Kommandozeile) und in `main.py` (Chat-App) |
| Antwortformat der Chat-App | `ANSWER_SCHEMA` in `main.py` |
| Testfragen | `EVALUATION_QUERIES` in `evaluation.py` (T04, T05) |
| Tools | `tool_agents.py` (T10 bis T14) |
| Modelle | `BACKEND`, `MODEL` und `EMBEDDING_MODEL`, siehe [Modelle](#modelle) |
| Name und Texte der Chat-App | `frontend/src/config.ts` und `frontend/src/lib/lang/` |

Nach Änderungen an den Dokumenten oder am Chunking die Stores neu bauen: `load_or_build_vector_store(reset=True)` für die Notebooks und `make build-db` für die Chat-App.

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

## Kommandozeile

```bash
BACKEND=openai uv run python -m ki_kompass_rag.baseline_rag
BACKEND=openai QUERY="Which tape products have a verified EPD?" uv run python -m ki_kompass_rag.baseline_rag
```

`RESET_VS=1` baut den Store der Notebooks neu.

## Chat-App

`make build-db` baut `vs_text` und `vs_image`; `uv run python -m ki_kompass_rag.build_app_db --skip-images` baut nur den Text-Store. `make run` stellt die API auf Port 8080 bereit. Der Agent kombiniert hybrides Retrieval über `vs_text` (semantische Suche plus BM25) mit Bild-Retrieval über `vs_image` und antwortet nach `ANSWER_SCHEMA`. Konversationen werden als JSON-Dateien in `DB_DIR` gespeichert, Logs in `<repo>/logs/api.log`.

## Modelle

- `BACKEND`: `openai` (Standardmodell `gpt-4o-mini`) oder `ollama`. Für die Kommandozeile Pflicht; die Chat-App verwendet standardmässig `openai`.
- `MODEL`: ersetzt das LLM. Pflicht bei `ollama`: ein beliebiges Chat-Modell aus der [Ollama-Bibliothek](https://ollama.com/library), heruntergeladen und über `localhost:11434` bereitgestellt.
- `EMBEDDING_MODEL`: Standard `text-embedding-3-small`. Namen, die mit `text-embedding-` beginnen, verwenden OpenAI; jeder andere Name lädt ein lokales [sentence-transformers](https://huggingface.co/models?library=sentence-transformers)-Modell. Jeder Store speichert sein Modell: Der Store der Notebooks wird bei einem Modellwechsel neu gebaut, die Chat-App verlangt `make build-db`. Bild-Embeddings verwenden immer Qwen3-VL.
