# conversational_toolkit

[English version](README.md)

Bausteine für konversationelle Agenten mit Retrieval: Chunker, Embedding-Modelle, Vector Stores, Retriever, LLM-Clients, Agenten, Evaluationsmetriken, Speicherung von Konversationen und ein FastAPI-Server für das Chat-Frontend in `../frontend`.

`uv sync` im Hauptverzeichnis des Repos installiert es als editierbares Workspace-Paket. Der Postgres-Vector-Store und die Postgres-Konversationsdatenbank brauchen das Extra `postgres`: `uv sync --all-packages --extra postgres`.

## Architektur

```
documents -> Chunker -> EmbeddingsModel -> VectorStore
                                               |
query -> Agent <-------------------------- Retriever(s)
           |
          LLM (+ Tools)
           |
ConversationalToolkitController (conversation storage) -> create_app (FastAPI) -> frontend
```

Jede Komponente ist eine abstrakte Basisklasse; pro Zeile eine Implementierung wählen und kombinieren.

| Komponente | Basisklasse | Implementierungen | Modul |
|---|---|---|---|
| Chunker | `Chunker` | `PDFChunker`, `MarkdownChunker`, `ExcelChunker`, `JSONLinesChunker` | `chunking` |
| Embeddings | `EmbeddingsModel` | `OpenAIEmbeddings`, `SentenceTransformerEmbeddings`, `Qwen3VLEmbeddings` (Text und Bilder); `CLIPEmbeddings` (eigenständig) | `embeddings` |
| Vector Store | `VectorStore` | `ChromaDBVectorStore`, `PGVectorStore` | `vectorstores` |
| Retriever | `Retriever[T]` | `VectorStoreRetriever`, `CompositeVectorStoreRetriever`, `BM25Retriever`, `HybridRetriever`, `RerankingRetriever`, `ContextWindowRetriever` | `retriever` |
| LLM | `LLM` | `OpenAILLM` (über `base_url` auch jeder OpenAI-kompatible Endpunkt), `OllamaLLM` | `llms` |
| Agent | `Agent` | `RAG`, `ToolAgent`, `Router` | `agents` |
| Tool | `Tool` | `RetrieverTool`, `EmbeddingsTool` | `tools` |
| Metrik | `Metric` | `HitRate`, `MRR`, `PrecisionAtK`, `RecallAtK`, `NDCGAtK`; RAGAS-Metriken über `evaluate_with_ragas` | `evaluation` |
| Speicherung | `ConversationDatabase`, `MessageDatabase`, `ReactionDatabase`, `SourceDatabase`, `UserDatabase` | `InMemory*` (JSON-Dateien), `PostgreSQL*` | `conversation_database` |
| Authentifizierung | `AuthProvider` | `SessionCookieProvider`, `PasscodeProvider` | `api.auth` |

## Minimale RAG-App

```python
from conversational_toolkit.agents.rag import RAG
from conversational_toolkit.api.server import create_app
from conversational_toolkit.chunking.pdf_chunker import PDFChunker
from conversational_toolkit.conversation_database.controller import ConversationalToolkitController
from conversational_toolkit.conversation_database.in_memory.conversation import InMemoryConversationDatabase
from conversational_toolkit.conversation_database.in_memory.message import InMemoryMessageDatabase
from conversational_toolkit.conversation_database.in_memory.reactions import InMemoryReactionDatabase
from conversational_toolkit.conversation_database.in_memory.source import InMemorySourceDatabase
from conversational_toolkit.conversation_database.in_memory.user import InMemoryUserDatabase
from conversational_toolkit.embeddings.openai import OpenAIEmbeddings
from conversational_toolkit.llms.openai import OpenAILLM
from conversational_toolkit.retriever.vectorstore_retriever import VectorStoreRetriever
from conversational_toolkit.vectorstores.chromadb import ChromaDBVectorStore

embeddings = OpenAIEmbeddings(model_name="text-embedding-3-small")
store = ChromaDBVectorStore(db_path="chunks.db")

# Indexing, once (inside an async function):
# chunks = PDFChunker().make_chunks("document.pdf", write_images=False)
# await store.insert_chunks(chunks, await embeddings.get_embeddings([c.content for c in chunks]))

agent = RAG(
    llm=OpenAILLM(model_name="gpt-4o-mini"),
    utility_llm=OpenAILLM(model_name="gpt-4o-mini"),
    retrievers=[VectorStoreRetriever(embeddings, store, top_k=5)],
    system_prompt="Answer only from the provided sources and cite them.",
)

controller = ConversationalToolkitController(
    conversation_db=InMemoryConversationDatabase("conversations.json"),
    message_db=InMemoryMessageDatabase("messages.json"),
    reaction_db=InMemoryReactionDatabase("reactions.json"),
    source_db=InMemorySourceDatabase("sources.json"),
    user_db=InMemoryUserDatabase("users.json"),
    agent=agent,
)
app = create_app(controller)  # serve with: uv run uvicorn myapp:app
```

`examples/` enthält dieses Setup (`simple_rag.py`), einen `ToolAgent` mit einem `RetrieverTool` auf Ollama (`simple_agent.py`) und denselben Agenten hinter einem Passcode (`simple_agent_with_passcode.py`).

## Komponenten

**Agenten.** `answer(QueryWithContext(query=..., history=[...]))` liefert eine `AgentAnswer` (Inhalt plus Quellen); `answer_stream` gibt Teilantworten aus.
- `RAG` schreibt die Frage anhand des Verlaufs um, ergänzt optional `number_query_expansion` vom LLM erzeugte Teilfragen und eine HyDE-Frage (`enable_hyde=True`), sucht mit jedem Retriever, vereint die Ergebnisse mit RRF und generiert die Antwort aus den vereinten Quellen.
- `ToolAgent(system_prompt, llm, max_steps=20)` ruft die Tools des `llm` auf, bis das Modell ohne Tool-Call antwortet. Chunks, die ein Tool unter `_sources` zurückgibt, werden der Antwort angehängt.
- `Router(llm, agents)` lässt das LLM anhand der `description` einen Subagenten wählen; das LLM muss in JSON antworten (z. B. `response_format={"type": "json_object"}`).

**LLMs.** `generate(messages)` liefert eine `LLMMessage`; `generate_stream(messages)` gibt sie in Teilen aus.
- `OpenAILLM(model_name="gpt-4o-mini", temperature, seed, tools, tool_choice, response_format, api_key=None, base_url=None)` funktioniert mit OpenAI oder jedem OpenAI-kompatiblen Server (vLLM usw.).
- `OllamaLLM(model_name="llama3.1", temperature, seed, tools, response_format="json" | None, host=None)`.

**Embeddings.** `get_embeddings(text_or_list)` liefert ein Array der Form `(n, dim)`. `OpenAIEmbeddings(model_name, dimensions=None)` liefert die native Grösse des Modells (1536 bei `text-embedding-3-small`), ausser `dimensions` ist gesetzt.

**Chunker.** `make_chunks(path)` liefert `Chunk`-Objekte mit `title`, `content` (Text oder base64 bei Bildern), `mime_type` und `metadata`. `PDFChunker` und `MarkdownChunker` setzen `source_file` und `chunk_index` auf Text-Chunks; `ContextWindowRetriever` verwendet diese Felder. `PDFChunker(do_ocr=True, table_mode=ACCURATE, table_cell_matching=True)` konvertiert mit docling (oder mit markitdown über `engine=MarkdownConverterEngine.MARKITDOWN`) und teilt an den Überschriften; `to_markdown(path)` liefert den konvertierten Text, und `write_images=True` (Standard) ergänzt Bild-Chunks. `MarkdownChunker` teilt Markdown-Dateien auf dieselbe Weise, `ExcelChunker` erzeugt eine Markdown-Tabelle pro Tabellenblatt, und `JSONLinesChunker.make_chunks(path, title_key, content_key, source_key)` erzeugt einen Chunk pro Zeile.

**Vector Stores.** `insert_chunks(chunks, embedding)` speichert Chunks, und `get_chunks_by_embedding(embedding, top_k, filters=None)` liefert `ChunkMatch`-Objekte, deren `score` eine Ähnlichkeit ist (höher ist besser).
- `ChromaDBVectorStore(db_path, collection_name="default_collection")` legt neue Collections mit Kosinus-Distanz an.
- `PGVectorStore(engine, table_name, embeddings_size)` nimmt eine SQLAlchemy-`AsyncEngine`; `enable_vector_extension()` und `create_table()` einmal aufrufen.

**Retriever.** `retrieve(query)` liefert sortierte `ChunkMatch`-Objekte.
- `VectorStoreRetriever(embedding_model, vector_store, top_k)`.
- `BM25Retriever(vector_store, top_k)` baut einen BM25-Index über die Chunks des Stores.
- `HybridRetriever(retrievers, top_k, rrf_k=60)` vereint die Ergebnisse mehrerer Retriever mit Reciprocal Rank Fusion.
- `RerankingRetriever(retriever, llm, top_k)` sortiert eine grössere Kandidatenmenge mit einem LLM neu und behält die ursprüngliche Reihenfolge, wenn die Ausgabe des LLM ungültig ist.
- `ContextWindowRetriever(retriever, vector_store, window_size=1, top_k=5)` ergänzt benachbarte Chunks derselben Datei.
- `CompositeVectorStoreRetriever(embedding_models, vector_stores, top_k)` fragt mehrere Stores ab, jeden mit seinem eigenen Embedding-Modell.

```python
from conversational_toolkit.retriever.bm25_retriever import BM25Retriever
from conversational_toolkit.retriever.hybrid_retriever import HybridRetriever
from conversational_toolkit.retriever.reranking_retriever import RerankingRetriever

hybrid = HybridRetriever([VectorStoreRetriever(embeddings, store, top_k=15), BM25Retriever(store, top_k=15)], top_k=20)
retriever = RerankingRetriever(hybrid, llm=OpenAILLM(model_name="gpt-4o-mini"), top_k=5)
```

**Tools.** Ein `Tool` legt `name`, `description` und `parameters` (JSON-Schema) fest und implementiert `async call(args) -> dict`. `RetrieverTool(name, description, parameters, llm, retriever, number_query_expansion=0)` stellt einem `ToolAgent` einen Retriever zur Verfügung.

**Evaluation.** `Evaluator(metrics).evaluate(samples)` berechnet Retrieval-Metriken (`HitRate(k)`, `MRR`, `PrecisionAtK`, `RecallAtK`, `NDCGAtK`) über `EvaluationSample`-Objekte. `Evaluator.build_samples_from_agent` und `build_samples_from_retriever` erzeugen die Beispiele. `evaluation.adapters.ragas_adapter.evaluate_with_ragas(samples, metrics)` wendet RAGAS-Metriken wie `Faithfulness` auf dieselben Beispiele an.

**Speicherung.** `InMemory*Database(json_file_path)` hält die Daten im Arbeitsspeicher und speichert sie in einer JSON-Datei. Die Klassen `PostgreSQL*Database(engine)` teilen sich eine SQLAlchemy-Modellbasis; `create_table()` einmal auf einer davon aufrufen, um alle Tabellen anzulegen.

**API.** `create_app(controller, auth_provider=None, allow_origins=None, dist_path=..., env="local")` liefert eine FastAPI-App. Standardmässig verwendet sie `SessionCookieProvider` (JWT in einem HttpOnly-Cookie) und liefert ein gebautes Frontend aus `dist_path` aus. `PasscodeProvider(auth_provider, passcode)` schaltet einem anderen Provider eine Passcode-Seite vor.

| Methode | Pfad | |
|---|---|---|
| `POST` | `/api/v1/messages` | Nachricht senden, fertige Antwort erhalten |
| `POST` | `/api/v1/messages/stream` | Nachricht senden, Antwort streamen |
| `GET` | `/api/v1/conversations` | Konversationen des aktuellen Users |
| `GET`, `PUT`, `DELETE` | `/api/v1/conversations/{id}` | Konversation abrufen, umbenennen oder löschen |
| `GET` | `/api/v1/conversations/{id}/messages` | Nachrichten mit Quellen und Reaktionen |
| `POST` | `/api/v1/conversations/{id}/messages/{message_id}/reactions` | Reaktion hinzufügen oder ersetzen |
| `POST` | `/auth/refresh` | Session-Token erneuern |

## Erweitern

Von der Basisklasse ableiten und ihre abstrakten Methoden implementieren: `LLM.generate` und `generate_stream`, `Agent.answer_stream`, `Retriever.retrieve`, `EmbeddingsModel.get_embeddings`, `VectorStore.insert_chunks`, `get_chunks_by_embedding` und `get_chunks_by_filter`, `Tool.call` oder alle fünf Speicherklassen. Jede Unterklasse von `Retriever` lässt sich an `RAG`, `HybridRetriever` und `RerankingRetriever` übergeben.

Die Tests liegen in `../tests`; die Befehle für die Entwicklung stehen im README im Hauptverzeichnis.
