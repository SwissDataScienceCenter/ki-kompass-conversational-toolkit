# conversational_toolkit

[English version](README.md)

Eine Python-Bibliothek für Chat-Assistenten, die Fragen aus eigenen Dokumenten beantworten (Retrieval-Augmented Generation, RAG) oder Tools verwenden. Sie deckt den ganzen Weg ab: Dokumente in Chunks aufteilen, diese embedden und speichern, die relevanten wiederfinden, mit einem LLM eine Antwort generieren und die Konversation über eine API für das Chat-Frontend in `../frontend` bereitstellen. Jeder Schritt ist eine austauschbare Komponente; so lässt sich zum Beispiel der LLM-Anbieter oder der Vector Store wechseln, ohne den Rest zu ändern.

## Verwendung

Das Toolkit enthält nichts, was an einen bestimmten Anwendungsfall gebunden ist. Dokumente, Prompts, Tools und Testfragen gehören zur Anwendung, die darauf aufbaut. `ki_kompass_rag` in `../backend` ist eine solche Anwendung, für die Beispielfirma PrimePack AG; ihre Notebooks führen die Komponenten Schritt für Schritt ein.

Für eine eigene RAG-Anwendung muss das Toolkit nicht verändert werden. Ein eigenes Paket anlegen, oder von einer Kopie von `../backend` ausgehen, und dort die vorhandenen Komponenten kombinieren:

1. **Dokumente indexieren:** mit einem `Chunker` aufteilen, die Chunks mit einem `EmbeddingsModel` embedden und in einem `VectorStore` speichern.
2. **Agenten bauen:** den Store in einen `Retriever` packen und ihn zusammen mit einem `LLM` und einem System-Prompt einem `RAG`-Agenten übergeben. Oder einem `ToolAgent` Tools zum Aufrufen geben.
3. **Agenten verwenden:** `agent.answer(...)` aus eigenem Code oder Notebooks aufrufen, oder ihn mit `ConversationalToolkitController` und `create_app` als Chat-App bereitstellen.

`examples/simple_rag.py` zeigt alle drei Schritte in einer Datei.

Fehlt etwas, zum Beispiel ein weiterer Dateityp, eine andere Vektordatenbank oder ein anderer LLM-Anbieter, lässt sich das Toolkit erweitern: von der passenden Basisklasse ableiten (siehe [Erweitern](#erweitern)). Die neue Klasse kann in der eigenen Anwendung liegen oder im Toolkit, damit auch andere Anwendungen sie nutzen können.

`uv sync` im Hauptverzeichnis des Repos installiert es als editierbares Workspace-Paket.

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

## Beispiele

`examples/` enthält vollständige Apps mit der Chat-API: einen `RAG`-Agenten über einem ChromaDB-Store (`simple_rag.py`, gestartet mit `make dev` auf Port 8080), einen `ToolAgent` mit einem `RetrieverTool` auf Ollama (`simple_agent.py`) und denselben Agenten hinter einem Passcode (`simple_agent_with_passcode.py`).

## Gut zu wissen

- `ChunkMatch.score` ist in jedem Vector Store eine Ähnlichkeit: höher ist besser. Neue ChromaDB-Collections verwenden die Kosinus-Distanz.
- `RAG` schreibt Folgefragen anhand des Verlaufs in eigenständige Fragen um, kann vom LLM erzeugte Teilfragen (`number_query_expansion`) und eine HyDE-Frage (`enable_hyde`) ergänzen und vereint die Ergebnisse aller Retriever mit Reciprocal Rank Fusion.
- `ToolAgent` schickt Fehler von Tools an das LLM zurück. Schlüssel, die ein Tool mit einem führenden `_` zurückgibt, werden nicht an das LLM geschickt; Chunks unter `_sources` werden der Antwort angehängt.
- `Router` braucht ein LLM, das in JSON antwortet, z. B. `response_format={"type": "json_object"}`.
- Die Postgres-Klassen brauchen das Extra `postgres`. `create_table()` einmal aufrufen (und `enable_vector_extension()` für `PGVectorStore`).
- `create_app` authentifiziert standardmässig über ein Session-Cookie; `PasscodeProvider` schaltet eine Passcode-Seite vor.

## Erweitern

Von der Basisklasse ableiten und ihre abstrakten Methoden implementieren: `LLM.generate` und `generate_stream`, `Agent.answer_stream`, `Retriever.retrieve`, `EmbeddingsModel.get_embeddings`, `VectorStore.insert_chunks`, `get_chunks_by_embedding` und `get_chunks_by_filter`, `Tool.call` oder alle fünf Speicherklassen. Jede Unterklasse von `Retriever` lässt sich an `RAG`, `HybridRetriever` und `RerankingRetriever` übergeben.

Die Tests liegen in `../tests`; die Befehle für die Entwicklung stehen im README im Hauptverzeichnis.
