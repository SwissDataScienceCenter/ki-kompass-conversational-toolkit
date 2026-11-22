# conversational_toolkit

[Deutsche Version](README.de.md)

Building blocks for retrieval-augmented conversational agents: chunkers, embedding models, vector stores, retrievers, LLM clients, agents, evaluation metrics, conversation storage and a FastAPI server for the chat frontend in `../frontend`.

It is installed as an editable workspace member by `uv sync` at the repository root. The Postgres vector store and conversation database need the `postgres` extra: `uv sync --all-packages --extra postgres`.

## Architecture

```
documents -> Chunker -> EmbeddingsModel -> VectorStore
                                               |
query -> Agent <-------------------------- Retriever(s)
           |
          LLM (+ Tools)
           |
ConversationalToolkitController (conversation storage) -> create_app (FastAPI) -> frontend
```

Each component is an abstract base class; pick an implementation per row and combine them.

| Component | Base class | Implementations | Module |
|---|---|---|---|
| Chunker | `Chunker` | `PDFChunker`, `MarkdownChunker`, `ExcelChunker`, `JSONLinesChunker` | `chunking` |
| Embeddings | `EmbeddingsModel` | `OpenAIEmbeddings`, `SentenceTransformerEmbeddings`, `Qwen3VLEmbeddings` (text and images); `CLIPEmbeddings` (standalone) | `embeddings` |
| Vector store | `VectorStore` | `ChromaDBVectorStore`, `PGVectorStore` | `vectorstores` |
| Retriever | `Retriever[T]` | `VectorStoreRetriever`, `CompositeVectorStoreRetriever`, `BM25Retriever`, `HybridRetriever`, `RerankingRetriever`, `ContextWindowRetriever` | `retriever` |
| LLM | `LLM` | `OpenAILLM` (also any OpenAI-compatible endpoint via `base_url`), `OllamaLLM` | `llms` |
| Agent | `Agent` | `RAG`, `ToolAgent`, `Router` | `agents` |
| Tool | `Tool` | `RetrieverTool`, `EmbeddingsTool` | `tools` |
| Metric | `Metric` | `HitRate`, `MRR`, `PrecisionAtK`, `RecallAtK`, `NDCGAtK`; RAGAS metrics through `evaluate_with_ragas` | `evaluation` |
| Storage | `ConversationDatabase`, `MessageDatabase`, `ReactionDatabase`, `SourceDatabase`, `UserDatabase` | `InMemory*` (JSON files), `PostgreSQL*` | `conversation_database` |
| Auth | `AuthProvider` | `SessionCookieProvider`, `PasscodeProvider` | `api.auth` |

## Minimal RAG app

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

`examples/` contains this setup (`simple_rag.py`), a `ToolAgent` with a `RetrieverTool` on Ollama (`simple_agent.py`) and the same agent behind a passcode (`simple_agent_with_passcode.py`).

## Components

**Agents.** `answer(QueryWithContext(query=..., history=[...]))` returns an `AgentAnswer` (content plus sources); `answer_stream` yields partial answers.
- `RAG` rewrites the query using the history, optionally adds `number_query_expansion` LLM-generated sub-queries and a HyDE query (`enable_hyde=True`), retrieves from every retriever, fuses the results with RRF and generates from the fused sources.
- `ToolAgent(system_prompt, llm, max_steps=20)` calls the tools attached to `llm` until the model answers without a tool call. Chunks a tool returns under `_sources` are attached to the answer.
- `Router(llm, agents)` lets the LLM pick one sub-agent by its `description`; the LLM must answer in JSON (e.g. `response_format={"type": "json_object"}`).

**LLMs.** `generate(messages)` returns an `LLMMessage`; `generate_stream(messages)` yields chunks.
- `OpenAILLM(model_name="gpt-4o-mini", temperature, seed, tools, tool_choice, response_format, api_key=None, base_url=None)` works with OpenAI or any OpenAI-compatible server (vLLM, etc.).
- `OllamaLLM(model_name="llama3.1", temperature, seed, tools, response_format="json" | None, host=None)`.

**Embeddings.** `get_embeddings(text_or_list)` returns an array of shape `(n, dim)`. `OpenAIEmbeddings(model_name, dimensions=None)` returns the model's native size (1536 for `text-embedding-3-small`) unless `dimensions` is set.

**Chunkers.** `make_chunks(path)` returns `Chunk` objects with `title`, `content` (text, or base64 for images), `mime_type` and `metadata`. `PDFChunker` and `MarkdownChunker` set `source_file` and `chunk_index` on text chunks, which `ContextWindowRetriever` uses. `PDFChunker(do_ocr=True, table_mode=ACCURATE, table_cell_matching=True)` converts with docling (or markitdown via `engine=MarkdownConverterEngine.MARKITDOWN`) and splits on headings; `to_markdown(path)` returns the converted text, and `write_images=True` (the default) adds image chunks. `MarkdownChunker` applies the same splitting to Markdown files, `ExcelChunker` produces one Markdown table per sheet, and `JSONLinesChunker.make_chunks(path, title_key, content_key, source_key)` produces one chunk per line.

**Vector stores.** `insert_chunks(chunks, embedding)` stores chunks, and `get_chunks_by_embedding(embedding, top_k, filters=None)` returns `ChunkMatch` objects whose `score` is a similarity (higher is better).
- `ChromaDBVectorStore(db_path, collection_name="default_collection")` creates new collections with cosine distance.
- `PGVectorStore(engine, table_name, embeddings_size)` takes a SQLAlchemy `AsyncEngine`; call `enable_vector_extension()` and `create_table()` once.

**Retrievers.** `retrieve(query)` returns ranked `ChunkMatch` objects.
- `VectorStoreRetriever(embedding_model, vector_store, top_k)`.
- `BM25Retriever(vector_store, top_k)` builds a BM25 index over the store's chunks.
- `HybridRetriever(retrievers, top_k, rrf_k=60)` fuses the results of several retrievers with Reciprocal Rank Fusion.
- `RerankingRetriever(retriever, llm, top_k)` reorders a larger candidate set with an LLM and keeps the base order if the LLM output is invalid.
- `ContextWindowRetriever(retriever, vector_store, window_size=1, top_k=5)` adds neighbouring chunks of the same file.
- `CompositeVectorStoreRetriever(embedding_models, vector_stores, top_k)` queries several stores, each with its own embedding model.

```python
from conversational_toolkit.retriever.bm25_retriever import BM25Retriever
from conversational_toolkit.retriever.hybrid_retriever import HybridRetriever
from conversational_toolkit.retriever.reranking_retriever import RerankingRetriever

hybrid = HybridRetriever([VectorStoreRetriever(embeddings, store, top_k=15), BM25Retriever(store, top_k=15)], top_k=20)
retriever = RerankingRetriever(hybrid, llm=OpenAILLM(model_name="gpt-4o-mini"), top_k=5)
```

**Tools.** A `Tool` declares `name`, `description` and `parameters` (JSON schema) and implements `async call(args) -> dict`. `RetrieverTool(name, description, parameters, llm, retriever, number_query_expansion=0)` exposes a retriever to a `ToolAgent`.

**Evaluation.** `Evaluator(metrics).evaluate(samples)` computes retrieval metrics (`HitRate(k)`, `MRR`, `PrecisionAtK`, `RecallAtK`, `NDCGAtK`) over `EvaluationSample` objects. `Evaluator.build_samples_from_agent` and `build_samples_from_retriever` create the samples. `evaluation.adapters.ragas_adapter.evaluate_with_ragas(samples, metrics)` runs RAGAS metrics such as `Faithfulness` on the same samples.

**Storage.** `InMemory*Database(json_file_path)` keeps data in memory and persists it to a JSON file. The `PostgreSQL*Database(engine)` classes share one SQLAlchemy model base; call `create_table()` on any of them once to create all tables.

**API.** `create_app(controller, auth_provider=None, allow_origins=None, dist_path=..., env="local")` returns a FastAPI app. It uses `SessionCookieProvider` (JWT in an HttpOnly cookie) by default and serves a built frontend from `dist_path`. `PasscodeProvider(auth_provider, passcode)` adds a passcode page in front of another provider.

| Method | Path | |
|---|---|---|
| `POST` | `/api/v1/messages` | Send a message, get the final answer |
| `POST` | `/api/v1/messages/stream` | Send a message, stream the answer |
| `GET` | `/api/v1/conversations` | The current user's conversations |
| `GET`, `PUT`, `DELETE` | `/api/v1/conversations/{id}` | Get, rename or delete a conversation |
| `GET` | `/api/v1/conversations/{id}/messages` | Messages with sources and reactions |
| `POST` | `/api/v1/conversations/{id}/messages/{message_id}/reactions` | Add or replace a reaction |
| `POST` | `/auth/refresh` | Refresh the session token |

## Extending

Subclass the base class and implement its abstract methods: `LLM.generate` and `generate_stream`, `Agent.answer_stream`, `Retriever.retrieve`, `EmbeddingsModel.get_embeddings`, `VectorStore.insert_chunks`, `get_chunks_by_embedding` and `get_chunks_by_filter`, `Tool.call`, or all five storage classes. Any `Retriever` subclass can be passed to `RAG`, `HybridRetriever` and `RerankingRetriever`.

Tests are in `../tests`; development commands are in the root README.
