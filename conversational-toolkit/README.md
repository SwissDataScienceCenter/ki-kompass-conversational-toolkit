# conversational_toolkit

[Deutsche Version](README.de.md)

A Python library for building chat assistants that answer from your own documents (retrieval-augmented generation, RAG) or use tools. It covers the whole path: splitting documents into chunks, embedding and storing them, retrieving the relevant ones, generating an answer with an LLM, and serving the conversation through an API to the chat frontend in `../frontend`. Each step is an interchangeable component, so you can swap, for example, the LLM provider or the vector store without changing the rest.

## How to use it

The toolkit is independent of any use case; documents, prompts and tools belong to the application built on it. `ki_kompass_rag` in `../backend` is one such application.

To build your own RAG application, you don't change the toolkit: combine its components in your own package. `examples/simple_rag.py` shows a complete app in one file. If a component is missing, such as another file type or LLM provider, subclass its base class (see [Extending](#extending)).

It is installed as an editable workspace member by `uv sync` at the repository root.

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

## Examples

`examples/` contains complete apps with the chat API: a `RAG` agent over a ChromaDB store (`simple_rag.py`, started by `make dev` on port 8080), a `ToolAgent` with a `RetrieverTool` on Ollama (`simple_agent.py`) and the same agent behind a passcode (`simple_agent_with_passcode.py`).

## Good to know

- `ChunkMatch.score` is a similarity in every vector store: higher is better. New ChromaDB collections use cosine distance.
- `RAG` rewrites follow-up queries into standalone ones using the history, can add LLM-generated sub-queries (`number_query_expansion`) and a HyDE query (`enable_hyde`), and fuses the results of all retrievers with Reciprocal Rank Fusion.
- `ToolAgent` sends tool errors back to the LLM. Keys a tool returns with a leading `_` are not sent to the LLM; chunks under `_sources` are attached to the answer.
- `Router` needs an LLM that answers in JSON, e.g. `response_format={"type": "json_object"}`.
- The Postgres classes need the `postgres` extra. Call `create_table()` once (and `enable_vector_extension()` for `PGVectorStore`).
- `create_app` authenticates with a session cookie by default; `PasscodeProvider` adds a passcode page in front.

## Extending

Subclass the base class and implement its abstract methods: `LLM.generate` and `generate_stream`, `Agent.answer_stream`, `Retriever.retrieve`, `EmbeddingsModel.get_embeddings`, `VectorStore.insert_chunks`, `get_chunks_by_embedding` and `get_chunks_by_filter`, `Tool.call`, or all five storage classes. Any `Retriever` subclass can be passed to `RAG`, `HybridRetriever` and `RerankingRetriever`.

Tests are in `../tests`; development commands are in the root README.
