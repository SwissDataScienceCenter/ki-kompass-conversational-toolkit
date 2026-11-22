"""Chat app server: uvicorn on :8080 over the stores built by ki_kompass_rag.build_app_db."""

import os
from textwrap import dedent
from typing import Any

import uvicorn
from loguru import logger

from conversational_toolkit.agents.base import AgentAnswer
from conversational_toolkit.agents.rag import RAG
from conversational_toolkit.api.server import create_app
from conversational_toolkit.conversation_database.controller import (
    ConversationalToolkitController,
)
from conversational_toolkit.conversation_database.in_memory.conversation import (
    InMemoryConversationDatabase,
)
from conversational_toolkit.conversation_database.in_memory.message import (
    InMemoryMessageDatabase,
)
from conversational_toolkit.conversation_database.in_memory.reactions import (
    InMemoryReactionDatabase,
)
from conversational_toolkit.conversation_database.in_memory.source import (
    InMemorySourceDatabase,
)
from conversational_toolkit.conversation_database.in_memory.user import (
    InMemoryUserDatabase,
)
from conversational_toolkit.embeddings.qwen_vl import Qwen3VLEmbeddings
from conversational_toolkit.llms.base import MessageContent
from conversational_toolkit.retriever.bm25_retriever import BM25Retriever
from conversational_toolkit.retriever.hybrid_retriever import HybridRetriever
from conversational_toolkit.retriever.vectorstore_retriever import VectorStoreRetriever
from conversational_toolkit.vectorstores.chromadb import ChromaDBVectorStore

from ki_kompass_rag.baseline_rag import (
    DB_DIR,
    EMBEDDING_MODEL,
    RETRIEVER_TOP_K,
    ROOT,
    build_embeddings,
    build_llm,
    store_embedding_model,
)
from ki_kompass_rag.build_app_db import IMAGE_VS_PATH, TEXT_VS_PATH
from ki_kompass_rag.utils.json import parse_llm_json_stream

logger.add(ROOT / "logs" / "api.log", rotation="50 MB")

BACKEND = os.getenv("BACKEND", "openai")
DB_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = dedent("""
    You are a sustainability compliance assistant for PrimePack AG.
    Answer questions using ONLY the provided sources. If no sources are relevant, say you don't know.
    NEVER use general knowledge unless the user explicitly asks for it.

    RULES (apply in order):
    1. Identify the key entity in the question (product name, supplier, product ID).
    2. Check that this exact entity appears in the retrieved sources.
       If it does NOT appear, respond: "The sources do not contain information about
       [entity]. I cannot answer this question." Do not substitute other products.
    3. Distinguish clearly between:
       VERIFIED: backed by a third-party EPD or independent audit
       CLAIMED: supplier self-declaration, not independently verified
       MISSING: not found in sources
    4. Label forward-looking targets (e.g. "carbon neutral by 2025") as targets,
       not as current verified status.
    5. Always cite the source document for each claim.
""").strip()


class CustomRAG(RAG):
    async def _answer_post_processing(self, answer: AgentAnswer) -> AgentAnswer:
        raw_text = (answer.content[0].text if answer.content else "") or ""
        json_answer: dict[str, Any] = parse_llm_json_stream(raw_text) or {}

        used_ids = set(json_answer.get("used_sources_id", []))
        sources = {
            source_id: s
            for s in answer.sources
            if (source_id := getattr(s, "id", None)) in used_ids
        }
        return answer.model_copy(
            update={
                "content": [
                    MessageContent(type="text", text=json_answer.get("answer", ""))
                ],
                "sources": list(sources.values()),
                "follow_up_questions": json_answer.get("follow_up_questions", []),
            }
        )


ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "The answer to the user's question in markdown format. When referencing a supplier or material source, provide a clickable link to the relevant compliance documentation or evidence page.",
        },
        "used_sources_id": {
            "type": "array",
            "description": "List of source IDs used to generate the answer. Do not invent them and give the exact id of the source which was used.",
            "items": {"type": "string"},
        },
        "follow_up_questions": {
            "type": "array",
            "description": "Follow-up questions the USER might want to ask, based on the sources used for this answer. Empty if no sources were used.",
            "items": {"type": "string"},
        },
    },
    "required": ["answer", "used_sources_id", "follow_up_questions"],
    "additionalProperties": False,
}


def build_server():
    text_vs = ChromaDBVectorStore(db_path=str(TEXT_VS_PATH))
    if (built_with := store_embedding_model(text_vs)) != EMBEDDING_MODEL:
        raise SystemExit(
            f"{TEXT_VS_PATH} was built with {built_with}, not {EMBEDDING_MODEL}; run make build-db."
        )
    text_embedding_model = build_embeddings()
    image_embedding_model = Qwen3VLEmbeddings()
    image_vs = ChromaDBVectorStore(db_path=str(IMAGE_VS_PATH))

    hybrid_retriever = HybridRetriever(
        retrievers=[
            VectorStoreRetriever(text_embedding_model, text_vs, top_k=RETRIEVER_TOP_K),
            BM25Retriever(text_vs, top_k=RETRIEVER_TOP_K),
        ],
        top_k=RETRIEVER_TOP_K,
    )

    image_retriever = VectorStoreRetriever(
        image_embedding_model, image_vs, top_k=RETRIEVER_TOP_K
    )

    llm = build_llm(
        backend=BACKEND,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "AnswerSchema",
                "schema": ANSWER_SCHEMA,
                "strict": True,
            },
        },
    )
    agent = CustomRAG(
        llm=llm,
        # Query rewriting needs plain text, not the answer schema.
        utility_llm=build_llm(backend=BACKEND),
        system_prompt=SYSTEM_PROMPT,
        retrievers=[hybrid_retriever, image_retriever],
    )

    controller = ConversationalToolkitController(
        conversation_db=InMemoryConversationDatabase(
            str(DB_DIR / "conversations.json")
        ),
        message_db=InMemoryMessageDatabase(str(DB_DIR / "messages.json")),
        reaction_db=InMemoryReactionDatabase(str(DB_DIR / "reactions.json")),
        source_db=InMemorySourceDatabase(str(DB_DIR / "sources.json")),
        user_db=InMemoryUserDatabase(str(DB_DIR / "users.json")),
        agent=agent,
    )

    return create_app(controller=controller)


app = build_server()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
