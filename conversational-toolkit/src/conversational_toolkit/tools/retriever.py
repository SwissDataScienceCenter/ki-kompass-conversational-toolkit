import asyncio
from typing import Any

from conversational_toolkit.llms.base import LLM
from conversational_toolkit.retriever.base import Retriever
from conversational_toolkit.utils.retriever import (
    make_query_standalone,
    query_expansion,
    reciprocal_rank_fusion,
)
from conversational_toolkit.tools.base import Tool
from conversational_toolkit.vectorstores.base import ChunkRecord


class RetrieverTool(Tool):
    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        llm: LLM,
        retriever: Retriever[ChunkRecord],
        number_query_expansion: int = 0,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.llm = llm
        self.retriever = retriever
        self.number_query_expansion = number_query_expansion

    async def call(self, args: dict[str, Any]) -> dict[str, Any]:
        history = args.get("_history", [])
        query = str(args.get("_query"))

        if history:
            query = await make_query_standalone(self.llm, history, query)
        queries = [query]
        if self.number_query_expansion > 0:
            queries += await query_expansion(query, self.llm, self.number_query_expansion)

        retrieved = await asyncio.gather(*(self.retriever.retrieve(q) for q in queries))
        sources = reciprocal_rank_fusion(retrieved)[: self.retriever.top_k]
        return {
            "result": "\n\n".join(f"<source id='{s.id}'>{s.content}</source>" for s in sources),
            "_sources": [
                {
                    "id": s.id,
                    "title": s.title,
                    "content": s.content,
                    "mime_type": s.mime_type,
                    "metadata": {**s.metadata, "id": s.id},
                }
                for s in sources
            ],
        }
