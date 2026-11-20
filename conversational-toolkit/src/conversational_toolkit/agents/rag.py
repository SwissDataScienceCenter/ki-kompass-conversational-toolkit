import asyncio
from collections.abc import AsyncGenerator
from typing import Any

from conversational_toolkit.agents.base import Agent, AgentAnswer, QueryWithContext
from conversational_toolkit.llms.base import LLM, LLMMessage, MessageContent, Roles
from conversational_toolkit.retriever.base import Retriever
from conversational_toolkit.utils.retriever import (
    hyde_expansion,
    make_query_standalone,
    query_expansion,
    reciprocal_rank_fusion,
)
from conversational_toolkit.vectorstores.base import ChunkRecord


def _source_message(source: ChunkRecord) -> LLMMessage:
    if "text" in source.mime_type:
        content = [MessageContent(type="text", text=f"<source id='{source.id}'>{source.content}</source>")]
    elif "image" in source.mime_type:
        content = [
            MessageContent(type="text", text=f"<source id='{source.id}' type='image'>"),
            MessageContent(type="image", image_url=source.content),
            MessageContent(type="text", text="</source>"),
        ]
    else:
        raise ValueError(f"Unsupported MIME type: {source.mime_type}")
    return LLMMessage(role=Roles.USER, content=content)


class RAG(Agent):
    """Rewrites the query, retrieves from every retriever, fuses the rankings with RRF and answers from the sources.

    Attributes:
        utility_llm: Cheaper LLM for query rewriting, expansion and HyDE.
        number_query_expansion: Extra search queries to generate (0 disables expansion).
    """

    def __init__(
        self,
        llm: LLM,
        utility_llm: LLM,
        retrievers: list[Retriever[Any]],
        system_prompt: str,
        description: str = "",
        number_query_expansion: int = 0,
        enable_hyde: bool = False,
    ):
        super().__init__(system_prompt, llm, description)
        self.utility_llm = utility_llm
        self.retrievers = retrievers
        self.number_query_expansion = number_query_expansion
        self.enable_hyde = enable_hyde

    async def answer_stream(self, query_with_context: QueryWithContext) -> AsyncGenerator[AgentAnswer, None]:
        query = query_with_context.query
        history = query_with_context.history

        if history:
            query = await make_query_standalone(self.utility_llm, history, query)

        queries = [query]
        if self.number_query_expansion > 0:
            queries += await query_expansion(query, self.utility_llm, self.number_query_expansion)
        if self.enable_hyde:
            queries.append(await hyde_expansion(query, self.utility_llm))

        per_retriever = await asyncio.gather(
            *(asyncio.gather(*(r.retrieve(q) for q in queries)) for r in self.retrievers)
        )
        sources: list[ChunkRecord] = []
        for retriever, retrieved in zip(self.retrievers, per_retriever, strict=True):
            sources += reciprocal_rank_fusion(retrieved)[: retriever.top_k]

        response_stream = self.llm.generate_stream(
            [
                LLMMessage(role=Roles.SYSTEM, content=[MessageContent(type="text", text=self.system_prompt)]),
                *history,
                *map(_source_message, sources),
                LLMMessage(role=Roles.USER, content=[MessageContent(type="text", text=query)]),
            ]
        )

        content = ""
        async for response_chunk in response_stream:
            if not response_chunk.content:
                continue
            for message_content in response_chunk.content:
                if message_content.type == "text" and message_content.text:
                    content += message_content.text
                elif message_content.type == "image" and message_content.image_url:
                    raise NotImplementedError("Image output from LLM is not supported.")
            yield await self._answer_post_processing(
                AgentAnswer(content=[MessageContent(type="text", text=content)], role=Roles.ASSISTANT, sources=sources)
            )
