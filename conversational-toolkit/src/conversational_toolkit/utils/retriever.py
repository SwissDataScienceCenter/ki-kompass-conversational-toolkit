import re
from collections import defaultdict
from collections.abc import Sequence
from textwrap import dedent

from loguru import logger

from conversational_toolkit.llms.base import LLM, LLMMessage, MessageContent, Roles
from conversational_toolkit.vectorstores.base import ChunkMatch, ChunkRecord


async def make_query_standalone(llm: LLM, history: list[LLMMessage], query: str) -> str:
    template_query_standalone = dedent(
        """
        Objective: Your task is to analyze the input query and the provided conversation history. When the user mentions something that was mentioned in the conversation, but not clearly said in the current query, rewrite it.

        Example:

            User Query: How much does it cost?
            Conversation History:
                - User: Can you tell me about the price of the new iPhone?
                - Assistant: The new iPhone costs around $999.
            Reformulated Query: How much does the new iPhone cost?


        Input:

            User Query: {query}
            Conversation History:
                {chat_history}

        Reformulated Query (ONLY the reformulated query, without any explanation):
    """
    )
    chat_history = "\n".join(f"{m.role}: {' '.join(c.text for c in m.content if c.text)}" for m in history)

    conversation = [
        LLMMessage(
            role=Roles.SYSTEM,
            content=[
                MessageContent(
                    type="text",
                    text="You are a helpful assistant that transforms a message from a user to be independent from the conversation history given.",
                )
            ],
        ),
        LLMMessage(
            role=Roles.USER,
            content=[
                MessageContent(
                    type="text", text=template_query_standalone.format(query=query, chat_history=chat_history)
                )
            ],
        ),
    ]
    reformulated_query = (await llm.generate(conversation)).content[0].text or ""

    logger.debug(f"Original query: {query}")
    logger.debug(f"Reformulated query: {reformulated_query}")

    return reformulated_query


_LIST_MARKER = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s*")


async def query_expansion(query: str, llm: LLM, expansion_number: int = 2) -> list[str]:
    template_query_expansion = """
        Generate multiple search queries related to: {query}, and translate them in english if they are not already in english. Only output {expansion_number} queries in english.
        OUTPUT ({expansion_number} queries):
    """
    conversation = [
        LLMMessage(
            role=Roles.SYSTEM,
            content=[
                MessageContent(
                    type="text",
                    text="You are a focused assistant designed to generate multiple, relevant search queries based solely on a single input query. Your task is to produce a list of these queries in English, without adding any further explanations or information.",
                )
            ],
        ),
        LLMMessage(
            role=Roles.USER,
            content=[
                MessageContent(
                    type="text", text=template_query_expansion.format(query=query, expansion_number=expansion_number)
                )
            ],
        ),
    ]

    output = (await llm.generate(conversation)).content[0].text or ""
    generated_queries = [_LIST_MARKER.sub("", line).strip() for line in output.splitlines()]
    generated_queries = [q for q in generated_queries if q]

    logger.debug(f"Original query for expansion: {query}")
    for i, generated_query in enumerate(generated_queries, start=1):
        logger.debug(f"Generated query {i}: {generated_query}")

    return generated_queries


async def hyde_expansion(query: str, llm: LLM) -> str:
    conversation = [
        LLMMessage(
            role=Roles.SYSTEM,
            content=[
                MessageContent(
                    type="text",
                    text="You are a helpful assistant. Provide an example of answer to the provided query. Only output an hypothetical explanation to the query. Concise, only a few sentences, without any introduction or conclusion.",
                )
            ],
        ),
        LLMMessage(role=Roles.USER, content=[MessageContent(type="text", text=query)]),
    ]
    hyde_expansion_message = (await llm.generate(conversation)).content[0].text or ""

    logger.debug(f"Original query for HyDE expansion: {query}")
    logger.debug(f"HyDE expansion: {hyde_expansion_message}")

    return hyde_expansion_message


def reciprocal_rank_fusion(rankings: Sequence[Sequence[ChunkRecord]], k: int = 60) -> list[ChunkMatch]:
    """Merge rankings with Reciprocal Rank Fusion; each chunk's score is the sum of 1 / (k + rank)."""
    scores: dict[str, float] = defaultdict(float)
    chunks: dict[str, ChunkRecord] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            scores[chunk.id] += 1 / (k + rank)
            chunks.setdefault(chunk.id, chunk)
    return [
        ChunkMatch(**chunks[cid].model_dump(exclude={"score"}), score=score)
        for cid, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
    ]


def build_query_with_chunks(user_query: str = "", chunks: list[ChunkRecord] | None = None) -> str:
    """Format the query followed by the chunks wrapped in '<source id=...>' tags."""
    if not chunks:
        sources_xml = "No sources found."
    else:
        sources_xml = "\n".join(f'<source id="{chunk.id}">\n{chunk.content}\n</source>' for chunk in chunks)

    return dedent(
        f"""
        User Query: {user_query}\n
        Here are the sources I found for you:
        {sources_xml}
        """
    )
