import json
from textwrap import dedent
from typing import Any

from loguru import logger

from conversational_toolkit.llms.base import LLM, LLMMessage, MessageContent, Roles
from conversational_toolkit.retriever.base import Retriever
from conversational_toolkit.vectorstores.base import ChunkMatch, ChunkRecord


class RerankingRetriever(Retriever[ChunkMatch]):
    """Asks an LLM to reorder the base retriever's candidates; falls back to the original order if that fails.

    Set the base retriever's 'top_k' to the candidate pool size (e.g. 3-4x this 'top_k'). Scores decay linearly
    from 1.0 at rank 1.
    """

    def __init__(self, retriever: Retriever[Any], llm: LLM, top_k: int) -> None:
        super().__init__(top_k)
        self.retriever = retriever
        self.llm = llm

    async def retrieve(self, query: str) -> list[ChunkMatch]:
        candidates: list[ChunkRecord] = await self.retriever.retrieve(query)  # type: ignore[assignment]
        if not candidates:
            return []

        ranking = await self._llm_rerank(query, candidates)
        n = len(ranking)
        return [
            ChunkMatch(**candidates[i].model_dump(exclude={"score"}), score=(n - position) / n)
            for position, i in enumerate(ranking[: self.top_k])
        ]

    async def _llm_rerank(self, query: str, candidates: list[ChunkRecord]) -> list[int]:
        numbered = "\n\n".join(
            f"[{i}] {chunk.title or '(no title)'}\n{chunk.content[:400]}" for i, chunk in enumerate(candidates)
        )
        prompt = dedent(f"""
            Query: {query}

            Rank the following {len(candidates)} document chunks from most to least relevant
            to the query. Output only a JSON object with a single key "ranking" whose value
            is a list of the chunk indices ordered from most to least relevant.

            Chunks:
            {numbered}

            Output format: {{"ranking": [most_relevant_index, second_index, ...]}}
        """).strip()

        messages = [
            LLMMessage(
                role=Roles.SYSTEM,
                content=[
                    MessageContent(
                        type="text", text="You are an expert at assessing document relevance. Output only valid JSON."
                    )
                ],
            ),
            LLMMessage(role=Roles.USER, content=[MessageContent(type="text", text=prompt)]),
        ]

        try:
            response = await self.llm.generate(messages)
            data = json.loads(response.content[0].text or "")
            ranking: list[int] = [int(i) for i in data["ranking"] if 0 <= int(i) < len(candidates)]
            seen = set(ranking)
            ranking += [i for i in range(len(candidates)) if i not in seen]
            return ranking
        except Exception as exc:
            logger.warning(f"RerankingRetriever LLM call failed, using original order: {exc}")
            return list(range(len(candidates)))
