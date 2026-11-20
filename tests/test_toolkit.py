import asyncio
import math
from collections.abc import AsyncGenerator
from typing import Any

import numpy as np
import pytest

from conversational_toolkit.agents.base import Agent, AgentAnswer, QueryWithContext
from conversational_toolkit.agents.router import Router
from conversational_toolkit.agents.tool_agent import ToolAgent
from conversational_toolkit.chunking.base import Chunk
from conversational_toolkit.conversation_database.data_models.message import Message
from conversational_toolkit.conversation_database.in_memory.message import (
    InMemoryMessageDatabase,
)
from conversational_toolkit.evaluation.adapters import ragas_adapter
from conversational_toolkit.evaluation.data_models import EvaluationSample
from conversational_toolkit.llms.base import (
    LLM,
    Function,
    LLMMessage,
    MessageContent,
    Roles,
    ToolCall,
)
from conversational_toolkit.tools.base import Tool
from conversational_toolkit.utils.retriever import (
    query_expansion,
    reciprocal_rank_fusion,
)
from conversational_toolkit.vectorstores.base import ChunkRecord
from conversational_toolkit.vectorstores.chromadb import ChromaDBVectorStore


def text(message: str) -> list[MessageContent]:
    return [MessageContent(type="text", text=message)]


class ScriptedLLM(LLM):
    """Returns the scripted responses in order and records every conversation it sees."""

    def __init__(self, responses: list[LLMMessage], tools: list[Tool] | None = None):
        super().__init__()
        self.responses = responses
        self.tools = tools
        self.seen: list[list[LLMMessage]] = []

    async def generate(self, conversation: list[LLMMessage]) -> LLMMessage:
        self.seen.append(list(conversation))
        return self.responses.pop(0)

    async def generate_stream(
        self, conversation: list[LLMMessage]
    ) -> AsyncGenerator[LLMMessage, None]:
        yield await self.generate(conversation)


def record(chunk_id: str) -> ChunkRecord:
    return ChunkRecord(
        id=chunk_id, title="", content=chunk_id, mime_type="text/plain", embedding=[]
    )


def test_reciprocal_rank_fusion_orders_by_summed_score():
    fused = reciprocal_rank_fusion(
        [[record("a"), record("b")], [record("b"), record("c")]]
    )

    assert [c.id for c in fused] == ["b", "a", "c"]
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)


def test_query_expansion_strips_numbering_and_blank_lines():
    llm = ScriptedLLM(
        [LLMMessage(content=text("1. first query\n\n2) second query\n- third\n"))]
    )

    assert asyncio.run(query_expansion("q", llm, 3)) == [
        "first query",
        "second query",
        "third",
    ]


class Failing(Tool):
    name = "failing"
    description = ""
    parameters: dict[str, Any] = {}  # noqa: RUF012

    async def call(self, args: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("boom")


class Retrieving(Tool):
    name = "retrieving"
    description = ""
    parameters: dict[str, Any] = {}  # noqa: RUF012

    async def call(self, args: dict[str, Any]) -> dict[str, Any]:
        return {
            "result": "text",
            "_sources": [{"title": "t", "content": "c", "mime_type": "text/plain"}],
        }


def tool_call(name: str) -> LLMMessage:
    return LLMMessage(
        content=[],
        tool_calls=[
            ToolCall(
                id=name, type="function", function=Function(name=name, arguments="{}")
            )
        ],
    )


def test_tool_agent_reports_tool_errors_and_hides_private_keys():
    llm = ScriptedLLM(
        [
            tool_call("failing"),
            tool_call("retrieving"),
            LLMMessage(content=text("done")),
        ],
        tools=[Failing(), Retrieving()],
    )
    answer = asyncio.run(
        ToolAgent("system", llm).answer(QueryWithContext(query="q", history=[]))
    )

    tool_results = [m.content[0].text for m in llm.seen[-1] if m.role == Roles.TOOL]
    assert tool_results == [str({"error": "boom"}), str({"result": "text"})]
    assert answer.content[0].text == "done"
    assert [s.content for s in answer.sources] == ["c"]


class Named(Agent):
    def __init__(self, name: str):
        super().__init__("", ScriptedLLM([]), description=name)

    async def answer_stream(
        self, query_with_context: QueryWithContext
    ) -> AsyncGenerator[AgentAnswer, None]:
        yield AgentAnswer(content=text(self.description))


def test_router_maps_categories_from_one():
    llm = ScriptedLLM([LLMMessage(content=text('{"category": 2}'))])
    router = Router(llm, [Named("a"), Named("b")])

    assert "Category 1: a" in router.system_prompt
    assert (
        asyncio.run(router.answer(QueryWithContext(query="q", history=[])))
        .content[0]
        .text
        == "b"
    )


def test_chroma_scores_are_similarities_and_metadata_round_trips(tmp_path):
    store = ChromaDBVectorStore(str(tmp_path))
    chunks = [
        Chunk(title=t, content=t, mime_type="text/plain", metadata={"chapters": [t]})
        for t in "ab"
    ]
    asyncio.run(store.insert_chunks(chunks, np.array([[1.0, 0.0], [0.0, 1.0]])))

    matches = asyncio.run(store.get_chunks_by_embedding(np.array([1.0, 0.0]), top_k=2))

    assert [m.title for m in matches] == ["a", "b"]
    assert [m.score for m in matches] == pytest.approx([1.0, 0.0], abs=1e-6)
    assert matches[0].metadata["chapters"] == ["a"]


def test_in_memory_store_persists(tmp_path):
    path = str(tmp_path / "messages.json")
    message = Message(
        id="m",
        user_id=None,
        conversation_id="c",
        content="hi",
        role=Roles.USER,
        create_timestamp=0,
    )
    asyncio.run(InMemoryMessageDatabase(path).create_message(message))

    reloaded = InMemoryMessageDatabase(path)
    assert asyncio.run(reloaded.get_messages_by_conversation_id("c")) == [message]
    assert asyncio.run(reloaded.delete_message("m")) is True


def test_evaluate_with_ragas_reads_per_metric_scores(monkeypatch):
    class Faithfulness:
        name = "faithfulness"

    def fake_evaluate(dataset, metrics, llm, embeddings):
        return {
            "faithfulness": [1.0, math.nan]
        }  # EvaluationResult[name] returns the per-sample list

    monkeypatch.setattr(ragas_adapter.ragas, "evaluate", fake_evaluate)
    samples = [EvaluationSample(query="q", answer="a")] * 2
    report = ragas_adapter.evaluate_with_ragas(samples, [Faithfulness()])

    assert report.summary() == {"faithfulness": 1.0}
    assert report.results[0].per_sample_scores is not None
    assert math.isnan(report.results[0].per_sample_scores[1])
