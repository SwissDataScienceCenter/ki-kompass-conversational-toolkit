import asyncio
from collections.abc import Sequence
from typing import Any, cast

from conversational_toolkit.agents.base import Agent, QueryWithContext
from conversational_toolkit.evaluation.data_models import EvaluationReport, EvaluationSample, MetricResult
from conversational_toolkit.evaluation.metrics.base import Metric
from conversational_toolkit.llms.base import LLMMessage
from conversational_toolkit.retriever.base import Retriever
from conversational_toolkit.vectorstores.base import ChunkRecord


class Evaluator:
    """Runs metrics concurrently over a batch of samples and builds samples from an agent or retriever."""

    def __init__(self, metrics: list[Metric]) -> None:
        self.metrics = metrics

    async def evaluate(
        self,
        samples: Sequence[EvaluationSample],
        metadata: dict[str, Any] | None = None,
    ) -> EvaluationReport:
        results: list[MetricResult] = list(await asyncio.gather(*(m.compute(samples) for m in self.metrics)))
        return EvaluationReport(
            results=results,
            num_samples=len(samples),
            metadata=metadata or {},
        )

    @staticmethod
    async def build_samples_from_agent(
        agent: Agent,
        queries: list[str],
        relevant_chunk_ids: list[set[str]] | None = None,
        ground_truth_answers: list[str | None] | None = None,
        history: list[LLMMessage] | None = None,
    ) -> list[EvaluationSample]:
        """Answer each query with the agent; 'history' is shared by all queries, the other lists align with 'queries'."""
        history = history or []
        samples: list[EvaluationSample] = []
        for i, query in enumerate(queries):
            answer = await agent.answer(QueryWithContext(query=query, history=history))
            samples.append(
                EvaluationSample(
                    query=query,
                    answer=answer.content[0].text or "",
                    retrieved_chunks=cast(list[ChunkRecord], list(answer.sources)),
                    history=list(history),
                    relevant_chunk_ids=relevant_chunk_ids[i] if relevant_chunk_ids else set(),
                    ground_truth_answer=ground_truth_answers[i] if ground_truth_answers else None,
                )
            )
        return samples

    @staticmethod
    async def build_samples_from_retriever(
        retriever: Retriever[ChunkRecord],
        queries: list[str],
        relevant_chunk_ids: list[set[str]] | None = None,
    ) -> list[EvaluationSample]:
        """Retrieve for each query, without generating answers, to evaluate retrieval on its own."""
        samples: list[EvaluationSample] = []
        for i, query in enumerate(queries):
            chunks = await retriever.retrieve(query)
            samples.append(
                EvaluationSample(
                    query=query,
                    answer="",
                    retrieved_chunks=list(chunks),
                    relevant_chunk_ids=relevant_chunk_ids[i] if relevant_chunk_ids else set(),
                )
            )
        return samples
