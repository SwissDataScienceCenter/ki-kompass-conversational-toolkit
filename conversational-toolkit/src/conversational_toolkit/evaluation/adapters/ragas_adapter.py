from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import ragas  # type: ignore[import-untyped]
from ragas import EvaluationDataset, MultiTurnSample, SingleTurnSample  # type: ignore[import-untyped]
from ragas.dataset_schema import EvaluationResult  # type: ignore[import-untyped]
from ragas.messages import AIMessage, HumanMessage, ToolMessage  # type: ignore[import-untyped]

from conversational_toolkit.evaluation.data_models import EvaluationReport, EvaluationSample, MetricResult
from conversational_toolkit.llms.base import LLMMessage, Roles


def _text(message: LLMMessage) -> str:
    return message.content[0].text or "" if message.content else ""


def _to_ragas_messages(sample: EvaluationSample) -> list[HumanMessage | AIMessage | ToolMessage]:
    """History plus the current turn; SYSTEM and TOOL messages have no RAGAS equivalent and are dropped."""
    messages: list[HumanMessage | AIMessage | ToolMessage] = []
    for msg in sample.history:
        if msg.role == Roles.USER:
            messages.append(HumanMessage(content=_text(msg)))
        elif msg.role == Roles.ASSISTANT:
            messages.append(AIMessage(content=_text(msg)))
    messages.append(HumanMessage(content=sample.query))
    if sample.answer:
        messages.append(AIMessage(content=sample.answer))
    return messages


def to_ragas_dataset(samples: Sequence[EvaluationSample]) -> EvaluationDataset:
    """Single-turn dataset: query, answer, retrieved chunk contents and reference answer."""
    return EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input=s.query,
                response=s.answer,
                retrieved_contexts=[chunk.content for chunk in s.retrieved_chunks],
                reference=s.ground_truth_answer,
            )
            for s in samples
        ]
    )


def to_ragas_multiturn_dataset(samples: Sequence[EvaluationSample]) -> EvaluationDataset:
    """Multi-turn dataset built from each sample's history, query and answer."""
    multiturn: list[SingleTurnSample | MultiTurnSample] = [
        MultiTurnSample(user_input=_to_ragas_messages(s), reference=s.ground_truth_answer) for s in samples
    ]
    return EvaluationDataset(samples=multiturn)


def evaluate_with_ragas(
    samples: Sequence[EvaluationSample],
    metrics: list[Any],
    llm: Any | None = None,
    embeddings: Any | None = None,
    multiturn: bool = False,
) -> EvaluationReport:
    """Run RAGAS 'metrics' (e.g. '[Faithfulness()]') on the samples; 'llm'/'embeddings' default to RAGAS's own."""
    dataset = to_ragas_multiturn_dataset(samples) if multiturn else to_ragas_dataset(samples)
    result = cast(EvaluationResult, ragas.evaluate(dataset=dataset, metrics=metrics, llm=llm, embeddings=embeddings))
    return EvaluationReport(
        results=[
            MetricResult(
                metric_name=m.name,
                score=float(np.nanmean(result[m.name])),
                per_sample_scores=[float(v) for v in result[m.name]],
            )
            for m in metrics
        ],
        num_samples=len(samples),
    )
