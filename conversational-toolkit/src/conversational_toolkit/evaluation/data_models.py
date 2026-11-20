from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field

from conversational_toolkit.llms.base import LLMMessage
from conversational_toolkit.vectorstores.base import ChunkRecord


class EvaluationSample(BaseModel):
    """One query with the system's answer and chunks, plus the labels the metrics compare against.

    Attributes:
        history: Turns preceding the query; needed for multi-turn RAGAS evaluation.
        relevant_chunk_ids: Ground-truth relevant chunks; needed for retrieval metrics.
    """

    query: str
    answer: str
    retrieved_chunks: Sequence[ChunkRecord] = Field(default_factory=list)
    history: list[LLMMessage] = Field(default_factory=list)
    ground_truth_answer: str | None = None
    relevant_chunk_ids: set[str] = Field(default_factory=set)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MetricResult(BaseModel):
    """Mean 'score' of one metric, with per-sample scores and free-form 'details'."""

    metric_name: str
    score: float
    per_sample_scores: list[float] | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class EvaluationReport(BaseModel):
    """Results of all metrics for one evaluation run."""

    results: list[MetricResult]
    num_samples: int
    metadata: dict[str, Any] = Field(default_factory=dict)

    def summary(self) -> dict[str, float]:
        """Metric name -> mean score."""
        return {r.metric_name: r.score for r in self.results}
