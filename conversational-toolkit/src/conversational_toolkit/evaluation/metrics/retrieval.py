import math
from abc import abstractmethod
from collections.abc import Sequence

from loguru import logger

from conversational_toolkit.evaluation.data_models import EvaluationSample, MetricResult
from conversational_toolkit.evaluation.metrics.base import Metric


class _RetrievalMetric(Metric):
    """Mean of a per-sample score over the top 'k' chunks; samples without 'relevant_chunk_ids' are skipped."""

    def __init__(self, k: int) -> None:
        self.k = k

    @property
    def name(self) -> str:
        return f"{self.__class__.__name__}@{self.k}"

    @abstractmethod
    def _score(self, top_k: list[str], relevant: set[str]) -> float: ...

    async def compute(self, samples: Sequence[EvaluationSample]) -> MetricResult:
        scores = []
        for s in samples:
            if not s.relevant_chunk_ids:
                logger.warning(f"{self.name}: sample '{s.query[:60]}' has no relevant_chunk_ids, skipping.")
                continue
            scores.append(self._score([c.id for c in s.retrieved_chunks[: self.k]], s.relevant_chunk_ids))
        mean = sum(scores) / len(scores) if scores else 0.0
        return MetricResult(metric_name=self.name, score=mean, per_sample_scores=scores)


class HitRate(_RetrievalMetric):
    """1 if any relevant chunk is in the top k."""

    def _score(self, top_k: list[str], relevant: set[str]) -> float:
        return float(any(cid in relevant for cid in top_k))


class MRR(_RetrievalMetric):
    """Reciprocal rank of the first relevant chunk in the top k."""

    def _score(self, top_k: list[str], relevant: set[str]) -> float:
        return next((1 / rank for rank, cid in enumerate(top_k, start=1) if cid in relevant), 0.0)


class PrecisionAtK(_RetrievalMetric):
    """Fraction of the top k that is relevant."""

    def _score(self, top_k: list[str], relevant: set[str]) -> float:
        return sum(cid in relevant for cid in top_k) / self.k


class RecallAtK(_RetrievalMetric):
    """Fraction of the relevant chunks found in the top k."""

    def _score(self, top_k: list[str], relevant: set[str]) -> float:
        return sum(cid in relevant for cid in top_k) / len(relevant)


class NDCGAtK(_RetrievalMetric):
    """NDCG with binary relevance."""

    def _score(self, top_k: list[str], relevant: set[str]) -> float:
        dcg = sum(1 / math.log2(rank + 1) for rank, cid in enumerate(top_k, start=1) if cid in relevant)
        idcg = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(relevant), self.k) + 1))
        return dcg / idcg
