from abc import ABC, abstractmethod
from collections.abc import Sequence

from conversational_toolkit.evaluation.data_models import EvaluationSample, MetricResult


class Metric(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier such as 'HitRate@5'."""

    @abstractmethod
    async def compute(self, samples: Sequence[EvaluationSample]) -> MetricResult:
        pass
