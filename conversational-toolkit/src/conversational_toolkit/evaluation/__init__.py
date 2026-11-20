"""Retrieval metrics and batch evaluation; RAGAS generation metrics live in 'evaluation.adapters'."""

from conversational_toolkit.evaluation.data_models import EvaluationReport, EvaluationSample, MetricResult
from conversational_toolkit.evaluation.evaluator import Evaluator
from conversational_toolkit.evaluation.metrics.base import Metric
from conversational_toolkit.evaluation.metrics.retrieval import HitRate, MRR, NDCGAtK, PrecisionAtK, RecallAtK

__all__ = [
    "MRR",
    "EvaluationReport",
    "EvaluationSample",
    "Evaluator",
    "HitRate",
    "Metric",
    "MetricResult",
    "NDCGAtK",
    "PrecisionAtK",
    "RecallAtK",
]
