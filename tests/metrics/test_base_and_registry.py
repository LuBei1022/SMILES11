import json

import pytest

from src.metrics.base import MetricResult
from src.metrics.registry import MetricRegistry


class ExampleMetric:
    name = "example.metric"
    stage = "example"

    def compute(self, trace, config):
        return MetricResult(
            trace_id=trace["trace_id"],
            metric_name=self.name,
            stage=self.stage,
            score=1.0,
            label="pass",
        )


def test_metric_result_serializes_all_required_fields():
    result = MetricResult(
        trace_id="trace-1",
        metric_name="retrieval.hit_at_k",
        stage="retrieval",
        score=1.0,
        label="hit",
    )

    payload = result.to_dict()

    assert set(payload) == {
        "trace_id",
        "metric_name",
        "stage",
        "score",
        "label",
        "status",
        "evidence",
        "config",
        "model",
        "warnings",
        "error",
        "runtime_ms",
    }
    assert json.loads(result.to_json()) == payload


def test_metric_result_rejects_invalid_status():
    with pytest.raises(ValueError, match="status"):
        MetricResult(
            trace_id="trace-1",
            metric_name="retrieval.hit_at_k",
            stage="retrieval",
            score=None,
            label="unknown",
            status="invalid",
        )


def test_metric_result_rejects_score_outside_normalized_range():
    with pytest.raises(ValueError, match="score"):
        MetricResult(
            trace_id="trace-1",
            metric_name="retrieval.hit_at_k",
            stage="retrieval",
            score=1.1,
            label="hit",
        )


def test_registry_returns_metrics_in_registration_order():
    registry = MetricRegistry()
    first = ExampleMetric()
    second = ExampleMetric()
    second.name = "example.second"

    registry.register(first)
    registry.register(second)

    assert registry.get("example.metric") is first
    assert registry.all() == [first, second]


def test_registry_rejects_duplicate_metric_names():
    registry = MetricRegistry()
    registry.register(ExampleMetric())

    with pytest.raises(ValueError, match="already registered"):
        registry.register(ExampleMetric())
