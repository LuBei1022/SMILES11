"""Batch runner and default registry for the Metric Engine."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import yaml

from .base import Metric, MetricResult
from .chunking.integrity import chunk_integrity
from .chunking.preservation import gold_evidence_preservation
from .common.trace_utils import safe_error
from .context.coverage import evidence_coverage
from .context.noise import noise_ratio
from .context.truncation import context_truncation
from .generation.faithfulness import faithfulness
from .generation.relevance import answer_relevance
from .registry import MetricRegistry
from .retrieval.lexical import hit_at_k, mrr, precision_at_k, recall_at_k


DEFAULT_CONFIG = {
    "metrics": {
        "retrieval": {"k": 5, "precision_high": 0.8},
        "chunking": {"integrity_intact": 0.8},
        "context": {"noise_low": 0.2, "noise_moderate": 0.5},
        "generation": {
            "relevance_relevant": 0.5,
            "sentence_support_threshold": 0.5,
            "faithfulness_supported": 0.8,
            "faithfulness_partial": 0.5,
        },
    }
}


@dataclass(frozen=True)
class FunctionMetric:
    name: str
    stage: str
    function: Callable[[dict[str, Any], dict[str, Any]], MetricResult]

    def compute(self, trace: dict[str, Any], config: dict[str, Any]) -> MetricResult:
        return self.function(trace, config)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_metric_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return deepcopy(DEFAULT_CONFIG)
    with open(path, "r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError("metric config must contain a YAML object")
    return _deep_merge(DEFAULT_CONFIG, loaded)


def default_registry() -> MetricRegistry:
    registry = MetricRegistry()
    definitions = [
        ("retrieval.hit_at_k", "retrieval", hit_at_k),
        ("retrieval.recall_at_k", "retrieval", recall_at_k),
        ("retrieval.precision_at_k", "retrieval", precision_at_k),
        ("retrieval.mrr", "retrieval", mrr),
        ("chunking.gold_evidence_preservation", "chunking", gold_evidence_preservation),
        ("chunking.chunk_integrity", "chunking", chunk_integrity),
        ("context.evidence_coverage", "context", evidence_coverage),
        ("context.noise_ratio", "context", noise_ratio),
        ("context.context_truncation", "context", context_truncation),
        ("generation.answer_relevance", "generation", answer_relevance),
        ("generation.faithfulness", "generation", faithfulness),
    ]
    for name, stage, function in definitions:
        registry.register(FunctionMetric(name, stage, function))
    return registry


class MetricRunner:
    def __init__(self, metrics: list[Metric], config: dict[str, Any]) -> None:
        self.metrics = list(metrics)
        self.config = deepcopy(config)

    def run_trace(self, trace: dict[str, Any]) -> list[MetricResult]:
        trace_id = trace.get("trace_id") if isinstance(trace.get("trace_id"), str) else "unknown"
        results: list[MetricResult] = []
        for metric in self.metrics:
            started = perf_counter()
            stage_config = self.config.get("metrics", {}).get(metric.stage, {})
            try:
                result = metric.compute(trace, deepcopy(stage_config))
            except Exception as exc:
                result = MetricResult(
                    trace_id=trace_id,
                    metric_name=metric.name,
                    stage=metric.stage,
                    score=None,
                    label="unknown",
                    status="error",
                    error=safe_error("metric_execution_error", str(exc),
                                     exception_type=type(exc).__name__),
                )
            result.runtime_ms = (perf_counter() - started) * 1000
            results.append(result)
        return results

    def run_jsonl(self, input_path: str, output_path: str) -> dict[str, int]:
        input_count = result_count = skipped_count = error_count = 0
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with open(input_path, "r", encoding="utf-8") as source, destination.open(
            "w", encoding="utf-8"
        ) as output:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                input_count += 1
                try:
                    trace = json.loads(line)
                    if not isinstance(trace, dict):
                        raise ValueError("trace must be a JSON object")
                except (json.JSONDecodeError, ValueError) as exc:
                    trace = {"trace_id": f"line-{line_number}"}
                    line_error = safe_error("invalid_jsonl_record", str(exc), line_number=line_number)
                    results = [
                        MetricResult(
                            trace_id=trace["trace_id"], metric_name=metric.name, stage=metric.stage,
                            score=None, label="unknown", status="error", error=line_error,
                        )
                        for metric in self.metrics
                    ]
                else:
                    results = self.run_trace(trace)
                for result in results:
                    output.write(result.to_json() + "\n")
                    result_count += 1
                    skipped_count += int(result.status == "skipped")
                    error_count += int(result.status == "error")
        return {
            "input_traces": input_count,
            "results": result_count,
            "skipped_metrics": skipped_count,
            "errors": error_count,
        }
