"""CPU-first lexical answer relevance baseline."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..base import MetricResult
from ..common.text import tokenize


def answer_relevance(trace: dict[str, Any], config: dict[str, Any]) -> MetricResult:
    trace_id = trace.get("trace_id") if isinstance(trace.get("trace_id"), str) else "unknown"
    query = trace.get("query")
    generation = trace.get("generation")
    answer = generation.get("final_answer") if isinstance(generation, Mapping) else None
    if not isinstance(query, str) or not query.strip() or not isinstance(answer, str) or not answer.strip():
        return MetricResult(
            trace_id=trace_id, metric_name="generation.answer_relevance", stage="generation",
            score=None, label="unknown", status="skipped", warnings=["empty_query_or_answer"],
        )
    language = trace.get("language") if isinstance(trace.get("language"), str) else None
    query_tokens = set(tokenize(query, language))
    answer_tokens = set(tokenize(answer, language))
    if not query_tokens:
        return MetricResult(
            trace_id=trace_id, metric_name="generation.answer_relevance", stage="generation",
            score=None, label="unknown", status="skipped", warnings=["no_valid_query_tokens"],
        )
    shared = sorted(query_tokens & answer_tokens)
    score = len(shared) / len(query_tokens)
    relevant = float(config.get("relevance_relevant", 0.5))
    label = "relevant" if score >= relevant else "weakly_relevant" if score > 0 else "unrelated"
    return MetricResult(
        trace_id=trace_id, metric_name="generation.answer_relevance", stage="generation",
        score=score, label=label,
        evidence={"query_tokens": sorted(query_tokens), "answer_tokens": sorted(answer_tokens),
                  "shared_tokens": shared},
        config={"threshold": relevant, "implementation": "rule_v1"},
    )
