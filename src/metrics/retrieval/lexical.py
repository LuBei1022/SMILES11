"""Gold-evidence retrieval metrics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..base import MetricResult
from ..common.trace_utils import gold_chunk_ids, safe_error


def _trace_id(trace: Mapping[str, Any]) -> str:
    value = trace.get("trace_id")
    return value if isinstance(value, str) and value else "unknown"


def _inputs(trace: Mapping[str, Any], config: Mapping[str, Any]):
    gold = gold_chunk_ids(trace)
    k = config.get("k", 5)
    if not isinstance(k, int) or k <= 0:
        return gold, [], 5, safe_error("invalid_config", "k must be a positive integer", k=k)
    if not gold:
        return gold, [], k, None

    retrieval = trace.get("retrieval")
    if not isinstance(retrieval, Mapping):
        return gold, [], k, safe_error(
            "invalid_field_type", "retrieval must be an object", field="retrieval"
        )
    chunks = retrieval.get("retrieved_chunks")
    if not isinstance(chunks, list):
        return gold, [], k, safe_error(
            "invalid_field_type",
            "retrieved_chunks must be a list",
            field="retrieval.retrieved_chunks",
        )
    ranked = [chunk for chunk in chunks if isinstance(chunk, Mapping)][:k]
    return gold, ranked, k, None


def _unavailable(trace, name, k, error=None):
    if error is not None:
        return MetricResult(
            trace_id=_trace_id(trace), metric_name=name, stage="retrieval",
            score=None, label="unknown", status="error", config={"k": k}, error=error,
        )
    return MetricResult(
        trace_id=_trace_id(trace), metric_name=name, stage="retrieval",
        score=None, label="unknown", status="skipped", config={"k": k},
        warnings=["no_gold_evidence"],
    )


def _matched(gold, ranked):
    retrieved_ids = [chunk.get("chunk_id") for chunk in ranked if isinstance(chunk.get("chunk_id"), str)]
    matches = [chunk_id for chunk_id in retrieved_ids if chunk_id in gold]
    return retrieved_ids, matches


def hit_at_k(trace: dict[str, Any], config: dict[str, Any]) -> MetricResult:
    gold, ranked, k, error = _inputs(trace, config)
    if error or not gold:
        return _unavailable(trace, "retrieval.hit_at_k", k, error)
    retrieved_ids, matches = _matched(gold, ranked)
    score = 1.0 if matches else 0.0
    return MetricResult(
        trace_id=_trace_id(trace), metric_name="retrieval.hit_at_k", stage="retrieval",
        score=score, label="hit" if matches else "miss",
        evidence={"gold_chunk_ids": sorted(gold), "retrieved_chunk_ids": retrieved_ids,
                  "matched_chunk_ids": matches}, config={"k": k, "implementation": "rule_v1"},
    )


def recall_at_k(trace: dict[str, Any], config: dict[str, Any]) -> MetricResult:
    gold, ranked, k, error = _inputs(trace, config)
    if error or not gold:
        return _unavailable(trace, "retrieval.recall_at_k", k, error)
    retrieved_ids, matches = _matched(gold, ranked)
    score = len(set(matches)) / len(gold)
    label = "complete" if score == 1.0 else "partial" if score > 0 else "miss"
    return MetricResult(
        trace_id=_trace_id(trace), metric_name="retrieval.recall_at_k", stage="retrieval",
        score=score, label=label,
        evidence={"gold_count": len(gold), "matched_count": len(set(matches)),
                  "retrieved_chunk_ids": retrieved_ids, "matched_chunk_ids": matches},
        config={"k": k, "implementation": "rule_v1"},
    )


def precision_at_k(trace: dict[str, Any], config: dict[str, Any]) -> MetricResult:
    gold, ranked, k, error = _inputs(trace, config)
    if error or not gold:
        return _unavailable(trace, "retrieval.precision_at_k", k, error)
    retrieved_ids, matches = _matched(gold, ranked)
    if not retrieved_ids:
        return MetricResult(
            trace_id=_trace_id(trace), metric_name="retrieval.precision_at_k", stage="retrieval",
            score=None, label="unknown", status="error", config={"k": k},
            error=safe_error("empty_retrieval", "retrieved_chunks is empty"),
        )
    score = len(matches) / len(retrieved_ids)
    threshold = float(config.get("precision_high", 0.8))
    label = "high" if score >= threshold else "partial" if score > 0 else "none"
    return MetricResult(
        trace_id=_trace_id(trace), metric_name="retrieval.precision_at_k", stage="retrieval",
        score=score, label=label,
        evidence={"retrieved_count": len(retrieved_ids), "matched_count": len(matches),
                  "matched_chunk_ids": matches},
        config={"k": k, "threshold": threshold, "implementation": "rule_v1"},
    )


def mrr(trace: dict[str, Any], config: dict[str, Any]) -> MetricResult:
    gold, ranked, k, error = _inputs(trace, config)
    if error or not gold:
        return _unavailable(trace, "retrieval.mrr", k, error)
    retrieved_ids, _ = _matched(gold, ranked)
    first_rank = next((index for index, chunk_id in enumerate(retrieved_ids, 1) if chunk_id in gold), None)
    score = 1.0 / first_rank if first_rank else 0.0
    return MetricResult(
        trace_id=_trace_id(trace), metric_name="retrieval.mrr", stage="retrieval",
        score=score, label="ranked" if first_rank else "miss",
        evidence={"first_matching_rank": first_rank,
                  "first_matching_chunk_id": retrieved_ids[first_rank - 1] if first_rank else None},
        config={"k": k, "implementation": "rule_v1"},
    )
