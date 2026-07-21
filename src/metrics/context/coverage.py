"""Gold evidence coverage in the final selected context."""

from __future__ import annotations

from typing import Any

from ..base import MetricResult
from ..common.trace_utils import gold_chunk_ids, selected_chunk_ids


def evidence_coverage(trace: dict[str, Any], config: dict[str, Any]) -> MetricResult:
    trace_id = trace.get("trace_id") if isinstance(trace.get("trace_id"), str) else "unknown"
    gold = gold_chunk_ids(trace)
    if not gold:
        return MetricResult(
            trace_id=trace_id, metric_name="context.evidence_coverage", stage="context",
            score=None, label="unknown", status="skipped", warnings=["no_gold_evidence"],
        )
    selected = selected_chunk_ids(trace)
    matched = sorted(gold & selected)
    score = len(matched) / len(gold)
    label = "complete" if score == 1.0 else "partial" if score > 0 else "missing"
    return MetricResult(
        trace_id=trace_id, metric_name="context.evidence_coverage", stage="context",
        score=score, label=label,
        evidence={"gold_chunk_ids": sorted(gold), "selected_chunk_ids": sorted(selected),
                  "matched_chunk_ids": matched},
        config={"implementation": "rule_v1"},
    )
