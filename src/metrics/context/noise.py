"""Gold-based noise ratio for selected context chunks."""

from __future__ import annotations

from typing import Any

from ..base import MetricResult
from ..common.trace_utils import gold_chunk_ids, selected_chunk_ids


def noise_ratio(trace: dict[str, Any], config: dict[str, Any]) -> MetricResult:
    trace_id = trace.get("trace_id") if isinstance(trace.get("trace_id"), str) else "unknown"
    gold = gold_chunk_ids(trace)
    if not gold:
        return MetricResult(
            trace_id=trace_id, metric_name="context.noise_ratio", stage="context",
            score=None, label="unknown", status="skipped", warnings=["no_gold_evidence"],
        )
    selected = selected_chunk_ids(trace)
    if not selected:
        return MetricResult(
            trace_id=trace_id, metric_name="context.noise_ratio", stage="context",
            score=None, label="unknown", status="skipped", warnings=["no_selected_chunks"],
        )
    noise_ids = sorted(selected - gold)
    score = len(noise_ids) / len(selected)
    low = float(config.get("noise_low", 0.2))
    moderate = float(config.get("noise_moderate", 0.5))
    label = "low_noise" if score <= low else "moderate_noise" if score <= moderate else "high_noise"
    return MetricResult(
        trace_id=trace_id, metric_name="context.noise_ratio", stage="context",
        score=score, label=label,
        evidence={"selected_chunk_count": len(selected), "noise_chunk_count": len(noise_ids),
                  "noise_chunk_ids": noise_ids},
        config={"low_threshold": low, "moderate_threshold": moderate,
                "implementation": "rule_v1"},
    )
