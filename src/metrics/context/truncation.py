"""Context truncation detection using explicit metadata and visible markers."""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any

from ..base import MetricResult
from ..common.trace_utils import safe_error


TRUNCATION_PATTERN = re.compile(r"(?:\.\.\.|…|\[TRUNCATED\]|\[\.\.\.\])\s*$", re.IGNORECASE)


def context_truncation(trace: dict[str, Any], config: dict[str, Any]) -> MetricResult:
    trace_id = trace.get("trace_id") if isinstance(trace.get("trace_id"), str) else "unknown"
    context = trace.get("context_construction")
    if not isinstance(context, Mapping):
        return MetricResult(
            trace_id=trace_id, metric_name="context.context_truncation", stage="context",
            score=None, label="unknown", status="error",
            error=safe_error("invalid_field_type", "context_construction must be an object",
                             field="context_construction"),
        )
    explicit = context.get("truncated")
    if isinstance(explicit, bool):
        return MetricResult(
            trace_id=trace_id, metric_name="context.context_truncation", stage="context",
            score=0.0 if explicit else 1.0,
            label="truncated" if explicit else "not_truncated",
            evidence={"source": "explicit_flag", "truncated": explicit},
            config={"implementation": "rule_v1"},
        )
    final_context = context.get("final_context")
    if isinstance(final_context, str) and TRUNCATION_PATTERN.search(final_context):
        return MetricResult(
            trace_id=trace_id, metric_name="context.context_truncation", stage="context",
            score=0.0, label="truncated",
            evidence={"source": "text_marker", "truncated": True},
            config={"implementation": "rule_v1"},
        )
    return MetricResult(
        trace_id=trace_id, metric_name="context.context_truncation", stage="context",
        score=None, label="unknown", status="skipped", warnings=["insufficient_truncation_metadata"],
        config={"implementation": "rule_v1"},
    )
