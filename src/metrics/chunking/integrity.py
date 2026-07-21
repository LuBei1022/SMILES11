"""Rule-based chunk structural integrity metric."""

from __future__ import annotations

import re
from typing import Any

from ..base import MetricResult
from ..common.trace_utils import safe_error


TRUNCATION_PATTERN = re.compile(r"(?:\.\.\.|…|\[TRUNCATED\]|\[\.\.\.\])\s*$", re.IGNORECASE)


def chunk_integrity(trace: dict[str, Any], config: dict[str, Any]) -> MetricResult:
    trace_id = trace.get("trace_id") if isinstance(trace.get("trace_id"), str) else "unknown"
    retrieval = trace.get("retrieval")
    retrieved_chunks = retrieval.get("retrieved_chunks") if isinstance(retrieval, dict) else None
    if isinstance(retrieved_chunks, list):
        chunks = retrieved_chunks
        checked_source = "retrieved_chunks"
    else:
        chunks = trace.get("chunks")
        checked_source = "chunks"
    if not isinstance(chunks, list):
        return MetricResult(
            trace_id=trace_id, metric_name="chunking.chunk_integrity", stage="chunking",
            score=None, label="unknown", status="error",
            error=safe_error("invalid_field_type", "chunks must be a list", field="chunks"),
        )
    if not chunks:
        return MetricResult(
            trace_id=trace_id, metric_name="chunking.chunk_integrity", stage="chunking",
            score=None, label="unknown", status="skipped", warnings=["no_chunks"],
        )

    seen_ids: set[str] = set()
    defects: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        types: list[str] = []
        if not isinstance(chunk, dict):
            defects.append({"chunk_id": None, "index": index, "types": ["invalid_chunk_type"]})
            continue
        chunk_id = chunk.get("chunk_id")
        text = chunk.get("text")
        if not isinstance(text, str) or not text.strip():
            types.append("empty_text")
        elif TRUNCATION_PATTERN.search(text):
            types.append("truncated")
        if not isinstance(chunk.get("document_id"), str) or not chunk.get("document_id"):
            types.append("missing_document_id")
        start = chunk.get("start_char")
        end = chunk.get("end_char")
        if isinstance(start, int) and isinstance(end, int) and (start < 0 or end < start):
            types.append("invalid_offsets")
        if isinstance(chunk_id, str):
            if chunk_id in seen_ids:
                types.append("duplicate_chunk_id")
            seen_ids.add(chunk_id)
        else:
            types.append("missing_chunk_id")
        if types:
            defects.append({"chunk_id": chunk_id, "index": index, "types": types})

    score = 1.0 - len(defects) / len(chunks)
    threshold = float(config.get("integrity_intact", 0.8))
    label = "intact" if score >= threshold else "degraded" if score > 0 else "broken"
    return MetricResult(
        trace_id=trace_id, metric_name="chunking.chunk_integrity", stage="chunking",
        score=score, label=label,
            evidence={"checked_source": checked_source, "checked_chunk_count": len(chunks),
                      "defective_chunk_count": len(defects), "defects": defects},
        config={"threshold": threshold, "implementation": "rule_v1"},
    )
