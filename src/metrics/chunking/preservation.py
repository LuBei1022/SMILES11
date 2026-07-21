"""Gold evidence preservation metric."""

from __future__ import annotations

from typing import Any

from ..base import MetricResult


def gold_evidence_preservation(trace: dict[str, Any], config: dict[str, Any]) -> MetricResult:
    trace_id = trace.get("trace_id") if isinstance(trace.get("trace_id"), str) else "unknown"
    evidence = trace.get("gold_evidence")
    if not isinstance(evidence, list) or not evidence:
        return MetricResult(
            trace_id=trace_id, metric_name="chunking.gold_evidence_preservation",
            stage="chunking", score=None, label="unknown", status="skipped",
            warnings=["no_gold_evidence"],
        )

    chunks = trace.get("chunks")
    known_ids = {
        chunk.get("chunk_id") for chunk in chunks or []
        if isinstance(chunk, dict) and isinstance(chunk.get("chunk_id"), str)
    } if isinstance(chunks, list) else set()
    preserved_ids: list[str] = []
    lost_indexes: list[int] = []
    for index, item in enumerate(evidence):
        if not isinstance(item, dict):
            lost_indexes.append(index)
            continue
        covering = item.get("covering_chunk_ids")
        valid_ids = [chunk_id for chunk_id in covering or [] if isinstance(chunk_id, str)]
        exists = bool(valid_ids) and (not known_ids or any(chunk_id in known_ids for chunk_id in valid_ids))
        if item.get("is_preserved") is True and exists:
            preserved_ids.extend(valid_ids)
        else:
            lost_indexes.append(index)

    score = (len(evidence) - len(lost_indexes)) / len(evidence)
    label = "complete" if score == 1.0 else "partial" if score > 0 else "lost"
    return MetricResult(
        trace_id=trace_id, metric_name="chunking.gold_evidence_preservation",
        stage="chunking", score=score, label=label,
        evidence={"total_count": len(evidence), "preserved_count": len(evidence) - len(lost_indexes),
                  "preserved_chunk_ids": sorted(set(preserved_ids)), "lost_evidence_indexes": lost_indexes},
        config={"implementation": "rule_v1"},
    )
