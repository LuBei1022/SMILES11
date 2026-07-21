"""Small, defensive helpers for reading full traces."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def gold_chunk_ids(trace: Mapping[str, Any]) -> set[str]:
    evidence = trace.get("gold_evidence")
    if not isinstance(evidence, list):
        return set()

    result: set[str] = set()
    for item in evidence:
        if not isinstance(item, Mapping):
            continue
        chunk_ids = item.get("covering_chunk_ids")
        if isinstance(chunk_ids, list):
            result.update(chunk_id for chunk_id in chunk_ids if isinstance(chunk_id, str))
    return result


def retrieved_chunks(trace: Mapping[str, Any]) -> list[dict[str, Any]]:
    retrieval = trace.get("retrieval")
    if not isinstance(retrieval, Mapping):
        return []
    chunks = retrieval.get("retrieved_chunks")
    if not isinstance(chunks, list):
        return []
    return [chunk for chunk in chunks if isinstance(chunk, dict)]


def selected_chunk_ids(trace: Mapping[str, Any]) -> set[str]:
    context = trace.get("context_construction")
    if not isinstance(context, Mapping):
        return set()
    chunk_ids = context.get("selected_chunk_ids")
    if not isinstance(chunk_ids, list):
        return set()
    return {chunk_id for chunk_id in chunk_ids if isinstance(chunk_id, str)}


def safe_error(code: str, message: str, **details: Any) -> dict[str, Any]:
    return {"code": code, "message": message, "details": details}
