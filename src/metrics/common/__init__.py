"""Shared read-only helpers used by metric implementations."""

from .trace_utils import gold_chunk_ids, retrieved_chunks, safe_error, selected_chunk_ids

__all__ = ["gold_chunk_ids", "retrieved_chunks", "safe_error", "selected_chunk_ids"]
