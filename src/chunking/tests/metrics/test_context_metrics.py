from src.metrics.context.coverage import evidence_coverage
from src.metrics.context.noise import noise_ratio
from src.metrics.context.truncation import context_truncation


def context_trace():
    return {
        "trace_id": "context-1",
        "gold_evidence": [
            {"covering_chunk_ids": ["gold-1"]},
            {"covering_chunk_ids": ["gold-2"]},
        ],
        "retrieval": {
            "retrieved_chunks": [
                {"chunk_id": "gold-1"},
                {"chunk_id": "gold-2"},
            ]
        },
        "context_construction": {
            "selected_chunk_ids": ["gold-1", "noise-1"],
            "final_context": "Useful evidence followed by unrelated content.",
            "truncated": False,
            "context_token_count": 20,
        },
    }


def test_evidence_coverage_uses_selected_context_not_retrieval_results():
    result = evidence_coverage(context_trace(), {})

    assert result.score == 0.5
    assert result.label == "partial"
    assert result.evidence["matched_chunk_ids"] == ["gold-1"]


def test_noise_ratio_counts_non_gold_selected_chunks():
    result = noise_ratio(context_trace(), {})

    assert result.score == 0.5
    assert result.label == "moderate_noise"
    assert result.evidence["noise_chunk_ids"] == ["noise-1"]


def test_noise_ratio_skips_without_gold_evidence():
    trace = context_trace()
    trace.pop("gold_evidence")

    result = noise_ratio(trace, {})

    assert result.status == "skipped"
    assert result.score is None
    assert result.label == "unknown"


def test_context_truncation_prefers_explicit_flag():
    trace = context_trace()
    trace["context_construction"]["truncated"] = True

    result = context_truncation(trace, {})

    assert result.score == 0.0
    assert result.label == "truncated"
    assert result.evidence["source"] == "explicit_flag"


def test_context_truncation_is_unknown_without_enough_information():
    trace = {"trace_id": "context-unknown", "context_construction": {"final_context": "complete"}}

    result = context_truncation(trace, {})

    assert result.status == "skipped"
    assert result.score is None
    assert result.label == "unknown"
