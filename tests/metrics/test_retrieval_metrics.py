from src.metrics.retrieval.lexical import (
    hit_at_k,
    mrr,
    precision_at_k,
    recall_at_k,
)


def trace_with_ranked_results():
    return {
        "trace_id": "trace-retrieval",
        "gold_evidence": [
            {"covering_chunk_ids": ["gold-1"]},
            {"covering_chunk_ids": ["gold-2"]},
        ],
        "retrieval": {
            "retrieved_chunks": [
                {"chunk_id": "other-1", "rank": 1, "text": "a"},
                {"chunk_id": "gold-1", "rank": 2, "text": "b"},
                {"chunk_id": "other-2", "rank": 3, "text": "c"},
                {"chunk_id": "other-3", "rank": 4, "text": "d"},
                {"chunk_id": "other-4", "rank": 5, "text": "e"},
            ]
        },
    }


def test_retrieval_metrics_use_top_k_and_gold_chunk_ids():
    trace = trace_with_ranked_results()
    config = {"k": 5}

    assert hit_at_k(trace, config).score == 1.0
    assert recall_at_k(trace, config).score == 0.5
    assert precision_at_k(trace, config).score == 0.2
    assert mrr(trace, config).score == 0.5


def test_retrieval_metrics_report_miss_without_gold_match():
    trace = trace_with_ranked_results()
    trace["retrieval"]["retrieved_chunks"] = [
        {"chunk_id": "other", "rank": 1, "text": "x"}
    ]

    assert hit_at_k(trace, {"k": 1}).label == "miss"
    assert recall_at_k(trace, {"k": 1}).score == 0.0
    assert precision_at_k(trace, {"k": 1}).score == 0.0
    assert mrr(trace, {"k": 1}).score == 0.0


def test_retrieval_metrics_skip_when_gold_evidence_is_missing():
    result = recall_at_k({"trace_id": "missing-gold"}, {"k": 5})

    assert result.status == "skipped"
    assert result.score is None
    assert result.label == "unknown"
