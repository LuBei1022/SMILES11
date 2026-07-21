from src.metrics.chunking.integrity import chunk_integrity
from src.metrics.chunking.preservation import gold_evidence_preservation


def trace_with_chunking_results():
    return {
        "trace_id": "trace-chunking",
        "gold_evidence": [
            {
                "document_id": "doc-1",
                "gold_text": "complete evidence",
                "covering_chunk_ids": ["chunk-1"],
                "is_preserved": True,
            },
            {
                "document_id": "doc-2",
                "gold_text": "lost evidence",
                "covering_chunk_ids": ["chunk-2"],
                "is_preserved": False,
            },
        ],
        "chunks": [
            {
                "chunk_id": "chunk-1",
                "document_id": "doc-1",
                "text": "A complete chunk.",
                "start_char": 0,
                "end_char": 18,
            },
            {
                "chunk_id": "chunk-2",
                "document_id": "doc-2",
                "text": "A truncated chunk...",
                "start_char": 10,
                "end_char": 5,
            },
        ],
    }


def test_gold_evidence_preservation_reports_partial_result():
    result = gold_evidence_preservation(trace_with_chunking_results(), {})

    assert result.score == 0.5
    assert result.label == "partial"
    assert result.evidence["preserved_count"] == 1


def test_chunk_integrity_reports_defect_types_and_score():
    result = chunk_integrity(trace_with_chunking_results(), {})

    assert result.score == 0.5
    assert result.label == "degraded"
    assert result.evidence["defective_chunk_count"] == 1
    assert result.evidence["defects"][0]["chunk_id"] == "chunk-2"
    assert "truncated" in result.evidence["defects"][0]["types"]
    assert "invalid_offsets" in result.evidence["defects"][0]["types"]


def test_chunk_integrity_prefers_retrieved_chunk_text_when_available():
    trace = trace_with_chunking_results()
    trace["retrieval"] = {
        "retrieved_chunks": [
            {
                "chunk_id": "chunk-1",
                "document_id": "doc-1",
                "text": "A complete chunk.",
            },
            {
                "chunk_id": "chunk-2",
                "document_id": "doc-2",
                "text": "Retrieved text is truncated...",
            },
        ]
    }

    result = chunk_integrity(trace, {})

    assert result.evidence["checked_source"] == "retrieved_chunks"
    assert "truncated" in result.evidence["defects"][0]["types"]


def test_chunking_metrics_skip_when_gold_evidence_is_missing():
    result = gold_evidence_preservation({"trace_id": "missing-gold"}, {})

    assert result.status == "skipped"
    assert result.score is None
    assert result.label == "unknown"
