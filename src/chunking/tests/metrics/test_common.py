import copy

from src.metrics.common.trace_utils import (
    gold_chunk_ids,
    retrieved_chunks,
    safe_error,
    selected_chunk_ids,
)


def test_trace_helpers_extract_expected_ids_without_mutation():
    trace = {
        "gold_evidence": [
            {"covering_chunk_ids": ["c1", "c2"]},
            {"covering_chunk_ids": ["c2", "c3"]},
        ],
        "retrieval": {"retrieved_chunks": [{"chunk_id": "c3"}]},
        "context_construction": {"selected_chunk_ids": ["c1", "c3"]},
        "injected_fault": {"fault_type": "must-not-be-read"},
    }
    original = copy.deepcopy(trace)

    assert gold_chunk_ids(trace) == {"c1", "c2", "c3"}
    assert retrieved_chunks(trace) == [{"chunk_id": "c3"}]
    assert selected_chunk_ids(trace) == {"c1", "c3"}
    assert trace == original


def test_trace_helpers_return_empty_collections_for_missing_fields():
    assert gold_chunk_ids({}) == set()
    assert retrieved_chunks({}) == []
    assert selected_chunk_ids({}) == set()


def test_safe_error_has_stable_structure():
    assert safe_error(
        "invalid_field_type",
        "retrieved_chunks must be a list",
        field="retrieval.retrieved_chunks",
        expected="array",
        actual="string",
    ) == {
        "code": "invalid_field_type",
        "message": "retrieved_chunks must be a list",
        "details": {
            "field": "retrieval.retrieved_chunks",
            "expected": "array",
            "actual": "string",
        },
    }
