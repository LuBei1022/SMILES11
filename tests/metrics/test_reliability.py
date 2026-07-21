import json

from src.metrics.reliability.repeatability import compare_runs, validate_against_labels


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_compare_runs_reports_exact_matches_and_mean_absolute_delta(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write_jsonl(first, [
        {"trace_id": "t1", "metric_name": "m", "score": 0.8},
        {"trace_id": "t2", "metric_name": "m", "score": 0.2},
    ])
    write_jsonl(second, [
        {"trace_id": "t1", "metric_name": "m", "score": 0.8},
        {"trace_id": "t2", "metric_name": "m", "score": 0.4},
    ])

    report = compare_runs([str(first), str(second)])

    assert report["result_count"] == 2
    assert report["exact_match_rate"] == 0.5
    assert report["mean_absolute_score_delta"] == 0.1
    assert report["missing_results"] == []


def test_compare_runs_reports_missing_result_keys(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write_jsonl(first, [{"trace_id": "t1", "metric_name": "m", "score": 0.8}])
    write_jsonl(second, [{"trace_id": "t2", "metric_name": "m", "score": 0.8}])

    report = compare_runs([str(first), str(second)])

    assert report["missing_results"] == ["t1|m", "t2|m"]
    assert report["warnings"]


def test_validate_against_labels_is_separate_offline_comparison(tmp_path):
    metrics = tmp_path / "diagnosis.jsonl"
    labels = tmp_path / "labels.csv"
    write_jsonl(metrics, [
        {"trace_id": "t1", "fault_label": "retrieval"},
        {"trace_id": "t2", "fault_label": "generation"},
    ])
    labels.write_text(
        "trace_id,expected_diagnosis,fault_class\n"
        "t1,retrieval,retrieval\n"
        "t2,chunking,chunking\n",
        encoding="utf-8",
    )

    report = validate_against_labels(str(metrics), str(labels))

    assert report["total"] == 2
    assert report["correct"] == 1
    assert report["accuracy"] == 0.5
    assert report["compared_field"] == "fault_label"
