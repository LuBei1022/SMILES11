"""Compare repeated runs and offline labels without affecting metric execution."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(value)
    return rows


def _result_key(row: dict[str, Any]) -> str:
    return f"{row.get('trace_id', 'unknown')}|{row.get('metric_name', 'unknown')}"


def compare_runs(result_paths: list[str]) -> dict[str, Any]:
    if len(result_paths) < 2:
        raise ValueError("compare_runs requires at least two result files")
    indexed_runs = [
        {_result_key(row): row for row in _read_jsonl(path)} for path in result_paths
    ]
    all_keys = sorted(set().union(*(set(run) for run in indexed_runs)))
    common_keys = [key for key in all_keys if all(key in run for run in indexed_runs)]
    missing_keys = [key for key in all_keys if key not in common_keys]

    exact_matches = 0
    deltas: list[float] = []
    for key in common_keys:
        scores = [run[key].get("score") for run in indexed_runs]
        statuses = [run[key].get("status") for run in indexed_runs]
        labels = [run[key].get("label") for run in indexed_runs]
        if len(set(map(repr, scores))) == 1 and len(set(statuses)) == 1 and len(set(labels)) == 1:
            exact_matches += 1
        numeric_scores = [float(score) for score in scores if isinstance(score, (int, float))]
        if len(numeric_scores) == len(scores):
            baseline = numeric_scores[0]
            deltas.extend(abs(score - baseline) for score in numeric_scores[1:])

    warnings = []
    if missing_keys:
        warnings.append("some trace/metric results are missing from one or more runs")
    return {
        "run_count": len(result_paths),
        "result_count": len(common_keys),
        "exact_matches": exact_matches,
        "exact_match_rate": exact_matches / len(common_keys) if common_keys else None,
        "mean_absolute_score_delta": round(sum(deltas) / len(deltas), 12) if deltas else None,
        "missing_results": missing_keys,
        "warnings": warnings,
    }


def validate_against_labels(metric_path: str, label_path: str) -> dict[str, Any]:
    predictions = _read_jsonl(metric_path)
    prediction_fields = ("fault_label", "diagnosis_label", "label")
    compared_field = next(
        (field for field in prediction_fields if any(field in row for row in predictions)), None
    )
    if compared_field is None:
        raise ValueError("prediction file has no fault_label, diagnosis_label, or label field")
    predicted_by_trace = {
        str(row.get("trace_id")): row.get(compared_field)
        for row in predictions
        if row.get("trace_id") is not None and row.get(compared_field) is not None
    }

    with open(label_path, "r", encoding="utf-8-sig", newline="") as handle:
        labels = list(csv.DictReader(handle))
    compared: list[dict[str, Any]] = []
    missing_predictions: list[str] = []
    for row in labels:
        trace_id = row.get("trace_id")
        expected = row.get("expected_diagnosis") or row.get("expected") or row.get("fault_class") or row.get("human")
        if not trace_id or not expected:
            continue
        predicted = predicted_by_trace.get(trace_id)
        if predicted is None:
            missing_predictions.append(trace_id)
            continue
        compared.append({"trace_id": trace_id, "expected": expected, "predicted": predicted,
                         "correct": predicted == expected})

    correct = sum(int(row["correct"]) for row in compared)
    return {
        "total": len(compared),
        "correct": correct,
        "accuracy": correct / len(compared) if compared else None,
        "compared_field": compared_field,
        "missing_predictions": sorted(missing_predictions),
        "disagreements": [row for row in compared if not row["correct"]],
        "note": "Offline validation only; normal Metric Engine execution does not read labels.",
    }
