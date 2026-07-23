"""Paired controlled-failure sensitivity analysis for Metric Engine outputs."""

from __future__ import annotations

from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import random
from statistics import mean, median, pstdev
import tempfile
from typing import Any

from ..base import Metric
from ..runner import MetricRunner


EXPECTED_DIRECTIONS = {
    "missing_evidence": {
        "retrieval.hit_at_k": "decrease",
        "retrieval.recall_at_k": "decrease",
        "retrieval.precision_at_k": "decrease",
        "retrieval.mrr": "decrease",
        "context.evidence_coverage": "decrease",
        "context.noise_ratio": "increase",
    },
    "irrelevant_document": {
        "retrieval.hit_at_k": "decrease",
        "retrieval.recall_at_k": "decrease",
        "retrieval.precision_at_k": "decrease",
        "retrieval.mrr": "decrease",
        "context.noise_ratio": "increase",
    },
    "chunk_truncation": {"chunking.chunk_integrity": "decrease"},
    "chunk_merge": {"chunking.chunk_integrity": "decrease"},
    "distractor_context": {
        "context.noise_ratio": "increase",
        "generation.faithfulness": "decrease",
    },
    "unsupported_answer": {"generation.faithfulness": "decrease"},
    "contradictory_answer": {"generation.faithfulness": "decrease"},
    "corrupted_query": {"generation.answer_relevance": "decrease"},
    "out_of_scope": {"generation.answer_relevance": "decrease"},
}

LOWER_IS_BETTER = {"context.noise_ratio"}

CSV_FIELDS = [
    "retriever", "fault_type", "metric_name", "sample_count", "healthy_mean",
    "fault_mean", "mean_delta", "median_delta", "std_delta", "improved_count",
    "degraded_count", "unchanged_count", "expected_direction", "direction_match",
    "ci_95_lower", "ci_95_upper", "standardized_effect", "rank_biserial_effect",
    "quality_mean_delta", "skipped_count", "error_count",
]

CORE_FIELDS = [
    "retriever", "fault_type", "metric_name", "sample_count", "healthy_mean",
    "fault_mean", "mean_delta", "ci_95_lower", "ci_95_upper",
    "standardized_effect", "rank_biserial_effect", "expected_direction",
    "direction_match", "skipped_count", "error_count",
]


def sanitize_trace(trace: dict[str, Any]) -> dict[str, Any]:
    clean = deepcopy(trace)
    clean.pop("injected_fault", None)
    clean.pop("_is_healthy", None)
    return clean


def pair_controlled_records(
    healthy_by_id: dict[str, dict[str, Any]],
    wrappers: list[dict[str, Any]],
) -> tuple[list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]], list[str]]:
    pairs = []
    unmatched = []
    for wrapper in wrappers:
        original_id = wrapper.get("original_trace_id")
        fault_trace = wrapper.get("trace")
        healthy = healthy_by_id.get(original_id) if isinstance(original_id, str) else None
        if healthy is None or not isinstance(fault_trace, dict):
            unmatched.append(str(original_id or "unknown"))
            continue
        pairs.append((healthy, fault_trace, wrapper))
    return pairs, sorted(unmatched)


def _result_map(runner: MetricRunner, trace: dict[str, Any]):
    return {result.metric_name: result for result in runner.run_trace(sanitize_trace(trace))}


def _direction_match(expected: str, delta: float | None) -> bool | None:
    if expected == "unspecified" or delta is None:
        return None
    if expected == "decrease":
        return delta < 0
    if expected == "increase":
        return delta > 0
    return None


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bootstrap_mean_ci(
    deltas: list[float], iterations: int = 2000, seed: int = 42,
) -> tuple[float | None, float | None]:
    if not deltas:
        return None, None
    generator = random.Random(seed)
    size = len(deltas)
    means = [mean(generator.choice(deltas) for _ in range(size)) for _ in range(iterations)]
    return _percentile(means, 0.025), _percentile(means, 0.975)


def _rank_biserial_effect(deltas: list[float]) -> float | None:
    nonzero = [(abs(delta), delta) for delta in deltas if delta != 0]
    if not nonzero:
        return 0.0

    ordered = sorted(nonzero, key=lambda item: item[0])
    signed_ranks: list[tuple[float, float]] = []
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][0] == ordered[start][0]:
            end += 1
        average_rank = ((start + 1) + end) / 2
        signed_ranks.extend((delta, average_rank) for _, delta in ordered[start:end])
        start = end

    positive_ranks = sum(rank for delta, rank in signed_ranks if delta > 0)
    negative_ranks = sum(rank for delta, rank in signed_ranks if delta < 0)
    return (positive_ranks - negative_ranks) / (positive_ranks + negative_ranks)


def analyze_controlled_failures(
    healthy_traces: list[dict[str, Any]],
    wrappers: list[dict[str, Any]],
    retriever: str,
    metrics: list[Metric],
    config: dict[str, Any],
) -> dict[str, Any]:
    healthy_by_id = {
        trace["trace_id"]: trace for trace in healthy_traces
        if isinstance(trace, dict) and isinstance(trace.get("trace_id"), str)
    }
    pairs, unmatched = pair_controlled_records(healthy_by_id, wrappers)
    runner = MetricRunner(metrics, config)
    healthy_cache: dict[str, dict[str, Any]] = {}
    grouped: dict[tuple[str, str], dict[str, Any]] = {}

    for healthy, fault_trace, wrapper in pairs:
        trace_id = healthy["trace_id"]
        if trace_id not in healthy_cache:
            healthy_cache[trace_id] = _result_map(runner, healthy)
        healthy_results = healthy_cache[trace_id]
        fault_results = _result_map(runner, fault_trace)
        fault_type = wrapper.get("injected_fault")
        if not isinstance(fault_type, str):
            fault_type = Path(str(wrapper.get("source_file", "unknown"))).stem

        for metric in metrics:
            key = (fault_type, metric.name)
            bucket = grouped.setdefault(key, {
                "healthy_scores": [], "fault_scores": [], "deltas": [],
                "skipped_count": 0, "error_count": 0,
            })
            healthy_result = healthy_results[metric.name]
            fault_result = fault_results[metric.name]
            statuses = {healthy_result.status, fault_result.status}
            if "error" in statuses:
                bucket["error_count"] += 1
                continue
            if "skipped" in statuses:
                bucket["skipped_count"] += 1
                continue
            if healthy_result.score is None or fault_result.score is None:
                bucket["skipped_count"] += 1
                continue
            healthy_score = float(healthy_result.score)
            fault_score = float(fault_result.score)
            bucket["healthy_scores"].append(healthy_score)
            bucket["fault_scores"].append(fault_score)
            bucket["deltas"].append(fault_score - healthy_score)

    summary = []
    for (fault_type, metric_name), bucket in sorted(grouped.items()):
        deltas = bucket["deltas"]
        expected = EXPECTED_DIRECTIONS.get(fault_type, {}).get(metric_name, "unspecified")
        mean_delta = mean(deltas) if deltas else None
        std_delta = pstdev(deltas) if deltas else None
        ci_lower, ci_upper = bootstrap_mean_ci(deltas)
        orientation = -1 if metric_name in LOWER_IS_BETTER else 1
        quality_deltas = [delta * orientation for delta in deltas]
        summary.append({
            "retriever": retriever,
            "fault_type": fault_type,
            "metric_name": metric_name,
            "sample_count": len(deltas),
            "healthy_mean": mean(bucket["healthy_scores"]) if deltas else None,
            "fault_mean": mean(bucket["fault_scores"]) if deltas else None,
            "mean_delta": mean_delta,
            "median_delta": median(deltas) if deltas else None,
            "std_delta": std_delta,
            "ci_95_lower": ci_lower,
            "ci_95_upper": ci_upper,
            "standardized_effect": mean_delta / std_delta if std_delta else None,
            "rank_biserial_effect": _rank_biserial_effect(deltas),
            "quality_mean_delta": mean(quality_deltas) if quality_deltas else None,
            "improved_count": sum(delta > 0 for delta in quality_deltas),
            "degraded_count": sum(delta < 0 for delta in quality_deltas),
            "unchanged_count": sum(delta == 0 for delta in quality_deltas),
            "expected_direction": expected,
            "direction_match": _direction_match(expected, mean_delta),
            "skipped_count": bucket["skipped_count"],
            "error_count": bucket["error_count"],
        })

    return {
        "metadata": {
            "retriever": retriever,
            "healthy_records": len(healthy_traces),
            "controlled_records": len(wrappers),
            "paired_records": len(pairs),
            "unmatched_records": len(unmatched),
            "metric_count": len(metrics),
            "label_isolation": "injected_fault removed before MetricRunner",
        },
        "summary": summary,
        "unmatched_trace_ids": unmatched,
    }


def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(value)
    return rows


def read_controlled_directory(path: str) -> list[dict[str, Any]]:
    wrappers = []
    for source in sorted(Path(path).glob("*.jsonl")):
        for wrapper in read_jsonl(str(source)):
            wrapper = dict(wrapper)
            wrapper["source_file"] = source.name
            wrappers.append(wrapper)
    return wrappers


def write_experiment_outputs(report: dict[str, Any], json_path: str, csv_path: str) -> None:
    json_output = Path(json_path)
    csv_output = Path(csv_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(report.get("summary", []))


def write_core_results(summary: list[dict[str, Any]], path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [row for row in summary if row.get("expected_direction") != "unspecified"]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CORE_FIELDS, extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(rows)


def write_heatmap(summary: list[dict[str, Any]], png_path: str, pdf_path: str) -> None:
    mpl_config = Path(tempfile.gettempdir()) / "smiles11-matplotlib"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    retrievers = sorted({str(row["retriever"]) for row in summary})
    faults = sorted({str(row["fault_type"]) for row in summary})
    metrics = sorted({str(row["metric_name"]) for row in summary})
    values = {
        (str(row["retriever"]), str(row["fault_type"]), str(row["metric_name"])):
        row.get("quality_mean_delta")
        for row in summary
    }
    numeric = [abs(float(value)) for value in values.values() if isinstance(value, (int, float))]
    limit = max(numeric, default=1.0) or 1.0
    fig, axes = plt.subplots(
        1, len(retrievers), figsize=(max(12, len(metrics) * 1.25 * len(retrievers)), 7),
        squeeze=False, constrained_layout=True,
    )
    image = None
    for column, retriever in enumerate(retrievers):
        matrix = np.full((len(faults), len(metrics)), np.nan)
        for row_index, fault in enumerate(faults):
            for metric_index, metric in enumerate(metrics):
                value = values.get((retriever, fault, metric))
                if isinstance(value, (int, float)):
                    matrix[row_index, metric_index] = value
        axis = axes[0][column]
        image = axis.imshow(matrix, cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto")
        axis.set_title(retriever.upper())
        axis.set_xticks(range(len(metrics)), [name.split(".")[-1] for name in metrics], rotation=55, ha="right")
        axis.set_yticks(range(len(faults)), faults if column == 0 else [""] * len(faults))
        axis.set_xlabel("Metric")
        if column == 0:
            axis.set_ylabel("Controlled fault")
        for row_index in range(len(faults)):
            for metric_index in range(len(metrics)):
                value = matrix[row_index, metric_index]
                if not np.isnan(value) and abs(value) >= 0.005:
                    axis.text(metric_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=7)
    fig.suptitle("Controlled-failure metric sensitivity\nQuality-normalized paired delta (fault - healthy)")
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8)
        colorbar.set_label("Quality delta: negative = degradation")
    for path in (Path(png_path), Path(pdf_path)):
        path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
