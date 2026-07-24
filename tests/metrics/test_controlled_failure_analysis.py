import csv
import json
import os
from pathlib import Path

from src.metrics.base import MetricResult
from src.metrics.reliability.controlled_failures import (
    analyze_controlled_failures,
    bootstrap_mean_ci,
    pair_controlled_records,
    write_core_results,
    write_experiment_outputs,
    write_heatmap,
)


class InjectedFaultGuardMetric:
    name = "generation.faithfulness"
    stage = "generation"

    def compute(self, trace, config):
        assert "injected_fault" not in trace
        score = float(trace["test_score"])
        return MetricResult(
            trace_id=trace["trace_id"],
            metric_name=self.name,
            stage=self.stage,
            score=score,
            label="test",
        )


class CountingMetric(InjectedFaultGuardMetric):
    def __init__(self):
        self.call_count = 0

    def compute(self, trace, config):
        self.call_count += 1
        return super().compute(trace, config)


def test_pair_controlled_records_uses_original_trace_id():
    healthy = {
        "trace-1": {"trace_id": "trace-1", "test_score": 1.0},
    }
    wrappers = [
        {
            "original_trace_id": "trace-1",
            "injected_fault": "unsupported_answer",
            "trace": {"trace_id": "trace-1", "test_score": 0.0},
        },
        {
            "original_trace_id": "missing",
            "injected_fault": "unsupported_answer",
            "trace": {"trace_id": "missing", "test_score": 0.0},
        },
    ]

    pairs, unmatched = pair_controlled_records(healthy, wrappers)

    assert len(pairs) == 1
    assert pairs[0][0]["trace_id"] == "trace-1"
    assert pairs[0][1]["test_score"] == 0.0
    assert unmatched == ["missing"]


def test_analysis_computes_paired_delta_and_expected_direction():
    healthy = [{"trace_id": "trace-1", "test_score": 1.0, "injected_fault": None}]
    wrappers = [{
        "original_trace_id": "trace-1",
        "injected_fault": "unsupported_answer",
        "expected_diagnosis": "generation",
        "trace": {"trace_id": "trace-1", "test_score": 0.0, "injected_fault": {"type": "unsupported_answer"}},
    }]

    report = analyze_controlled_failures(
        healthy,
        wrappers,
        retriever="bm25",
        metrics=[InjectedFaultGuardMetric()],
        config={"metrics": {"generation": {}}},
    )
    row = report["summary"][0]

    assert row["fault_type"] == "unsupported_answer"
    assert row["metric_name"] == "generation.faithfulness"
    assert row["sample_count"] == 1
    assert row["healthy_mean"] == 1.0
    assert row["fault_mean"] == 0.0
    assert row["mean_delta"] == -1.0
    assert row["expected_direction"] == "decrease"
    assert row["direction_match"] is True
    assert row["degraded_count"] == 1
    assert row["ci_95_lower"] == -1.0
    assert row["ci_95_upper"] == -1.0
    assert row["rank_biserial_effect"] == -1.0


def test_analysis_caches_reused_healthy_trace_results():
    metric = CountingMetric()
    healthy = [{"trace_id": "trace-1", "test_score": 1.0}]
    wrappers = [
        {"original_trace_id": "trace-1", "injected_fault": "unsupported_answer",
         "trace": {"trace_id": "trace-1", "test_score": 0.0}},
        {"original_trace_id": "trace-1", "injected_fault": "contradictory_answer",
         "trace": {"trace_id": "trace-1", "test_score": 0.0}},
    ]

    analyze_controlled_failures(
        healthy, wrappers, retriever="bm25", metrics=[metric],
        config={"metrics": {"generation": {}}},
    )

    assert metric.call_count == 3


def test_analysis_rank_biserial_effect_uses_absolute_delta_ranks():
    healthy_scores = (0.0, 0.0, 1.0)
    fault_scores = (0.1, 0.2, 0.1)
    healthy = [
        {"trace_id": f"trace-{index}", "test_score": score}
        for index, score in enumerate(healthy_scores)
    ]
    wrappers = [
        {
            "original_trace_id": f"trace-{index}",
            "injected_fault": "unsupported_answer",
            "trace": {"trace_id": f"trace-{index}", "test_score": score},
        }
        for index, score in enumerate(fault_scores)
    ]

    report = analyze_controlled_failures(
        healthy, wrappers, retriever="bm25", metrics=[InjectedFaultGuardMetric()],
        config={"metrics": {"generation": {}}},
    )

    assert report["summary"][0]["rank_biserial_effect"] == 0.0


def test_write_experiment_outputs_has_stable_csv_and_json(tmp_path):
    report = {
        "metadata": {"paired_records": 1},
        "summary": [{
            "retriever": "bm25",
            "fault_type": "unsupported_answer",
            "metric_name": "generation.faithfulness",
            "sample_count": 1,
            "healthy_mean": 1.0,
            "fault_mean": 0.0,
            "mean_delta": -1.0,
            "median_delta": -1.0,
            "std_delta": 0.0,
            "ci_95_lower": -1.0,
            "ci_95_upper": -1.0,
            "standardized_effect": None,
            "rank_biserial_effect": -1.0,
            "quality_mean_delta": -1.0,
            "improved_count": 0,
            "degraded_count": 1,
            "unchanged_count": 0,
            "expected_direction": "decrease",
            "direction_match": True,
            "skipped_count": 0,
            "error_count": 0,
        }],
        "unmatched_trace_ids": [],
    }
    json_path = tmp_path / "report.json"
    csv_path = tmp_path / "summary.csv"

    write_experiment_outputs(report, str(json_path), str(csv_path))

    saved = json.loads(json_path.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert saved["metadata"]["paired_records"] == 1
    assert rows[0]["fault_type"] == "unsupported_answer"
    assert rows[0]["direction_match"] == "True"


def test_bootstrap_mean_ci_is_deterministic():
    first = bootstrap_mean_ci([-1.0, -0.5, 0.0], iterations=500, seed=7)
    second = bootstrap_mean_ci([-1.0, -0.5, 0.0], iterations=500, seed=7)

    assert first == second
    assert first[0] <= -0.5 <= first[1]


def test_core_results_and_heatmap_are_written(tmp_path, monkeypatch):
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)
    summary = [
        {
            "retriever": "bm25", "fault_type": "unsupported_answer",
            "metric_name": "generation.faithfulness", "mean_delta": -0.9,
            "quality_mean_delta": -0.9, "expected_direction": "decrease",
            "direction_match": True,
        },
        {
            "retriever": "bm25", "fault_type": "unsupported_answer",
            "metric_name": "retrieval.mrr", "mean_delta": 0.0,
            "quality_mean_delta": 0.0, "expected_direction": "unspecified",
            "direction_match": None,
        },
    ]
    core_path = tmp_path / "core.csv"
    png_path = tmp_path / "heatmap.png"
    pdf_path = tmp_path / "heatmap.pdf"

    write_core_results(summary, str(core_path))
    write_heatmap(summary, str(png_path), str(pdf_path))

    rows = list(csv.DictReader(core_path.open(encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["metric_name"] == "generation.faithfulness"
    assert png_path.stat().st_size > 1000
    assert pdf_path.stat().st_size > 1000
    mpl_config = Path(os.environ["MPLCONFIGDIR"])
    assert mpl_config.name == "smiles11-matplotlib"
    assert mpl_config.is_dir()
