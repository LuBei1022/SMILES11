import copy
import json
from pathlib import Path

from src.metrics.runner import MetricRunner, default_registry, load_metric_config


FIXTURES = Path(__file__).parent / "fixtures"


def read_fixture(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_default_registry_contains_all_eleven_metrics_in_stable_order():
    names = [metric.name for metric in default_registry().all()]

    assert names == [
        "retrieval.hit_at_k",
        "retrieval.recall_at_k",
        "retrieval.precision_at_k",
        "retrieval.mrr",
        "chunking.gold_evidence_preservation",
        "chunking.chunk_integrity",
        "context.evidence_coverage",
        "context.noise_ratio",
        "context.context_truncation",
        "generation.answer_relevance",
        "generation.faithfulness",
    ]


def test_runner_emits_eleven_results_without_mutating_trace():
    trace = read_fixture("healthy_trace.json")
    original = copy.deepcopy(trace)
    runner = MetricRunner(default_registry().all(), load_metric_config(None))

    results = runner.run_trace(trace)

    assert len(results) == 11
    assert {result.status for result in results} == {"ok"}
    assert trace == original


def test_runner_continues_after_malformed_trace(tmp_path):
    malformed = read_fixture("malformed_trace.json")
    healthy = read_fixture("healthy_trace.json")
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    input_path.write_text(
        "\n".join(json.dumps(value) for value in [malformed, healthy]) + "\n",
        encoding="utf-8",
    )
    runner = MetricRunner(default_registry().all(), load_metric_config(None))

    summary = runner.run_jsonl(str(input_path), str(output_path))
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

    assert summary["input_traces"] == 2
    assert summary["results"] == 22
    assert len(rows) == 22
    assert any(row["status"] == "error" for row in rows[:11])
    assert all(row["trace_id"] == "fixture-en" for row in rows[11:])


def test_load_metric_config_reads_yaml_and_merges_defaults(tmp_path):
    config_path = tmp_path / "metrics.yaml"
    config_path.write_text("metrics:\n  retrieval:\n    k: 3\n", encoding="utf-8")

    config = load_metric_config(str(config_path))

    assert config["metrics"]["retrieval"]["k"] == 3
    assert "generation" in config["metrics"]
