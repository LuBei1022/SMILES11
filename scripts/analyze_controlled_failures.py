#!/usr/bin/env python3
"""Run paired Metric Engine experiments on controlled RAG failures."""

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.metrics.reliability.controlled_failures import (
    analyze_controlled_failures,
    read_controlled_directory,
    read_jsonl,
    write_core_results,
    write_experiment_outputs,
    write_heatmap,
)
from src.metrics.runner import default_registry, load_metric_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze controlled failure metric sensitivity")
    parser.add_argument("--data-root", required=True, help="Root containing healthy_traces and controlled_failures_*")
    parser.add_argument("--config", default="configs/metrics_default.yaml")
    parser.add_argument("--json-output", default="outputs/controlled_failure_report.json")
    parser.add_argument("--csv-output", default="outputs/controlled_failure_summary.csv")
    parser.add_argument("--core-csv-output", default="outputs/controlled_failure_core_results.csv")
    parser.add_argument("--heatmap-output", default="outputs/controlled_failure_heatmap.png")
    parser.add_argument("--heatmap-pdf-output", default="outputs/controlled_failure_heatmap.pdf")
    args = parser.parse_args()

    root = Path(args.data_root)
    metrics = default_registry().all()
    config = load_metric_config(args.config)
    reports = []
    for retriever in ("bm25", "dense"):
        healthy_path = root / "healthy_traces" / f"healthy_{retriever}.jsonl"
        controlled_path = root / f"controlled_failures_{retriever}"
        if not healthy_path.is_file() or not controlled_path.is_dir():
            continue
        reports.append(analyze_controlled_failures(
            read_jsonl(str(healthy_path)),
            read_controlled_directory(str(controlled_path)),
            retriever,
            metrics,
            config,
        ))

    if not reports:
        parser.error("no BM25 or Dense healthy/controlled data found under --data-root")
    combined = {
        "metadata": {
            "data_root": str(root),
            "retrievers": [report["metadata"]["retriever"] for report in reports],
            "paired_records": sum(report["metadata"]["paired_records"] for report in reports),
            "unmatched_records": sum(report["metadata"]["unmatched_records"] for report in reports),
            "metric_count": len(metrics),
            "label_isolation": "injected_fault removed before MetricRunner",
        },
        "summary": [row for report in reports for row in report["summary"]],
        "unmatched_trace_ids": [
            {"retriever": report["metadata"]["retriever"], "trace_id": trace_id}
            for report in reports for trace_id in report["unmatched_trace_ids"]
        ],
        "per_retriever": {report["metadata"]["retriever"]: report["metadata"] for report in reports},
    }
    write_experiment_outputs(combined, args.json_output, args.csv_output)
    write_core_results(combined["summary"], args.core_csv_output)
    write_heatmap(combined["summary"], args.heatmap_output, args.heatmap_pdf_output)
    print(json.dumps(combined["metadata"], ensure_ascii=False, indent=2))
    print(f"JSON report: {args.json_output}")
    print(f"CSV summary: {args.csv_output}")
    print(f"Core results: {args.core_csv_output}")
    print(f"Heatmap PNG: {args.heatmap_output}")
    print(f"Heatmap PDF: {args.heatmap_pdf_output}")


if __name__ == "__main__":
    main()
