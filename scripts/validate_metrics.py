#!/usr/bin/env python3
"""Offline repeatability or label comparison for Metric Engine outputs."""

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.metrics.reliability import compare_runs, validate_against_labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Metric Engine outputs offline")
    parser.add_argument("--runs", nargs="+", help="Two or more repeated MetricResult JSONL files")
    parser.add_argument("--metrics", help="Prediction or diagnosis JSONL for label comparison")
    parser.add_argument("--labels", help="CSV containing trace_id and expected/human labels")
    parser.add_argument("--output", required=True, help="Output JSON report")
    args = parser.parse_args()

    if args.runs:
        report = compare_runs(args.runs)
    elif args.metrics and args.labels:
        report = validate_against_labels(args.metrics, args.labels)
    else:
        parser.error("provide --runs, or provide both --metrics and --labels")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
