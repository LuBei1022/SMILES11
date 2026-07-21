#!/usr/bin/env python3
"""Run all first-version metrics over a full-trace JSONL file."""

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.metrics.runner import MetricRunner, default_registry, load_metric_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SMILES RAG Metric Engine")
    parser.add_argument("--input", required=True, help="Input full-trace JSONL path")
    parser.add_argument("--output", required=True, help="Output MetricResult JSONL path")
    parser.add_argument("--config", default=None, help="Optional metric YAML config")
    args = parser.parse_args()

    runner = MetricRunner(default_registry().all(), load_metric_config(args.config))
    summary = runner.run_jsonl(args.input, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
