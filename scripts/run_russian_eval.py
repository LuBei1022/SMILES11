#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_russian_eval.py
在俄文完整 trace 上跑「评估层」实验（Part B 抽健康 + 注入 → Part C 指标 + 敏感度分析），
复用组里已有脚本，不修改它们。

链路:
  俄文 trace
    → extract_healthy.py        抽出健康 trace（金标准全在 context）
    → fault_injection.py        注入 9 类受控故障
    → analyze_controlled_failures.py  算指标 + 配对敏感度 + 热力图

关键点:
  - analyze_controlled_failures.py 用固定文件名找数据
    (<root>/healthy_traces/healthy_<method>.jsonl 和 <root>/controlled_failures_<method>/)，
    所以本脚本为俄文单开一个 --eval-root，避免覆盖英文结果。
  - quality_checker.py 写死 supported_language="en"，会把俄文标成 unsupported_language；
    它不是必经步骤（extract_healthy 直接读原始 trace），默认跳过；加 --run-quality 可选跑一次做参考。

用法:
  python scripts/run_russian_eval.py \
    --trace datasets/traces/russian_bm25_glm.jsonl \
    --method bm25 \
    --eval-root datasets/ru_eval \
    --count 20 --seed 42
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def run(cmd, desc):
    print(f"\n===== {desc} =====")
    print("$ " + " ".join(str(c) for c in cmd))
    # 从仓库根运行，保证 configs/ 与 src.metrics 导入正常
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print(f"!! 步骤失败（返回码 {result.returncode}）：{desc}")
        sys.exit(result.returncode)


def main():
    ap = argparse.ArgumentParser(description="俄文评估层实验一键运行")
    ap.add_argument("--trace", required=True, help="俄文完整 trace 文件（run_generation 的输出）")
    ap.add_argument("--method", default="bm25", choices=["bm25", "dense"],
                    help="检索方法名，须与 analyze 脚本约定一致")
    ap.add_argument("--eval-root", default="datasets/ru_eval",
                    help="俄文专用数据根目录（与英文分开）")
    ap.add_argument("--count", type=int, default=20, help="每类故障注入条数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", default="configs/metrics_default.yaml")
    ap.add_argument("--run-quality", action="store_true",
                    help="额外跑一次质量检查（俄文会被标 unsupported_language，仅供参考）")
    args = ap.parse_args()

    py = sys.executable
    root = Path(args.eval_root)
    healthy_path = root / "healthy_traces" / f"healthy_{args.method}.jsonl"
    controlled_dir = root / f"controlled_failures_{args.method}"
    os.makedirs(REPO_ROOT / "outputs", exist_ok=True)
    os.makedirs(REPO_ROOT / (root / "healthy_traces"), exist_ok=True)

    # 0.（可选）质量检查
    if args.run_quality:
        run([py, "src/data/quality_checker.py",
             "-i", args.trace,
             "-o", f"outputs/quality_ru_{args.method}.json",
             "-p", f"outputs/problematic_ru_{args.method}.jsonl"],
            "步骤0：质量检查（俄文语言标记为预期现象）")

    # 1. 抽健康 trace
    run([py, "scripts/extract_healthy.py",
         "-i", args.trace,
         "-o", str(healthy_path)],
        "步骤1：抽取健康 trace（金标准全在 context）")

    # 2. 注入受控故障
    run([py, "src/data/fault_injection.py",
         "-i", str(healthy_path),
         "-o", str(controlled_dir),
         "-c", str(args.count),
         "--seed", str(args.seed)],
        "步骤2：注入 9 类受控故障")

    # 3. 指标 + 配对敏感度分析 + 热力图（俄文专用输出名）
    run([py, "scripts/analyze_controlled_failures.py",
         "--data-root", str(root),
         "--config", args.config,
         "--json-output", "outputs/controlled_failure_report_ru.json",
         "--csv-output", "outputs/controlled_failure_summary_ru.csv",
         "--core-csv-output", "outputs/controlled_failure_core_results_ru.csv",
         "--heatmap-output", "outputs/controlled_failure_heatmap_ru.png",
         "--heatmap-pdf-output", "outputs/controlled_failure_heatmap_ru.pdf"],
        "步骤3：俄文指标敏感度分析 + 热力图")

    print("\n===== 俄文评估完成 =====")
    print(f"健康 trace : {healthy_path}")
    print(f"受控故障   : {controlled_dir}")
    print("热力图     : outputs/controlled_failure_heatmap_ru.png")
    print("核心结果   : outputs/controlled_failure_core_results_ru.csv")


if __name__ == "__main__":
    main()
