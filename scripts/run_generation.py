#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_generation.py
读取上下文构造后的样本 (*_context.jsonl)，用选定 LLM 依据 final_context 生成答案，
组装成【完整 Trace】写入 *_traces.jsonl。生成后每一行就是一条可评估的完整 trace。

用法（先离线打通）:
  python scripts/run_generation.py \
    --input datasets/context/english_bm25_context.jsonl \
    --output datasets/traces/english_bm25_gemini.jsonl \
    --pipeline-id en_bm25_gemini_baseline_v1 \
    --dry-run --limit 5

  # 真正调用 Gemini（先 export GEMINI_API_KEY=...）:
  python scripts/run_generation.py \
    --input datasets/context/english_bm25_context.jsonl \
    --output datasets/traces/english_bm25_gemini.jsonl \
    --pipeline-id en_bm25_gemini_baseline_v1 \
    --backend gemini --model gemini-3.5-flash --sleep 1 --limit 5
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from generation.prompt_builder import build_prompt  # noqa: E402
from generation.llm_client import build_client  # noqa: E402


def assemble_trace(sample, pipeline_id, generation, prompt_version, final_prompt, run_id):
    """把累积的各阶段结果组装成完整 trace（对应方案 §13）。"""
    sample_id = sample.get("sample_id")
    generation = dict(generation)
    generation["prompt_version"] = prompt_version
    generation["final_prompt"] = final_prompt
    return {
        "trace_id": f"{sample_id}__{pipeline_id}",
        "source_record_id": sample_id,
        "pipeline_id": pipeline_id,
        "language": sample.get("language"),
        "query": sample.get("question"),
        "reference_answer": sample.get("reference_answer"),
        "gold_evidence": sample.get("gold_evidence"),
        "source_documents": sample.get("source_documents"),
        "chunks": sample.get("chunks"),
        "retrieval": sample.get("retrieval"),
        "context_construction": sample.get("context_construction"),
        "generation": generation,
        "pipeline_config": (sample.get("metadata") or {}),
        "injected_fault": None,
        "runtime": {
            "run_id": run_id,
            "status": generation.get("status"),
            "error": generation.get("error"),
        },
    }


def main():
    ap = argparse.ArgumentParser(description="生成答案并保存完整 Trace")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--pipeline-id", required=True)
    ap.add_argument("--backend", choices=["gemini", "zhipu", "dry-run"], default="dry-run")
    ap.add_argument("--model", default="gemini-1.5-flash")
    ap.add_argument("--prompt-version", default="grounded_v1")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-output-tokens", type=int, default=256)
    ap.add_argument("--sleep", type=float, default=0.0, help="每次调用后 sleep 秒数（防限流）")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true",
                    help="断点续跑：跳过输出文件里已成功的样本，只补未完成/失败的")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    client = build_client(
        args.backend,
        model_name=args.model,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
    )
    run_id = time.strftime("run_%Y%m%d_%H%M%S")

    n = n_ok = n_err = 0
    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            if args.limit is not None and n >= args.limit:
                break
            sample = json.loads(line)
            final_context = (sample.get("context_construction") or {}).get("final_context", "")
            prompt = build_prompt(sample.get("question", ""), final_context, args.prompt_version)

            gen = client.generate(prompt)
            if gen.get("status") == "success":
                n_ok += 1
            else:
                n_err += 1

            trace = assemble_trace(sample, args.pipeline_id, gen,
                                   args.prompt_version, prompt, run_id)
            fout.write(json.dumps(trace, ensure_ascii=False) + "\n")
            n += 1
            if args.sleep:
                time.sleep(args.sleep)

    print(f"[生成] backend={args.backend} model={args.model} pipeline={args.pipeline_id}")
    print(f"[生成] 完整 trace {n} 条: 成功 {n_ok}, 失败 {n_err}")
    print(f"[生成] 输出文件: {args.output}")


if __name__ == "__main__":
    main()
