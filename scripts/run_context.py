#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_context.py
读取检索后的样本 (*_retrieved.jsonl)，把 retrieval.retrieved_chunks 组装成 final_context，
写入 sample["context_construction"]，并计算「金标准是否留在最终 context 里」。

context 级金标准保留:
  gold_chunk_ids（chunking 阶段标注的 covering_chunk_ids 并集）与
  selected_chunk_ids 的交集非空 => 金标准进入了 context（gold_in_context=1）。
  这是比检索 Recall 更靠后的一道指标：即使检索到了，也可能被去重/预算截断挤掉。

用法:
  python scripts/run_context.py \
    --input datasets/retrieved/english_bm25.jsonl \
    --output datasets/context/english_bm25_context.jsonl \
    [--max-tokens 3000] [--limit N] [--fallback-counter]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from context.context_builder import ContextBuilder  # noqa: E402


def fallback_counter(s: str) -> int:
    return max(1, len((s or "").split()))


def gold_chunk_set(sample):
    ids = set()
    for g in sample.get("gold_evidence") or []:
        for cid in g.get("covering_chunk_ids") or []:
            ids.add(cid)
    return ids


def main():
    ap = argparse.ArgumentParser(description="构造 final_context 并算金标准保留")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--no-dedup", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fallback-counter", action="store_true",
                    help="不用 tiktoken，用词数近似（仅调试）")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    builder = ContextBuilder(
        max_context_tokens=args.max_tokens,
        deduplicate=not args.no_dedup,
        token_counter=fallback_counter if args.fallback_counter else None,
    )

    n = 0
    n_with_gold = n_gold_in_ctx = n_truncated = 0
    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            if args.limit is not None and n >= args.limit:
                break
            sample = json.loads(line)
            retrieved = (sample.get("retrieval") or {}).get("retrieved_chunks", [])

            ctx = builder.build(retrieved)

            gold_ids = gold_chunk_set(sample)
            gold_in_ctx = None
            if gold_ids:
                gold_in_ctx = 1 if (gold_ids & set(ctx["selected_chunk_ids"])) else 0
                n_with_gold += 1
                n_gold_in_ctx += gold_in_ctx
            if ctx["truncated"]:
                n_truncated += 1

            ctx["eval"] = {"gold_in_context": gold_in_ctx, "n_gold_chunks": len(gold_ids)}
            sample["context_construction"] = ctx
            sample.setdefault("metadata", {})["context_builder"] = builder.config()

            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            n += 1

    print(f"[上下文] 处理 {n} 条, 发生预算截断 {n_truncated} 条")
    if n_with_gold:
        print(f"[上下文] 有金标准 {n_with_gold} 条: "
              f"金标准进入 context 比例={n_gold_in_ctx / n_with_gold * 100:.1f}%")
    print(f"[上下文] tokenizer: {'词数近似(降级)' if args.fallback_counter else 'tiktoken/cl100k_base'}")
    print(f"[上下文] 输出文件: {args.output}")


if __name__ == "__main__":
    main()
