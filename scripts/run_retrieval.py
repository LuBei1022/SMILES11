#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_retrieval.py
读取切块后的样本 (*_chunks.jsonl)，对每个样本用选定检索器在其自身 chunk 上检索 top_k，
把结果写入 sample["retrieval"]，并用金标准计算 Recall@k / Hit@k。

Recall 依据: chunking 阶段已给每条 gold_evidence 标注 covering_chunk_ids
（金标准落在哪些 chunk）。检索命中 = 检索到的 chunk_id 与这些金标准 chunk 有交集。

用法:
  python scripts/run_retrieval.py \
    --input datasets/chunked/english_chunks.jsonl \
    --output datasets/retrieved/english_bm25.jsonl \
    --method bm25 --top-k 5 [--limit N]

  # 稠密检索:
  python scripts/run_retrieval.py --input ... --output ... --method dense --top-k 5
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def build_retriever(method, chunks, model_name):
    if method == "bm25":
        from retrieval.bm25_retriever import BM25Retriever
        return BM25Retriever(chunks)
    elif method == "dense":
        from retrieval.dense_retriever import DenseRetriever
        return DenseRetriever(chunks, model_name=model_name)
    raise ValueError(f"未知检索方法: {method}")


def gold_chunk_set(sample):
    """收集该样本所有金标准 chunk_id（并集）。"""
    ids = set()
    for g in sample.get("gold_evidence") or []:
        for cid in g.get("covering_chunk_ids") or []:
            ids.add(cid)
    return ids


def main():
    ap = argparse.ArgumentParser(description="对切块样本做检索并算 Recall@k")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--method", choices=["bm25", "dense"], default="bm25")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--model", default="paraphrase-multilingual-MiniLM-L12-v2",
                    help="dense 检索用的句向量模型名")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    n = 0
    n_with_gold = n_hit = 0
    recall_sum = 0.0
    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            if args.limit is not None and n >= args.limit:
                break
            sample = json.loads(line)
            chunks = sample.get("chunks", [])

            retriever = build_retriever(args.method, chunks, args.model)
            retrieved = retriever.search(sample.get("question", ""), top_k=args.top_k)

            # ---- Recall@k / Hit@k ----
            gold_ids = gold_chunk_set(sample)
            retrieved_ids = {r["chunk_id"] for r in retrieved}
            recall = hit = None
            if gold_ids:
                inter = gold_ids & retrieved_ids
                recall = len(inter) / len(gold_ids)
                hit = 1 if inter else 0
                n_with_gold += 1
                n_hit += hit
                recall_sum += recall

            sample["retrieval"] = {
                **retriever.config(),
                "top_k": args.top_k,
                "retrieved_chunks": retrieved,
                "eval": {"recall_at_k": recall, "hit_at_k": hit,
                         "n_gold_chunks": len(gold_ids)},
            }
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            n += 1

    print(f"[检索] 方法={args.method} top_k={args.top_k} 处理 {n} 条")
    if n_with_gold:
        print(f"[检索] 有金标准 {n_with_gold} 条: "
              f"Hit@{args.top_k}={n_hit / n_with_gold * 100:.1f}%  "
              f"平均 Recall@{args.top_k}={recall_sum / n_with_gold * 100:.1f}%")
    print(f"[检索] 输出文件: {args.output}")


if __name__ == "__main__":
    main()
