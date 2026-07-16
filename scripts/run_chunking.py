#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_chunking.py
读取统一 source 样本 (*_source.jsonl)，对每篇 source_document 做正常切块，
并计算每条 gold_evidence 落在哪些 chunk、是否被完整保留，输出 *_chunks.jsonl。

金标准保留判断:
  - 先确定 gold 在所属文档中的字符区间 [gold_start, gold_end):
      俄文: 直接用 gold_evidence 里的 char_start / char_end;
      英文: 用 gold_text 在文档 text 中做子串定位得到区间。
  - 找出所有与该区间有重叠的 chunk => covering_chunk_ids;
  - 若存在某一个 chunk 完整包含该区间 => is_preserved = True（证据没被切开）;
    否则 False（证据被切到了多个 chunk / 被截断，属于潜在 chunking 隐患）。

用法:
  python scripts/run_chunking.py \
    --input datasets/normalized/english_source.jsonl \
    --output datasets/chunked/english_chunks.jsonl \
    [--chunk-size 400] [--overlap 80] [--limit N] [--fallback-counter]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from chunking.recursive_chunker import RecursiveChunker  # noqa: E402


def fallback_counter(s: str) -> int:
    """无 tiktoken 时的降级 token 估计：按空白词数近似。仅供调试，勿用于正式实验。"""
    return max(1, len(s.split()))


def locate_gold_span(doc_text, gold):
    """返回 (start, end) 或 None。"""
    cs, ce = gold.get("char_start"), gold.get("char_end")
    if isinstance(cs, int) and isinstance(ce, int) and 0 <= cs < ce <= len(doc_text):
        return (cs, ce)
    gt = gold.get("gold_text") or ""
    if gt:
        pos = doc_text.find(gt)
        if pos != -1:
            return (pos, pos + len(gt))
    return None


def annotate_gold(gold_list, doc_index, chunks_by_doc):
    """为每条 gold_evidence 加 covering_chunk_ids 与 is_preserved。"""
    n_gold = n_preserved = 0
    for gold in gold_list or []:
        n_gold += 1
        doc_id = gold["document_id"]
        doc_text = doc_index.get(doc_id, "")
        span = locate_gold_span(doc_text, gold)
        if span is None:
            gold["covering_chunk_ids"] = []
            gold["is_preserved"] = False
            gold["locate_ok"] = False
            continue
        gs, ge = span
        gold["locate_ok"] = True
        covering = []
        preserved = False
        for ch in chunks_by_doc.get(doc_id, []):
            # 区间重叠
            if ch["start_char"] < ge and gs < ch["end_char"]:
                covering.append(ch["chunk_id"])
                # 完整包含
                if ch["start_char"] <= gs and ge <= ch["end_char"]:
                    preserved = True
        gold["covering_chunk_ids"] = covering
        gold["is_preserved"] = preserved
        if preserved:
            n_preserved += 1
    return n_gold, n_preserved


def main():
    ap = argparse.ArgumentParser(description="对统一 source 样本做正常切块")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--chunk-size", type=int, default=400)
    ap.add_argument("--overlap", type=int, default=80)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fallback-counter", action="store_true",
                    help="不用 tiktoken，用词数近似（仅调试）")
    args = ap.parse_args()

    counter = fallback_counter if args.fallback_counter else None
    chunker = RecursiveChunker(
        chunk_size_tokens=args.chunk_size,
        chunk_overlap_tokens=args.overlap,
        token_counter=counter,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    n_in = n_out = 0
    tot_chunks = tot_gold = tot_preserved = 0
    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            if args.limit is not None and n_out >= args.limit:
                break
            n_in += 1
            sample = json.loads(line)

            chunks_by_doc = {}
            doc_index = {}
            all_chunks = []
            for doc in sample.get("source_documents", []):
                doc_id = doc["document_id"]
                doc_text = doc.get("text", "") or ""
                doc_index[doc_id] = doc_text
                dchunks = chunker.chunk_document(doc_id, doc_text)
                chunks_by_doc[doc_id] = dchunks
                all_chunks.extend(dchunks)

            n_gold, n_preserved = annotate_gold(
                sample.get("gold_evidence"), doc_index, chunks_by_doc)

            sample["chunks"] = all_chunks
            sample.setdefault("metadata", {})["chunking"] = chunker.config()

            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            n_out += 1
            tot_chunks += len(all_chunks)
            tot_gold += n_gold
            tot_preserved += n_preserved

    print(f"[切块] 处理 {n_out} 条样本, 共生成 {tot_chunks} 个 chunk "
          f"(平均 {tot_chunks / max(n_out,1):.1f}/条)")
    if tot_gold:
        print(f"[切块] 金标准证据 {tot_gold} 条, 完整保留 {tot_preserved} 条 "
              f"({tot_preserved / tot_gold * 100:.1f}%)")
    print(f"[切块] tokenizer: {'词数近似(降级)' if args.fallback_counter else 'tiktoken/cl100k_base'}")
    print(f"[切块] 输出文件: {args.output}")


if __name__ == "__main__":
    main()
