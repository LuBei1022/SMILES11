#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
normalize_russian_data.py
把俄文 retrieval_dataset.jsonl 转换为统一 source 样本格式。

原始字段: text / q / a / context
  q        -> question
  a        -> reference_answer
  context  -> source_documents   (整个 blob 作为单篇待检索源文档；后续 chunk 阶段再切分)
  text     -> gold_evidence       (金标准段落，用子串在 context 中定位，记录字符偏移)

设计说明:
  - 俄文 context 是「金标准段落 + 干扰段落」拼成的一坨字符串，角色等同英文的候选文档池。
  - text 是金标准段落（已验证 100% 为 context 子串），角色等同英文的 supporting_facts。
  - 俄文无结构化句子/标题，故 title=null, sentences=null, sentence_index=null，
    金标准用 char_start/char_end 定位，供 chunk 阶段计算「金标准是否被切坏/保留」。

用法:
  python scripts/normalize_russian_data.py \
    --input data/raw/retrieval_dataset.jsonl \
    --output data/normalized/russian_source.jsonl \
    [--limit N]
"""
import argparse
import json


def convert_record(rec, line_idx, warnings):
    q = rec.get("q", "")
    a = rec.get("a", "")
    context = rec.get("context", "") or ""
    text = rec.get("text", "") or ""

    sample_id = f"ru__retrieval_dataset__{line_idx:06d}"
    doc_id = f"{sample_id}__doc_000"

    source_documents = [{
        "document_id": doc_id,
        "title": None,
        "text": context,
        "sentences": None,
    }]

    # ---- gold_evidence: 在 context 中定位 text ----
    gold_evidence = None
    if text:
        char_start = context.find(text)
        if char_start == -1:
            warnings.append(f"{sample_id}: gold text 未在 context 中找到（子串匹配失败）")
            char_start = None
            char_end = None
        else:
            char_end = char_start + len(text)
        gold_evidence = [{
            "document_id": doc_id,
            "sentence_index": None,
            "gold_text": text,
            "char_start": char_start,
            "char_end": char_end,
        }]

    metadata = {
        "source_line": line_idx,
        # raw_context 已等于 source_documents[0].text，故不重复存储以节省体积
    }

    return {
        "sample_id": sample_id,
        "source_dataset": "retrieval_ru",
        "language": "ru",
        "question": q,
        "reference_answer": a,
        "source_documents": source_documents,
        "gold_evidence": gold_evidence,
        "metadata": metadata,
    }


def main():
    ap = argparse.ArgumentParser(description="俄文 retrieval_dataset -> 统一 source 格式")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 条（调试用）")
    args = ap.parse_args()

    warnings = []
    n_in = n_out = n_err = n_gold_missing = 0
    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            idx = n_in
            n_in += 1
            if args.limit is not None and n_out >= args.limit:
                break
            try:
                rec = json.loads(line)
                out = convert_record(rec, idx, warnings)
                if out["gold_evidence"] and out["gold_evidence"][0]["char_start"] is None:
                    n_gold_missing += 1
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                n_out += 1
            except Exception as e:
                n_err += 1
                warnings.append(f"第 {n_in} 行解析失败: {e}")

    print(f"[俄文] 读取 {n_in} 行, 输出 {n_out} 条, 失败 {n_err} 条")
    print(f"[俄文] gold 子串定位失败 {n_gold_missing} 条")
    if warnings:
        print(f"[俄文] 警告 {len(warnings)} 条（前 10 条）:")
        for w in warnings[:10]:
            print("   -", w)
    print(f"[俄文] 输出文件: {args.output}")


if __name__ == "__main__":
    main()
