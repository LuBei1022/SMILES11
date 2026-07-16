#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
normalize_english_data.py
把英文 HotpotQA 数据 (gpt_3.5_turbo.jsonl) 转换为统一 source 样本格式。

映射规则:
  id                -> sample_id
  question          -> question
  answer            -> reference_answer   (原为嵌套 list，展平)
  context           -> source_documents   (title[] + sentences[] 逐篇拼成一个 doc)
  supporting_facts  -> gold_evidence       (title+sent_id 定位，附 gold_text)
  type/level        -> metadata
  response/short_answer/answer_exact_match/answer_f1
                    -> metadata.legacy_output  (旧 GPT-3.5 结果，仅作对照，不进入主流程)

用法:
  python scripts/normalize_english_data.py \
    --input data/raw/gpt_3.5_turbo.jsonl \
    --output data/normalized/english_source.jsonl \
    [--limit N]
"""
import argparse
import json
import sys


def flatten_answer(answer):
    """HotpotQA 的 answer 常为嵌套 list，如 [["yes"]]。展平为字符串。"""
    leaves = []

    def walk(x):
        if isinstance(x, list):
            for item in x:
                walk(item)
        else:
            leaves.append(str(x))

    walk(answer)
    uniq = list(dict.fromkeys(s for s in leaves if s != ""))
    if not uniq:
        return "", leaves
    return (" | ".join(uniq), leaves)


def convert_record(rec, warnings):
    sample_id = rec.get("id")
    if not sample_id:
        raise ValueError("缺少 id")

    question = rec.get("question", "")
    reference_answer, answer_raw = flatten_answer(rec.get("answer", []))

    # ---- source_documents ----
    ctx = rec.get("context", {}) or {}
    titles = ctx.get("title", []) or []
    sentences_lists = ctx.get("sentences", []) or []
    if len(titles) != len(sentences_lists):
        warnings.append(f"{sample_id}: context title/sentences 数量不一致 "
                        f"({len(titles)} vs {len(sentences_lists)})")

    source_documents = []
    title_to_docid = {}
    for i, title in enumerate(titles):
        sents = sentences_lists[i] if i < len(sentences_lists) else []
        doc_id = f"{sample_id}__doc_{i:03d}"
        source_documents.append({
            "document_id": doc_id,
            "title": title,
            "text": "".join(sents),
            "sentences": sents,
        })
        # 同名标题理论上唯一；若重复保留第一个
        title_to_docid.setdefault(title, (doc_id, sents))

    # ---- gold_evidence ----
    sf = rec.get("supporting_facts", {}) or {}
    sf_titles = sf.get("title", []) or []
    sf_sent_ids = sf.get("sent_id", []) or []
    gold_evidence = []
    for title, sent_id in zip(sf_titles, sf_sent_ids):
        if title not in title_to_docid:
            warnings.append(f"{sample_id}: supporting_fact 标题不在 context 中: {title!r}")
            continue
        doc_id, sents = title_to_docid[title]
        gold_text = ""
        if isinstance(sent_id, int) and 0 <= sent_id < len(sents):
            gold_text = sents[sent_id]
        else:
            warnings.append(f"{sample_id}: sent_id 越界 title={title!r} sent_id={sent_id}")
        gold_evidence.append({
            "document_id": doc_id,
            "sentence_index": sent_id if isinstance(sent_id, int) else None,
            "gold_text": gold_text,
            "char_start": None,
            "char_end": None,
        })

    # ---- metadata (含旧结果对照) ----
    metadata = {
        "type": rec.get("type"),
        "level": rec.get("level"),
        "answer_raw": answer_raw,
        "legacy_output": {
            "model": "gpt-3.5-turbo",
            "response": rec.get("response"),
            "short_answer": rec.get("short_answer"),
            "answer_exact_match": rec.get("answer_exact_match"),
            "answer_f1": rec.get("answer_f1"),
        },
    }

    return {
        "sample_id": sample_id,
        "source_dataset": "hotpot_en",
        "language": "en",
        "question": question,
        "reference_answer": reference_answer,
        "source_documents": source_documents,
        "gold_evidence": gold_evidence if gold_evidence else None,
        "metadata": metadata,
    }


def main():
    ap = argparse.ArgumentParser(description="英文 HotpotQA -> 统一 source 格式")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 条（调试用）")
    args = ap.parse_args()

    warnings = []
    n_in = n_out = n_err = 0
    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            if args.limit is not None and n_out >= args.limit:
                break
            try:
                rec = json.loads(line)
                out = convert_record(rec, warnings)
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                n_out += 1
            except Exception as e:
                n_err += 1
                warnings.append(f"第 {n_in} 行解析失败: {e}")

    print(f"[英文] 读取 {n_in} 行, 输出 {n_out} 条, 失败 {n_err} 条")
    if warnings:
        print(f"[英文] 警告 {len(warnings)} 条（前 10 条）:")
        for w in warnings[:10]:
            print("   -", w)
    print(f"[英文] 输出文件: {args.output}")


if __name__ == "__main__":
    main()
