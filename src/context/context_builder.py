# -*- coding: utf-8 -*-
"""
context_builder.py
上下文构造：把检索到的 top-k chunk 组装成最终喂给生成器的 final_context。

流程（对应方案 §11）:
  retrieved top-k
    -> 按 rank 排序
    -> 去重（内容归一化后相同的丢弃）
    -> 控制总 token 预算（超预算的靠后 chunk 丢弃）
    -> 加 [编号] 与来源标签，拼成 final_context

输出记录（全程留痕，便于诊断）:
  selected_chunk_ids       最终进入 context 的 chunk
  dropped_chunk_ids        因超 token 预算被丢弃的 chunk
  deduplicated_chunk_ids   因内容重复被丢弃的 chunk
  context_token_count      final_context 的 token 数
  truncated                是否发生了预算截断
  final_context            最终文本
"""
import re
from typing import List, Dict, Callable, Optional


def _normalize(text: str) -> str:
    """内容归一化：小写、压缩空白，用于去重判断。"""
    return re.sub(r"\s+", " ", (text or "").strip().lower())


class ContextBuilder:
    def __init__(self,
                 max_context_tokens: int = 3000,
                 deduplicate: bool = True,
                 preserve_rank_order: bool = True,
                 token_counter: Optional[Callable[[str], int]] = None,
                 tokenizer_name: str = "cl100k_base"):
        self.max_tokens = max_context_tokens
        self.deduplicate = deduplicate
        self.preserve_rank_order = preserve_rank_order
        self.tokenizer_name = tokenizer_name
        if token_counter is None:
            from chunking.recursive_chunker import get_tiktoken_counter
            token_counter = get_tiktoken_counter(tokenizer_name)
        self.count = token_counter

    def build(self, retrieved_chunks: List[Dict]) -> Dict:
        chunks = list(retrieved_chunks)
        if self.preserve_rank_order:
            chunks.sort(key=lambda c: c.get("rank", 1e9))

        selected, dropped, deduped = [], [], []
        seen = set()
        parts = []
        total_tokens = 0
        truncated = False

        for c in chunks:
            cid = c["chunk_id"]
            text = c.get("text", "") or ""

            if self.deduplicate:
                key = _normalize(text)
                if key in seen:
                    deduped.append(cid)
                    continue
                seen.add(key)

            # 该 chunk 加入后的片段（带编号与来源）
            piece = f"[{len(selected) + 1}] (source: {c.get('document_id', '?')})\n{text}"
            piece_tokens = self.count(piece)

            if total_tokens + piece_tokens > self.max_tokens and selected:
                # 预算已满，后续 chunk 丢弃
                dropped.append(cid)
                truncated = True
                continue

            parts.append(piece)
            selected.append(cid)
            total_tokens += piece_tokens

        final_context = "\n\n".join(parts)
        return {
            "selected_chunk_ids": selected,
            "dropped_chunk_ids": dropped,
            "deduplicated_chunk_ids": deduped,
            "context_token_count": self.count(final_context) if final_context else 0,
            "truncated": truncated,
            "final_context": final_context,
        }

    def config(self) -> Dict:
        return {
            "max_context_tokens": self.max_tokens,
            "deduplicate": self.deduplicate,
            "preserve_rank_order": self.preserve_rank_order,
            "tokenizer": self.tokenizer_name,
        }
