# -*- coding: utf-8 -*-
"""
recursive_chunker.py
正常（healthy）切块器：段落/句子感知的递归切块，使用 tiktoken 计 token。

设计要点:
  - 目标 chunk 大小按 token 计（默认 400），相邻 chunk 有 overlap（默认 80 token）。
  - 尽量不切断句子：先把文本切成句子级"原子"，超长句子再退化为按词切；
    然后贪心地把原子打包进 chunk，保证每个 chunk 不超过目标 token 数。
  - 每个 chunk 精确记录 start_char / end_char，chunk 文本 == 原文切片，
    因此可以和 gold_evidence 的字符偏移对齐，用于"金标准是否被切坏"的判断。

token 计数:
  - 默认用 tiktoken 的 cl100k_base 编码（英文/俄文通用）。
  - 若环境没有 tiktoken，可传入其它计数函数（降级用，不建议正式实验使用）。
"""
import re
from typing import Callable, List, Dict, Optional


# 句子结束标点（含中英俄常见），后跟空白或结尾即视为一句边界；换行也断句
_SENT_BOUNDARY_RE = re.compile(r'[.!?。！？…]+["\')\]]*\s+|\n+', re.UNICODE)
_WORD_RE = re.compile(r'\S+\s*', re.UNICODE)


def get_tiktoken_counter(encoding_name: str = "cl100k_base") -> Callable[[str], int]:
    """返回一个基于 tiktoken 的 token 计数函数。"""
    import tiktoken
    enc = tiktoken.get_encoding(encoding_name)
    return lambda s: len(enc.encode(s))


def _sentence_spans(text: str) -> List[tuple]:
    """把 text 切成句子级 (start, end) 区间，完整覆盖全文（含标点与空白）。"""
    spans = []
    start = 0
    for m in _SENT_BOUNDARY_RE.finditer(text):
        end = m.end()
        if end > start:
            spans.append((start, end))
            start = end
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def _word_spans(text: str, s: int, e: int) -> List[tuple]:
    """把 [s, e) 区间按词切成更小的 (start, end) 原子（用于超长句子的退化）。"""
    spans = []
    for m in _WORD_RE.finditer(text[s:e]):
        spans.append((s + m.start(), s + m.end()))
    return spans or [(s, e)]


class RecursiveChunker:
    def __init__(self,
                 chunk_size_tokens: int = 400,
                 chunk_overlap_tokens: int = 80,
                 token_counter: Optional[Callable[[str], int]] = None,
                 tokenizer_name: str = "cl100k_base"):
        assert chunk_overlap_tokens < chunk_size_tokens, "overlap 必须小于 chunk_size"
        self.chunk_size = chunk_size_tokens
        self.overlap = chunk_overlap_tokens
        self.tokenizer_name = tokenizer_name
        self.count = token_counter or get_tiktoken_counter(tokenizer_name)

    def _atoms(self, text: str) -> List[tuple]:
        """生成 (start, end, ntok) 原子列表：句子优先，超长句退化为词。"""
        atoms = []
        for (s, e) in _sentence_spans(text):
            ntok = self.count(text[s:e])
            if ntok <= self.chunk_size:
                atoms.append((s, e, ntok))
            else:
                for (ws, we) in _word_spans(text, s, e):
                    atoms.append((ws, we, self.count(text[ws:we])))
        return atoms

    def chunk_document(self, document_id: str, text: str) -> List[Dict]:
        """对单篇文档切块，返回 chunk 字典列表。"""
        if not text:
            return []
        atoms = self._atoms(text)
        chunks = []
        i = 0
        idx = 0
        n = len(atoms)
        while i < n:
            cur_tokens = 0
            j = i
            # 至少放入一个原子，之后在不超过 chunk_size 的前提下继续打包
            while j < n and (j == i or cur_tokens + atoms[j][2] <= self.chunk_size):
                cur_tokens += atoms[j][2]
                j += 1
            start_char = atoms[i][0]
            end_char = atoms[j - 1][1]
            chunk_text = text[start_char:end_char]
            chunks.append({
                "chunk_id": f"{document_id}__chunk_{idx:03d}",
                "document_id": document_id,
                "text": chunk_text,
                "chunk_index": idx,
                "start_char": start_char,
                "end_char": end_char,
                "token_count": self.count(chunk_text),
            })
            idx += 1
            if j >= n:
                break
            # 计算 overlap：从 j-1 往回退，累计到约 overlap 个 token
            ov = 0
            k = j - 1
            while k > i and ov < self.overlap:
                ov += atoms[k][2]
                k -= 1
            next_i = max(k + 1, i + 1)  # 保证向前推进，避免死循环
            i = next_i
        return chunks

    def config(self) -> Dict:
        return {
            "method": "recursive_sentence_aware",
            "chunk_size_tokens": self.chunk_size,
            "chunk_overlap_tokens": self.overlap,
            "tokenizer": self.tokenizer_name,
            "preserve_sentence_boundary": True,
        }
