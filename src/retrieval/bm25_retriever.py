# -*- coding: utf-8 -*-
"""
bm25_retriever.py
稀疏检索（BM25）。per-sample：在单个样本自己的 chunk 集合上检索。

依赖: rank_bm25  (pip install rank_bm25)
"""
import re
from typing import List, Dict

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> List[str]:
    """简单的多语言分词：抽取 Unicode 词字符，转小写。英俄通用。"""
    return _TOKEN_RE.findall((text or "").lower())


class BM25Retriever:
    method = "bm25"

    def __init__(self, chunks: List[Dict]):
        """chunks: 该样本的 chunk 列表，每个含 chunk_id/document_id/text。"""
        from rank_bm25 import BM25Okapi
        self.chunks = chunks
        self.corpus = [tokenize(c.get("text", "")) for c in chunks]
        # 语料非空且至少有一个非空文档时才建索引
        self.bm25 = BM25Okapi(self.corpus) if any(self.corpus) else None

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if not self.bm25 or not self.chunks:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for rank, i in enumerate(order, start=1):
            c = self.chunks[i]
            results.append({
                "chunk_id": c["chunk_id"],
                "document_id": c["document_id"],
                "rank": rank,
                "score": float(scores[i]),
                "text": c.get("text", ""),
            })
        return results

    def config(self) -> Dict:
        return {"method": "bm25", "tokenizer": "unicode_word_lower"}
