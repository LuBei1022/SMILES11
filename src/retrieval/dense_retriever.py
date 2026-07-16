# -*- coding: utf-8 -*-
"""
dense_retriever.py
稠密（语义）检索。用多语言句向量 + 余弦相似度，per-sample 在样本自身 chunk 上检索。

- 默认模型: paraphrase-multilingual-MiniLM-L12-v2（支持英文/俄文）。
- 模型只加载一次（类级缓存），避免每个样本重复加载。
- 向量做 L2 归一化，余弦相似度 = 点积。

依赖: sentence-transformers, numpy
  pip install sentence-transformers numpy

说明: 方案原写 "Dense + Chroma"。在 per-sample（每题仅几个 chunk）规模下，
      向量库属于过度设计，这里直接用 numpy 余弦；接口与 BM25 一致，
      若将来要换 Chroma 作为存储后端，只需替换 search 内部实现。
"""
from typing import List, Dict, Optional

DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


class DenseRetriever:
    method = "dense"
    _model = None
    _model_name = None

    def __init__(self, chunks: List[Dict], model_name: str = DEFAULT_MODEL):
        import numpy as np
        self.np = np
        self.model_name = model_name
        self.model = self._get_model(model_name)
        self.chunks = chunks
        texts = [c.get("text", "") for c in chunks]
        if texts:
            self.emb = self.model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False)
            self.emb = np.asarray(self.emb, dtype="float32")
        else:
            self.emb = np.zeros((0, self.model.get_sentence_embedding_dimension()),
                                dtype="float32")

    @classmethod
    def _get_model(cls, model_name: str):
        """加载并缓存模型，跨样本复用。"""
        from sentence_transformers import SentenceTransformer
        if cls._model is None or cls._model_name != model_name:
            cls._model = SentenceTransformer(model_name)
            cls._model_name = model_name
        return cls._model

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if len(self.chunks) == 0:
            return []
        q = self.model.encode([query], normalize_embeddings=True,
                              show_progress_bar=False)[0]
        q = self.np.asarray(q, dtype="float32")
        sims = self.emb @ q  # 已归一化 -> 点积即余弦
        order = self.np.argsort(-sims)[:top_k]
        results = []
        for rank, i in enumerate(order.tolist(), start=1):
            c = self.chunks[i]
            results.append({
                "chunk_id": c["chunk_id"],
                "document_id": c["document_id"],
                "rank": rank,
                "score": float(sims[i]),
                "text": c.get("text", ""),
            })
        return results

    def config(self) -> Dict:
        return {"method": "dense", "embedding_model": self.model_name,
                "normalize_embeddings": True, "similarity": "cosine"}
