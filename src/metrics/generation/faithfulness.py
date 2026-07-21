"""Sentence-level lexical context-support baseline."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..base import MetricResult
from ..common.text import split_sentences, tokenize


def faithfulness(trace: dict[str, Any], config: dict[str, Any]) -> MetricResult:
    trace_id = trace.get("trace_id") if isinstance(trace.get("trace_id"), str) else "unknown"
    context_data = trace.get("context_construction")
    generation = trace.get("generation")
    context = context_data.get("final_context") if isinstance(context_data, Mapping) else None
    answer = generation.get("final_answer") if isinstance(generation, Mapping) else None
    if not isinstance(context, str) or not context.strip() or not isinstance(answer, str) or not answer.strip():
        return MetricResult(
            trace_id=trace_id, metric_name="generation.faithfulness", stage="generation",
            score=None, label="unknown", status="skipped", warnings=["empty_context_or_answer"],
        )
    language = trace.get("language") if isinstance(trace.get("language"), str) else None
    context_tokens = set(tokenize(context, language))
    sentences = split_sentences(answer)
    support_threshold = float(config.get("sentence_support_threshold", 0.5))
    sentence_results: list[dict[str, Any]] = []
    supported_count = 0
    for sentence in sentences:
        tokens = set(tokenize(sentence, language))
        overlap = len(tokens & context_tokens) / len(tokens) if tokens else 0.0
        supported = bool(tokens) and overlap >= support_threshold
        supported_count += int(supported)
        sentence_results.append({"sentence": sentence, "support_score": overlap,
                                 "supported": supported})
    if not sentences:
        return MetricResult(
            trace_id=trace_id, metric_name="generation.faithfulness", stage="generation",
            score=None, label="unknown", status="skipped", warnings=["no_answer_sentences"],
        )
    score = supported_count / len(sentences)
    supported_threshold = float(config.get("faithfulness_supported", 0.8))
    partial_threshold = float(config.get("faithfulness_partial", 0.5))
    label = "supported" if score >= supported_threshold else (
        "partially_supported" if score >= partial_threshold else "unsupported"
    )
    return MetricResult(
        trace_id=trace_id, metric_name="generation.faithfulness", stage="generation",
        score=score, label=label,
        evidence={"sentence_count": len(sentences), "supported_sentence_count": supported_count,
                  "sentences": sentence_results},
        config={"sentence_support_threshold": support_threshold,
                "supported_threshold": supported_threshold,
                "partial_threshold": partial_threshold,
                "implementation": "rule_v1", "method": "lexical_support"},
    )
