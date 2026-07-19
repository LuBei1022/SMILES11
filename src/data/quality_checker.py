#!/usr/bin/env python3
"""
Quality Checker for RAG traces (English only).
Checks data integrity, completeness, structural validity, and specific failure patterns.
"""
import json
import argparse
import re
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))


class QualityChecker:
    """Проверяет качество данных на уровне отдельных трасс (только английский)."""
    
    def __init__(self):
        self.required_fields = [
            "trace_id", "source_record_id", "pipeline_id", "language",
            "query", "reference_answer", "gold_evidence", 
            "retrieval", "context_construction", "generation"
        ]
        
        self.supported_language = "en"
        
        self.min_lengths = {
            "query": 3,
            "answer": 2,
            "chunk": 20,
            "context": 50
        }
        
        self.max_lengths = {
            "query": 500,
            "answer": 500,
            "chunk": 2000,
            "context": 10000
        }
        
        self.readable_pattern = re.compile(r'^[\w\s.,!?;:\'\"()\-\u0400-\u04FF0-9]+$', re.UNICODE)
        self.api_error_patterns = [
            r'api_quota_exceeded',
            r'rate_limit',
            r'quota',
            r'api key',
            r'authentication',
            r'permission denied',
            r'access denied',
            r'forbidden',
            r'429',
            r'503',
            r'timeout'
        ]
        self.safety_refusal_patterns = [
            r'safety',
            r'refusal',
            r'policy',
            r'violation',
            r'content policy',
            r'cannot answer',
            r'not appropriate',
            r'harmful',
            r'offensive',
            r'unsafe',
            r'sensitive',
            r'inappropriate',
            r'not allowed',
            r'prohibited',
            r'restrict'
        ]
        self.truncation_patterns = [
            r'\.\.\.$',
            r'\[TRUNCATED\]',
            r'…$',
            r'\[...\]$'
        ]
    
    def check_gold_in_context(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        """
        Проверяет наличие золотых доказательств в контексте.
        Использует правильную логику через chunk_id, а не substring match.
        """
        gold_evidence = trace.get('gold_evidence', [])
        context = trace.get('context_construction', {})
        
        result = {
            'gold_present_any': False,
            'gold_present_all': False,
            'n_gold_chunks': 0,
            'n_present_chunks': 0
        }
        
        if not gold_evidence or not context:
            return result
        
        gold_chunk_ids = set()
        for ev in gold_evidence:
            if isinstance(ev, dict):
                chunk_ids = ev.get('covering_chunk_ids', [])
                if chunk_ids:
                    gold_chunk_ids.update(chunk_ids)
        
        selected_chunk_ids = set(context.get('selected_chunk_ids', []))
        present_chunks = gold_chunk_ids.intersection(selected_chunk_ids)
        
        result['n_gold_chunks'] = len(gold_chunk_ids)
        result['n_present_chunks'] = len(present_chunks)
        result['gold_present_any'] = len(present_chunks) > 0
        result['gold_present_all'] = len(present_chunks) == len(gold_chunk_ids) if gold_chunk_ids else False
        
        return result
    
    def check_trace(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        trace_id = trace.get("trace_id", "unknown")
        quality_tags = []
        warnings = []
        issues = defaultdict(list)
        
        # 1. Проверка обязательных полей
        missing_fields = [f for f in self.required_fields if f not in trace or trace[f] is None]
        if missing_fields:
            quality_tags.append("missing_fields")
            issues["missing_fields"] = missing_fields
            warnings.append(f"Missing fields: {', '.join(missing_fields)}")
        
        # 2. Проверка query
        query = trace.get("query", "")
        if not query or not query.strip():
            quality_tags.append("empty_query")
            issues["query"] = ["empty or whitespace-only"]
            warnings.append("Query is empty")
        elif len(query) < self.min_lengths["query"]:
            quality_tags.append("query_too_short")
            issues["query"] = [f"length {len(query)} < {self.min_lengths['query']}"]
        elif len(query) > self.max_lengths["query"]:
            quality_tags.append("query_too_long")
            issues["query"] = [f"length {len(query)} > {self.max_lengths['query']}"]
        elif not self.is_readable(query):
            quality_tags.append("unreadable_query")
            issues["query"] = ["contains non-readable characters"]
        
        # 3. Проверка reference_answer
        ref_answer = trace.get("reference_answer", "")
        if not ref_answer or not ref_answer.strip():
            quality_tags.append("empty_reference_answer")
            issues["reference_answer"] = ["empty or whitespace-only"]
            warnings.append("Reference answer is empty")
        elif len(ref_answer) < self.min_lengths["answer"]:
            quality_tags.append("answer_too_short")
            issues["reference_answer"] = [f"length {len(ref_answer)} < {self.min_lengths['answer']}"]
        elif not self.is_readable(ref_answer):
            quality_tags.append("unreadable_answer")
            issues["reference_answer"] = ["contains non-readable characters"]
        
        # 4. Проверка gold_evidence
        gold_evidence = trace.get("gold_evidence", [])
        if not gold_evidence:
            quality_tags.append("no_gold_evidence")
            issues["gold_evidence"] = ["empty list"]
            warnings.append("No gold evidence provided")
        else:
            if not isinstance(gold_evidence, list):
                quality_tags.append("gold_evidence_not_list")
                issues["gold_evidence"] = [f"expected list, got {type(gold_evidence).__name__}"]
            else:
                for idx, ev in enumerate(gold_evidence):
                    if not isinstance(ev, dict):
                        quality_tags.append("gold_evidence_invalid_item")
                        issues[f"gold_evidence[{idx}]"] = [f"not a dict, got {type(ev).__name__}"]
                        continue
                    
                    if "document_id" not in ev:
                        quality_tags.append("evidence_missing_document_id")
                        issues[f"gold_evidence[{idx}]"] = ["missing document_id"]
                    
                    if "gold_text" not in ev or not ev["gold_text"].strip():
                        quality_tags.append("evidence_empty_gold_text")
                        issues[f"gold_evidence[{idx}]"] = ["empty gold_text"]
                    
                    if "covering_chunk_ids" not in ev or not ev["covering_chunk_ids"]:
                        quality_tags.append("gold_evidence_no_chunk_ids")
                        issues[f"gold_evidence[{idx}]"] = ["missing covering_chunk_ids"]
                    
                    if ev.get("is_preserved") is False:
                        quality_tags.append("gold_evidence_lost_in_chunking")
                        issues[f"gold_evidence[{idx}]"] = ["gold evidence not preserved in chunks"]
                        warnings.append(f"Gold evidence {idx} lost in chunking")
        
        # 5. Проверка retrieval
        retrieval = trace.get("retrieval", {})
        if not retrieval:
            quality_tags.append("no_retrieval_data")
            issues["retrieval"] = ["empty or missing"]
            warnings.append("No retrieval data")
        else:
            retrieved_chunks = retrieval.get("retrieved_chunks", [])
            if not retrieved_chunks:
                quality_tags.append("no_retrieved_chunks")
                issues["retrieval"] = ["retrieved_chunks is empty"]
                warnings.append("No retrieved chunks")
            else:
                if not isinstance(retrieved_chunks, list):
                    quality_tags.append("retrieved_chunks_not_list")
                    issues["retrieval"] = [f"expected list, got {type(retrieved_chunks).__name__}"]
                else:
                    for idx, chunk in enumerate(retrieved_chunks[:10]):
                        if not isinstance(chunk, dict):
                            quality_tags.append("chunk_not_dict")
                            issues[f"retrieved_chunks[{idx}]"] = [f"not a dict, got {type(chunk).__name__}"]
                            continue
                        
                        chunk_text = chunk.get("text", "")
                        if not chunk_text or not chunk_text.strip():
                            quality_tags.append("empty_retrieved_chunk")
                            issues[f"retrieved_chunks[{idx}]"] = ["empty text"]
                        elif len(chunk_text) < self.min_lengths["chunk"]:
                            quality_tags.append("chunk_too_short")
                            issues[f"retrieved_chunks[{idx}]"] = [f"length {len(chunk_text)} < {self.min_lengths['chunk']}"]
                        elif self.is_truncated(chunk_text):
                            quality_tags.append("chunk_truncated")
                            issues[f"retrieved_chunks[{idx}]"] = ["appears truncated"]
                            warnings.append(f"Chunk {idx} appears truncated")
                    
                    chunk_texts = [c.get("text", "") for c in retrieved_chunks if isinstance(c, dict)]
                    if len(chunk_texts) != len(set(chunk_texts)):
                        quality_tags.append("duplicate_retrieved_chunks")
                        issues["retrieval"] = ["duplicate chunk texts found"]
                        warnings.append("Duplicate retrieved chunks found")
        
        # 6. Проверка context_construction
        context_data = trace.get("context_construction", {})
        if context_data:
            if not isinstance(context_data, dict):
                quality_tags.append("context_not_dict")
                issues["context_construction"] = [f"expected dict, got {type(context_data).__name__}"]
            else:
                final_context = context_data.get("final_context", "")
                if not final_context or not final_context.strip():
                    quality_tags.append("empty_final_context")
                    issues["context_construction"] = ["final_context is empty"]
                    warnings.append("Final context is empty")
                elif len(final_context) < self.min_lengths["context"]:
                    quality_tags.append("context_too_short")
                    issues["context_construction"] = [f"length {len(final_context)} < {self.min_lengths['context']}"]
                elif self.is_truncated(final_context):
                    quality_tags.append("context_truncated")
                    issues["context_construction"] = ["context appears truncated"]
                    warnings.append("Context appears truncated")
                
                # ✅ ПРАВИЛЬНАЯ ПРОВЕРКА: через chunk_id
                gold_status = self.check_gold_in_context(trace)
                
                if gold_status['n_gold_chunks'] > 0:
                    if not gold_status['gold_present_any']:
                        quality_tags.append("gold_evidence_not_in_context")
                        issues["context_construction"] = [
                            f"no gold evidence chunks in context (gold chunks: {gold_status['n_gold_chunks']})"
                        ]
                        warnings.append("Gold evidence chunks not found in context")
                    elif not gold_status['gold_present_all']:
                        quality_tags.append("gold_evidence_partial_in_context")
                        issues["context_construction"] = [
                            f"partial gold evidence: {gold_status['n_present_chunks']}/{gold_status['n_gold_chunks']} chunks in context"
                        ]
                        warnings.append(f"Partial gold evidence: {gold_status['n_present_chunks']}/{gold_status['n_gold_chunks']}")
        
        # 7. Проверка generation
        generation = trace.get("generation", {})
        if generation:
            if not isinstance(generation, dict):
                quality_tags.append("generation_not_dict")
                issues["generation"] = [f"expected dict, got {type(generation).__name__}"]
            else:
                final_answer = generation.get("final_answer", "")
                status = generation.get("status", "")
                error_msg = generation.get("error", "")
                final_prompt = generation.get("final_prompt", "")
                
                if status != "success":
                    quality_tags.append(f"generation_status_{status}")
                    issues["generation"] = [f"status: {status}"]
                    if error_msg:
                        issues["generation"].append(f"error: {error_msg}")
                        
                        if self.is_api_quota_error(error_msg):
                            quality_tags.append("api_quota_exceeded")
                            warnings.append("API quota exceeded")
                        elif self.is_safety_refusal(error_msg):
                            quality_tags.append("generation_refused_content_policy")
                            warnings.append("Generation refused due to content policy")
                        elif "token" in error_msg.lower() and "max" in error_msg.lower():
                            quality_tags.append("hit_max_tokens_limit")
                            warnings.append("Hit max tokens limit")
                
                if not final_answer or not final_answer.strip():
                    quality_tags.append("empty_generated_answer")
                    issues["generation"] = ["final_answer is empty"]
                    warnings.append("Generated answer is empty")
                    
                    safety_indicators = (
                        self.is_safety_refusal(final_prompt) or 
                        self.is_safety_refusal(error_msg) or
                        "refused" in status.lower() or
                        "blocked" in status.lower()
                    )
                    
                    if safety_indicators:
                        quality_tags.append("generation_refused_content_policy")
                        issues["generation"].append("likely safety refusal (empty answer + safety indicators)")
                        warnings.append("Empty answer likely due to content policy refusal")
                else:
                    if len(final_answer) < self.min_lengths["answer"]:
                        quality_tags.append("answer_too_short")
                        issues["generation"] = [f"length {len(final_answer)} < {self.min_lengths['answer']}"]
                    elif self.is_truncated(final_answer):
                        quality_tags.append("answer_truncated")
                        issues["generation"] = ["answer appears truncated"]
                        warnings.append("Answer appears truncated")
                    
                    if self.is_safety_refusal(final_answer):
                        quality_tags.append("possible_safety_refusal")
                        issues["generation"] = ["possible safety refusal in answer"]
                        warnings.append("Possible safety refusal in answer")
        
        # 8. Проверка pipeline_id
        pipeline_id = trace.get("pipeline_id", "")
        if not pipeline_id:
            quality_tags.append("missing_pipeline_id")
            issues["pipeline_id"] = ["empty or missing"]
            warnings.append("Missing pipeline_id")
        
        # 9. Проверка языка
        language = trace.get("language", "")
        if language != "en":
            quality_tags.append(f"unsupported_language_{language}")
            issues["language"] = [f"language '{language}' not supported (only 'en' allowed)"]
            warnings.append(f"Unsupported language: {language}")
        
        # 10. Проверка source_record_id
        if not trace.get("source_record_id"):
            quality_tags.append("missing_source_record_id")
            issues["source_record_id"] = ["empty or missing"]
        
        # Определяем usable и severity
        high_severity_tags = [
            "empty_query", "no_gold_evidence", "no_retrieved_chunks", 
            "missing_fields", "empty_generated_answer", "empty_final_context",
            "api_quota_exceeded", "generation_refused_content_policy",
            "gold_evidence_lost_in_chunking"
        ]
        medium_severity_tags = [
            "gold_evidence_not_in_context", "generation_status_error", 
            "possible_safety_refusal", "context_truncated", "answer_truncated",
            "gold_evidence_partial_in_context"
        ]
        
        is_usable = len(quality_tags) == 0 or all(
            t not in high_severity_tags for t in quality_tags
        )
        
        if quality_tags:
            if any(t in high_severity_tags for t in quality_tags):
                severity = "high"
            elif any(t in medium_severity_tags for t in quality_tags):
                severity = "medium"
            else:
                severity = "low"
        else:
            severity = "none"
        
        # Получаем gold_status для статистики (без set-ов)
        gold_status = self.check_gold_in_context(trace)
        
        return {
            "trace_id": trace_id,
            "quality_tags": quality_tags,
            "usable": is_usable,
            "severity": severity,
            "warnings": warnings,
            "issues": dict(issues),
            "has_issues": len(quality_tags) > 0,
            "gold_present_any": gold_status['gold_present_any'],
            "gold_present_all": gold_status['gold_present_all'],
            "n_gold_chunks": gold_status['n_gold_chunks'],
            "n_present_chunks": gold_status['n_present_chunks']
        }
    
    def is_readable(self, text: str) -> bool:
        if not text:
            return False
        readable_chars = sum(1 for c in text if self.readable_pattern.match(c))
        return readable_chars / max(len(text), 1) > 0.7
    
    def is_truncated(self, text: str) -> bool:
        if not text:
            return False
        for pattern in self.truncation_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def is_api_quota_error(self, error_msg: str) -> bool:
        if not error_msg:
            return False
        error_lower = error_msg.lower()
        for pattern in self.api_error_patterns:
            if re.search(pattern, error_lower):
                return True
        return False
    
    def is_safety_refusal(self, text: str) -> bool:
        if not text:
            return False
        text_lower = text.lower()
        for pattern in self.safety_refusal_patterns:
            if re.search(pattern, text_lower):
                return True
        return False
    
    def check_batch(self, traces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        trace_ids_seen = set()
        
        for trace in traces:
            result = self.check_trace(trace)
            
            trace_id = result["trace_id"]
            if trace_id in trace_ids_seen:
                result["quality_tags"].append("duplicate_trace_id")
                result["issues"]["trace_id"] = ["duplicate trace_id found"]
                result["warnings"].append(f"Duplicate trace_id: {trace_id}")
            else:
                trace_ids_seen.add(trace_id)
            
            results.append(result)
        
        return results
    
    def generate_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(results)
        usable_count = sum(1 for r in results if r["usable"])
        
        all_tags = []
        for r in results:
            all_tags.extend(r["quality_tags"])
        tag_distribution = Counter(all_tags)
        
        # Статистика по gold
        gold_any = sum(1 for r in results if r.get("gold_present_any", False))
        gold_all = sum(1 for r in results if r.get("gold_present_all", False))
        gold_none = sum(1 for r in results if not r.get("gold_present_any", False) and r.get("n_gold_chunks", 0) > 0)
        
        by_pipeline = defaultdict(lambda: {"total": 0, "usable": 0, "tags": Counter()})
        for r, trace in zip(results, self._last_traces):
            pipeline_id = trace.get("pipeline_id", "unknown")
            by_pipeline[pipeline_id]["total"] += 1
            if r["usable"]:
                by_pipeline[pipeline_id]["usable"] += 1
            by_pipeline[pipeline_id]["tags"].update(r["quality_tags"])
        
        examples = defaultdict(list)
        for r in results:
            for tag in r["quality_tags"]:
                if len(examples[tag]) < 3:
                    examples[tag].append({
                        "trace_id": r["trace_id"],
                        "issues": r["issues"],
                        "severity": r["severity"]
                    })
        
        return {
            "report_generated": datetime.now().isoformat(),
            "total_traces": total,
            "usable_traces": usable_count,
            "problematic_traces": total - usable_count,
            "usable_ratio": usable_count / total if total > 0 else 0,
            "severity_distribution": dict(Counter(r["severity"] for r in results)),
            "tag_distribution": dict(tag_distribution),
            "gold_statistics": {
                "gold_present_any": gold_any,
                "gold_present_any_ratio": gold_any / total if total > 0 else 0,
                "gold_present_all": gold_all,
                "gold_present_all_ratio": gold_all / total if total > 0 else 0,
                "gold_none": gold_none,
                "gold_none_ratio": gold_none / total if total > 0 else 0
            },
            "by_pipeline": {
                pid: {
                    "total": data["total"],
                    "usable": data["usable"],
                    "usable_ratio": data["usable"] / data["total"] if data["total"] > 0 else 0,
                    "top_issues": dict(data["tags"].most_common(5))
                }
                for pid, data in by_pipeline.items()
            },
            "example_problems": {tag: examples[tag] for tag in list(tag_distribution.keys())[:10]}
        }
    
    _last_traces = []


def main():
    parser = argparse.ArgumentParser(description="Check quality of RAG traces (English only)")
    parser.add_argument("--input", "-i", required=True, help="Input JSONL file with traces")
    parser.add_argument("--output-report", "-o", default="outputs/data_quality_report.json", 
                        help="Output report JSON file")
    parser.add_argument("--output-problematic", "-p", default="outputs/problematic_traces.jsonl",
                        help="Output problematic traces file")
    args = parser.parse_args()
    
    traces = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    traces.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse line: {e}")
    
    print(f"Loaded {len(traces)} traces from {args.input}")
    
    checker = QualityChecker()
    checker._last_traces = traces
    results = checker.check_batch(traces)
    
    report = checker.generate_report(results)
    output_path = Path(args.output_report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {output_path}")
    
    problematic = [r for r in results if r["has_issues"]]
    prob_path = Path(args.output_problematic)
    prob_path.parent.mkdir(parents=True, exist_ok=True)
    with open(prob_path, 'w', encoding='utf-8') as f:
        for r in problematic:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"Problematic traces saved to {prob_path} ({len(problematic)} records)")
    
    print("\n" + "="*50)
    print("QUALITY CHECK SUMMARY")
    print("="*50)
    print(f"Total traces: {report['total_traces']}")
    print(f"Usable traces: {report['usable_traces']} ({report['usable_ratio']*100:.1f}%)")
    print(f"Problematic traces: {report['problematic_traces']}")
    
    print("\nGold Evidence Statistics:")
    gold_stats = report.get('gold_statistics', {})
    print(f"  Gold present (any chunk): {gold_stats.get('gold_present_any', 0)} ({gold_stats.get('gold_present_any_ratio', 0)*100:.1f}%)")
    print(f"  Gold present (all chunks): {gold_stats.get('gold_present_all', 0)} ({gold_stats.get('gold_present_all_ratio', 0)*100:.1f}%)")
    print(f"  Gold missing: {gold_stats.get('gold_none', 0)} ({gold_stats.get('gold_none_ratio', 0)*100:.1f}%)")
    
    print(f"\nSeverity distribution: {report['severity_distribution']}")
    print("\nTop issues:")
    for tag, count in list(report['tag_distribution'].items())[:10]:
        print(f"  {tag}: {count}")


if __name__ == "__main__":
    main()