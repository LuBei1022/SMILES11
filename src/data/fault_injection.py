#!/usr/bin/env python3
"""
Fault Injection for RAG traces (English only).
Generates controlled failures for testing diagnosis accuracy.
"""
import json
import argparse
import random
import copy
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict


class FaultInjector:
    """Генерирует контролируемые неисправности в трассах RAG (только английский)."""
    
    def __init__(self, random_seed: int = 42):
        random.seed(random_seed)
        
        # Описание типов неисправностей
        self.fault_types = {
            "missing_evidence": {
                "description": "Remove chunks containing gold evidence from retrieval",
                "expected_diagnosis": "retrieval",
                "expected_metric_change": "Recall@K decreases"
            },
            "chunk_truncation": {
                "description": "Truncate a chunk in the middle of a sentence",
                "expected_diagnosis": "chunking",
                "expected_metric_change": "Chunk coherence decreases"
            },
            "chunk_merge": {
                "description": "Merge two unrelated chunks together",
                "expected_diagnosis": "chunking",
                "expected_metric_change": "Context relevance decreases"
            },
            "distractor_context": {
                "description": "Add irrelevant chunks to context",
                "expected_diagnosis": "chunking/retrieval",
                "expected_metric_change": "Context precision decreases"
            },
            "corrupted_query": {
                "description": "Corrupt/truncate the query",
                "expected_diagnosis": "unknown",
                "expected_metric_change": "All metrics deteriorate"
            },
            "out_of_scope": {
                "description": "Replace query with out-of-scope question",
                "expected_diagnosis": "out_of_scope",
                "expected_metric_change": "Retrieval relevance low, generation hallucinates"
            },
            "irrelevant_document": {
                "description": "Replace retrieved chunk with chunk from another sample",
                "expected_diagnosis": "retrieval",
                "expected_metric_change": "Context relevance drops significantly"
            },
            "unsupported_answer": {
                "description": "Replace answer with unsupported information",
                "expected_diagnosis": "generation",
                "expected_metric_change": "Faithfulness decreases"
            },
            "contradictory_answer": {
                "description": "Generate answer that contradicts context",
                "expected_diagnosis": "generation",
                "expected_metric_change": "Faithfulness decreases, contradiction detected"
            }
        }
    
    def inject_missing_evidence(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        """Удаляет чанки с золотыми доказательствами из результатов поиска."""
        faulty = copy.deepcopy(trace)
        
        # Получаем ID документов с золотыми доказательствами
        gold_doc_ids = set()
        for ev in trace.get("gold_evidence", []):
            if isinstance(ev, dict) and "document_id" in ev:
                gold_doc_ids.add(ev["document_id"])
        
        if not gold_doc_ids:
            return self._create_result(faulty, "missing_evidence", "no_gold_evidence_found")
        
        # Удаляем чанки, содержащие золотые доказательства
        retrieval = faulty.get("retrieval", {})
        retrieved_chunks = retrieval.get("retrieved_chunks", [])
        
        chunks_to_remove = []
        for chunk in retrieved_chunks:
            if isinstance(chunk, dict):
                doc_id = chunk.get("document_id", "")
                if doc_id in gold_doc_ids:
                    chunks_to_remove.append(chunk)
        
        if not chunks_to_remove:
            return self._create_result(faulty, "missing_evidence", "no_chunks_with_gold_evidence")
        
        retrieval["retrieved_chunks"] = [
            c for c in retrieved_chunks if c not in chunks_to_remove
        ]
        
        # Обновляем контекст
        context_data = faulty.get("context_construction", {})
        if context_data:
            chunk_ids_to_remove = set()
            for chunk in chunks_to_remove:
                chunk_id = chunk.get("chunk_id", "")
                if chunk_id:
                    chunk_ids_to_remove.add(chunk_id)
            
            if chunk_ids_to_remove:
                selected = context_data.get("selected_chunk_ids", [])
                context_data["selected_chunk_ids"] = [
                    cid for cid in selected if cid not in chunk_ids_to_remove
                ]
                context_data["final_context"] = self._rebuild_context_from_trace(
                    faulty, context_data["selected_chunk_ids"]
                )
        
        return self._create_result(faulty, "missing_evidence", 
                                   f"removed {len(chunks_to_remove)} chunks with gold evidence")
    
    def inject_chunk_truncation(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        """Усекает случайный чанк посередине предложения."""
        faulty = copy.deepcopy(trace)
        
        retrieval = faulty.get("retrieval", {})
        retrieved_chunks = retrieval.get("retrieved_chunks", [])
        
        if not retrieved_chunks:
            return self._create_result(faulty, "chunk_truncation", "no_chunks_to_truncate")
        
        # Выбираем случайный чанк
        chunk_idx = random.randint(0, len(retrieved_chunks) - 1)
        chunk = retrieved_chunks[chunk_idx]
        
        if not isinstance(chunk, dict) or not chunk.get("text"):
            return self._create_result(faulty, "chunk_truncation", "invalid_chunk")
        
        text = chunk["text"]
        if len(text) < 50:
            return self._create_result(faulty, "chunk_truncation", "chunk_too_short_to_truncate")
        
        # Находим середину предложения (между 30% и 70% длины)
        mid_point = int(len(text) * random.uniform(0.3, 0.7))
        
        # Ищем место для обрезания (конец предложения или после запятой)
        cut_positions = [mid_point]
        for pattern in ['. ', '! ', '? ', ', ', '; ']:
            pos = text.find(pattern, max(0, mid_point - 50), min(len(text), mid_point + 50))
            if pos != -1:
                cut_positions.append(pos + len(pattern.rstrip()))
        
        cut_at = max(cut_positions) if cut_positions else mid_point
        cut_at = min(cut_at, len(text) - 10)  # Не обрезаем совсем в конец
        
        # Обрезаем текст
        chunk["text"] = text[:cut_at] + " [TRUNCATED]"
        
        return self._create_result(faulty, "chunk_truncation", 
                                   f"truncated chunk at position {cut_at}")
    
    def inject_chunk_merge(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        """Объединяет два несвязанных чанка."""
        faulty = copy.deepcopy(trace)
        
        retrieval = faulty.get("retrieval", {})
        retrieved_chunks = retrieval.get("retrieved_chunks", [])
        
        if len(retrieved_chunks) < 2:
            return self._create_result(faulty, "chunk_merge", "not_enough_chunks")
        
        # Выбираем два случайных разных чанка
        idx1, idx2 = random.sample(range(len(retrieved_chunks)), 2)
        chunk1 = retrieved_chunks[idx1]
        chunk2 = retrieved_chunks[idx2]
        
        if not isinstance(chunk1, dict) or not isinstance(chunk2, dict):
            return self._create_result(faulty, "chunk_merge", "invalid_chunks")
        
        text1 = chunk1.get("text", "")
        text2 = chunk2.get("text", "")
        
        if not text1 or not text2:
            return self._create_result(faulty, "chunk_merge", "empty_chunk_text")
        
        # Объединяем тексты
        merged_text = text1 + " [MERGED] " + text2
        chunk1["text"] = merged_text
        # Удаляем второй чанк
        retrieved_chunks.pop(idx2)
        
        return self._create_result(faulty, "chunk_merge", 
                                   f"merged chunks from docs {chunk1.get('document_id')} and {chunk2.get('document_id')}")
    
    def inject_distractor_context(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        """Добавляет нерелевантные чанки в контекст."""
        faulty = copy.deepcopy(trace)
        
        context_data = faulty.get("context_construction", {})
        if not context_data:
            return self._create_result(faulty, "distractor_context", "no_context_to_modify")
        
        # Создаем отвлекающий чанк
        distractor_text = "This is irrelevant information that does not help answer the question. It is just random text inserted as a distractor."
        distractor_chunk_id = f"distractor_{random.randint(1000, 9999)}"
        distractor_doc_id = f"distractor_doc_{random.randint(1000, 9999)}"
        
        # Добавляем в selected_chunk_ids
        selected = context_data.get("selected_chunk_ids", [])
        selected.append(distractor_chunk_id)
        context_data["selected_chunk_ids"] = selected
        
        # Перестраиваем контекст
        context_data["final_context"] = self._rebuild_context_from_trace(
            faulty, selected
        )
        
        # Добавляем в retrieved_chunks для согласованности
        retrieval = faulty.get("retrieval", {})
        retrieval.setdefault("retrieved_chunks", []).append({
            "chunk_id": distractor_chunk_id,
            "document_id": distractor_doc_id,
            "rank": len(retrieval["retrieved_chunks"]) + 1,
            "score": 0.5,
            "text": distractor_text
        })
        
        return self._create_result(faulty, "distractor_context", "added distractor chunk")
    
    def inject_corrupted_query(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        """Портит запрос."""
        faulty = copy.deepcopy(trace)
        
        query = faulty.get("query", "")
        if not query:
            return self._create_result(faulty, "corrupted_query", "empty_query")
        
        # Разные способы порчи
        corruption_type = random.choice(["truncate", "scramble", "insert_garbage"])
        
        if corruption_type == "truncate":
            # Обрезаем до 30-50% длины
            cut_at = int(len(query) * random.uniform(0.2, 0.5))
            faulty["query"] = query[:cut_at] + "..."
        elif corruption_type == "scramble":
            # Перемешиваем слова
            words = query.split()
            if len(words) > 2:
                random.shuffle(words)
                faulty["query"] = " ".join(words)
            else:
                # Если мало слов, просто обрезаем
                cut_at = int(len(query) * 0.3)
                faulty["query"] = query[:cut_at] + "..."
        else:  # insert_garbage
            # Вставляем случайные символы
            garbage = "".join(random.choices("!@#$%^&*()_+[]{}|;:',.<>?/", k=10))
            pos = random.randint(0, len(query))
            faulty["query"] = query[:pos] + garbage + query[pos:]
        
        # Обновляем final_prompt в generation
        generation = faulty.get("generation", {})
        if generation and "final_prompt" in generation:
            generation["final_prompt"] = generation["final_prompt"].replace(
                trace.get("query", ""), faulty["query"]
            )
            faulty["generation"] = generation
        
        return self._create_result(faulty, "corrupted_query", f"{corruption_type} applied")
    
    def inject_out_of_scope(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        """Заменяет вопрос на вопрос вне области знаний."""
        faulty = copy.deepcopy(trace)
        
        oos_questions = [
            "What is the weather like today?",
            "Can you recommend a good restaurant?",
            "How do I bake a cake?",
            "What is the meaning of life?",
            "Tell me a joke",
            "What should I wear tomorrow?",
            "How to fix a broken pipe?",
            "What are today's stock prices?",
            "Can you predict the future?",
            "What is the best movie of all time?",
            "Who will win the next election?",
            "What is the price of Bitcoin?",
            "How to become a millionaire?",
            "What is the secret to happiness?",
            "Should I buy a new car?"
        ]
        
        faulty["query"] = random.choice(oos_questions)
        faulty["original_query"] = trace.get("query", "")
        
        # Обновляем generation
        generation = faulty.get("generation", {})
        if generation and "final_prompt" in generation:
            generation["final_prompt"] = generation["final_prompt"].replace(
                trace.get("query", ""), faulty["query"]
            )
            faulty["generation"] = generation
        
        return self._create_result(faulty, "out_of_scope", "replaced with OOS question")
    
    def inject_irrelevant_document(self, trace: Dict[str, Any], 
                                   other_traces: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Заменяет один чанк на чанк из другого примера."""
        if not other_traces:
            return self._create_result(trace, "irrelevant_document", "no_other_traces")
        
        faulty = copy.deepcopy(trace)
        
        # Находим другой trace с чанком
        other_traces_filtered = [t for t in other_traces if t.get("trace_id") != trace.get("trace_id")]
        if not other_traces_filtered:
            return self._create_result(trace, "irrelevant_document", "no_other_traces_with_chunks")
        
        other = random.choice(other_traces_filtered)
        other_chunks = other.get("retrieval", {}).get("retrieved_chunks", [])
        
        if not other_chunks:
            return self._create_result(trace, "irrelevant_document", "other_trace_no_chunks")
        
        # Заменяем случайный чанк
        retrieval = faulty.get("retrieval", {})
        retrieved_chunks = retrieval.get("retrieved_chunks", [])
        
        if not retrieved_chunks:
            return self._create_result(trace, "irrelevant_document", "no_chunks_to_replace")
        
        idx = random.randint(0, len(retrieved_chunks) - 1)
        other_chunk = random.choice(other_chunks)
        
        # Сохраняем оригинальный чанк для контекста
        original_chunk = retrieved_chunks[idx].copy()
        
        retrieved_chunks[idx] = {
            "chunk_id": other_chunk.get("chunk_id", "irrelevant_chunk"),
            "document_id": other_chunk.get("document_id", "irrelevant_doc"),
            "rank": retrieved_chunks[idx].get("rank", idx + 1),
            "score": 0.1,
            "text": other_chunk.get("text", "Irrelevant content that does not answer the question.")
        }
        retrieval["retrieved_chunks"] = retrieved_chunks
        
        # Обновляем контекст
        context_data = faulty.get("context_construction", {})
        if context_data:
            selected = context_data.get("selected_chunk_ids", [])
            if idx < len(selected):
                selected[idx] = retrieved_chunks[idx]["chunk_id"]
                context_data["selected_chunk_ids"] = selected
                context_data["final_context"] = self._rebuild_context_from_trace(
                    faulty, selected
                )
        
        return self._create_result(faulty, "irrelevant_document", 
                                   f"replaced chunk at position {idx} with chunk from {other.get('trace_id', 'unknown')}")
    
    def inject_unsupported_answer(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        """Заменяет ответ на неподтвержденный."""
        faulty = copy.deepcopy(trace)
        
        unsupported_answers = [
            "The answer cannot be determined from the context.",
            "I don't have enough information to answer this question.",
            "The context doesn't mention this topic.",
            "This is not covered in the provided documents.",
            "I'm sorry, but I don't know the answer.",
            "The information is not available in the given context.",
            "I cannot answer this question based on the provided context.",
            "The context does not contain the necessary information."
        ]
        
        generation = faulty.get("generation", {})
        if generation:
            generation["final_answer"] = random.choice(unsupported_answers)
            generation["original_answer"] = trace.get("generation", {}).get("final_answer", "")
            faulty["generation"] = generation
        
        return self._create_result(faulty, "unsupported_answer", 
                                   "replaced with unsupported answer")
    
    def inject_contradictory_answer(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        """Создает ответ, противоречащий контексту."""
        faulty = copy.deepcopy(trace)
        
        contradictory_answers = [
            "The answer is the complete opposite of what the context states.",
            "No, this is not correct according to the information provided.",
            "The context actually says the opposite of this statement.",
            "This contradicts the information in the documents.",
            "The evidence suggests a different conclusion.",
            "The context clearly refutes this claim.",
            "This is contradicted by the provided information."
        ]
        
        generation = faulty.get("generation", {})
        if generation:
            generation["final_answer"] = random.choice(contradictory_answers)
            generation["original_answer"] = trace.get("generation", {}).get("final_answer", "")
            faulty["generation"] = generation
        
        return self._create_result(faulty, "contradictory_answer", 
                                   "replaced with contradictory answer")
    
    def _rebuild_context_from_trace(self, trace: Dict[str, Any], chunk_ids: List[str]) -> str:
        """Перестраивает контекст по chunk_ids из трассы."""
        if not chunk_ids:
            return ""
        
        # Получаем тексты чанков из retrieval
        retrieval = trace.get("retrieval", {})
        retrieved_chunks = retrieval.get("retrieved_chunks", [])
        
        # Создаем словарь chunk_id -> chunk
        chunk_map = {}
        for chunk in retrieved_chunks:
            if isinstance(chunk, dict) and "chunk_id" in chunk:
                chunk_map[chunk["chunk_id"]] = chunk
        
        # Собираем контекст
        context_parts = []
        for idx, chunk_id in enumerate(chunk_ids):
            chunk = chunk_map.get(chunk_id, {})
            doc_id = chunk.get("document_id", "unknown")
            text = chunk.get("text", f"Chunk {chunk_id}")
            
            if text:
                context_parts.append(f"[{idx+1}] (source: {doc_id})\n{text}")
        
        return "\n\n".join(context_parts)
    
    def _create_result(self, faulty_trace: Dict[str, Any], 
                       fault_type: str, 
                       details: str) -> Dict[str, Any]:
        """Создает запись о внедренной неисправности."""
        return {
            "trace": faulty_trace,
            "original_trace_id": faulty_trace.get("trace_id", "unknown"),
            "injected_fault": fault_type,
            "injection_method": details,
            "expected_metric_change": self.fault_types.get(fault_type, {}).get("expected_metric_change", "unknown"),
            "expected_diagnosis": self.fault_types.get(fault_type, {}).get("expected_diagnosis", "unknown"),
            "fault_description": self.fault_types.get(fault_type, {}).get("description", "no_description")
        }
    
    def inject_batch(self, traces: List[Dict[str, Any]], 
                     fault_type: str, 
                     count: int = 10,
                     other_traces: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """Внедряет неисправности в пакет трасс."""
        results = []
        injection_method = getattr(self, f"inject_{fault_type}", None)
        
        if injection_method is None:
            print(f"Warning: Unknown fault type '{fault_type}'")
            return results
        
        # Фильтруем только здоровые трассы (без injected_fault)
        healthy_traces = [t for t in traces if t.get("injected_fault") is None]
        
        if not healthy_traces:
            print(f"Warning: No healthy traces found for fault type '{fault_type}'")
            return results
        
        # Берем случайные трассы для инъекции
        available = min(count, len(healthy_traces))
        selected_traces = random.sample(healthy_traces, available)
        
        for trace in selected_traces:
            if fault_type == "irrelevant_document":
                result = injection_method(trace, other_traces)
            else:
                result = injection_method(trace)
            results.append(result)
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Inject controlled failures into RAG traces (English only)")
    parser.add_argument("--input", "-i", required=True, help="Input JSONL file with healthy traces")
    parser.add_argument("--output-dir", "-o", default="data/controlled_failures", 
                        help="Output directory for fault files")
    parser.add_argument("--count", "-c", type=int, default=20, 
                        help="Number of injections per fault type")
    parser.add_argument("--fault-types", "-f", nargs="+", 
                        default=["missing_evidence", "chunk_truncation", "chunk_merge", 
                                "distractor_context", "corrupted_query", "out_of_scope",
                                "irrelevant_document", "unsupported_answer", "contradictory_answer"],
                        help="Fault types to inject")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--language", default="en",
                        help="Only inject into traces of this language; use 'any' to accept all")
    args = parser.parse_args()
    
    # Чтение данных
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
    
    # Language filter (default English; --language any accepts all)
    if args.language != "any":
        kept = [t for t in traces if t.get("language") == args.language]
        if len(kept) < len(traces):
            print(f"Warning: filtered to {len(kept)} traces with language "
                  f"'{args.language}' (dropped {len(traces) - len(kept)}).")
        traces = kept

    if not traces:
        print(f"Error: no traces with language '{args.language}' found!")
        return
    
    # Инициализация инжектора
    injector = FaultInjector(random_seed=args.seed)
    
    # Создание выходной директории
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Статистика
    stats = defaultdict(int)
    other_traces = traces
    
    # Инъекция
    for fault_type in args.fault_types:
        print(f"Injecting {fault_type}...")
        
        results = injector.inject_batch(
            traces, fault_type, 
            count=args.count,
            other_traces=other_traces
        )
        
        if results:
            output_file = output_dir / f"{fault_type}.jsonl"
            with open(output_file, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            stats[fault_type] = len(results)
            print(f"  Created {len(results)} {fault_type} failures in {output_file}")
        else:
            print(f"  No {fault_type} failures created")
    
    # Сводная статистика
    print("\n" + "="*50)
    print("FAULT INJECTION SUMMARY")
    print("="*50)
    print(f"Total traces processed: {len(traces)}")
    print(f"Fault types: {len([f for f, c in stats.items() if c > 0])}")
    print(f"Total injections: {sum(stats.values())}")
    print("\nPer fault type:")
    for fault_type, count in stats.items():
        print(f"  {fault_type}: {count}")


if __name__ == "__main__":
    main()