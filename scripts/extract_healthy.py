#!/usr/bin/env python3
"""
Extract healthy traces where ALL gold evidence is present in context.
"""
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any


def is_healthy(trace: Dict[str, Any]) -> bool:
    """
    Проверяет, является ли трасса healthy.
    Healthy = все золотые доказательства присутствуют в контексте.
    """
    gold_evidence = trace.get('gold_evidence', [])
    context = trace.get('context_construction', {})
    
    if not gold_evidence or not context:
        return False
    
    # Собираем все chunk_id из gold evidence
    gold_chunk_ids = set()
    for ev in gold_evidence:
        if isinstance(ev, dict):
            chunk_ids = ev.get('covering_chunk_ids', [])
            gold_chunk_ids.update(chunk_ids)
    
    if not gold_chunk_ids:
        return False
    
    # Получаем chunk_id из контекста
    selected_chunk_ids = set(context.get('selected_chunk_ids', []))
    
    # Проверяем, все ли gold chunk_ids присутствуют в контексте
    present_chunks = gold_chunk_ids.intersection(selected_chunk_ids)
    
    return len(present_chunks) == len(gold_chunk_ids)


def extract_healthy_traces(input_file: str, output_file: str, max_count: int = None) -> int:
    """
    Извлекает healthy трассы из входного файла.
    """
    healthy_traces = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                trace = json.loads(line)
                
                if is_healthy(trace):
                    # Добавляем флаг, что это healthy трасса
                    trace['_is_healthy'] = True
                    healthy_traces.append(trace)
                    
                    if max_count and len(healthy_traces) >= max_count:
                        break
                        
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse line: {e}")
                continue
    
    # Сохраняем результат
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for trace in healthy_traces:
            f.write(json.dumps(trace, ensure_ascii=False) + '\n')
    
    return len(healthy_traces)


def main():
    parser = argparse.ArgumentParser(description="Extract healthy RAG traces")
    parser.add_argument("--input", "-i", required=True, help="Input JSONL file with traces")
    parser.add_argument("--output", "-o", default="data/healthy_traces/healthy.jsonl", 
                        help="Output JSONL file for healthy traces")
    parser.add_argument("--max-count", "-n", type=int, default=None,
                        help="Maximum number of traces to extract (default: all)")
    args = parser.parse_args()
    
    print(f"Extracting healthy traces from {args.input}...")
    count = extract_healthy_traces(args.input, args.output, args.max_count)
    
    print(f"\nExtracted {count} healthy traces")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()