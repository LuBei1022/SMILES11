#!/usr/bin/env python3
"""
Create annotation template ONLY from injected traces.
This ensures trace_ids match between annotation and injected files.
"""
import json
import csv
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any


def load_injected_traces(injected_dir: str) -> List[Dict[str, Any]]:
    """Загружает все инжектированные трассы."""
    all_traces = []
    injected_path = Path(injected_dir)
    
    if not injected_path.exists():
        print(f"Directory not found: {injected_dir}")
        return all_traces
    
    for fault_file in injected_path.glob("*.jsonl"):
        fault_type = fault_file.stem
        with open(fault_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    trace = data.get('trace', {})
                    trace_id = trace.get('trace_id', 'unknown')
                    
                    all_traces.append({
                        'trace_id': trace_id,
                        'query': trace.get('query', '')[:200],
                        'fault_type': fault_type,
                        'expected_diagnosis': data.get('expected_diagnosis', 'unknown'),
                        'injection_method': data.get('injection_method', ''),
                        'trace': trace
                    })
                except json.JSONDecodeError:
                    continue
    
    print(f"Loaded {len(all_traces)} injected traces from {injected_dir}")
    return all_traces


def create_annotation_template(injected_dir: str, output_file: str, n_samples: int = 30):
    """Создает шаблон для аннотации только из инжектированных трасс."""
    random.seed(42)
    
    all_traces = load_injected_traces(injected_dir)
    
    if not all_traces:
        print("No injected traces found!")
        return
    
    # Группируем по типу ошибки
    by_fault = {}
    for trace in all_traces:
        fault_type = trace['fault_type']
        if fault_type not in by_fault:
            by_fault[fault_type] = []
        by_fault[fault_type].append(trace)
    
    print(f"\nFound {len(by_fault)} fault types:")
    for fault_type, traces in sorted(by_fault.items()):
        print(f"  {fault_type}: {len(traces)} traces")
    
    # Выбираем равномерно из каждого типа
    selected = []
    fault_types = list(by_fault.keys())
    samples_per_type = max(1, n_samples // len(fault_types))
    
    for fault_type in fault_types:
        traces = by_fault[fault_type]
        n_take = min(samples_per_type, len(traces))
        selected.extend(random.sample(traces, n_take))
    
    # Если нужно больше, добираем случайными
    remaining = n_samples - len(selected)
    if remaining > 0:
        all_remaining = [t for t in all_traces if t not in selected]
        if all_remaining:
            extra = random.sample(all_remaining, min(remaining, len(all_remaining)))
            selected.extend(extra)
    
    # Перемешиваем
    random.shuffle(selected)
    
    # Создаем CSV
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'trace_id',
            'annotator',
            'fault_class',
            'confidence',
            'notes',
            'fault_type',
            'expected_diagnosis',
            'query_snippet'
        ])
        
        for trace in selected[:n_samples]:
            writer.writerow([
                trace['trace_id'],
                '',  # Заполняет аннотатор
                '',  # Заполняет аннотатор
                '',  # Заполняет аннотатор
                '',
                trace['fault_type'],
                trace['expected_diagnosis'],
                trace['query']
            ])
    
    print(f"\nCreated annotation template: {output_file}")
    print(f"Total samples: {len(selected[:n_samples])}")
    print("\nDistribution:")
    fault_counts = {}
    for trace in selected[:n_samples]:
        ft = trace['fault_type']
        fault_counts[ft] = fault_counts.get(ft, 0) + 1
    for ft, count in sorted(fault_counts.items()):
        print(f"  {ft}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Create annotation template from injected traces")
    parser.add_argument("--injected", "-i", default="data/controlled_failures",
                        help="Directory with injected traces")
    parser.add_argument("--output", "-o", default="experiments/human_validation/injection_annotation_template.csv",
                        help="Output CSV file")
    parser.add_argument("--samples", "-n", type=int, default=30,
                        help="Number of samples")
    args = parser.parse_args()
    
    create_annotation_template(args.injected, args.output, args.samples)


if __name__ == "__main__":
    main()