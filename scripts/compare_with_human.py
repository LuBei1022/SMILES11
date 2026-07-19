#!/usr/bin/env python3
"""
Compare human annotation with expected classes from fault injection.
"""
import json
import csv
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Optional


def safe_strip(value: Optional[str]) -> str:
    if value is None:
        return ''
    return str(value).strip()


def load_annotation_results(csv_file: str) -> Dict[str, Dict[str, Any]]:
    annotations = {}
    
    if not Path(csv_file).exists():
        print(f"File not found: {csv_file}")
        return annotations
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            trace_id = safe_strip(row.get('trace_id', ''))
            if not trace_id:
                continue
            
            annotations[trace_id] = {
                'annotator': safe_strip(row.get('annotator', '')),
                'fault_class': safe_strip(row.get('fault_class', '')),
                'confidence': int(safe_strip(row.get('confidence', '0'))) if safe_strip(row.get('confidence', '')).isdigit() else 0,
                'notes': safe_strip(row.get('notes', '')),
                'fault_type': safe_strip(row.get('fault_type', '')),
                'expected_diagnosis': safe_strip(row.get('expected_diagnosis', '')),
                'query_snippet': safe_strip(row.get('query_snippet', ''))
            }
    
    print(f"Loaded {len(annotations)} annotations from {csv_file}")
    return annotations


def compare(annotations: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    results = {
        'total': 0,
        'exact_matches': 0,
        'partial_matches': 0,
        'mismatches': 0,
        'by_class': defaultdict(lambda: {'total': 0, 'correct': 0}),
        'by_fault_type': defaultdict(lambda: {'total': 0, 'correct': 0}),
        'confusion_matrix': defaultdict(lambda: defaultdict(int)),
        'disagreements': []
    }
    
    for trace_id, human in annotations.items():
        expected = human.get('expected_diagnosis', '').strip()
        human_class = human.get('fault_class', '').strip()
        fault_type = human.get('fault_type', 'unknown')
        
        if not human_class or not expected:
            continue
        
        results['total'] += 1
        results['by_class'][expected]['total'] += 1
        results['by_fault_type'][fault_type]['total'] += 1
        
        if human_class == expected:
            results['exact_matches'] += 1
            results['by_class'][expected]['correct'] += 1
            results['by_fault_type'][fault_type]['correct'] += 1
        elif human_class in ['chunking', 'retrieval'] and expected in ['chunking', 'retrieval']:
            results['partial_matches'] += 1
        else:
            results['mismatches'] += 1
            results['confusion_matrix'][expected][human_class] += 1
            results['disagreements'].append({
                'trace_id': trace_id,
                'expected': expected,
                'human': human_class,
                'fault_type': fault_type
            })
    
    results['accuracy'] = results['exact_matches'] / results['total'] if results['total'] > 0 else 0
    results['partial_accuracy'] = (results['exact_matches'] + results['partial_matches']) / results['total'] if results['total'] > 0 else 0
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare human annotation with expected classes")
    parser.add_argument("--human", "-hm", required=True, help="Human annotation CSV file")
    parser.add_argument("--output", "-o", default="experiments/human_validation/comparison_report.json")
    parser.add_argument("--csv-report", "-c", default="experiments/human_validation/comparison_results.csv")
    args = parser.parse_args()
    
    annotations = load_annotation_results(args.human)
    
    if not annotations:
        print("Failed to load annotations. Exiting.")
        return
    
    results = compare(annotations)
    
    print("\n" + "="*60)
    print("COMPARISON REPORT: Human vs Expected")
    print("="*60)
    print(f"\nTotal: {results['total']}")
    print(f"Exact matches: {results['exact_matches']} ({results['accuracy']*100:.1f}%)")
    print(f"Partial matches: {results['partial_matches']}")
    print(f"Mismatches: {results['mismatches']}")
    print(f"\nAccuracy: {results['accuracy']*100:.1f}%")
    print(f"Partial accuracy: {results['partial_accuracy']*100:.1f}%")
    
    print("\nBy expected class:")
    for cls, stats in sorted(results['by_class'].items(), key=lambda x: x[1]['total'], reverse=True):
        acc = stats['correct']/stats['total']*100 if stats['total'] > 0 else 0
        print(f"  {cls}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")
    
    print("\nBy fault type:")
    for ft, stats in sorted(results['by_fault_type'].items(), key=lambda x: x[1]['total'], reverse=True):
        acc = stats['correct']/stats['total']*100 if stats['total'] > 0 else 0
        print(f"  {ft}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")
    
    if results['disagreements']:
        print("\nDisagreements:")
        for item in results['disagreements'][:10]:
            print(f"  {item['trace_id'][:36]}... Expected: {item['expected']} | Human: {item['human']}")
    
    # Сохраняем отчет
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump({
            'total': results['total'],
            'exact_matches': results['exact_matches'],
            'partial_matches': results['partial_matches'],
            'mismatches': results['mismatches'],
            'accuracy': results['accuracy'],
            'partial_accuracy': results['partial_accuracy'],
            'by_class': {k: {'total': v['total'], 'correct': v['correct']} for k, v in results['by_class'].items()},
            'by_fault_type': {k: {'total': v['total'], 'correct': v['correct']} for k, v in results['by_fault_type'].items()},
            'confusion_matrix': {k: dict(v) for k, v in results['confusion_matrix'].items()},
            'disagreements': results['disagreements'][:20]
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\nReport saved to: {args.output}")
    
    # Сохраняем CSV
    if results['disagreements']:
        with open(args.csv_report, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['trace_id', 'expected', 'human', 'fault_type'])
            for item in results['disagreements']:
                writer.writerow([item['trace_id'], item['expected'], item['human'], item['fault_type']])
        print(f"Detailed CSV saved to: {args.csv_report}")


if __name__ == "__main__":
    main()