# Quality Check & Fault Injection Framework

**Part of the SMILES 2026 project "Automatic Evaluation &
Fault-Diagnosis Framework for RAG".**

This repository contains data quality validation and fault injection
modules for RAG traces. It takes complete RAG traces, performs quality
checks, extracts healthy traces, and generates controlled failures to
validate diagnosis systems.

------------------------------------------------------------------------

## Repository Layout

``` text
SMILES11/
├── README.md

├── src/
│   └── data/
│       ├── quality_checker.py
│       └── fault_injection.py
│
├── scripts/
│   ├── extract_healthy.py
│   ├── create_injection_template.py
│   └── compare_with_human.py
│
├── data/
│   ├── traces/
│   │   ├── english_bm25_glm.jsonl
│   │   └── english_dense_glm.jsonl
│   ├── healthy_traces/
│   │   ├── healthy_bm25.jsonl
│   │   └── healthy_dense.jsonl
│   └── controlled_failures/
│       ├── bm25/
│       │   └── 9 fault files
│       └── dense/
│           └── 9 fault files
├── docs/
│   └──  annotation_guideline.md
│
├── outputs/
│   ├── data_quality_report_bm25.json
│   ├── data_quality_report_dense.json
│   ├── problematic_traces_bm25.jsonl
│   └── problematic_traces_dense.jsonl
│
└── experiments/
    └── human_validation/
        ├── annotation_results_bm25.csv
        ├── annotation_results_dense.csv
        ├── comparison_report_bm25.json
        └── comparison_report_dense.json
```

------------------------------------------------------------------------

## How to Run

### Install Dependencies

```bash
pip install numpy
```

### 1. Quality Check

#### BM25

```bash
python src/data/quality_checker.py \
  -i data/traces/english_bm25_glm.jsonl \
  -o outputs/data_quality_report_bm25.json \
  -p outputs/problematic_traces_bm25.jsonl
```

#### Dense

```bash
python src/data/quality_checker.py \
  -i data/traces/english_dense_glm.jsonl \
  -o outputs/data_quality_report_dense.json \
  -p outputs/problematic_traces_dense.jsonl
```

---

### 2. Extract Healthy Traces

#### BM25

```bash
python scripts/extract_healthy.py \
  -i data/traces/english_bm25_glm.jsonl \
  -o data/healthy_traces/healthy_bm25.jsonl
```

#### Dense

```bash
python scripts/extract_healthy.py \
  -i data/traces/english_dense_glm.jsonl \
  -o data/healthy_traces/healthy_dense.jsonl
```

---

### 3. Fault Injection

#### BM25

```bash
python src/data/fault_injection.py \
  -i data/healthy_traces/healthy_bm25.jsonl \
  -o data/controlled_failures_bm25 \
  -c 20 \
  --seed 42
```

#### Dense

```bash
python src/data/fault_injection.py \
  -i data/healthy_traces/healthy_dense.jsonl \
  -o data/controlled_failures_dense \
  -c 20 \
  --seed 42
```

---

### 4. Create Annotation Templates

#### BM25

```bash
python scripts/create_injection_template.py \
  -i data/controlled_failures_bm25 \
  -o experiments/human_validation/annotation_template_bm25.csv \
  -n 30
```

#### Dense

```bash
python scripts/create_injection_template.py \
  -i data/controlled_failures_dense \
  -o experiments/human_validation/annotation_template_dense.csv \
  -n 30
```

---

### 5. Human Validation

Open the generated annotation template (`annotation_template_*.csv`) and manually assign the fault class for each sample.

Supported labels:

- retrieval
- chunking
- generation
- out_of_scope
- unknown

Save the completed file as:

- `annotation_results_bm25.csv`
- `annotation_results_dense.csv`

---

### 6. Compare Human vs Expected

#### BM25

```bash
python scripts/compare_with_human.py \
  --human experiments/human_validation/annotation_results_bm25.csv \
  --output experiments/human_validation/comparison_report_bm25.json \
  --csv-report experiments/human_validation/comparison_results_bm25.csv
```

#### Dense

```bash
python scripts/compare_with_human.py \
  --human experiments/human_validation/annotation_results_dense.csv \
  --output experiments/human_validation/comparison_report_dense.json \
  --csv-report experiments/human_validation/comparison_results_dense.csv
```
------------------------------------------------------------------------

## Fault Types

| Fault Type | Expected Diagnosis |
|------------|--------------------|
| missing_evidence | retrieval |
| chunk_truncation | chunking |
| chunk_merge | chunking |
| distractor_context | chunking / retrieval |
| corrupted_query | unknown |
| out_of_scope | out_of_scope |
| irrelevant_document | retrieval |
| unsupported_answer | generation |
| contradictory_answer | generation |

------------------------------------------------------------------------


## Results Summary

### Quality Check (500 traces)

| Metric | BM25 | Dense |
|--------|------|------|
| Usable traces | 496 (99.2%) | 497 (99.4%) |
| Gold present (any) | 487 (97.4%) | 482 (96.4%) |
| Gold present (all) | 293 (58.6%) | 291 (58.2%) |
| Gold missing | 13 (2.6%) | 18 (3.6%) |

### Fault Injection

| Method | Healthy traces | Fault types | Total injections |
|--------|----------------:|------------:|-----------------:|
| BM25 | 293 | 9 | 180 |
| Dense | 291 | 9 | 180 |

## Human Validation

| Method | Samples | Matches | Accuracy |
|--------|---------:|---------:|---------:|
| BM25 | 18 | 17 | **94.4%** |
| Dense | 17 | 14 | **82.4%** |

---

## Comparison: BM25 vs Dense

| Metric | BM25 | Dense |
|--------|------:|------:|
| Gold present (any) | 97.4% | 96.4% |
| Healthy traces | 293 | 291 |
| Fault injections | 180 | 180 |
| Human validation accuracy | **94.4%** | **82.4%** |

## Conclusion

-   Quality Checker validates data with 99%+ usability.
-   Nine fault types were successfully injected for both retrieval
    methods.
-   Human validation confirms the reliability of the framework.
-   BM25 slightly outperforms Dense retrieval in gold evidence
    preservation.
-   The framework is ready for downstream diagnosis modules.
