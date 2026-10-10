# RAG Trace Platform

**SMILES 2026 · Automatic Evaluation & Fault-Diagnosis Framework for RAG**

A modular platform for tracing, evaluating, and diagnosing failures in
Retrieval-Augmented Generation (RAG) systems.

The project was developed for **SMILES 2026** and focuses on a practical
question:

> **When a RAG answer is wrong, which stage actually failed?**

Instead of reducing an entire RAG run to a single quality score, this project
records the full execution trace, evaluates individual pipeline components, and
uses controlled failures to identify where the system broke.

---

## Overview

A RAG system usually performs several dependent stages:

```text
Raw Data
   ↓
Normalization
   ↓
Chunking
   ↓
Retrieval
   ↓
Context Construction
   ↓
Generation
   ↓
Full Trace
   ↓
Metric Engine
   ↓
Fault Diagnosis
```

A final answer may be wrong because of:

- corrupted or out-of-scope input,
- bad chunking,
- missing or irrelevant retrieval results,
- noisy context,
- unsupported generation,
- contradictory generation,
- or other upstream failures.

A single end-to-end score cannot tell us which component caused the problem.

This project therefore turns every RAG execution into a **complete trace** and
performs **component-level fault diagnosis** rather than only reporting a final
quality score.

---

## Key Features

- End-to-end RAG trace generation
- Unified English and Russian data representation
- BM25 and dense retrieval behind a common interface
- Logged chunking, retrieval, context, and generation stages
- 9 controlled fault types
- 11 component-level diagnostic metrics
- Deterministic metric evaluation
- Rule-based fault localization
- Trace-level diagnosis
- Pipeline-level aggregation
- Honest abstention when evidence is insufficient
- Controlled-failure sensitivity analysis
- Cross-lingual portability experiments on Russian data
- Human validation of controlled fault labels

---

## System Architecture

```text
                         ┌──────────────────────┐
                         │       Raw Data       │
                         │      EN + RU QA      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    Normalization     │
                         │ Unified Data Schema  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │       Chunking       │
                         │ sentence/token aware │
                         └──────────┬───────────┘
                                    │
                                    ▼
                    ┌──────────────────────────────┐
                    │          Retrieval           │
                    │        BM25 / Dense          │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │    Context Construction      │
                    │ ranked + deduped + budgeted  │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                         ┌──────────────────────┐
                         │      Generation      │
                         │     GLM / Gemini     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      Full Trace      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    Metric Engine     │
                         │     11 metrics       │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │  Rule-based Diagnosis│
                         │ earliest-fault-first │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Diagnosis / Abstain  │
                         └──────────────────────┘
```

The central design principle is **traceability**.

Each stage records its inputs, outputs, and configuration so that later
diagnostic logic can inspect where the quality degradation originated.

---

## Repository Structure

```text
SMILES11/
├── README.md
├── LICENSE
├── configs/
├── schemas/
│
├── src/
│   ├── chunking/
│   ├── retrieval/
│   ├── context/
│   ├── generation/
│   └── data/
│
├── scripts/
│   ├── normalize_english_data.py
│   ├── normalize_russian_data.py
│   ├── run_chunking.py
│   ├── run_retrieval.py
│   ├── run_context.py
│   ├── run_generation.py
│   ├── run_metrics.py
│   ├── validate_metrics.py
│   ├── extract_healthy.py
│   └── analyze_controlled_failures.py
│
├── data/
│   ├── raw/
│   ├── normalized/
│   ├── chunked/
│   ├── retrieved/
│   ├── context/
│   └── traces/
│
├── experiments/
│   └── human_validation/
│
├── outputs/
├── docs/
└── tests/
```

Some large/raw datasets may be kept locally rather than committed to the
repository.

---

## Trace Platform

The first part of the system converts heterogeneous QA datasets into a unified,
fully logged RAG trace.

### Unified source schema

The English and Russian datasets have different original structures.

The platform normalizes them into a shared representation so that downstream
pipeline stages remain dataset-independent.

Gold evidence is preserved through normalized identifiers and evidence text.

---

### Chunking

The chunking stage converts source documents into retrievable text segments.

The trace records:

- chunk identifiers,
- chunk text,
- offsets,
- gold-evidence relationships,
- chunking configuration.

This makes it possible to diagnose structural failures such as truncation.

---

### Retrieval

Two retrieval strategies are supported:

- **BM25** — sparse lexical retrieval
- **Dense Retrieval** — embedding-based semantic retrieval

Both expose a common retrieval interface.

The platform records the ranked results and retrieval configuration for every
question.

---

### Context Construction

Retrieved chunks are:

- ranked,
- deduplicated,
- constrained by the context token budget,
- assembled into the final context passed to the generator.

The complete context-construction trace is preserved.

---

### Generation

The generation layer supports swappable backends.

The current implementation includes:

- Zhipu GLM
- Google Gemini
- Dry-run mode for offline pipeline validation

Generation configuration such as model name, prompt version, temperature, and
token limits is recorded in the trace.

---

## Controlled Fault Injection

To evaluate whether diagnostic metrics actually respond to the correct type of
failure, the project injects known faults into healthy RAG traces.

The controlled fault families include:

```text
chunk_merge
chunk_truncation
contradictory_answer
corrupted_query
distractor_context
irrelevant_document
missing_evidence
out_of_scope
unsupported_answer
```

Each injected example carries a known ground-truth fault label.

This allows the system to compare:

```text
healthy trace
      vs
controlled faulty trace
```

and directly measure which metrics react to each failure.

---

## Metric Engine

The Metric Engine computes **11 component-level metrics** across the main RAG
stages.

| Stage | Metric | Purpose |
|---|---|---|
| Retrieval | Hit@K | Whether at least one gold chunk appears in the top-K results |
| Retrieval | Recall@K | Fraction of gold chunks successfully retrieved |
| Retrieval | Precision@K | Fraction of retrieved chunks that correspond to gold evidence |
| Retrieval | MRR | Reciprocal rank of the first retrieved gold chunk |
| Chunking | Gold Evidence Preservation | Whether gold evidence survives chunking |
| Chunking | Chunk Integrity | Detects structural chunk defects such as truncation |
| Context | Evidence Coverage | Fraction of gold evidence included in final context |
| Context | Noise Ratio | Fraction of selected context that is not gold evidence |
| Context | Context Truncation | Detects explicit final-context truncation |
| Generation | Answer Relevance | Measures query-answer lexical relevance |
| Generation | Faithfulness | Measures lexical support from the final context |

The first-generation relevance and faithfulness metrics are deliberately
transparent and lightweight.

They are useful diagnostic baselines, but they should **not** be interpreted as
equivalent to multilingual NLI or LLM-as-a-Judge evaluation.

---

## Fault Diagnosis

The final diagnosis module consumes the metric bundle produced for each trace.

It uses a deterministic **earliest-fault-first rule cascade**:

```text
input / scope
      ↓
chunking
      ↓
retrieval
      ↓
context
      ↓
generation
```

The system returns the most likely failing component when the evidence is strong
enough.

When evidence is weak or conflicting, the diagnosis engine **abstains instead
of forcing a potentially incorrect explanation**.

This makes every decision traceable and avoids presenting uncertain root-cause
predictions as facts.

---

## Final Results

### Trace Quality

On the English trace set, each retriever was evaluated on 500 examples.

| Metric | BM25 | Dense Retrieval |
|---|---:|---:|
| Usable traces | 496 / 500 (99.2%) | 497 / 500 (99.4%) |
| At least one gold evidence item present | 487 / 500 (97.4%) | 482 / 500 (96.4%) |
| Full gold evidence present | 293 / 500 (58.6%) | 291 / 500 (58.2%) |

The healthy traces with full gold evidence were used as the main pool for
controlled fault injection.

---

### Manual Validation of Fault Labels

A manually checked subset was used to validate whether the injected labels
matched the intended failure.

| Retriever | Matches | Accuracy |
|---|---:|---:|
| BM25 | 17 / 18 | **94.4%** |
| Dense Retrieval | 14 / 17 | **82.4%** |

---

### Metric Sensitivity

The controlled-failure experiment showed clear component-specific responses.

Representative results include:

- `missing_evidence` strongly degrades retrieval metrics such as Hit@K,
  Recall@K, and MRR.
- `contradictory_answer` strongly decreases faithfulness.
- `unsupported_answer` decreases both faithfulness and answer relevance.
- `out_of_scope` primarily decreases answer relevance.
- `chunk_truncation` reduces chunk integrity.
- BM25 and Dense Retrieval show broadly consistent diagnostic behavior.

Known blind spots are reported explicitly instead of being hidden.

For example, semantic `chunk_merge` failures are not yet reliably detected by
the current chunk-integrity baseline.

---

### Diagnosis Quality

Final controlled validation produced:

| Metric | Result |
|---|---:|
| Diagnostic coverage | **78.33%** |
| Supported accuracy | **64.58%** |
| Abstention rate | **21.7%** |

The abstention behavior is intentional.

When diagnostic evidence is weak or conflicting, the system returns
"uncertain" rather than inventing a root cause.

---

## Cross-Lingual Portability

The same core pipeline was also executed on Russian data without changing the
pipeline architecture.

### Russian BM25 experiment

| Metric | Result |
|---|---:|
| Full traces generated | 300 |
| Generation success | **300 / 300 (100%)** |
| Retrieval Hit@5 | **98.7%** |
| Gold in context | **98.7%** |
| Healthy traces used for fault injection | 145 |

The same major controlled-failure signatures observed in English were also
reproduced in Russian.

Examples include:

- `missing_evidence` collapsing retrieval metrics,
- `contradictory_answer` and `unsupported_answer` degrading faithfulness,
- `chunk_truncation` affecting chunk integrity.

This suggests that **structure-sensitive diagnostic signals transfer across
languages**, while lexical generation metrics remain more language-dependent.

---

## Installation

Install the main pipeline dependencies:

```bash
pip install tiktoken rank_bm25 sentence-transformers numpy jsonschema zhipuai
```

Optional Gemini backend:

```bash
pip install google-genai
```

Metric Engine dependencies:

```bash
pip install pytest PyYAML matplotlib
```

---

## Running the Pipeline

### 1. Normalize English data

```bash
python scripts/normalize_english_data.py \
  --input data/raw/gpt_3.5_turbo.jsonl \
  --output data/normalized/english_source.jsonl
```

### 2. Chunk

```bash
python scripts/run_chunking.py \
  --input data/normalized/english_source.jsonl \
  --output data/chunked/english_chunks.jsonl
```

### 3. Retrieve

BM25 example:

```bash
python scripts/run_retrieval.py \
  --input data/chunked/english_chunks.jsonl \
  --output data/retrieved/english_bm25.jsonl \
  --method bm25 \
  --top-k 5
```

Dense retrieval:

```bash
python scripts/run_retrieval.py \
  --input data/chunked/english_chunks.jsonl \
  --output data/retrieved/english_dense.jsonl \
  --method dense \
  --top-k 5
```

### 4. Build context

```bash
python scripts/run_context.py \
  --input data/retrieved/english_bm25.jsonl \
  --output data/context/english_bm25_context.jsonl
```

### 5. Generate full traces

Set the API key first:

```bash
export ZHIPU_API_KEY=YOUR_API_KEY
```

Then run:

```bash
python scripts/run_generation.py \
  --input data/context/english_bm25_context.jsonl \
  --output data/traces/english_bm25_glm.jsonl \
  --pipeline-id en_bm25_glm45flash_baseline_v1 \
  --backend zhipu \
  --model glm-4.5-flash \
  --max-output-tokens 512 \
  --sleep 1 \
  --resume
```

Useful generation options:

```text
--limit N
--backend zhipu
--backend gemini
--backend dry-run
--resume
--sleep S
```

`dry-run` mode can be used to validate pipeline plumbing without making an API
request.

---

## Running the Metric Engine

```bash
python scripts/run_metrics.py \
  --input data/traces/english_bm25_glm.jsonl \
  --output outputs/metrics_english_bm25.jsonl \
  --config configs/metrics_default.yaml
```

Each result contains:

- metric name,
- score,
- label,
- execution status,
- evidence,
- effective configuration,
- warnings,
- runtime metadata.

---

## Controlled-Failure Analysis

Run paired controlled-failure analysis:

```bash
python scripts/analyze_controlled_failures.py \
  --data-root data \
  --config configs/metrics_default.yaml \
  --json-output outputs/controlled_failure_report.json \
  --csv-output outputs/controlled_failure_summary.csv \
  --core-csv-output outputs/controlled_failure_core_results.csv \
  --heatmap-output outputs/controlled_failure_heatmap.png \
  --heatmap-pdf-output outputs/controlled_failure_heatmap.pdf
```

The analysis compares each faulty trace with its original healthy trace and
computes paired metric changes.

Reported statistics include:

- healthy and faulty means,
- mean and median paired deltas,
- standard deviation,
- expected-direction agreement,
- skipped/error counts,
- paired-bootstrap confidence intervals,
- standardized paired effect,
- rank-biserial effect.

---

## Russian Experiment

The same downstream pipeline can be used after Russian normalization.

Example:

```bash
python scripts/run_chunking.py \
  --input datasets/normalized/russian_source.jsonl \
  --output datasets/chunked/russian_chunks.jsonl \
  --limit 300

python scripts/run_retrieval.py \
  --input datasets/chunked/russian_chunks.jsonl \
  --output datasets/retrieved/russian_bm25.jsonl \
  --method bm25 \
  --top-k 5

python scripts/run_context.py \
  --input datasets/retrieved/russian_bm25.jsonl \
  --output datasets/context/russian_bm25_context.jsonl

python scripts/run_generation.py \
  --input datasets/context/russian_bm25_context.jsonl \
  --output datasets/traces/russian_bm25_glm.jsonl \
  --pipeline-id ru_bm25_glm45flash_baseline_v1 \
  --backend zhipu \
  --model glm-4.5-flash \
  --max-output-tokens 512 \
  --sleep 1 \
  --resume
```

For Russian controlled-failure evaluation:

```bash
python scripts/extract_healthy.py \
  -i datasets/traces/russian_bm25_glm.jsonl \
  -o datasets/ru_eval/healthy_traces/healthy_bm25.jsonl

python src/data/fault_injection.py \
  -i datasets/ru_eval/healthy_traces/healthy_bm25.jsonl \
  -o datasets/ru_eval/controlled_failures_bm25 \
  -c 20 \
  --seed 42 \
  --language ru

python scripts/analyze_controlled_failures.py \
  --data-root datasets/ru_eval \
  --core-csv-output outputs/controlled_failure_core_results_ru.csv \
  --heatmap-output outputs/controlled_failure_heatmap_ru.png
```

---

## Validation & Reproducibility

The pipeline was designed so that diagnostic experiments remain auditable and
reproducible.

Key safeguards include:

- fixed prompt versions,
- explicit pipeline configuration,
- deterministic metric implementations,
- fault labels separated from metric computation,
- reproducible controlled-failure generation,
- documented skipped/error cases,
- repeated-run comparison.

The final project validation reported:

- **39 automated tests passed**
- **11,000 metric results**
- **0 metric execution errors**
- repeated runs produced matching results

---

## Known Limitations

The current system intentionally reports several limitations.

### Chunk-merge detection

The current `chunk_integrity` metric detects explicit truncation effectively but
does not reliably detect semantic chunk fusion.

### Generation metrics

The current answer relevance and faithfulness metrics are lexical,
interpretable baselines.

They are not replacements for:

- Natural Language Inference (NLI),
- semantic entailment models,
- multilingual LLM-as-a-Judge evaluation.

### Context-truncation faults

Context truncation is measured, but a dedicated controlled
context-truncation fault family remains future work.

### Russian evaluation

The Russian experiment successfully reproduces the major metric-sensitivity
patterns, but:

- the healthy pool is smaller than the English pool,
- lexical generation metrics are primarily calibrated on English,
- full Russian trace-level diagnosis and human validation remain future work.

---

## Future Work

Planned extensions include:

- rerunning the full pipeline after each injected fault,
- improving detection of semantic chunk merging,
- adding dedicated context-truncation faults,
- replacing lexical faithfulness with multilingual NLI or LLM-Judge metrics,
- completing full Russian diagnosis and human validation,
- expanding the controlled-failure benchmark.

---

## Team & Contributions

This project was developed collaboratively for **SMILES 2026**.

| Contributor | Main Contribution |
|---|---|
| [Lu Ke](https://github.com/LuBei1022) | **Trace Platform** — unified trace schema, full-trace logging for English and Russian, and execution of the full Russian pipeline |
| Khazratova Guzel | **Quality & Fault Injection** — data quality checks, 9 controlled fault types, and manual validation |
| Shao Shiyun | **Metric Engine** — 11 component-level metrics, reproducible test suite, and controlled-failure sensitivity analysis |
| [Hao Xiaohua](https://github.com/hxhua380) | **Final Diagnosis** — rule-cascade diagnosis, trace-level results, pipeline-level aggregation, and paper preparation |

**Mentor:** Samsonova Darya — methodological guidance and research direction.

**Paper:** Khazratova Guzel · Hao Xiaohua  
**Presentation:** Lu Ke · Shao Shiyun

---

## Project Summary

The project demonstrates that RAG evaluation can move beyond a single
end-to-end score.

By combining:

```text
complete execution traces
        +
controlled failures
        +
component-level metrics
        +
interpretable diagnosis rules
```

the system can identify where a RAG pipeline is likely to have failed and
abstain when the available evidence is insufficient.

The controlled-failure signatures are broadly consistent across BM25 and Dense
Retrieval and reproduce on Russian data, suggesting that many structural
failure signals are not specific to a single retriever or language.

---

## License

This project is released under the MIT License.
