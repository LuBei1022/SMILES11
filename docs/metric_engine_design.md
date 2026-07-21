# Metric Engine Design

## 1. Scope

This document defines the first version of the component-level Metric Engine for the SMILES RAG evaluation project.

The engine evaluates English and Russian full RAG traces on CPU-first infrastructure. It computes lightweight metrics for retrieval, chunking, context construction, and answer generation. Complex semantic models and LLM Judge integrations are optional later extensions and must not block the rule-based baseline.

The engine must not read `injected_fault` during normal evaluation. That field is reserved for offline validation against controlled-failure ground truth.

## 1.1 First-Version Scope and Test Boundary

The first version is a working CPU-first Metric Engine, not the complete reliability research platform. The following boundary makes the delivery target explicit.

### Core deliverables required in the first version

- The 11 metrics defined in Section 5.
- `MetricResult`, the metric interface, registry, and batch runner.
- Configuration for K values, thresholds, and implementation versions.
- Rule-based English and Russian processing for the full trace input format.
- Basic shared helpers in `common/` that are actually used by the metrics.
- Unit tests for metric formulas, boundary cases, and shared helpers.
- One minimal dependency-chain test: fixture trace -> registry -> runner -> 11 results -> JSONL output -> output schema validation.
- A README usage example and a representative metric output.

### Components that may be minimal in the first version

- `common/` does not require a complex caching subsystem; simple deterministic text and validation helpers are sufficient.
- `reliability/` may contain only a small repeatability or score-comparison utility, or the first validation may remain in `experiments/`. It is not required to deliver automatic threshold calibration in this version.
- Semantic-model adapters may expose an optional interface, but the rule-based baseline must run without downloading or loading a model.

### Explicitly deferred to later iterations

- Automatic threshold calibration and large-scale repeatability reports.
- Multilingual embedding, NLI, and LLM Judge implementations as required dependencies.
- Dedicated contradiction, `unknown`, and `out_of_scope` semantic metrics.
- Full integration with `diagnosis/` and `reporting/`, including pipeline-level prevalence aggregation.
- Production monitoring, web interfaces, and cost/latency optimization beyond basic runtime recording.

### Test boundary for the first version

The first version tests the Metric Engine contract and its metric calculations. It does not claim to validate the final diagnosis system. Tests must cover:

- individual metric formulas and edge cases;
- English and Russian text handling;
- shared helper behavior in `common/`;
- stable `MetricResult` fields and status values;
- input immutability;
- continuation after one malformed trace;
- complete execution of all 11 metrics through the runner;
- output JSONL structure and compatibility with the future diagnosis input contract.

Reliability experiments, controlled-fault discrimination, human agreement, and pipeline-level diagnosis are validation activities around the Metric Engine, not prerequisites for the runner to execute. They must be reported separately and must not be confused with unit or integration-test coverage.

## 2. Repository Architecture

Core metric code belongs under `src/metrics/`, alongside the existing `src/retrieval/`, `src/chunking/`, `src/context/`, and `src/generation/` modules.

```text
src/metrics/
├── __init__.py
├── base.py                 # MetricResult and metric interface
├── registry.py             # metric registration and lookup
├── runner.py               # batch execution and output writing
├── common/                 # text, language, validation, and cache helpers
├── retrieval/              # Hit@K, Recall@K, Precision@K, MRR
├── chunking/               # evidence preservation and chunk integrity
├── context/                # evidence coverage, noise, truncation
├── generation/             # answer relevance and faithfulness
└── reliability/            # repeatability and calibration experiments
```

The future integration boundary is:

```text
Metric Engine -> metric scores and evidence
             -> diagnosis rules and combination logic
             -> trace-level diagnosis
             -> pipeline-level aggregation
```

The Metric Engine produces a separate output file and does not mutate the input traces.

## 3. Input Contract

The primary input is one JSON object per line conforming to `schemas/full_trace.schema.json`.

The engine reads, as available:

- `trace_id`, `pipeline_id`, and `language`
- `query` and `reference_answer`
- `gold_evidence`
- `chunks`
- `retrieval.retrieved_chunks`
- `context_construction.selected_chunk_ids`, `final_context`, and truncation metadata
- `generation.final_answer` and generation status

The engine must never use `injected_fault` as a feature or shortcut for diagnosis.

## 4. MetricResult

Every metric returns one result object with the following fields:

```json
{
  "trace_id": "sample_001__en_bm25_v1",
  "metric_name": "retrieval.recall_at_k",
  "stage": "retrieval",
  "score": 0.5,
  "label": "partial",
  "status": "ok",
  "evidence": {
    "gold_count": 2,
    "matched_count": 1,
    "matched_chunk_ids": ["chunk_03"]
  },
  "config": {
    "k": 5,
    "threshold": null,
    "implementation": "rule_v1"
  },
  "model": null,
  "warnings": [],
  "error": null,
  "runtime_ms": 0
}
```

Field rules:

- `score` is a normalized number in `[0, 1]`, or `null` when the metric cannot be computed.
- `label` is a stable human-readable category.
- `status` is exactly one of `ok`, `skipped`, or `error`.
- `evidence` contains auditable intermediate values and relevant IDs.
- `config` records K values, thresholds, and implementation version.
- `model` is `null` for rule metrics and records provider/model/version for optional model metrics.
- `warnings` contains non-fatal issues; `error` contains a structured fatal input/program error.
- `runtime_ms` records the elapsed time for the individual metric.

## 5. First-Version Metrics

### 5.1 Retrieval

#### `retrieval.hit_at_k`

Let `G` be the set of gold evidence chunk IDs and `R_k` the set of the first K retrieved chunk IDs.

```text
Hit@K = 1 if G ∩ R_k is non-empty, otherwise 0
```

Labels: `hit` or `miss`.

#### `retrieval.recall_at_k`

```text
Recall@K = |G ∩ R_k| / |G|
```

Labels: `complete` for 1, `partial` for a value between 0 and 1, and `miss` for 0.

#### `retrieval.precision_at_k`

```text
Precision@K = |G ∩ R_k| / |R_k|
```

The denominator is the actual number of retrieved results when fewer than K are returned.

Labels: `high`, `partial`, or `none`, using configurable thresholds.

#### `retrieval.mrr`

If the first gold chunk appears at rank `r`, then `MRR = 1 / r`; if no gold chunk appears, `MRR = 0`.

The evidence records the first matching rank and chunk ID.

### 5.2 Chunking

#### `chunking.gold_evidence_preservation`

For each gold evidence item, check `is_preserved`, non-empty `covering_chunk_ids`, and existence of the referenced chunk. The score is:

```text
preserved evidence count / total gold evidence count
```

Labels: `complete`, `partial`, `lost`, or `unknown` when no gold evidence exists.

#### `chunking.chunk_integrity`

Inspect retrieved chunks for empty text, explicit truncation markers, invalid character offsets, duplicate IDs, missing document IDs, and obvious fusion anomalies.

```text
integrity = 1 - defective chunk count / checked chunk count
```

Labels: `intact`, `degraded`, or `broken` using configurable thresholds. Evidence lists each defective chunk and defect type.

### 5.3 Context

#### `context.evidence_coverage`

Let `C` be `context_construction.selected_chunk_ids`.

```text
coverage = |G ∩ C| / |G|
```

This differs from retrieval recall because it evaluates the final selected context, not the retrieved candidate list.

Labels: `complete`, `partial`, or `missing`.

#### `context.noise_ratio`

For traces with gold evidence:

```text
noise_ratio = non-gold selected chunks / all selected chunks
```

If gold evidence is absent, return `score: null`, `label: unknown`, and `status: skipped` rather than treating every chunk as noise.

Labels: `low_noise`, `moderate_noise`, or `high_noise` using configurable thresholds.

#### `context.context_truncation`

Use, in order, the explicit `truncated` flag, token count and budget metadata, and visible truncation markers. Return `1` for confirmed not truncated, `0` for confirmed truncated, and `null` when evidence is insufficient.

Labels: `not_truncated`, `truncated`, or `unknown`.

### 5.4 Generation

#### `generation.answer_relevance`

The CPU-first baseline tokenizes query and answer, removes language-appropriate stop words, and computes:

```text
query-answer overlap = shared valid tokens / valid query tokens
```

Answer length and explicit refusal checks prevent a short generic answer from receiving a misleading score. Labels are `relevant`, `weakly_relevant`, `unrelated`, or `unknown` for empty/error cases.

#### `generation.faithfulness`

Split the answer into sentences. For each sentence, calculate whether its valid tokens receive sufficient lexical support from the final context. The score is:

```text
supported answer sentences / total answer sentences
```

Labels: `supported`, `partially_supported`, `unsupported`, or `unknown`.

This baseline is an interpretable support heuristic, not a formal NLI result. Future NLI and LLM Judge implementations must retain this baseline for comparison and fallback.

## 6. Threshold Configuration

Thresholds are configuration data, not constants embedded in metric code. The initial configuration may be represented as:

```yaml
metrics:
  retrieval:
    k: 5
    precision_high: 0.8
  chunking:
    integrity_intact: 0.8
  context:
    noise_low: 0.2
    noise_moderate: 0.5
  generation:
    relevance_relevant: 0.5
    faithfulness_supported: 0.8
    faithfulness_partial: 0.5
```

The first run uses common thresholds for English and Russian. Reliability experiments may later calibrate language-specific thresholds, and every result records the effective configuration.

## 7. Error Handling

- A malformed trace produces metric-level `error` results but never stops the batch.
- Missing `gold_evidence` makes evidence-dependent metrics `skipped` with `score: null`.
- Missing retrieval data makes retrieval metrics `error`; missing context or answer data makes dependent metrics `skipped`.
- Wrong field types produce structured `invalid_field_type` errors; they are not silently coerced.
- Empty denominators produce `unknown`/`skipped`, never an artificial zero score.
- Missing `trace_id` may use an internal line-number identifier with a warning.
- Optional semantic model failures fall back to the rule baseline and record the fallback warning.
- English and Russian are supported; unsupported or missing language values produce warnings and do not automatically stop rule metrics.

## 8. Testing Plan

```text
tests/metrics/
├── test_retrieval_metrics.py
├── test_chunking_metrics.py
├── test_context_metrics.py
├── test_generation_metrics.py
├── test_runner.py
└── fixtures/
    ├── healthy_trace.json
    ├── missing_fields_trace.json
    ├── retrieval_failure_trace.json
    └── generation_failure_trace.json
```

Tests cover formula correctness, empty and missing fields, English/Russian text, stable `MetricResult` structure, input immutability, and batch continuation after a bad trace. A small end-to-end test runs a JSONL input through `MetricRunner` and checks that all 11 metrics are emitted per trace.

## 9. Reliability Validation

Use existing healthy traces and controlled failures for offline validation. The evaluator computes metrics without reading fault labels; the validation script compares metric outputs with the known controlled-failure labels afterward.

Expected affected metrics:

| Controlled fault | Expected signals |
|---|---|
| `missing_evidence` | Hit@K, Recall@K, evidence coverage |
| `irrelevant_document` | retrieval metrics, answer relevance |
| `chunk_truncation` | evidence preservation, chunk integrity |
| `chunk_merge` | chunk integrity, noise ratio |
| `distractor_context` | noise ratio, faithfulness |
| `unsupported_answer` | faithfulness |
| `contradictory_answer` | faithfulness; future contradiction metric |
| `corrupted_query` | answer relevance; future unknown metric |
| `out_of_scope` | answer relevance; future out-of-scope metric |

Validation reports should include score distributions, precision/recall/F1 where labels are available, English/Russian comparisons, BM25/Dense comparisons, repeatability, and metric-conflict examples.

## 10. Future Integration and Iteration

The first version deliberately stops at 11 CPU-friendly metrics. Later iterations may add multilingual embeddings, NLI faithfulness, contradiction, out-of-scope and unknown detection, confidence calibration, caching, and LLM Judge sampling.

The required downstream contract remains:

```text
MetricResult records
    -> diagnosis rules and combination logic
    -> trace-level fault label, confidence, evidence, alternatives
    -> pipeline_id aggregation and prevalence
```

Diagnosis and pipeline aggregation are outside this module, but the output schema is designed for them from the beginning.

## 11. Delivery Checklist

The completed module should be delivered with:

- `src/metrics/` implementation;
- a CLI runner and configuration;
- unit, interface, and end-to-end tests;
- this design document;
- README usage instructions and a minimal example;
- sample metric output and reliability results;
- explicit limitations and model/version configuration.
