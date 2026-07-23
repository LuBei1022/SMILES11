# RAG Trace Platform

Part of the SMILES 2026 project *"Automatic Evaluation & Fault-Diagnosis Framework for RAG"*.

This repository is the **RAG experiment platform**: it takes heterogeneous raw QA
datasets (English and Russian), runs them through a fully configurable, fully
logged RAG pipeline, and emits **complete traces** — one record per question
capturing every stage from source documents to the final generated answer.

These traces are the input for the *separate* evaluation & diagnosis framework
(metrics, fault localization, pipeline-level aggregation), which is intentionally
kept in a different codebase so that fault-injection labels never leak into the
evaluator.

## Pipeline status

```
Source data  →  Chunking  →  Retrieval  →  Context  →  Generation  →  Full Trace
   [done]        [done]      [done]        [done]       [done]         [done]
```

Everything from raw data to a complete, schema-valid trace runs end to end.
Fault injection and the downstream metric/diagnosis modules are future work.

## Pipeline stages

| # | Stage | Input | Output | Key metric produced |
|---|-------|-------|--------|---------------------|
| 1 | Normalize | raw `.jsonl` (EN / RU) | unified *source samples* | — |
| 2 | Chunking | source samples | samples + `chunks` | gold-evidence preservation |
| 3 | Retrieval | chunked samples | samples + `retrieval` | Recall@k / Hit@k |
| 4 | Context | retrieved samples | samples + `context_construction` | gold-in-context |
| 5 | Generation | context samples | **full traces** | success rate, latency, tokens |

Each stage only adds its own fields and passes everything else through, so a
generated trace accumulates the full history of the run.

## Repository layout

```
SMILES11/
├── README.md
├── schemas/
│   └── source_sample.schema.json      # unified source-sample JSON Schema (stage 1 output)
│
├── src/
│   ├── chunking/
│   │   └── recursive_chunker.py       # sentence-aware, tiktoken-based chunker
│   ├── retrieval/
│   │   ├── bm25_retriever.py          # sparse / keyword retrieval (rank_bm25)
│   │   └── dense_retriever.py         # dense / semantic retrieval (multilingual embeddings + cosine)
│   ├── context/
│   │   └── context_builder.py         # ranked, deduped, token-budgeted context assembly
│   └── generation/
│       ├── prompt_builder.py          # versioned prompts (grounded_v1)
│       └── llm_client.py              # swappable LLM backends: ZhipuClient (GLM) / GeminiClient / DryRunClient
│
├── scripts/                           # one CLI per stage; each takes --input / --output
│   ├── normalize_english_data.py      # HotpotQA (EN)  -> source samples
│   ├── normalize_russian_data.py      # retrieval set (RU) -> source samples
│   ├── run_chunking.py                # + gold-evidence preservation check
│   ├── run_retrieval.py               # --method bm25|dense ; + Recall@k / Hit@k
│   ├── run_context.py                 # + gold-in-context check
│   └── run_generation.py              # calls the LLM, assembles & writes full traces
│
└── data/
    ├── raw/                           # original inputs (kept local, not committed)
    │   ├── gpt_3.5_turbo.jsonl        # English HotpotQA-style, GPT-3.5 legacy answers
    │   └── retrieval_dataset.jsonl    # Russian retrieval QA
    ├── normalized/                    # stage 1 output
    ├── chunked/                       # stage 2 output
    ├── retrieved/                     # stage 3 output
    ├── context/                       # stage 4 output
    └── traces/                        # stage 5 output — the deliverable
```

## Design notes

**Unified source schema.** The English and Russian datasets have very different
shapes (EN: structured multi-doc context + `supporting_facts`; RU: a single
concatenated context blob + a gold passage). Both are normalized into one
`source_sample` format via per-dataset adapters, so every downstream stage is
dataset-agnostic. Gold evidence is stored uniformly: `document_id` + `gold_text`
always, plus `sentence_index` (EN) or character offsets (RU).

**Per-sample retrieval.** Each question is retrieved against *its own* candidate
pool (mirroring how both datasets are built and guaranteeing the gold passage is
in-pool), rather than a global index.

**Config transparency & reproducibility.** Every stage records its configuration
into the trace (chunk size, tokenizer, retriever, top-k, embedding model, prompt
version, model name, temperature). Unknown upstream config is marked explicitly
rather than fabricated. `temperature=0` and fixed prompt versions keep runs
reproducible.

**Two-system boundary.** This platform produces traces; it does **not** score
them. Fault-injection labels (future) will live only in the trace as ground
truth and must never be read by the evaluator.

## How to run

Install dependencies:

```bash
pip install tiktoken rank_bm25 sentence-transformers numpy jsonschema zhipuai
# (google-genai only needed if using the optional --backend gemini)
```

Run the full pipeline (English, BM25 baseline shown):

```bash
# 1. normalize
python scripts/normalize_english_data.py \
  --input data/raw/gpt_3.5_turbo.jsonl \
  --output data/normalized/english_source.jsonl

# 2. chunk
python scripts/run_chunking.py \
  --input data/normalized/english_source.jsonl \
  --output data/chunked/english_chunks.jsonl

# 3. retrieve  (--method bm25 | dense)
python scripts/run_retrieval.py \
  --input data/chunked/english_chunks.jsonl \
  --output data/retrieved/english_bm25.jsonl \
  --method bm25 --top-k 5

# 4. build context
python scripts/run_context.py \
  --input data/retrieved/english_bm25.jsonl \
  --output data/context/english_bm25_context.jsonl

# 5. generate -> full traces  (Zhipu GLM, free tier)
export ZHIPU_API_KEY=...           # required for --backend zhipu
python scripts/run_generation.py \
  --input data/context/english_bm25_context.jsonl \
  --output data/traces/english_bm25_glm.jsonl \
  --pipeline-id en_bm25_glm45flash_baseline_v1 \
  --backend zhipu --model glm-4.5-flash \
  --max-output-tokens 512 --sleep 1 --resume
```

The generation backend is swappable: `--backend zhipu` (GLM, current default choice),
`--backend gemini` (Google Gemini), or `--backend dry-run` (offline placeholder).

Russian: swap the raw input and use `normalize_russian_data.py`; all later stages
are identical.

### Useful flags

- `--limit N` — process only the first N records (quick testing on any stage).
- `run_generation.py --backend dry-run` — assemble full traces offline without
  calling the API (validates plumbing, costs no quota).
- `run_generation.py --resume` — skip already-succeeded samples; safe to re-run
  after hitting rate limits.
- `--sleep S` on generation — pause between calls to respect free-tier rate limits.

## Current baselines (English, full 500 samples)

| Retriever | Hit@5 | Recall@5 |
|-----------|-------|----------|
| BM25 | 97.4% | 78.0% |
| Dense (multilingual) | 96.6% | 77.4% |

BM25 and dense retrieval are essentially tied on this multi-hop data, with BM25
marginally ahead.

## Metric Engine

The first-version Metric Engine evaluates full traces without changing them. It
computes 11 CPU-friendly metrics across retrieval, chunking, context
construction, and answer generation. English and Russian text are supported.

Install the Metric Engine test and configuration dependencies:

```bash
pip install pytest PyYAML matplotlib
```

Run all metrics:

```bash
python scripts/run_metrics.py \
  --input data/traces/english_bm25_glm.jsonl \
  --output outputs/metrics_english_bm25.jsonl \
  --config configs/metrics_default.yaml
```

The output contains one `MetricResult` JSON object per metric and trace. Every
record includes a normalized score or `null`, a label, `ok`/`skipped`/`error`
status, auditable evidence, effective configuration, warnings, and runtime.

The Metric Engine never reads `injected_fault`. Controlled-failure or human
labels may be read only by the separate offline validation command:

```bash
python scripts/validate_metrics.py \
  --metrics outputs/diagnosis.jsonl \
  --labels experiments/human_validation/annotation_results_bm25.csv \
  --output outputs/metric_label_validation.json
```

Compare repeated Metric Engine runs:

```bash
python scripts/validate_metrics.py \
  --runs outputs/metrics_run_1.jsonl outputs/metrics_run_2.jsonl \
  --output outputs/metric_repeatability.json
```

The rule-based answer relevance and faithfulness metrics are transparent
baselines. Faithfulness measures lexical context support and must not be
reported as an NLI or LLM-Judge result. Diagnosis, trace-level fault labels,
and pipeline-level aggregation remain downstream work.

### Metric Engine validation (2026-07-20)

The complete metric test suite passes with 32 tests. Full local runs produced:

| Input | Traces | Metric results | Skipped | Errors |
|---|---:|---:|---:|---:|
| English BM25 traces | 500 | 5,500 | 8 | 0 |
| English Dense traces | 500 | 5,500 | 6 | 0 |
| Healthy BM25 traces | 293 | 3,223 | 6 | 0 |
| Healthy Dense traces | 291 | 3,201 | 2 | 0 |

All skipped results were generation metrics where an upstream trace had an
empty answer or context. They remain `skipped/unknown` and are not converted to
zero scores.

Offline runs also processed 360 controlled-failure traces after extracting the
inner full trace from each validation wrapper. The truncation injection reduced
mean `chunking.chunk_integrity` from 1.0 to 0.8 for both BM25 and Dense samples;
unsupported-answer faithfulness fell to 0.0, and missing-evidence retrieval
recall fell to 0.0. Empty retrieval in two missing-evidence samples produced the
documented structured precision error rather than stopping the batch.

Current limitations:

- Russian is covered by a 300-trace batch with the full controlled-failure
  metric-sensitivity experiment (see *Cross-lingual portability* below), which
  reproduces the English pattern. It is a smaller-scale probe: the healthy pool is
  145 vs 293, and the lexical faithfulness/relevance metrics are English-calibrated,
  so absolute Russian metric values are indicative rather than fully calibrated.
  Trace-level **diagnosis** (rule engine, pipeline aggregation) is not yet run on
  Russian.
- The first rule-based `chunk_integrity` baseline detects explicit truncation
  but does not yet reliably identify semantic chunk fusion (`chunk_merge`).
- Lexical faithfulness detects many unsupported or contradictory answers but is
  not a replacement for multilingual NLI or an LLM Judge.

### Paired controlled-failure experiment

Run the 9 controlled fault types for BM25 and Dense as paired comparisons
against their original healthy traces:

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

On Yandex DataSphere, use `--data-root datasets` when the uploaded data folder
is named `datasets/`.

The experiment removes `injected_fault` before every Metric Engine call. Fault
labels are used only after scoring to group the paired deltas. For every
retriever, fault type, and metric, the CSV reports healthy/fault means, mean and
median paired delta, population standard deviation, quality improvement or
degradation counts, expected direction agreement, and skipped/error counts.
It also reports a deterministic paired-bootstrap 95% confidence interval,
standardized paired effect, and rank-biserial effect. The core CSV retains only
pre-declared fault/metric relationships, while the heatmap shows
quality-normalized deltas where negative values consistently mean degradation.

The first complete local run paired all 360 controlled records with their
healthy originals and had zero unmatched records. Of 38 pre-declared
fault/metric direction checks, 32 matched the expected direction. Strong signals
included Recall@K falling to 0 for `missing_evidence`, faithfulness falling to 0
for `unsupported_answer`, and chunk integrity falling from 1.0 to 0.8 for
`chunk_truncation`. The unchanged `chunk_merge` integrity and distractor-context
faithfulness results are reported as limitations rather than hidden.

### Cross-lingual portability (Russian, 300 traces)

The **entire pipeline — trace generation and the controlled-failure evaluation —
was run unchanged on Russian** (300 samples, BM25), validating portability at both
the data layer and the evaluation layer.

**Trace generation (data layer):**

| Metric | Russian (BM25) |
|--------|---------------:|
| Full traces generated | 300 |
| Generation success | 300 / 300 (100%) |
| Retrieval Hit@5 | 98.7% |
| Retrieval Recall@5 | 77.2% |
| Gold in context | 98.7% |

The platform produces schema-valid, fully-populated Russian traces with correct
Russian answers, confirming that the per-dataset adapters, multilingual
chunking/retrieval, and generation transfer across languages without pipeline
changes.

**Controlled-failure sensitivity (evaluation layer):** from 145 healthy Russian
traces, the 9 fault types were injected and the same metric-sensitivity analysis
was run. The Russian sensitivity pattern closely matches English:

- `missing_evidence` collapses the retrieval metrics (`hit_at_k`, `recall_at_k`
  → −1.00; `mrr` → −0.93) without touching generation metrics.
- `contradictory_answer` and `unsupported_answer` collapse `faithfulness`
  (−0.94) without touching retrieval metrics.
- `chunk_truncation` moves only `chunk_integrity` (−0.20).

The same known metric gaps (`chunk_merge` vs `chunk_integrity`, unexercised
`context_truncation`/`gold_evidence_preservation`, weak `corrupted_query`)
reproduce identically in both languages, confirming they are method-level, not
language-specific. This upgrades the portability claim from "data layer only" to
**"the metric sensitivity experiment reproduces the English diagnostic behavior
on Russian."** Caveats: the Russian healthy pool is smaller (145 vs 293), and the
lexical `faithfulness`/`answer_relevance` metrics are calibrated on English, so
their absolute Russian values should be read as a probe rather than a fully
calibrated measurement.

Reproduce — generation (first stage caps the run at 300; later stages inherit the count):

```bash
python scripts/run_chunking.py   --input datasets/normalized/russian_source.jsonl --output datasets/chunked/russian_chunks.jsonl --limit 300
python scripts/run_retrieval.py  --input datasets/chunked/russian_chunks.jsonl    --output datasets/retrieved/russian_bm25.jsonl --method bm25 --top-k 5
python scripts/run_context.py    --input datasets/retrieved/russian_bm25.jsonl    --output datasets/context/russian_bm25_context.jsonl
python scripts/run_generation.py --input datasets/context/russian_bm25_context.jsonl --output datasets/traces/russian_bm25_glm.jsonl \
  --pipeline-id ru_bm25_glm45flash_baseline_v1 --backend zhipu --model glm-4.5-flash --max-output-tokens 512 --sleep 1 --resume
```

Reproduce — evaluation (fault injection uses `--language ru`; the analysis reuses
the English scripts against a Russian-only data root):

```bash
python scripts/extract_healthy.py    -i datasets/traces/russian_bm25_glm.jsonl -o datasets/ru_eval/healthy_traces/healthy_bm25.jsonl
python src/data/fault_injection.py   -i datasets/ru_eval/healthy_traces/healthy_bm25.jsonl -o datasets/ru_eval/controlled_failures_bm25 -c 20 --seed 42 --language ru
python scripts/analyze_controlled_failures.py --data-root datasets/ru_eval \
  --core-csv-output outputs/controlled_failure_core_results_ru.csv \
  --heatmap-output outputs/controlled_failure_heatmap_ru.png
```
