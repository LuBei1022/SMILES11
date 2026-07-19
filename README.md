# SMILES 2026 — RAG Trace Platform & Fault-Diagnosis Data Framework

Part of the SMILES 2026 project *"Automatic Evaluation & Fault-Diagnosis Framework for RAG"*.

This repository covers the **data side** of the project, in two parts:

- **Part A — RAG Trace Platform.** Takes heterogeneous raw QA datasets (English
  and Russian), runs them through a fully configurable, fully logged RAG
  pipeline, and emits **complete traces** — one record per question capturing
  every stage from source documents to the generated answer.
- **Part B — Quality Check & Fault Injection.** Takes those complete traces,
  validates their quality, extracts *healthy* traces (gold evidence present),
  injects nine controlled fault types, and cross-checks against human
  annotation to produce a labeled test set.

The resulting labeled traces feed the *separate* evaluation & diagnosis
framework (metrics, fault localization, pipeline-level aggregation). Fault
labels live in the trace as ground truth and must never be read by the
evaluator.

## End-to-end pipeline

```
Part A — Trace generation
  Raw data → Chunking → Retrieval → Context → Generation → Full traces
     [done]    [done]     [done]      [done]     [done]       [done]

Part B — Quality & fault injection
  Full traces → Quality check → Healthy traces → Fault injection →
                Annotation templates → Human validation → Comparison
     [done]        [done]           [done]           [done]        [done]

Downstream (separate repo): Metric Engine → Diagnosis → Pipeline aggregation
     [future]
```

## Repository layout

```
SMILES11/
├── README.md
├── schemas/
│   ├── source_sample.schema.json      # unified source-sample schema (Part A, stage 1)
│   └── full_trace.schema.json         # complete-trace schema (Part A output / Part B input)
│
├── src/
│   ├── chunking/
│   │   └── recursive_chunker.py       # sentence-aware, tiktoken-based chunker
│   ├── retrieval/
│   │   ├── bm25_retriever.py          # sparse / keyword retrieval (rank_bm25)
│   │   └── dense_retriever.py         # dense / semantic retrieval (multilingual embeddings + cosine)
│   ├── context/
│   │   └── context_builder.py         # ranked, deduped, token-budgeted context assembly
│   ├── generation/
│   │   ├── prompt_builder.py          # versioned prompts (grounded_v1)
│   │   └── llm_client.py              # swappable backends: ZhipuClient (GLM) / GeminiClient / DryRunClient
│   └── data/
│       ├── quality_checker.py         # trace quality validation (Part B)
│       └── fault_injection.py         # controlled fault generation (Part B)
│
├── scripts/                           # one CLI per stage; each takes --input / --output
│   ├── normalize_english_data.py      # HotpotQA (EN)  -> source samples
│   ├── normalize_russian_data.py      # retrieval set (RU) -> source samples
│   ├── run_chunking.py                # + gold-evidence preservation check
│   ├── run_retrieval.py               # --method bm25|dense ; + Recall@k / Hit@k
│   ├── run_context.py                 # + gold-in-context check
│   ├── run_generation.py              # calls the LLM, assembles & writes full traces
│   ├── extract_healthy.py             # keep traces whose gold evidence is in context
│   ├── create_injection_template.py   # build human-annotation CSV templates
│   └── compare_with_human.py          # expected vs human labels
│
├── data/
│   ├── raw/                           # original inputs (kept local, not committed)
│   │   ├── gpt_3.5_turbo.jsonl        # English HotpotQA-style, GPT-3.5 legacy answers
│   │   └── retrieval_dataset.jsonl    # Russian retrieval QA
│   ├── normalized/                    # stage 1 output
│   ├── chunked/                       # stage 2 output
│   ├── retrieved/                     # stage 3 output
│   ├── context/                       # stage 4 output
│   ├── traces/                        # stage 5 output — full traces
│   │   ├── english_bm25_glm.jsonl
│   │   └── english_dense_glm.jsonl
│   ├── healthy_traces/                # Part B: gold-present traces
│   └── controlled_failures/           # Part B: injected-fault traces (bm25/ , dense/)
│
├── docs/
│   └── annotation_guideline.md        # annotation instructions (Part B)
│
├── outputs/                           # quality reports & problematic traces
└── experiments/
    └── human_validation/              # annotation results & comparison reports
```

## Install dependencies

```bash
pip install tiktoken rank_bm25 sentence-transformers numpy jsonschema zhipuai
# (google-genai only needed if using the optional --backend gemini)
```

---

# Part A — Trace generation

Each stage only adds its own fields and passes everything else through, so a
generated trace accumulates the full history of the run.

| # | Stage | Input | Output | Key metric produced |
|---|-------|-------|--------|---------------------|
| 1 | Normalize | raw `.jsonl` (EN / RU) | unified *source samples* | — |
| 2 | Chunking | source samples | samples + `chunks` | gold-evidence preservation |
| 3 | Retrieval | chunked samples | samples + `retrieval` | Recall@k / Hit@k |
| 4 | Context | retrieved samples | samples + `context_construction` | gold-in-context |
| 5 | Generation | context samples | **full traces** | success rate, latency, tokens |

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

Russian: swap the raw input and use `normalize_russian_data.py`; all later stages
are identical. The generation backend is swappable: `--backend zhipu` (GLM,
current default choice), `--backend gemini` (Google Gemini), or `--backend
dry-run` (offline placeholder).

### Useful flags

- `--limit N` — process only the first N records (quick testing on any stage).
- `run_generation.py --backend dry-run` — assemble full traces offline without
  calling the API (validates plumbing, costs no quota).
- `run_generation.py --resume` — skip already-succeeded samples; safe to re-run
  after hitting rate limits.
- `--sleep S` on generation — pause between calls to respect free-tier rate limits.

---

# Part B — Quality check & fault injection

Consumes the full traces from Part A. Commands below show BM25; for Dense,
replace `bm25` with `dense` in the file paths.

### 1. Quality check

```bash
python src/data/quality_checker.py \
  -i data/traces/english_bm25_glm.jsonl \
  -o outputs/data_quality_report_bm25.json \
  -p outputs/problematic_traces_bm25.jsonl
```

### 2. Extract healthy traces

Keeps only traces whose gold evidence is present in the context (needed as a
clean baseline for controlled fault injection).

```bash
python scripts/extract_healthy.py \
  -i data/traces/english_bm25_glm.jsonl \
  -o data/healthy_traces/healthy_bm25.jsonl
```

### 3. Fault injection

```bash
python src/data/fault_injection.py \
  -i data/healthy_traces/healthy_bm25.jsonl \
  -o data/controlled_failures_bm25 \
  -c 20 \
  --seed 42
```

### 4. Create annotation templates

```bash
python scripts/create_injection_template.py \
  -i data/controlled_failures_bm25 \
  -o experiments/human_validation/annotation_template_bm25.csv \
  -n 30
```

### 5. Human validation

Open the generated `annotation_template_*.csv` and manually assign the fault
class for each sample. Supported labels: `retrieval`, `chunking`, `generation`,
`out_of_scope`, `unknown`. Save the completed file as
`annotation_results_bm25.csv` / `annotation_results_dense.csv`.

### 6. Compare human vs expected

```bash
python scripts/compare_with_human.py \
  --human experiments/human_validation/annotation_results_bm25.csv \
  --output experiments/human_validation/comparison_report_bm25.json \
  --csv-report experiments/human_validation/comparison_results_bm25.csv
```

### Fault types

| Fault type | Expected diagnosis |
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

---

# Schemas

- `schemas/source_sample.schema.json` — unified source-sample format (Part A stage 1 output).
- `schemas/full_trace.schema.json` — complete trace format (Part A output, Part B input).
  Downstream code should validate traces against this and read fields from it.

```python
import json, jsonschema
schema = json.load(open("schemas/full_trace.schema.json"))
for line in open("data/traces/english_bm25_glm.jsonl", encoding="utf-8"):
    jsonschema.validate(json.loads(line), schema)
```

# Design notes

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
rather than fabricated. Chunking and retrieval are deterministic; `temperature=0`
and fixed prompt versions minimize generation variance.

**Two-system boundary.** Part A produces traces; Part B labels them. Fault-injection
labels live in the trace only as ground truth and must never be read by the
evaluation/diagnosis framework, to avoid cheating.

---

# Results (English, 500 traces)

### Retrieval baselines (Part A)

| Retriever | Hit@5 | Recall@5 |
|-----------|-------|----------|
| BM25 | 97.4% | 78.0% |
| Dense (multilingual) | 96.6% | 77.4% |

BM25 and dense retrieval are essentially tied on this multi-hop data, with BM25
marginally ahead.

### Quality check (Part B)

| Metric | BM25 | Dense |
|--------|------|-------|
| Usable traces | 496 (99.2%) | 497 (99.4%) |
| Gold present (any) | 487 (97.4%) | 482 (96.4%) |
| Gold present (all) | 293 (58.6%) | 291 (58.2%) |
| Gold missing | 13 (2.6%) | 18 (3.6%) |

The "gold present (any)" figures match the Part A retrieval Hit@5, confirming the
two measurements are consistent. "Gold present (all)" (both supporting docs for a
multi-hop question) defines the healthy-trace pool used for fault injection.

### Fault injection & human validation (Part B)

| Method | Healthy traces | Fault types | Total injections | Human-validation accuracy |
|--------|---------------:|------------:|-----------------:|--------------------------:|
| BM25 | 293 | 9 | 180 | **94.4%** (18 samples) |
| Dense | 291 | 9 | 180 | **82.4%** (17 samples) |

# Conclusion

- The RAG platform produces complete, schema-valid traces end to end for both
  retrieval baselines.
- Quality checking validates data at 99%+ usability; the healthy-trace pool is
  extracted from the traces where all gold evidence is in context.
- Nine controlled fault types are injected for both retrieval methods, and human
  validation confirms the framework's reliability.
- The labeled traces are ready for the downstream diagnosis modules.
