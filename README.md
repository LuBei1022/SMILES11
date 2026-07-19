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
