# AI Search Engine

A production-oriented search stack inspired by Perplexity-style systems. The system is built around a strict pipeline:

`User Query -> Retrieve Documents -> Rank Results -> Generate Grounded Answer -> Evaluate Quality -> Improve Ranking`

## Quick Start

1. Install dependencies: `pip install -e .[dev]`
2. Run the API: `uvicorn searchengine.app.main:app --reload`
3. Run tests: `pytest`
4. Run the benchmark: `python -m searchengine.cli.run_benchmark --dataset tests/fixtures/ab_benchmark.json --output benchmark_results.json --report-json benchmark_report.json --summary benchmark_summary.html --json-status`

## What’s Included

This repository now contains the full core search loop and the benchmark tooling around it:

- Document ingestion for PDFs, text files, and URLs.
- Intelligent sentence-aware chunking with overlap.
- Transformer embeddings with a Chroma vector store backend.
- Semantic retrieval plus a heuristic reranking layer.
- Grounded answer generation with abstention when evidence is weak.
- Evaluation metrics for retrieval precision@k, answer relevance, and hallucination rate.
- Offline A/B benchmark runner for baseline vs improved ranking.
- Human-readable benchmark summaries in text, Markdown, and HTML.
- Structured benchmark JSON output for scripts and automation.
- FastAPI endpoints for ingest, search, evaluation, and A/B testing.
- Regression tests covering the core pipeline and benchmark outputs.

## Architecture

### Core Services
- Ingestion service for PDFs, text files, and URLs.
- Chunking and embedding service using transformer models.
- Chroma vector store for semantic retrieval.
- Ranking layer with baseline semantic and improved heuristic strategies.
- Grounded answer generator that only uses retrieved context.
- Evaluation framework for retrieval and answer quality metrics.
- A/B testing runner for ranking experiments.
- FastAPI layer for `/search`, `/ingest`, `/evaluate`, and `/ab-test` endpoints.

### Data Flow

```text
[PDF/Text/URL]
      |
      v
[Loader + Cleaner]
      |
      v
[Intelligent Chunker]
      |
      v
[Transformer Embeddings]
      |
      v
[Chroma Vector Store]
      |
      v
[Semantic Retrieval top-k]
      |
      v
[Ranking Layer]
      |
      v
[Grounded Context Filter]
      |
      v
[LLM Answer Composer]
      |
      v
[Evaluation + A/B Logging]
      |
      v
[Ranking Improvements]
```

### Ranking Strategy
The ranker uses a weighted blend of:
- semantic similarity from the retriever
- recency weighting from document metadata
- keyword overlap with query terms
- source trust / relevance priors

The default improved ranker is intentionally deterministic and interpretable. A baseline semantic-only ranker is included for comparison.

The benchmark runner compares both strategies on the same labeled dataset and reports deltas for the improved ranker.

### Evaluation Methodology
The evaluation module measures:
- precision@k for retrieval quality
- answer relevance score
- hallucination rate based on unsupported answer claims

The A/B runner compares baseline vs improved ranking on the same query set and logs deltas over time.

The offline benchmark also produces a structured JSON report with raw results, threshold configuration, comparison deltas, and pass/fail status.

### Scaling Notes
- Chunking is designed to keep context windows bounded.
- Retrieval uses vector search plus metadata filters.
- Ranking is CPU-light and can run in-process.
- The grounded answer path falls back to abstention if support is weak.
- For scale-out, split ingestion and retrieval into separate services and store embeddings in a persistent Chroma deployment or FAISS shard layout.

## Run Locally

1. Create a virtual environment.
2. Install dependencies: `pip install -e .[dev]`
3. Start the API: `uvicorn searchengine.app.main:app --reload`

## Test

Run the lightweight regression suite with `pytest`.

Current test coverage includes chunking, reranking, grounded generation, evaluation, benchmark summaries, CLI status output, and structured benchmark JSON.

## Offline Benchmark

Run the deterministic A/B benchmark with the bundled fixture:

`python -m searchengine.cli.run_benchmark --dataset tests/fixtures/ab_benchmark.json --output benchmark_results.json`

To also generate a human-readable comparison report, add `--summary benchmark_summary.txt`, `benchmark_summary.md`, or `benchmark_summary.html`. You can tune the pass criteria with `--min-precision-delta`, `--min-answer-relevance-delta`, and `--max-hallucination-delta`.

To write a single machine-readable benchmark bundle containing raw results plus report metadata, add `--report-json benchmark_report.json`.

To emit a parseable stdout line for automation, add `--json-status`.

Example:

```bash
python -m searchengine.cli.run_benchmark \
      --dataset tests/fixtures/ab_benchmark.json \
      --output benchmark_results.json \
      --report-json benchmark_report.json \
      --summary benchmark_summary.html \
      --json-status
```

## Endpoints
- `POST /ingest` to ingest text, PDF, or URL documents.
- `POST /search` to run retrieval, ranking, generation, and scoring.
- `POST /evaluate` to benchmark a query set.
- `POST /ab-test` to compare ranking strategies.

## Current Status

The implemented project is validated with regression tests and benchmark smoke tests. The offline benchmark workflow now supports:

- raw JSON results
- structured bundled JSON reports
- text, Markdown, and HTML summaries
- JSON stdout status output for scripts

## Project Layout

- `src/searchengine/ingestion`: loaders and chunking for PDFs, text files, and URLs.
- `src/searchengine/embeddings`: transformer embedding adapters.
- `src/searchengine/vectorstore`: Chroma persistence and retrieval.
- `src/searchengine/ranking`: baseline and heuristic reranking.
- `src/searchengine/generation`: grounded answer generation.
- `src/searchengine/evaluation`: retrieval and answer metrics.
- `src/searchengine/ab_testing`: online A/B comparison runner.
- `src/searchengine/benchmarking`: offline benchmark dataset, report, and structured output.
- `src/searchengine/cli`: benchmark command-line entrypoints.
- `src/searchengine/api`: FastAPI service layer and routes.
- `tests`: regression tests and benchmark fixtures.

## Notes
This project is structured as a production foundation. The LLM layer is provider-agnostic, so you can wire it to an OpenAI-compatible endpoint or any internal model gateway.
