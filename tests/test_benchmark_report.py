from searchengine.benchmarking.report import (
    BenchmarkThresholds,
    build_report,
    render_summary_report,
    render_text_report,
)
from searchengine.schemas import ABStrategyResult


def test_benchmark_report_renders_pass_fail_summary():
    strategy_results = [
        ABStrategyResult(
            strategy="baseline",
            average_precision_at_k=0.50,
            average_answer_relevance_score=0.40,
            average_hallucination_rate=0.20,
            query_count=3,
        ),
        ABStrategyResult(
            strategy="improved",
            average_precision_at_k=0.65,
            average_answer_relevance_score=0.55,
            average_hallucination_rate=0.10,
            query_count=3,
        ),
    ]

    report = build_report(
        strategy_results,
        thresholds=BenchmarkThresholds(
            min_precision_delta=0.05,
            min_answer_relevance_delta=0.05,
            max_hallucination_delta=0.0,
        ),
    )

    text = render_text_report(report)

    assert "Offline Benchmark Summary" in text
    assert "Precision delta: +0.1500" in text
    assert "Answer relevance delta: +0.1500" in text
    assert "Hallucination delta: -0.1000" in text
    assert "Result: PASS" in text


def test_benchmark_report_renders_markdown_for_md_paths():
    strategy_results = [
        ABStrategyResult(
            strategy="baseline",
            average_precision_at_k=0.50,
            average_answer_relevance_score=0.40,
            average_hallucination_rate=0.20,
            query_count=3,
        ),
        ABStrategyResult(
            strategy="improved",
            average_precision_at_k=0.65,
            average_answer_relevance_score=0.55,
            average_hallucination_rate=0.10,
            query_count=3,
        ),
    ]

    report = build_report(strategy_results)
    markdown = render_summary_report(report, "benchmark_summary.md")

    assert markdown.startswith("# Offline Benchmark Summary")
    assert "| metric | baseline | improved | delta |".lower() in markdown.lower()
    assert "**Result:** PASS" in markdown


def test_benchmark_report_renders_html_for_html_paths():
    strategy_results = [
        ABStrategyResult(
            strategy="baseline",
            average_precision_at_k=0.50,
            average_answer_relevance_score=0.40,
            average_hallucination_rate=0.20,
            query_count=3,
        ),
        ABStrategyResult(
            strategy="improved",
            average_precision_at_k=0.65,
            average_answer_relevance_score=0.55,
            average_hallucination_rate=0.10,
            query_count=3,
        ),
    ]

    report = build_report(strategy_results)
    html = render_summary_report(report, "benchmark_summary.html")

    assert html.startswith("<html>")
    assert "Offline Benchmark Summary" in html
    assert "<table>" in html
    assert "Result: PASS" in html


def test_cli_json_status_payload(tmp_path, capsys):
    import asyncio

    from searchengine.cli.run_benchmark import _run_async

    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        """
        {
          "cases": [
            {
              "query": {
                "query": "What is the capital of France?",
                "expected_facts": ["Paris is the capital of France"],
                "expected_answer": "Paris is the capital of France."
              },
              "candidate_hits": [
                {
                  "id": "p1",
                  "content": "France's capital is Paris.",
                  "source_name": "wikipedia.org",
                  "score": 0.82,
                  "rerank_score": 0.82,
                  "metadata": {"topic": "geography"},
                  "published_at": "2026-01-01T00:00:00+00:00"
                }
              ]
            }
          ]
        }
        """,
        encoding="utf-8",
    )
    output_path = tmp_path / "results.json"
    summary_path = tmp_path / "summary.md"

    class Args:
        dataset = str(dataset_path)
        output = str(output_path)
        report_json = str(tmp_path / "report.json")
        summary = str(summary_path)
        top_k = 1
        strategies = ["baseline", "improved"]
        min_precision_delta = 0.0
        min_answer_relevance_delta = 0.0
        max_hallucination_delta = 0.0
        json_status = True

    asyncio.run(_run_async(Args()))
    captured = capsys.readouterr().out.strip()

    assert captured.startswith("{")
    assert '"json_output"' in captured
    assert '"summary_output"' in captured
    assert '"status"' in captured


def test_structured_report_json_contains_raw_and_report(tmp_path):
    import asyncio
    import json

    from searchengine.benchmarking.dataset import BenchmarkDataset
    from searchengine.benchmarking.offline_runner import OfflineBenchmarkRunner

    dataset = BenchmarkDataset.from_json_file("tests/fixtures/ab_benchmark.json")
    runner = OfflineBenchmarkRunner()
    result = asyncio.run(runner.run(dataset, strategies=["baseline", "improved"], top_k=2))

    output_path = tmp_path / "structured_report.json"
    runner.write_structured_json(result, output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert "raw" in payload
    assert "report" in payload
    assert "strategies" in payload["raw"]
    assert "per_query" in payload["raw"]
    assert "comparisons" in payload["report"]
    assert "status" in payload["report"]
