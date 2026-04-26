from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from searchengine.schemas import ABStrategyResult


@dataclass(slots=True)
class BenchmarkThresholds:
    min_precision_delta: float = 0.0
    min_answer_relevance_delta: float = 0.0
    max_hallucination_delta: float = 0.0


@dataclass(slots=True)
class BenchmarkComparison:
    baseline: ABStrategyResult
    improved: ABStrategyResult
    precision_delta: float
    answer_relevance_delta: float
    hallucination_delta: float
    passed: bool


@dataclass(slots=True)
class BenchmarkReport:
    comparisons: list[BenchmarkComparison]
    thresholds: BenchmarkThresholds


def build_report(strategy_results: list[ABStrategyResult], thresholds: BenchmarkThresholds | None = None) -> BenchmarkReport:
    thresholds = thresholds or BenchmarkThresholds()
    by_strategy = {result.strategy.lower().strip(): result for result in strategy_results}
    baseline = by_strategy.get("baseline")
    improved = by_strategy.get("improved")
    comparisons: list[BenchmarkComparison] = []

    if baseline and improved:
        precision_delta = improved.average_precision_at_k - baseline.average_precision_at_k
        answer_relevance_delta = improved.average_answer_relevance_score - baseline.average_answer_relevance_score
        hallucination_delta = improved.average_hallucination_rate - baseline.average_hallucination_rate
        passed = (
            precision_delta >= thresholds.min_precision_delta
            and answer_relevance_delta >= thresholds.min_answer_relevance_delta
            and hallucination_delta <= thresholds.max_hallucination_delta
        )
        comparisons.append(
            BenchmarkComparison(
                baseline=baseline,
                improved=improved,
                precision_delta=precision_delta,
                answer_relevance_delta=answer_relevance_delta,
                hallucination_delta=hallucination_delta,
                passed=passed,
            )
        )

    return BenchmarkReport(comparisons=comparisons, thresholds=thresholds)


def build_report_payload(strategy_results: list[ABStrategyResult], thresholds: BenchmarkThresholds | None = None) -> dict:
    report = build_report(strategy_results, thresholds=thresholds)
    comparison_payloads = [
        {
            "baseline": comparison.baseline.model_dump(),
            "improved": comparison.improved.model_dump(),
            "precision_delta": comparison.precision_delta,
            "answer_relevance_delta": comparison.answer_relevance_delta,
            "hallucination_delta": comparison.hallucination_delta,
            "passed": comparison.passed,
        }
        for comparison in report.comparisons
    ]
    return {
        "thresholds": {
            "min_precision_delta": report.thresholds.min_precision_delta,
            "min_answer_relevance_delta": report.thresholds.min_answer_relevance_delta,
            "max_hallucination_delta": report.thresholds.max_hallucination_delta,
        },
        "comparisons": comparison_payloads,
        "status": "PASS" if any(comparison["passed"] for comparison in comparison_payloads) else ("FAIL" if comparison_payloads else "N/A"),
    }


def render_text_report(report: BenchmarkReport) -> str:
    lines: list[str] = []
    lines.append("Offline Benchmark Summary")
    lines.append("=" * 26)
    lines.append(
        "Thresholds: "
        f"precision_delta>={report.thresholds.min_precision_delta:.4f}, "
        f"answer_relevance_delta>={report.thresholds.min_answer_relevance_delta:.4f}, "
        f"hallucination_delta<={report.thresholds.max_hallucination_delta:.4f}"
    )
    lines.append("")

    if not report.comparisons:
        lines.append("No baseline/improved comparison was available.")
        return "\n".join(lines)

    comparison = report.comparisons[0]
    lines.append(f"Baseline precision@k: {comparison.baseline.average_precision_at_k:.4f}")
    lines.append(f"Improved precision@k: {comparison.improved.average_precision_at_k:.4f}")
    lines.append(f"Precision delta: {comparison.precision_delta:+.4f}")
    lines.append(f"Baseline answer relevance: {comparison.baseline.average_answer_relevance_score:.4f}")
    lines.append(f"Improved answer relevance: {comparison.improved.average_answer_relevance_score:.4f}")
    lines.append(f"Answer relevance delta: {comparison.answer_relevance_delta:+.4f}")
    lines.append(f"Baseline hallucination rate: {comparison.baseline.average_hallucination_rate:.4f}")
    lines.append(f"Improved hallucination rate: {comparison.improved.average_hallucination_rate:.4f}")
    lines.append(f"Hallucination delta: {comparison.hallucination_delta:+.4f}")
    lines.append(f"Result: {'PASS' if comparison.passed else 'FAIL'}")
    return "\n".join(lines)


def render_markdown_report(report: BenchmarkReport) -> str:
    lines: list[str] = []
    lines.append("# Offline Benchmark Summary")
    lines.append("")
    lines.append("## Thresholds")
    lines.append(
        f"- Precision delta >= `{report.thresholds.min_precision_delta:.4f}`\n"
        f"- Answer relevance delta >= `{report.thresholds.min_answer_relevance_delta:.4f}`\n"
        f"- Hallucination delta <= `{report.thresholds.max_hallucination_delta:.4f}`"
    )
    lines.append("")

    if not report.comparisons:
        lines.append("No baseline/improved comparison was available.")
        return "\n".join(lines)

    comparison = report.comparisons[0]
    lines.append("## Comparison")
    lines.append("")
    lines.append("| Metric | Baseline | Improved | Delta |")
    lines.append("| --- | ---: | ---: | ---: |")
    lines.append(
        f"| precision@k | {comparison.baseline.average_precision_at_k:.4f} | {comparison.improved.average_precision_at_k:.4f} | {comparison.precision_delta:+.4f} |"
    )
    lines.append(
        f"| answer relevance | {comparison.baseline.average_answer_relevance_score:.4f} | {comparison.improved.average_answer_relevance_score:.4f} | {comparison.answer_relevance_delta:+.4f} |"
    )
    lines.append(
        f"| hallucination rate | {comparison.baseline.average_hallucination_rate:.4f} | {comparison.improved.average_hallucination_rate:.4f} | {comparison.hallucination_delta:+.4f} |"
    )
    lines.append("")
    lines.append(f"**Result:** {'PASS' if comparison.passed else 'FAIL'}")
    return "\n".join(lines)


def render_html_report(report: BenchmarkReport) -> str:
        if not report.comparisons:
                return (
                        "<html><head><meta charset='utf-8'><title>Offline Benchmark Summary</title></head><body>"
                        "<h1>Offline Benchmark Summary</h1><p>No baseline/improved comparison was available.</p>"
                        "</body></html>"
                )

        comparison = report.comparisons[0]
        result_label = "PASS" if comparison.passed else "FAIL"
        return f"""
<html>
    <head>
        <meta charset="utf-8">
        <title>Offline Benchmark Summary</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 2rem; color: #1f2937; }}
            h1, h2 {{ color: #111827; }}
            table {{ border-collapse: collapse; width: 100%; max-width: 900px; }}
            th, td {{ border: 1px solid #d1d5db; padding: 0.75rem; text-align: right; }}
            th:first-child, td:first-child {{ text-align: left; }}
            th {{ background: #f3f4f6; }}
            .pass {{ color: #166534; font-weight: bold; }}
            .fail {{ color: #991b1b; font-weight: bold; }}
            code {{ background: #f9fafb; padding: 0.15rem 0.35rem; border-radius: 0.25rem; }}
        </style>
    </head>
    <body>
        <h1>Offline Benchmark Summary</h1>
        <h2>Thresholds</h2>
        <ul>
            <li>Precision delta &gt;= <code>{report.thresholds.min_precision_delta:.4f}</code></li>
            <li>Answer relevance delta &gt;= <code>{report.thresholds.min_answer_relevance_delta:.4f}</code></li>
            <li>Hallucination delta &lt;= <code>{report.thresholds.max_hallucination_delta:.4f}</code></li>
        </ul>
        <h2>Comparison</h2>
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Baseline</th>
                    <th>Improved</th>
                    <th>Delta</th>
                </tr>
            </thead>
            <tbody>
                <tr><td>precision@k</td><td>{comparison.baseline.average_precision_at_k:.4f}</td><td>{comparison.improved.average_precision_at_k:.4f}</td><td>{comparison.precision_delta:+.4f}</td></tr>
                <tr><td>answer relevance</td><td>{comparison.baseline.average_answer_relevance_score:.4f}</td><td>{comparison.improved.average_answer_relevance_score:.4f}</td><td>{comparison.answer_relevance_delta:+.4f}</td></tr>
                <tr><td>hallucination rate</td><td>{comparison.baseline.average_hallucination_rate:.4f}</td><td>{comparison.improved.average_hallucination_rate:.4f}</td><td>{comparison.hallucination_delta:+.4f}</td></tr>
            </tbody>
        </table>
        <p class="{'pass' if comparison.passed else 'fail'}">Result: {result_label}</p>
    </body>
</html>
""".strip()


def render_summary_report(report: BenchmarkReport, output_path: str | Path) -> str:
    path = Path(output_path)
    if path.suffix.lower() in {".md", ".markdown"}:
        return render_markdown_report(report)
    if path.suffix.lower() in {".html", ".htm"}:
        return render_html_report(report)
    return render_text_report(report)