from __future__ import annotations

import json
import argparse
import asyncio
from pathlib import Path

from searchengine.benchmarking.dataset import BenchmarkDataset
from searchengine.benchmarking.offline_runner import OfflineBenchmarkRunner
from searchengine.benchmarking.report import BenchmarkThresholds, build_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an offline search ranking benchmark")
    parser.add_argument("--dataset", required=True, help="Path to benchmark dataset JSON")
    parser.add_argument("--output", required=True, help="Path to write benchmark results JSON")
    parser.add_argument("--report-json", help="Optional path to write a single structured benchmark JSON report")
    parser.add_argument("--summary", help="Optional path to write a human-readable summary report (.txt, .md, or .html)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of hits to evaluate")
    parser.add_argument("--strategies", nargs="*", default=["baseline", "improved"], help="Ranking strategies to compare")
    parser.add_argument("--min-precision-delta", type=float, default=0.0, help="Minimum precision@k delta required to pass")
    parser.add_argument("--min-answer-relevance-delta", type=float, default=0.0, help="Minimum answer relevance delta required to pass")
    parser.add_argument("--max-hallucination-delta", type=float, default=0.0, help="Maximum hallucination delta allowed to pass")
    parser.add_argument("--json-status", action="store_true", help="Emit a machine-readable JSON status line on stdout")
    return parser


async def _run_async(args: argparse.Namespace) -> None:
    dataset = BenchmarkDataset.from_json_file(args.dataset)
    runner = OfflineBenchmarkRunner()
    result = await runner.run(dataset, strategies=args.strategies, top_k=args.top_k)
    runner.write_json(result, args.output)
    status = "N/A"
    thresholds = None
    if args.report_json or args.summary:
        thresholds = BenchmarkThresholds(
            min_precision_delta=args.min_precision_delta,
            min_answer_relevance_delta=args.min_answer_relevance_delta,
            max_hallucination_delta=args.max_hallucination_delta,
        )
    if args.report_json:
        runner.write_structured_json(result, args.report_json, thresholds=thresholds)
        report = build_report(result.strategy_results, thresholds=thresholds)
        if report.comparisons:
            status = "PASS" if report.comparisons[0].passed else "FAIL"
    if args.summary:
        runner.write_summary(result, args.summary, thresholds=thresholds)
        report = build_report(result.strategy_results, thresholds=thresholds)
        if report.comparisons:
            status = "PASS" if report.comparisons[0].passed else "FAIL"
    else:
        thresholds = None

    if args.json_status:
        payload = {
            "json_output": args.output,
            "report_json_output": args.report_json,
            "summary_output": args.summary,
            "status": status,
        }
        print(json.dumps(payload))
    else:
        if args.report_json and args.summary:
            print(f"Benchmark complete. JSON: {args.output}; Report: {args.report_json}; Summary: {args.summary}; Status: {status}")
        elif args.report_json:
            print(f"Benchmark complete. JSON: {args.output}; Report: {args.report_json}; Status: {status}")
        elif args.summary:
            print(f"Benchmark complete. JSON: {args.output}; Summary: {args.summary}; Status: {status}")
        else:
            print(f"Benchmark complete. JSON: {args.output}; Status: {status}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(_run_async(args))


if __name__ == "__main__":
    main()
