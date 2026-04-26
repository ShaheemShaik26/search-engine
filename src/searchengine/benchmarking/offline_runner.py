from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

from searchengine.evaluation.runner import EvaluationRunner
from searchengine.generation.grounded import GroundedAnswerGenerator
from searchengine.ranking.strategies import BaselineSemanticRanker, HeuristicRanker
from searchengine.schemas import ABStrategyResult, EvaluationQuery, EvaluationResult, SearchHit
from searchengine.benchmarking.dataset import BenchmarkDataset
from searchengine.benchmarking.report import BenchmarkThresholds, build_report, build_report_payload, render_summary_report


@dataclass(slots=True)
class OfflineBenchmarkResult:
    strategy_results: list[ABStrategyResult]
    per_query_results: dict[str, list[EvaluationResult]]


class OfflineBenchmarkRunner:
    def __init__(self) -> None:
        self.evaluator = EvaluationRunner()
        self.generator = GroundedAnswerGenerator()
        self.rankers = {
            "baseline": BaselineSemanticRanker(),
            "improved": HeuristicRanker(),
        }

    async def run(self, dataset: BenchmarkDataset, strategies: list[str] | None = None, top_k: int = 5) -> OfflineBenchmarkResult:
        strategies = strategies or ["baseline", "improved"]
        strategy_results: list[ABStrategyResult] = []
        per_query_results: dict[str, list[EvaluationResult]] = {}

        for strategy in strategies:
            query_results: list[EvaluationResult] = []
            ranker = self._resolve_ranker(strategy)
            for case in dataset.cases:
                ranked_hits = ranker.rerank(case.query.query, case.candidate_hits)
                answer, _ = await self.generator.generate(case.query.query, ranked_hits[:top_k])
                evaluation_result = self.evaluator.evaluate_query(case.query, ranked_hits[:top_k], answer)
                query_results.append(evaluation_result)
                per_query_results.setdefault(case.query.query, []).append(evaluation_result)
            strategy_results.append(self.evaluator.aggregate(strategy, query_results))

        return OfflineBenchmarkResult(strategy_results=strategy_results, per_query_results=per_query_results)

    def _resolve_ranker(self, strategy: str):
        return self.rankers.get(strategy.lower().strip(), self.rankers["improved"])

    def write_json(self, result: OfflineBenchmarkResult, output_path: str | Path) -> None:
        payload = {
            "strategies": [result_item.model_dump() for result_item in result.strategy_results],
            "per_query": {
                query: [item.model_dump() for item in evaluations]
                for query, evaluations in result.per_query_results.items()
            },
        }
        Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def write_structured_json(
        self,
        result: OfflineBenchmarkResult,
        output_path: str | Path,
        thresholds: BenchmarkThresholds | None = None,
    ) -> None:
        payload = {
            "raw": {
                "strategies": [result_item.model_dump() for result_item in result.strategy_results],
                "per_query": {
                    query: [item.model_dump() for item in evaluations]
                    for query, evaluations in result.per_query_results.items()
                },
            },
            "report": build_report_payload(result.strategy_results, thresholds=thresholds),
        }
        Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def write_summary(
        self,
        result: OfflineBenchmarkResult,
        output_path: str | Path,
        thresholds: BenchmarkThresholds | None = None,
    ) -> None:
        report = build_report(result.strategy_results, thresholds=thresholds)
        Path(output_path).write_text(render_summary_report(report, output_path), encoding="utf-8")
