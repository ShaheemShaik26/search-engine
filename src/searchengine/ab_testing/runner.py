from __future__ import annotations

from searchengine.api.service import SearchService
from searchengine.evaluation.runner import EvaluationRunner
from searchengine.schemas import ABStrategyResult, EvaluationQuery


class ABTestRunner:
    def __init__(self, search_service: SearchService) -> None:
        self.search_service = search_service
        self.evaluation_runner = EvaluationRunner()

    async def run(self, queries: list[EvaluationQuery], strategies: list[str], top_k: int) -> list[ABStrategyResult]:
        results: list[ABStrategyResult] = []
        for strategy in strategies:
            query_results = []
            for query in queries:
                response = await self.search_service.search(query.query, top_k=top_k, strategy=strategy)
                query_results.append(
                    self.evaluation_runner.evaluate_query(query, response.results, response.answer)
                )
            results.append(self.evaluation_runner.aggregate(strategy, query_results))
        return results
