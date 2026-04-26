from __future__ import annotations

from collections import defaultdict

from searchengine.evaluation.metrics import EvaluationMetrics
from searchengine.schemas import ABStrategyResult, EvaluationQuery, EvaluationResult, SearchHit


class EvaluationRunner:
    def __init__(self) -> None:
        self.metrics = EvaluationMetrics()

    def evaluate_query(
        self,
        query: EvaluationQuery,
        hits: list[SearchHit],
        answer: str,
    ) -> EvaluationResult:
        bundle = self.metrics.evaluate(hits, answer, query.expected_facts, query.expected_answer)
        return EvaluationResult(
            query=query.query,
            precision_at_k=bundle.precision_at_k,
            answer_relevance_score=bundle.answer_relevance_score,
            hallucination_rate=bundle.hallucination_rate,
        )

    def aggregate(self, strategy: str, results: list[EvaluationResult]) -> ABStrategyResult:
        if not results:
            return ABStrategyResult(
                strategy=strategy,
                average_precision_at_k=0.0,
                average_answer_relevance_score=0.0,
                average_hallucination_rate=0.0,
                query_count=0,
            )
        count = len(results)
        return ABStrategyResult(
            strategy=strategy,
            average_precision_at_k=sum(result.precision_at_k for result in results) / count,
            average_answer_relevance_score=sum(result.answer_relevance_score for result in results) / count,
            average_hallucination_rate=sum(result.hallucination_rate for result in results) / count,
            query_count=count,
        )
