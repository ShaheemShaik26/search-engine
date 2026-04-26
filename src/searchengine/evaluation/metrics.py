from __future__ import annotations

from dataclasses import dataclass

from searchengine.schemas import EvaluationResult, SearchHit
from searchengine.utils.text import tokenize


@dataclass(slots=True)
class MetricBundle:
    precision_at_k: float
    answer_relevance_score: float
    hallucination_rate: float


class EvaluationMetrics:
    def precision_at_k(self, hits: list[SearchHit], expected_facts: list[str]) -> float:
        if not hits:
            return 0.0
        if not expected_facts:
            return 0.0
        hit_text = " ".join(hit.content.lower() for hit in hits)
        matches = sum(1 for fact in expected_facts if fact.lower() in hit_text)
        return matches / max(len(expected_facts), 1)

    def answer_relevance_score(self, answer: str, expected_answer: str | None) -> float:
        if not expected_answer:
            return 0.0
        answer_tokens = set(tokenize(answer))
        expected_tokens = set(tokenize(expected_answer))
        if not expected_tokens:
            return 0.0
        overlap = len(answer_tokens & expected_tokens)
        return overlap / len(expected_tokens)

    def hallucination_rate(self, answer: str, hits: list[SearchHit]) -> float:
        if not answer:
            return 0.0
        answer_tokens = set(tokenize(answer))
        context_tokens = set()
        for hit in hits:
            context_tokens.update(tokenize(hit.content))
        unsupported = answer_tokens - context_tokens
        return len(unsupported) / max(len(answer_tokens), 1)

    def evaluate(self, hits: list[SearchHit], answer: str, expected_facts: list[str], expected_answer: str | None) -> MetricBundle:
        return MetricBundle(
            precision_at_k=self.precision_at_k(hits, expected_facts),
            answer_relevance_score=self.answer_relevance_score(answer, expected_answer),
            hallucination_rate=self.hallucination_rate(answer, hits),
        )
