import asyncio

from searchengine.evaluation.metrics import EvaluationMetrics
from searchengine.generation.grounded import GroundedAnswerGenerator
from searchengine.schemas import SearchHit


def test_evaluation_metrics_measure_overlap_and_hallucination():
    hits = [
        SearchHit(
            id="1",
            content="Paris is the capital of France.",
            source_name="example.com",
            score=0.9,
            rerank_score=0.9,
        )
    ]
    metrics = EvaluationMetrics()

    bundle = metrics.evaluate(
        hits=hits,
        answer="Paris is the capital of France.",
        expected_facts=["capital of France"],
        expected_answer="Paris is the capital of France.",
    )

    assert bundle.precision_at_k == 1.0
    assert bundle.answer_relevance_score > 0.5
    assert bundle.hallucination_rate < 0.5


def test_grounded_generator_falls_back_when_context_is_missing():
    generator = GroundedAnswerGenerator()

    answer, grounded = asyncio.run(generator.generate("What is the answer?", []))

    assert grounded is False
    assert "could not find enough evidence" in answer.lower()