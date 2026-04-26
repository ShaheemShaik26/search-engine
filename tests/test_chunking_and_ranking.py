from datetime import datetime, timezone

from searchengine.ingestion.chunking import ChunkingPolicy, IntelligentChunker
from searchengine.ranking.strategies import BaselineSemanticRanker, HeuristicRanker
from searchengine.schemas import SearchHit


def test_chunker_keeps_sentence_boundaries_and_overlap():
    text = (
        "Alpha is the first sentence. Beta follows with more detail. Gamma adds context. "
        "Delta closes the paragraph."
    )
    chunker = IntelligentChunker(ChunkingPolicy(max_tokens=8, overlap_tokens=3))

    chunks = chunker.chunk(text)

    assert len(chunks) >= 2
    assert chunks[0].startswith("Alpha is the first sentence")
    assert "Beta follows" in chunks[1] or "Gamma adds" in chunks[1]


def test_baseline_ranker_sorts_by_semantic_score_only():
    hits = [
        SearchHit(id="1", content="low", source_name="source", score=0.2, rerank_score=0.2),
        SearchHit(id="2", content="high", source_name="source", score=0.9, rerank_score=0.9),
    ]

    ranked = BaselineSemanticRanker().rerank("query", hits)

    assert [hit.id for hit in ranked] == ["2", "1"]


def test_heuristic_ranker_prefers_recency_and_keyword_overlap():
    old_hit = SearchHit(
        id="old",
        content="This document discusses query matching and ranking strategy.",
        source_name="example.com",
        score=0.70,
        rerank_score=0.70,
        published_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )
    recent_hit = SearchHit(
        id="recent",
        content="This document mentions query matching with extra context.",
        source_name="example.com",
        score=0.68,
        rerank_score=0.68,
        published_at=datetime.now(timezone.utc),
    )

    ranked = HeuristicRanker().rerank("query matching", [old_hit, recent_hit])

    assert ranked[0].id == "recent"
    assert ranked[0].rerank_score >= ranked[1].rerank_score