from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import exp

from searchengine.ranking.base import Ranker
from searchengine.schemas import SearchHit
from searchengine.utils.text import tokenize


@dataclass(slots=True)
class RankWeights:
    semantic: float = 0.62
    recency: float = 0.18
    keyword: float = 0.15
    source: float = 0.05


class BaselineSemanticRanker(Ranker):
    def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        return sorted(hits, key=lambda hit: hit.score, reverse=True)


class HeuristicRanker(Ranker):
    def __init__(self, weights: RankWeights | None = None) -> None:
        self.weights = weights or RankWeights()

    def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        query_tokens = set(tokenize(query))
        now = datetime.now(timezone.utc)
        ranked: list[SearchHit] = []
        for hit in hits:
            semantic = self._clamp(hit.score)
            recency = self._recency_score(hit.published_at, now)
            keyword = self._keyword_overlap(query_tokens, hit.content)
            source = self._source_prior(hit.source_name)
            rerank_score = (
                self.weights.semantic * semantic
                + self.weights.recency * recency
                + self.weights.keyword * keyword
                + self.weights.source * source
            )
            ranked.append(hit.model_copy(update={"rerank_score": rerank_score}))
        return sorted(ranked, key=lambda hit: hit.rerank_score, reverse=True)

    def _clamp(self, value: float) -> float:
        return max(0.0, min(1.0, value))

    def _recency_score(self, published_at: datetime | None, now: datetime) -> float:
        if not published_at:
            return 0.5
        age_days = max(0.0, (now - published_at.astimezone(timezone.utc)).total_seconds() / 86400.0)
        return exp(-age_days / 30.0)

    def _keyword_overlap(self, query_tokens: set[str], content: str) -> float:
        if not query_tokens:
            return 0.0
        content_tokens = set(tokenize(content))
        if not content_tokens:
            return 0.0
        overlap = len(query_tokens & content_tokens)
        return overlap / max(len(query_tokens), 1)

    def _source_prior(self, source_name: str) -> float:
        trusted = {"wikipedia.org": 0.95, "arxiv.org": 0.9, "github.com": 0.7}
        return trusted.get(source_name.lower(), 0.6)
