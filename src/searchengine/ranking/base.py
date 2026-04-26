from __future__ import annotations

from abc import ABC, abstractmethod

from searchengine.schemas import SearchHit


class Ranker(ABC):
    @abstractmethod
    def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        raise NotImplementedError
