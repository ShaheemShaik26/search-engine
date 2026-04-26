from __future__ import annotations

from abc import ABC, abstractmethod

from searchengine.schemas import ChunkRecord, SearchHit


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, chunks: list[ChunkRecord], embeddings: list[list[float]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int) -> list[SearchHit]:
        raise NotImplementedError
