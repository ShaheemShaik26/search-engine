from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from searchengine.embeddings.base import EmbeddingModel
from searchengine.core.config import settings


class SentenceTransformerEmbeddingModel(EmbeddingModel):
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.embedding_model_name
        self._model = SentenceTransformer(self.model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]

    def embed_query(self, query: str) -> list[float]:
        vector = self._model.encode([query], normalize_embeddings=True)[0]
        return vector.tolist()


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformerEmbeddingModel:
    return SentenceTransformerEmbeddingModel()
