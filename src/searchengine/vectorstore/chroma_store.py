from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import chromadb
import numpy as np

from searchengine.core.config import settings
from searchengine.schemas import ChunkRecord, SearchHit
from searchengine.vectorstore.base import VectorStore


class ChromaVectorStore(VectorStore):
    def __init__(self, persist_directory: str | None = None, collection_name: str | None = None) -> None:
        self.persist_directory = persist_directory or settings.chroma_path
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name or settings.chroma_collection
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(name=self.collection_name, metadata={"hnsw:space": "cosine"})

    def upsert(self, chunks: list[ChunkRecord], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        ids = [chunk.id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = []
        for chunk in chunks:
            payload = dict(chunk.metadata)
            payload["source_name"] = chunk.source_name
            payload["published_at"] = chunk.published_at.isoformat() if chunk.published_at else None
            metadatas.append(payload)
        self.collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def search(self, query_embedding: list[float], top_k: int) -> list[SearchHit]:
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        hits: list[SearchHit] = []
        if not results.get("ids"):
            return hits
        ids = results["ids"][0]
        documents = results["documents"][0]
        distances = results["distances"][0]
        metadatas = results["metadatas"][0]

        for chunk_id, document, distance, metadata in zip(ids, documents, distances, metadatas):
            metadata = metadata or {}
            published_at_raw = metadata.pop("published_at", None)
            published_at = datetime.fromisoformat(published_at_raw) if published_at_raw else None
            source_name = metadata.pop("source_name", "document")
            similarity = float(1.0 - distance)
            hits.append(
                SearchHit(
                    id=chunk_id,
                    content=document,
                    source_name=source_name,
                    score=similarity,
                    rerank_score=similarity,
                    metadata=metadata,
                    published_at=published_at,
                )
            )
        return hits


@lru_cache(maxsize=1)
def get_vector_store() -> ChromaVectorStore:
    return ChromaVectorStore()
