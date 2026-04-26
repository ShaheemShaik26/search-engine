from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from searchengine.core.config import settings
from searchengine.embeddings.sentence_transformer import get_embedding_model
from searchengine.evaluation.runner import EvaluationRunner
from searchengine.generation.grounded import GroundedAnswerGenerator
from searchengine.ingestion.chunking import ChunkingPolicy, IntelligentChunker
from searchengine.ingestion.loaders import DocumentLoader
from searchengine.ranking.strategies import BaselineSemanticRanker, HeuristicRanker
from searchengine.schemas import ChunkRecord, EvaluationQuery, IngestResponse, SearchHit, SearchResponse
from searchengine.utils.text import detect_source_name, utc_now
from searchengine.vectorstore.chroma_store import get_vector_store


@dataclass(slots=True)
class SearchService:
    def __init__(self) -> None:
        self.loader = DocumentLoader()
        self.chunker = IntelligentChunker(ChunkingPolicy(max_tokens=settings.max_chunk_size, overlap_tokens=settings.chunk_overlap))
        self.embedding_model = get_embedding_model()
        self.vector_store = get_vector_store()
        self.semantic_ranker = BaselineSemanticRanker()
        self.heuristic_ranker = HeuristicRanker()
        self.generator = GroundedAnswerGenerator()
        self.evaluator = EvaluationRunner()

    async def ingest_document(
        self,
        text: str | None = None,
        url: str | None = None,
        file_path: str | None = None,
        source_name: str | None = None,
        published_at: datetime | None = None,
        metadata: dict | None = None,
    ) -> IngestResponse:
        if text is not None:
            content = self.loader.load_text(text)
        elif url is not None:
            content = await self.loader.load_url(url)
        elif file_path is not None:
            content = self.loader.load_file(file_path)
        else:
            raise ValueError("One of text, url, or file_path must be provided")

        source_name = source_name or detect_source_name(url or file_path, fallback="document")
        chunks = self.chunker.chunk(content)
        chunk_records: list[ChunkRecord] = []
        for index, chunk in enumerate(chunks):
            chunk_records.append(
                ChunkRecord(
                    id=str(uuid4()),
                    content=chunk,
                    source_name=source_name,
                    metadata={**(metadata or {}), "chunk_index": index, "source": url or file_path or "text"},
                    published_at=published_at or utc_now(),
                )
            )
        embeddings = self.embedding_model.embed_texts([chunk.content for chunk in chunk_records])
        self.vector_store.upsert(chunk_records, embeddings)
        return IngestResponse(ingested_chunks=len(chunk_records), source_name=source_name)

    async def search(self, query: str, top_k: int | None = None, strategy: str = "improved") -> SearchResponse:
        top_k = top_k or settings.top_k_retrieval
        query_embedding = self.embedding_model.embed_query(query)
        retrieved_hits = self.vector_store.search(query_embedding, top_k=top_k)
        reranked_hits = self._select_ranker(strategy).rerank(query, retrieved_hits)
        answer, grounded = await self.generator.generate(query, reranked_hits[: settings.top_k_rerank])
        return SearchResponse(
            query=query,
            answer=answer,
            grounded=grounded,
            retrieval_precision_at_k=None,
            answer_relevance_score=None,
            hallucination_rate=None,
            results=reranked_hits[:top_k],
        )

    def _select_ranker(self, strategy: str):
        normalized = strategy.lower().strip()
        if normalized == "baseline":
            return self.semantic_ranker
        return self.heuristic_ranker

search_service = SearchService()
