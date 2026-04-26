from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class DocumentIngestRequest(BaseModel):
    text: str | None = None
    url: str | None = None
    file_path: str | None = None
    source_name: str | None = None
    published_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    strategy: str = "improved"


class EvaluationQuery(BaseModel):
    query: str
    expected_facts: list[str] = Field(default_factory=list)
    expected_answer: str | None = None


class EvaluationRequest(BaseModel):
    queries: list[EvaluationQuery]
    top_k: int = 5
    strategy: str = "improved"


class ABTestRequest(BaseModel):
    queries: list[EvaluationQuery]
    strategies: list[str] = Field(default_factory=lambda: ["baseline", "improved"])
    top_k: int = 5


class ChunkRecord(BaseModel):
    id: str
    content: str
    source_name: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    published_at: datetime | None = None


class SearchHit(BaseModel):
    id: str
    content: str
    source_name: str
    score: float
    rerank_score: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    published_at: datetime | None = None


class SearchResponse(BaseModel):
    query: str
    answer: str
    grounded: bool
    retrieval_precision_at_k: float | None = None
    answer_relevance_score: float | None = None
    hallucination_rate: float | None = None
    results: list[SearchHit]


class IngestResponse(BaseModel):
    ingested_chunks: int
    source_name: str


class EvaluationResult(BaseModel):
    query: str
    precision_at_k: float
    answer_relevance_score: float
    hallucination_rate: float


class ABStrategyResult(BaseModel):
    strategy: str
    average_precision_at_k: float
    average_answer_relevance_score: float
    average_hallucination_rate: float
    query_count: int


class ABTestResponse(BaseModel):
    strategies: list[ABStrategyResult]
