from __future__ import annotations

from fastapi import APIRouter, HTTPException

from searchengine.ab_testing.runner import ABTestRunner
from searchengine.api.service import search_service
from searchengine.evaluation.runner import EvaluationRunner
from searchengine.schemas import ABTestRequest, DocumentIngestRequest, EvaluationRequest, IngestResponse, SearchRequest, SearchResponse

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest(request: DocumentIngestRequest):
    try:
        return await search_service.ingest_document(
            text=request.text,
            url=request.url,
            file_path=request.file_path,
            source_name=request.source_name,
            published_at=request.published_at,
            metadata=request.metadata,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    return await search_service.search(request.query, top_k=request.top_k, strategy=request.strategy)


@router.post("/evaluate")
async def evaluate(request: EvaluationRequest):
    runner = EvaluationRunner()
    results = []
    for query in request.queries:
        response = await search_service.search(query.query, top_k=request.top_k, strategy=request.strategy)
        results.append(runner.evaluate_query(query, response.results, response.answer))
    return {"results": results, "aggregate": runner.aggregate(request.strategy, results)}


@router.post("/ab-test")
async def ab_test(request: ABTestRequest):
    runner = ABTestRunner(search_service)
    results = await runner.run(request.queries, request.strategies, request.top_k)
    return {"strategies": results}


@router.get("/health")
async def health():
    return {"status": "ok"}
