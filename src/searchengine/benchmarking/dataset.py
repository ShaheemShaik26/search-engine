from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

from searchengine.schemas import EvaluationQuery, SearchHit


@dataclass(slots=True)
class BenchmarkCase:
    query: EvaluationQuery
    candidate_hits: list[SearchHit]


@dataclass(slots=True)
class BenchmarkDataset:
    cases: list[BenchmarkCase]

    @classmethod
    def from_json_file(cls, file_path: str | Path) -> "BenchmarkDataset":
        payload = json.loads(Path(file_path).read_text(encoding="utf-8"))
        cases: list[BenchmarkCase] = []
        for item in payload["cases"]:
            query = EvaluationQuery(**item["query"])
            candidate_hits = [SearchHit(**hit) for hit in item["candidate_hits"]]
            cases.append(BenchmarkCase(query=query, candidate_hits=candidate_hits))
        return cls(cases=cases)
