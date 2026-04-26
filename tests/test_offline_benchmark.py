import asyncio
from pathlib import Path

from searchengine.benchmarking.dataset import BenchmarkDataset
from searchengine.benchmarking.offline_runner import OfflineBenchmarkRunner


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "ab_benchmark.json"


def test_offline_benchmark_runs_and_compares_strategies():
    dataset = BenchmarkDataset.from_json_file(FIXTURE_PATH)
    runner = OfflineBenchmarkRunner()

    result = asyncio.run(runner.run(dataset, strategies=["baseline", "improved"], top_k=2))

    assert [item.strategy for item in result.strategy_results] == ["baseline", "improved"]
    assert all(item.query_count == 3 for item in result.strategy_results)
    assert set(result.per_query_results) == {
        "What is the capital of France?",
        "Which planet is known as the Red Planet?",
        "What protocol is used for secure web browsing?",
    }
