import pytest
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_harness_loads_dataset():
    dataset_path = ROOT / "eval" / "dataset.json"
    assert dataset_path.exists()
    with open(dataset_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    assert len(data) > 0
    assert "query" in data[0]

def test_metrics_aggregate():
    from eval.metrics import aggregate_metrics
    evals = [
        {"completion": True, "tool_calls_correct": True, "arguments_correct": True, "latency_ms": 100, "tokens": 50, "actual_actions": ["a"], "hard_failure": False, "soft_failure": False, "cascading_soft_failure": False, "iterations": 1, "tool_calls": 1},
        {"completion": False, "tool_calls_correct": False, "arguments_correct": False, "latency_ms": 200, "tokens": 50, "actual_actions": ["a", "b"], "hard_failure": False, "soft_failure": False, "cascading_soft_failure": False, "iterations": 2, "tool_calls": 2},
    ]
    agg = aggregate_metrics(evals)
    assert agg["task_completion_rate"] == 0.5
    assert agg["tool_call_correctness"] == 0.5
    assert agg["average_trajectory_length"] == 1.5
