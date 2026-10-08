import pytest


def test_metrics_aggregate():
    from eval.metrics import aggregate_metrics

    evals = [
        {
            "completion": True,
            "tool_case": True,
            "tool_calls_correct": True,
            "arguments_correct": True,
            "clarification_expected": False,
            "clarification_correct": False,
            "latency_ms": 100,
            "tokens": 50,
            "actual_actions": ["a"],
            "hard_failure": False,
            "soft_failure": False,
            "cascading_soft_failure": False,
            "iterations": 1,
            "tool_calls": 1,
        },
        {
            "completion": False,
            "tool_case": True,
            "tool_calls_correct": False,
            "arguments_correct": False,
            "clarification_expected": False,
            "clarification_correct": False,
            "latency_ms": 200,
            "tokens": 50,
            "actual_actions": ["a", "b"],
            "hard_failure": False,
            "soft_failure": False,
            "cascading_soft_failure": False,
            "iterations": 2,
            "tool_calls": 2,
        },
    ]

    agg = aggregate_metrics(evals)

    assert agg["total_cases"] == 2

    assert agg["task_completion_rate"] == 0.5

    assert agg["tool_case_count"] == 2

    assert agg["tool_call_correctness"] == 0.5

    assert agg["argument_correctness"] == 0.5

    assert agg["clarification_case_count"] == 0

    assert agg["clarification_correctness"] is None

    assert agg["average_trajectory_length"] == 1.5

    assert agg["average_tool_calls"] == 1.5

    assert agg["average_tokens"] == 50

    assert agg["average_latency_ms"] == 150

    assert agg["hard_failures"] == 0

    assert agg["soft_failures"] == 0

    assert agg["cascading_soft_failures"] == 0


def test_metrics_separate_clarification_cases():

    from eval.metrics import aggregate_metrics

    evals = [
        {
            "completion": True,
            "tool_case": True,
            "tool_calls_correct": True,
            "arguments_correct": True,
            "clarification_expected": False,
            "clarification_correct": False,
            "latency_ms": 100,
            "tokens": 50,
            "actual_actions": ["get_order_status"],
            "hard_failure": False,
            "soft_failure": False,
            "cascading_soft_failure": False,
            "iterations": 2,
            "tool_calls": 1,
        },
        {
            "completion": True,
            "tool_case": False,
            "tool_calls_correct": False,
            "arguments_correct": False,
            "clarification_expected": True,
            "clarification_correct": True,
            "latency_ms": 80,
            "tokens": 40,
            "actual_actions": [],
            "hard_failure": False,
            "soft_failure": False,
            "cascading_soft_failure": False,
            "iterations": 1,
            "tool_calls": 0,
        },
    ]

    agg = aggregate_metrics(evals)

    assert agg["total_cases"] == 2

    assert agg["task_completion_rate"] == 1.0

    # Only the actual tool case is included.
    assert agg["tool_case_count"] == 1
    assert agg["tool_call_correctness"] == 1.0
    assert agg["argument_correctness"] == 1.0

    # Only the clarification case is included.
    assert agg["clarification_case_count"] == 1
    assert agg["clarification_correctness"] == 1.0


def test_metrics_handle_empty_evaluations():

    from eval.metrics import aggregate_metrics

    assert aggregate_metrics([]) == {}


def test_metrics_detect_cascading_soft_failure():

    from eval.metrics import aggregate_metrics

    evals = [
        {
            "completion": False,
            "tool_case": True,
            "tool_calls_correct": False,
            "arguments_correct": False,
            "clarification_expected": False,
            "clarification_correct": False,
            "latency_ms": 100,
            "tokens": 50,
            "actual_actions": [
                "a",
                "b",
            ],
            "hard_failure": False,
            "soft_failure": True,
            "cascading_soft_failure": True,
            "iterations": 3,
            "tool_calls": 2,
        },
    ]

    agg = aggregate_metrics(evals)

    assert agg["task_completion_rate"] == 0.0

    assert agg["hard_failures"] == 0

    assert agg["soft_failures"] == 1

    assert agg["cascading_soft_failures"] == 1