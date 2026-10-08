from typing import Any, Dict, List


def action_sequence(result: Dict[str, Any]) -> List[str]:
    """
    Return only actions that resulted in actual tool execution.

    `final_answer` and `ask_clarification` are agent decisions,
    but they are not counted as tool executions.
    """
    return [
        item["action"]
        for item in result.get("trajectory", [])
        if item.get("type") == "tool_execution"
    ]


def argument_sequence(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Return arguments associated with actual tool executions.
    """
    return [
        item.get("arguments", {})
        for item in result.get("trajectory", [])
        if item.get("type") == "tool_execution"
    ]


def decision_sequence(result: Dict[str, Any]) -> List[str]:
    """
    Return every agent-selected action, including final_answer
    and ask_clarification.
    """
    return [
        item["action"]
        for item in result.get("trajectory", [])
        if item.get("type") == "decision"
    ]


def is_clarification_case(case: Dict[str, Any]) -> bool:
    return case.get("expected_status") == "clarification_required"


def expected_tool_case(case: Dict[str, Any]) -> bool:
    """
    A tool case is any case where the expected terminal behavior
    is not clarification_required.
    """
    return not is_clarification_case(case)


def evaluate_case(
    case: Dict[str, Any],
    result: Dict[str, Any],
) -> Dict[str, Any]:

    expected_status = case["expected_status"]

    expected_actions = case.get(
        "expected_actions",
        [],
    )

    expected_arguments = case.get(
        "expected_arguments",
        [],
    )

    actual_actions = action_sequence(result)

    actual_arguments = argument_sequence(result)

    actual_decisions = decision_sequence(result)

    status_correct = (
        result.get("status") == expected_status
    )

    actions_correct = (
        actual_actions == expected_actions
    )

    arguments_correct = (
        actual_arguments == expected_arguments
    )

    clarification_expected = (
        expected_status == "clarification_required"
    )

    clarification_correct = (
        clarification_expected
        and result.get("status")
        == "clarification_required"
        and "ask_clarification"
        in actual_decisions
    )

    completion = (
        status_correct
        and (
            actions_correct
            or (
                clarification_expected
                and clarification_correct
            )
        )
        and (
            arguments_correct
            or clarification_expected
        )
    )

    iterations = result.get(
        "iterations",
        0,
    )

    max_iterations = case.get(
        "max_iterations",
        6,
    )

    hard_failure = (
        result.get("status") == "failed"
        or iterations > max_iterations
    )

    soft_failure = (
        not hard_failure
        and not completion
        and result.get("status")
        in {
            "completed",
            "clarification_required",
        }
    )

    cascading_soft_failure = (
        soft_failure
        and len(actual_actions)
        > len(expected_actions)
    )

    return {
        "id": case["id"],
        "category": case["category"],

        "completion": completion,

        "status_correct": status_correct,

        "tool_case": expected_tool_case(case),

        "tool_calls_correct": actions_correct,
        "arguments_correct": arguments_correct,

        "clarification_expected": clarification_expected,
        "clarification_correct": clarification_correct,

        "expected_actions": expected_actions,
        "actual_actions": actual_actions,

        "expected_arguments": expected_arguments,
        "actual_arguments": actual_arguments,

        "decision_sequence": actual_decisions,

        "status": result.get("status"),

        "iterations": iterations,

        "tool_calls": result.get(
            "tool_calls",
            0,
        ),

        "tokens": result.get(
            "total_tokens",
            0,
        ),

        "latency_ms": result.get(
            "total_latency_ms",
            0,
        ),

        "hard_failure": hard_failure,
        "soft_failure": soft_failure,

        "cascading_soft_failure": (
            cascading_soft_failure
        ),
    }


def aggregate_metrics(
    evaluations: List[Dict[str, Any]],
) -> Dict[str, Any]:

    total = len(evaluations)

    if total == 0:
        return {}

    completed = sum(
        bool(item["completion"])
        for item in evaluations
    )

    tool_cases = [
        item
        for item in evaluations
        if item.get("tool_case")
    ]

    clarification_cases = [
        item
        for item in evaluations
        if item.get("clarification_expected")
    ]

    tool_case_count = len(tool_cases)

    clarification_case_count = len(
        clarification_cases
    )

    tool_correct = sum(
        bool(item["tool_calls_correct"])
        for item in tool_cases
    )

    argument_correct = sum(
        bool(item["arguments_correct"])
        for item in tool_cases
    )

    clarification_correct = sum(
        bool(item["clarification_correct"])
        for item in clarification_cases
    )

    hard_failures = sum(
        bool(item["hard_failure"])
        for item in evaluations
    )

    soft_failures = sum(
        bool(item["soft_failure"])
        for item in evaluations
    )

    cascading = sum(
        bool(item["cascading_soft_failure"])
        for item in evaluations
    )

    return {
        "total_cases": total,

        "task_completion_rate": round(
            completed / total,
            4,
        ),

        "tool_case_count": tool_case_count,

        "tool_call_correctness": (
            round(
                tool_correct / tool_case_count,
                4,
            )
            if tool_case_count
            else None
        ),

        "argument_correctness": (
            round(
                argument_correct / tool_case_count,
                4,
            )
            if tool_case_count
            else None
        ),

        "clarification_case_count": (
            clarification_case_count
        ),

        "clarification_correctness": (
            round(
                clarification_correct
                / clarification_case_count,
                4,
            )
            if clarification_case_count
            else None
        ),

        "average_trajectory_length": round(
            sum(
                item["iterations"]
                for item in evaluations
            )
            / total,
            4,
        ),

        "average_tool_calls": round(
            sum(
                item["tool_calls"]
                for item in evaluations
            )
            / total,
            4,
        ),

        "average_tokens": round(
            sum(
                item["tokens"]
                for item in evaluations
            )
            / total,
            4,
        ),

        "average_latency_ms": round(
            sum(
                item["latency_ms"]
                for item in evaluations
            )
            / total,
            4,
        ),

        "hard_failures": hard_failures,
        "soft_failures": soft_failures,
        "cascading_soft_failures": cascading,
    }