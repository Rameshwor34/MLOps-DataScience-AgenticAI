import json
from pathlib import Path
import sys
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from backend.agents.agent import AgentDecisionEngine
from backend.agents.executor import AgentActionExecutor
from backend.services.agentic_service import AgenticService
from backend.tests.mock_provider import MockAgentProvider


RESULTS_PATH = (
    ROOT
    / "eval"
    / "failure_results.json"
)


class UnavailableToolExecutor(AgentActionExecutor):
    """
    Simulates an unavailable transactional service.
    """

    def _get_order_status(
        self,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:

        return {
            "success": False,
            "type": "order_status",
            "error": (
                "Injected failure: order service unavailable."
            ),
        }


class MalformedToolExecutor(AgentActionExecutor):
    """
    Simulates a tool returning an unexpected payload.

    Important:
    The tool reports success=True, but the payload is malformed.
    This is intentionally different from an infrastructure failure.
    """

    def _get_product_info(
        self,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:

        return {
            "success": True,
            "malformed": True,
            "unexpected": object(),
        }


class RetrievalFailureExecutor(AgentActionExecutor):
    """
    Simulates a knowledge-retrieval infrastructure failure.
    """

    def _search_knowledge(
        self,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:

        return {
            "success": False,
            "type": "knowledge_search",
            "error": (
                "Injected failure: retrieval service timeout."
            ),
        }


def get_tool_executions(
    result: Dict[str, Any],
):
    """
    Extract actual tool execution events from the trajectory.
    """

    return [
        item
        for item in result.get(
            "trajectory",
            [],
        )
        if item.get("type") == "tool_execution"
    ]


def validate_infrastructure_failure(
    result: Dict[str, Any],
) -> None:
    """
    Validate failures where the underlying service explicitly
    reports success=False.
    """

    status = result.get("status")

    assert status == "completed", (
        f"Unexpected status: {status!r}"
    )

    answer = (
        result.get("answer")
        or ""
    ).strip()

    assert answer, (
        "Failure case returned an empty answer."
    )

    tool_executions = get_tool_executions(
        result
    )

    assert tool_executions, (
        "Expected at least one tool execution."
    )

    failed_tools = [
        item
        for item in tool_executions
        if item.get("success") is False
    ]

    assert failed_tools, (
        "Expected the injected infrastructure "
        "failure to appear as success=False."
    )

    validate_no_unsupported_success_claim(
        answer
    )


def validate_malformed_response(
    result: Dict[str, Any],
) -> None:
    """
    Validate the malformed-response scenario.

    The current W16 controller does not yet perform full schema
    validation at the executor boundary. Therefore this test does
    NOT claim that malformed=True was automatically detected.

    Instead, it verifies the safety property that matters at the
    final response boundary: malformed evidence must not result in
    an unsupported successful product claim.
    """

    status = result.get("status")

    assert status == "completed", (
        f"Unexpected status: {status!r}"
    )

    answer = (
        result.get("answer")
        or ""
    ).strip()

    assert answer, (
        "Malformed-response case returned an empty answer."
    )

    tool_executions = get_tool_executions(
        result
    )

    assert tool_executions, (
        "Expected at least one tool execution."
    )

    # The injected malformed executor intentionally returns
    # success=True. Therefore success=False would actually
    # indicate that the test setup behaved unexpectedly.
    assert tool_executions[0].get(
        "success"
    ) is True, (
        "Malformed response test expected the injected "
        "tool to report success=True."
    )

    validate_no_unsupported_success_claim(
        answer
    )


def validate_no_unsupported_success_claim(
    answer: str,
) -> None:
    """
    Ensure a failed or malformed evidence path does not produce
    an unsupported successful customer-facing claim.
    """

    normalized_answer = answer.lower()

    unsupported_success_phrases = [
        "successfully verified",
        "verified successfully",
        "confirmed successfully",
        "is delivered",
        "is available",
    ]

    for phrase in unsupported_success_phrases:
        assert phrase not in normalized_answer, (
            "Failure case appears to make an unsupported "
            f"successful claim: {phrase!r}"
        )


def run_failure_case(
    name: str,
    query: str,
    decisions,
    executor: AgentActionExecutor,
    final_answer: str,
    validator,
) -> Dict[str, Any]:

    provider = MockAgentProvider(
        decisions=decisions,
        final_answer=final_answer,
    )

    engine = AgentDecisionEngine(
        provider=provider,
    )

    service = AgenticService(
        decision_engine=engine,
        executor=executor,
        provider=provider,
    )

    result = service.process(
        query,
    )

    validator(
        result
    )

    return {
        "name": name,
        "query": query,
        "status": result.get(
            "status",
        ),
        "answer": result.get(
            "answer",
        ),
        "trajectory": result.get(
            "trajectory",
            [],
        ),
        "evidence": result.get(
            "evidence",
            [],
        ),
        "tokens": result.get(
            "total_tokens",
            0,
        ),
        "latency_ms": result.get(
            "total_latency_ms",
            0,
        ),
        "safe_failure": True,
    }


def main():

    print("=" * 70)
    print("W16 FAILURE INJECTION TESTS")
    print("=" * 70)

    test_cases = [
        {
            "name": "tool_unavailable",
            "query": (
                "What is the status of order ORD-1003?"
            ),
            "decisions": [
                {
                    "action": "get_order_status",
                    "arguments": {
                        "order_id": "ORD-1003",
                    },
                    "reason": (
                        "The order status must be retrieved."
                    ),
                    "confidence": 0.99,
                },
                {
                    "action": "final_answer",
                    "arguments": {},
                    "reason": (
                        "The order service failed, so "
                        "the response must acknowledge "
                        "that limitation."
                    ),
                    "confidence": 0.99,
                },
            ],
            "executor": UnavailableToolExecutor(),
            "final_answer": (
                "I could not verify the order status "
                "because the order service is unavailable."
            ),
            "validator": validate_infrastructure_failure,
        },
        {
            "name": "malformed_tool_response",
            "query": (
                "Give me information about product PROD-001."
            ),
            "decisions": [
                {
                    "action": "get_product_info",
                    "arguments": {
                        "product_id": "PROD-001",
                    },
                    "reason": (
                        "Product information is required."
                    ),
                    "confidence": 0.99,
                },
                {
                    "action": "final_answer",
                    "arguments": {},
                    "reason": (
                        "The product response is not "
                        "safe to use as verified evidence."
                    ),
                    "confidence": 0.99,
                },
            ],
            "executor": MalformedToolExecutor(),
            "final_answer": (
                "I could not safely verify the "
                "product information."
            ),
            "validator": validate_malformed_response,
        },
        {
            "name": "retrieval_timeout",
            "query": (
                "What is the return policy?"
            ),
            "decisions": [
                {
                    "action": "search_knowledge",
                    "arguments": {
                        "query": "return policy",
                    },
                    "reason": (
                        "The documented return policy "
                        "must be retrieved."
                    ),
                    "confidence": 0.99,
                },
                {
                    "action": "final_answer",
                    "arguments": {},
                    "reason": (
                        "Knowledge retrieval failed, "
                        "so the policy cannot be verified."
                    ),
                    "confidence": 0.99,
                },
            ],
            "executor": RetrievalFailureExecutor(),
            "final_answer": (
                "I could not verify the return policy "
                "because the knowledge retrieval "
                "service timed out."
            ),
            "validator": validate_infrastructure_failure,
        },
    ]

    results = []

    for case in test_cases:

        print()
        print(
            f"TEST: {case['name']}"
        )

        try:

            result = run_failure_case(
                name=case["name"],
                query=case["query"],
                decisions=case["decisions"],
                executor=case["executor"],
                final_answer=case["final_answer"],
                validator=case["validator"],
            )

            results.append(
                result
            )

            print(
                "STATUS:",
                result["status"],
            )

            print(
                "ANSWER:",
                result["answer"],
            )

            print(
                "TOKENS:",
                result["tokens"],
            )

            print(
                "LATENCY:",
                result["latency_ms"],
            )

            print(
                "SAFE FAILURE:",
                result["safe_failure"],
            )

        except AssertionError as exc:

            print(
                "ASSERTION FAILED:",
                exc,
            )

            raise

        except Exception as exc:

            print(
                "HARNESS ERROR:",
                type(exc).__name__,
                str(exc),
            )

            raise

    output = {
        "evaluation_type": (
            "failure_injection_safety"
        ),
        "provider": "MockAgentProvider",
        "total_cases": len(results),
        "all_safe": all(
            item["safe_failure"]
            for item in results
        ),
        "cases": results,
    }

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            default=str,
        )

    print()
    print("=" * 70)
    print("FAILURE INJECTION SUMMARY")
    print("=" * 70)

    print(
        "total_cases:",
        output["total_cases"],
    )

    print(
        "all_safe:",
        output["all_safe"],
    )

    print()
    print(
        f"Results written to: {RESULTS_PATH}"
    )


if __name__ == "__main__":
    main()