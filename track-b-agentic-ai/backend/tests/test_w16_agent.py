from backend.agents.agent import AgentDecisionEngine
from backend.agents.executor import AgentActionExecutor
from backend.services.agentic_service import AgenticService
from backend.tests.mock_provider import MockAgentProvider


def build_service(
    decisions,
    final_answer="Mock verified answer.",
    executor=None,
):
    provider = MockAgentProvider(
        decisions=decisions,
        final_answer=final_answer,
    )

    engine = AgentDecisionEngine(
        provider=provider,
    )

    service = AgenticService(
        decision_engine=engine,
        executor=(
            executor
            if executor is not None
            else AgentActionExecutor()
        ),
        provider=provider,
    )

    return service


def test_single_step_agentic_loop_completes():

    service = build_service(
        decisions=[
            {
                "action": "get_order_status",
                "arguments": {
                    "order_id": "ORD-1003",
                },
                "reason": (
                    "The order status is required."
                ),
                "confidence": 0.99,
            },
            {
                "action": "final_answer",
                "arguments": {},
                "reason": (
                    "The order status is now verified."
                ),
                "confidence": 0.99,
            },
        ],
        final_answer=(
            "Order ORD-1003 has been verified."
        ),
    )

    result = service.process(
        "What is the status of ORD-1003?"
    )

    assert result["status"] == "completed"
    assert result["iterations"] == 2
    assert result["tool_calls"] == 1
    assert result["answer"] == (
        "Order ORD-1003 has been verified."
    )


def test_multi_step_agentic_loop_replans_after_observation():

    service = build_service(
        decisions=[
            {
                "action": "check_return_eligibility",
                "arguments": {
                    "order_id": "ORD-1003",
                },
                "reason": (
                    "Order-specific return eligibility "
                    "must be checked."
                ),
                "confidence": 0.98,
            },
            {
                "action": "search_knowledge",
                "arguments": {
                    "query": (
                        "return policy eligibility "
                        "and return requirements"
                    ),
                },
                "reason": (
                    "The customer also requested "
                    "the documented return policy."
                ),
                "confidence": 0.97,
            },
            {
                "action": "final_answer",
                "arguments": {},
                "reason": (
                    "Both requested evidence sources "
                    "are available."
                ),
                "confidence": 0.98,
            },
        ],
        final_answer=(
            "The order-specific eligibility and "
            "return policy have been verified."
        ),
    )

    result = service.process(
        "Can I return ORD-1003, and what does "
        "the return policy say?"
    )

    assert result["status"] == "completed"
    assert result["iterations"] == 3
    assert result["tool_calls"] == 2

    assert [
        step["action"]
        for step in result["trajectory"]
        if step["type"] == "tool_execution"
    ] == [
        "check_return_eligibility",
        "search_knowledge",
    ]


def test_missing_information_requests_clarification():

    service = build_service(
        decisions=[
            {
                "action": "ask_clarification",
                "arguments": {
                    "question": (
                        "Could you provide the order ID "
                        "so I can check its status?"
                    ),
                },
                "reason": (
                    "An order ID is required."
                ),
                "confidence": 0.99,
            },
        ],
    )

    result = service.process(
        "Can you check the status of my order?"
    )

    assert result["status"] == (
        "clarification_required"
    )

    assert result["iterations"] == 1
    assert result["tool_calls"] == 0

    assert "order ID" in result["answer"]


def test_product_lookup_completes():

    service = build_service(
        decisions=[
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
                    "The product information is available."
                ),
                "confidence": 0.99,
            },
        ],
        final_answer=(
            "Product PROD-001 information "
            "has been verified."
        ),
    )

    result = service.process(
        "Tell me about product PROD-001."
    )

    assert result["status"] == "completed"
    assert result["iterations"] == 2
    assert result["tool_calls"] == 1


def test_return_eligibility_completes():

    service = build_service(
        decisions=[
            {
                "action": "check_return_eligibility",
                "arguments": {
                    "order_id": "ORD-1003",
                },
                "reason": (
                    "Return eligibility must be checked."
                ),
                "confidence": 0.99,
            },
            {
                "action": "final_answer",
                "arguments": {},
                "reason": (
                    "Return eligibility is available."
                ),
                "confidence": 0.99,
            },
        ],
        final_answer=(
            "The return eligibility has been verified."
        ),
    )

    result = service.process(
        "Can I return ORD-1003?"
    )

    assert result["status"] == "completed"
    assert result["iterations"] == 2
    assert result["tool_calls"] == 1


def test_agent_records_decision_trajectory():

    service = build_service(
        decisions=[
            {
                "action": "get_order_status",
                "arguments": {
                    "order_id": "ORD-1003",
                },
                "reason": (
                    "The order status is required."
                ),
                "confidence": 0.99,
            },
            {
                "action": "final_answer",
                "arguments": {},
                "reason": (
                    "Sufficient evidence is available."
                ),
                "confidence": 0.99,
            },
        ],
    )

    result = service.process(
        "Check ORD-1003."
    )

    decisions = [
        item
        for item in result["trajectory"]
        if item["type"] == "decision"
    ]

    assert len(decisions) == 2

    assert decisions[0]["action"] == (
        "get_order_status"
    )

    assert decisions[1]["action"] == (
        "final_answer"
    )


def test_agent_records_token_usage():

    service = build_service(
        decisions=[
            {
                "action": "get_order_status",
                "arguments": {
                    "order_id": "ORD-1003",
                },
                "reason": (
                    "The order status is required."
                ),
                "confidence": 0.99,
            },
            {
                "action": "final_answer",
                "arguments": {},
                "reason": (
                    "Sufficient evidence is available."
                ),
                "confidence": 0.99,
            },
        ],
    )

    result = service.process(
        "Check ORD-1003."
    )

    assert result["total_tokens"] > 0

    tokenized_steps = [
        item
        for item in result["trajectory"]
        if "tokens" in item
    ]

    assert tokenized_steps


def test_agent_respects_bounded_loop():

    decisions = [
        {
            "action": "search_knowledge",
            "arguments": {
                "query": "return policy",
            },
            "reason": (
                "Additional policy evidence is required."
            ),
            "confidence": 0.50,
        }
        for _ in range(10)
    ]

    service = build_service(
        decisions=decisions,
    )

    result = service.process(
        "Tell me everything about returns."
    )

    assert result["iterations"] <= 6

    assert result["status"] == (
        "max_iterations"
    )


def test_malformed_order_observation_is_rejected():

    malformed = {
        "success": True,
        "unexpected": "data",
    }

    result = AgentActionExecutor._validate_observation(
        action="get_order_status",
        observation=malformed,
    )

    assert result["success"] is False

    assert result["type"] == (
        "malformed_observation"
    )

    assert "order_id" in result["error"]

    assert "status" in result["error"]


def test_malformed_product_observation_is_rejected():

    malformed = {
        "success": True,
        "malformed": True,
    }

    result = AgentActionExecutor._validate_observation(
        action="get_product_info",
        observation=malformed,
    )

    assert result["success"] is False

    assert result["type"] == (
        "malformed_observation"
    )

    assert "product_id" in result["error"]


def test_valid_failed_observation_is_preserved():

    failed = {
        "success": False,
        "error": "Service unavailable.",
    }

    result = AgentActionExecutor._validate_observation(
        action="get_order_status",
        observation=failed,
    )

    assert result == failed
def test_agent_stops_at_max_iterations():
    from backend.agents.agent import AgentDecisionEngine
    from backend.agents.executor import AgentActionExecutor
    from backend.services.agentic_service import (
        AgenticService,
        MAX_ITERATIONS,
    )
    from backend.tests.mock_provider import MockAgentProvider

    decisions = [
        {
            "action": "get_order_status",
            "arguments": {"order_id": "ORD-1003"},
            "reason": "Need the current order status.",
            "confidence": 0.99,
        }
        for _ in range(MAX_ITERATIONS)
    ]

    provider = MockAgentProvider(
        decisions=decisions,
        final_answer="",
    )

    decision_engine = AgentDecisionEngine(
        provider=provider,
    )

    executor = AgentActionExecutor()

    service = AgenticService(
        decision_engine=decision_engine,
        executor=executor,
        provider=provider,
    )

    result = service.process(
        "Keep checking the order status."
    )

    assert result["status"] == "max_iterations"
    assert result["iterations"] == MAX_ITERATIONS
    assert result["tool_calls"] == MAX_ITERATIONS
    assert result["decision_steps"] == MAX_ITERATIONS
    assert result["answer"] == (
        "I could not safely complete "
        "the request within the allowed "
        "number of reasoning steps."
    )
