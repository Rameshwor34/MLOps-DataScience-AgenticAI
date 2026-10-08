from backend.agents.observation_schemas import (
    validate_observation,
)


def test_valid_order_observation():
    observation = validate_observation(
        "get_order_status",
        {
            "success": True,
            "type": "order_status",
            "order_id": "ORD-1003",
            "status": "shipped",
        },
    )

    assert observation["success"] is True
    assert observation["order_id"] == "ORD-1003"
    assert observation["status"] == "shipped"


def test_invalid_order_observation():
    observation = validate_observation(
        "get_order_status",
        {
            "success": True,
            "type": "order_status",
            "order_id": "ORD-1003",
        },
    )

    assert observation["success"] is False
    assert observation["type"] == "malformed_observation"


def test_valid_product_observation():
    observation = validate_observation(
        "get_product_info",
        {
            "success": True,
            "type": "product_info",
            "product_id": "PROD-001",
            "name": "Laptop",
            "price": 1000,
            "description": "A laptop.",
        },
    )

    assert observation["success"] is True
    assert observation["product_id"] == "PROD-001"
    assert observation["price"] == 1000


def test_invalid_product_observation():
    observation = validate_observation(
        "get_product_info",
        {
            "success": True,
            "type": "product_info",
            "product_id": "PROD-001",
            "name": "Laptop",
            "price": "unknown",
            "description": "A laptop.",
        },
    )

    assert observation["success"] is False
    assert observation["type"] == "malformed_observation"


def test_valid_return_observation():
    observation = validate_observation(
        "check_return_eligibility",
        {
            "success": True,
            "type": "return_eligibility",
            "eligible": True,
            "reason": "Within return window.",
        },
    )

    assert observation["success"] is True
    assert observation["eligible"] is True


def test_invalid_return_observation():
    observation = validate_observation(
        "check_return_eligibility",
        {
            "success": True,
            "type": "return_eligibility",
            "eligible": "yes",
            "reason": "Within return window.",
        },
    )

    assert observation["success"] is False
    assert observation["type"] == "malformed_observation"


def test_failed_observation_is_preserved():
    observation = validate_observation(
        "get_order_status",
        {
            "success": False,
            "error": "Order service unavailable.",
        },
    )

    assert observation["success"] is False
    assert observation["error"] == "Order service unavailable."