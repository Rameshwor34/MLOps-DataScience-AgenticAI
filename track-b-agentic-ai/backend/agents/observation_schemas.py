from typing import Any, Dict, List

from pydantic import BaseModel, Field, StrictBool


class OrderStatusObservation(BaseModel):
    order_id: str = Field(min_length=1)
    status: str = Field(min_length=1)


class ProductInfoObservation(BaseModel):
    product_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    price: float = Field(ge=0)
    description: str = Field(min_length=1)


class ReturnEligibilityObservation(BaseModel):
    eligible: StrictBool
    reason: str = Field(min_length=1)


class KnowledgeSearchObservation(BaseModel):
    query: str = Field(min_length=1)
    results: List[Dict[str, Any]]


OBSERVATION_SCHEMAS = {
    "get_order_status": OrderStatusObservation,
    "get_product_info": ProductInfoObservation,
    "check_return_eligibility": ReturnEligibilityObservation,
    "search_knowledge": KnowledgeSearchObservation,
}


def validate_observation(
    action: str,
    observation: Dict[str, Any],
) -> Dict[str, Any]:

    if not isinstance(observation, dict):
        return {
            "success": False,
            "type": "malformed_observation",
            "error": "Tool observation must be a dictionary.",
            "original_action": action,
        }

    if observation.get("success") is not True:
        return observation

    schema = OBSERVATION_SCHEMAS.get(action)

    if schema is None:
        return observation

    payload = {
        key: value
        for key, value in observation.items()
        if key not in {"success", "type"}
    }

    try:
        validated = schema.model_validate(payload)

        result = validated.model_dump()

        result["success"] = True

        if "type" in observation:
            result["type"] = observation["type"]

        return result

    except Exception as exc:
        return {
            "success": False,
            "type": "malformed_observation",
            "error": (
                f"Observation validation failed: "
                f"{type(exc).__name__}: {exc}"
            ),
            "original_action": action,
        }