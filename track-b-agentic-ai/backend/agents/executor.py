from typing import Any, Dict

from backend.tools.orders import get_order_status
from backend.tools.products import get_product_info
from backend.tools.returns import check_return_eligibility


class AgentActionExecutor:
    """
    Executes validated W16 agent actions.

    The executor is responsible for:
      1. validating action arguments,
      2. calling the underlying application tool,
      3. validating the returned observation,
      4. converting malformed observations into safe failures.

    Heavy RAG dependencies are imported lazily only when the agent
    actually chooses search_knowledge.
    """

    def execute(
        self,
        action: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:

        if not isinstance(arguments, dict):
            return {
                "success": False,
                "error": (
                    "Action arguments must be a dictionary."
                ),
            }

        if action == "search_knowledge":
            return self._search_knowledge(arguments)

        if action == "get_order_status":
            return self._get_order_status(arguments)

        if action == "get_product_info":
            return self._get_product_info(arguments)

        if action == "check_return_eligibility":
            return self._check_return_eligibility(arguments)

        if action == "ask_clarification":
            return {
                "success": True,
                "type": "clarification",
                "message": arguments.get(
                    "question",
                    "Could you provide more information so I can help?",
                ),
            }

        if action == "final_answer":
            return {
                "success": True,
                "type": "final_answer",
            }

        raise ValueError(
            f"Unsupported action: {action}"
        )

    def _search_knowledge(
        self,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:

        query = arguments.get("query")

        if not query:
            return {
                "success": False,
                "error": (
                    "Knowledge search requires a query."
                ),
            }

        if not isinstance(query, str):
            return {
                "success": False,
                "error": (
                    "Knowledge search query must be a string."
                ),
            }

        try:
            from backend.rag.retrieval import retrieve

            results = retrieve(
                query,
                top_k=3,
            )

            observation = {
                "success": True,
                "type": "knowledge_search",
                "query": query,
                "results": results,
            }

            return self._validate_observation(
                action="search_knowledge",
                observation=observation,
            )

        except Exception as exc:
            return {
                "success": False,
                "type": "knowledge_search",
                "error": (
                    f"Knowledge search failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
            }

    def _get_order_status(
        self,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:

        order_id = arguments.get("order_id")

        if not order_id:
            return {
                "success": False,
                "error": (
                    "Order status requires an order_id."
                ),
            }

        if not isinstance(order_id, str):
            return {
                "success": False,
                "error": (
                    "Order ID must be a string."
                ),
            }

        try:
            observation = get_order_status(
                order_id
            )

            return self._validate_observation(
                action="get_order_status",
                observation=observation,
            )

        except Exception as exc:
            return {
                "success": False,
                "error": (
                    f"Order lookup failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
            }

    def _get_product_info(
        self,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:

        product_id = arguments.get("product_id")

        if not product_id:
            return {
                "success": False,
                "error": (
                    "Product lookup requires a product_id."
                ),
            }

        if not isinstance(product_id, str):
            return {
                "success": False,
                "error": (
                    "Product ID must be a string."
                ),
            }

        try:
            observation = get_product_info(
                product_id
            )

            return self._validate_observation(
                action="get_product_info",
                observation=observation,
            )

        except Exception as exc:
            return {
                "success": False,
                "error": (
                    f"Product lookup failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
            }

    def _check_return_eligibility(
        self,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:

        order_id = arguments.get("order_id")

        if not order_id:
            return {
                "success": False,
                "error": (
                    "Return eligibility requires an order_id."
                ),
            }

        if not isinstance(order_id, str):
            return {
                "success": False,
                "error": (
                    "Order ID must be a string."
                ),
            }

        try:
            observation = check_return_eligibility(
                order_id
            )

            return self._validate_observation(
                action="check_return_eligibility",
                observation=observation,
            )

        except Exception as exc:
            return {
                "success": False,
                "error": (
                    f"Return eligibility check failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
            }

    @staticmethod
    def _validate_observation(
        action: str,
        observation: Any,
    ) -> Dict[str, Any]:
        """
        Validate the structural contract of a tool observation.

        The LLM must never receive an apparently successful
        observation that does not contain the fields required
        to interpret it safely.
        """

        if not isinstance(observation, dict):
            return {
                "success": False,
                "error": (
                    f"Malformed observation from "
                    f"{action}: expected a dictionary."
                ),
            }

        if not isinstance(
            observation.get("success"),
            bool,
        ):
            return {
                "success": False,
                "error": (
                    f"Malformed observation from "
                    f"{action}: 'success' must be boolean."
                ),
            }

        # An explicit application-level failure is already a
        # valid observation. Preserve it without further schema
        # requirements.
        if observation["success"] is False:
            return observation

        validators = {
            "get_order_status": (
                AgentActionExecutor._validate_order_status
            ),
            "get_product_info": (
                AgentActionExecutor._validate_product_info
            ),
            "check_return_eligibility": (
                AgentActionExecutor._validate_return_eligibility
            ),
            "search_knowledge": (
                AgentActionExecutor._validate_knowledge_search
            ),
        }

        validator = validators.get(action)

        if validator is None:
            return observation

        validation_error = validator(
            observation
        )

        if validation_error is not None:
            return {
                "success": False,
                "type": "malformed_observation",
                "error": validation_error,
                "original_action": action,
            }

        return observation

    @staticmethod
    def _validate_order_status(
        observation: Dict[str, Any],
    ) -> str | None:
        """
        Expected successful order-status observation.

        Required:
          success
          order_id
          status
        """

        required = [
            "order_id",
            "status",
        ]

        missing = [
            key
            for key in required
            if key not in observation
        ]

        if missing:
            return (
                "Malformed get_order_status observation: "
                f"missing required field(s): {missing}."
            )

        if not isinstance(
            observation["order_id"],
            str,
        ):
            return (
                "Malformed get_order_status observation: "
                "'order_id' must be a string."
            )

        if not isinstance(
            observation["status"],
            str,
        ):
            return (
                "Malformed get_order_status observation: "
                "'status' must be a string."
            )

        return None

    @staticmethod
    def _validate_product_info(
        observation: Dict[str, Any],
    ) -> str | None:
        """
        Expected successful product observation.

        Required:
          success
          product_id
          name
          price
          description
        """

        required = [
            "product_id",
            "name",
            "price",
            "description",
        ]

        missing = [
            key
            for key in required
            if key not in observation
        ]

        if missing:
            return (
                "Malformed get_product_info observation: "
                f"missing required field(s): {missing}."
            )

        if not isinstance(
            observation["product_id"],
            str,
        ):
            return (
                "Malformed get_product_info observation: "
                "'product_id' must be a string."
            )

        if not isinstance(
            observation["name"],
            str,
        ):
            return (
                "Malformed get_product_info observation: "
                "'name' must be a string."
            )

        if not isinstance(
            observation["description"],
            str,
        ):
            return (
                "Malformed get_product_info observation: "
                "'description' must be a string."
            )

        if not isinstance(
            observation["price"],
            (int, float),
        ):
            return (
                "Malformed get_product_info observation: "
                "'price' must be numeric."
            )

        return None

    @staticmethod
    def _validate_return_eligibility(
        observation: Dict[str, Any],
    ) -> str | None:
        """
        Expected successful return-eligibility observation.

        Required:
          success
          eligible
          reason
        """

        required = [
            "eligible",
            "reason",
        ]

        missing = [
            key
            for key in required
            if key not in observation
        ]

        if missing:
            return (
                "Malformed check_return_eligibility "
                "observation: "
                f"missing required field(s): {missing}."
            )

        if not isinstance(
            observation["eligible"],
            bool,
        ):
            return (
                "Malformed check_return_eligibility "
                "observation: 'eligible' must be boolean."
            )

        if not isinstance(
            observation["reason"],
            str,
        ):
            return (
                "Malformed check_return_eligibility "
                "observation: 'reason' must be a string."
            )

        return None

    @staticmethod
    def _validate_knowledge_search(
        observation: Dict[str, Any],
    ) -> str | None:
        """
        Expected successful knowledge-search observation.

        Required:
          success
          type
          query
          results
        """

        required = [
            "query",
            "results",
        ]

        missing = [
            key
            for key in required
            if key not in observation
        ]

        if missing:
            return (
                "Malformed search_knowledge observation: "
                f"missing required field(s): {missing}."
            )

        if not isinstance(
            observation["query"],
            str,
        ):
            return (
                "Malformed search_knowledge observation: "
                "'query' must be a string."
            )

        if not isinstance(
            observation["results"],
            list,
        ):
            return (
                "Malformed search_knowledge observation: "
                "'results' must be a list."
            )

        return None