from backend.agents.agent import AgentDecisionEngine
from backend.agents.executor import AgentActionExecutor
from backend.services.agentic_service import AgenticService
from backend.llm.gemini_provider import GeminiProvider


def create_agentic_service() -> AgenticService:
    """
    Build the production W16 agent dependency graph.

    Provider
        -> Decision Engine
        -> Action Executor
        -> Agentic Service
    """

    provider = GeminiProvider()

    decision_engine = AgentDecisionEngine(
        provider=provider,
    )

    executor = AgentActionExecutor()

    return AgenticService(
        decision_engine=decision_engine,
        executor=executor,
        provider=provider,
    )