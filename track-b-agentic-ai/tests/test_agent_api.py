from fastapi.testclient import TestClient

from backend.main import app, agentic_service
from backend.llm.gemini_provider import (
    GeminiAuthenticationError,
    GeminiModelError,
    GeminiProviderError,
    GeminiQuotaError,
    GeminiTimeoutError,
)


client = TestClient(app)


def _assert_provider_error(
    monkeypatch,
    exception,
    expected_status,
    expected_error,
):
    def failing_process(_message):
        raise exception

    monkeypatch.setattr(
        agentic_service,
        "process",
        failing_process,
    )

    response = client.post(
        "/agent/chat",
        json={"message": "Test agent request"},
    )

    assert response.status_code == expected_status

    body = response.json()

    assert body["detail"]["error"] == expected_error
    assert "message" in body["detail"]


def test_agent_api_handles_quota_error(monkeypatch):
    _assert_provider_error(
        monkeypatch,
        GeminiQuotaError("quota exceeded"),
        429,
        "provider_rate_limited",
    )


def test_agent_api_handles_timeout_error(monkeypatch):
    _assert_provider_error(
        monkeypatch,
        GeminiTimeoutError("request timed out"),
        504,
        "provider_timeout",
    )


def test_agent_api_handles_authentication_error(monkeypatch):
    _assert_provider_error(
        monkeypatch,
        GeminiAuthenticationError("invalid credentials"),
        503,
        "provider_authentication_failed",
    )


def test_agent_api_handles_model_error(monkeypatch):
    _assert_provider_error(
        monkeypatch,
        GeminiModelError("model unavailable"),
        503,
        "provider_model_unavailable",
    )


def test_agent_api_handles_generic_provider_error(monkeypatch):
    _assert_provider_error(
        monkeypatch,
        GeminiProviderError("provider unavailable"),
        503,
        "provider_unavailable",
    )


def test_agent_api_handles_unexpected_error(monkeypatch):
    def failing_process(_message):
        raise RuntimeError("unexpected internal failure")

    monkeypatch.setattr(
        agentic_service,
        "process",
        failing_process,
    )

    response = client.post(
        "/agent/chat",
        json={"message": "Test agent request"},
    )

    assert response.status_code == 500

    body = response.json()

    assert body["detail"]["error"] == "agent_generation_failed"
    assert (
        body["detail"]["message"]
        == "The agent could not safely complete this request."
    )
def test_agent_api_success_contract(monkeypatch):
    expected = {
        "answer": "Your order ORD-1003 is delivered.",
        "status": "completed",
        "iterations": 2,
        "tool_calls": 1,
        "decision_steps": 2,
        "total_tokens": 120,
        "total_latency_ms": 15.5,
        "sources": ["orders.json"],
        "trajectory": [],
        "evidence": [],
        "agentic": True,
    }

    def successful_process(_message):
        return dict(expected)

    monkeypatch.setattr(
        agentic_service,
        "process",
        successful_process,
    )

    response = client.post(
        "/agent/chat",
        json={"message": "Where is my order ORD-1003?"},
    )

    assert response.status_code == 200

    body = response.json()

    assert body["agentic"] is True
    assert body["answer"] == expected["answer"]
    assert body["status"] == expected["status"]
    assert body["iterations"] == expected["iterations"]
    assert body["tool_calls"] == expected["tool_calls"]
    assert body["decision_steps"] == expected["decision_steps"]
    assert body["sources"] == expected["sources"]
