from backend.llm.gemini_provider import (
    GeminiAuthenticationError,
    GeminiModelError,
    GeminiProvider,
    GeminiProviderError,
    GeminiProviderErrorResponse,
    GeminiQuotaError,
    GeminiTimeoutError,
)


def test_quota_error_is_classified():
    error = GeminiProvider._translate_exception(
        Exception(
            "429 RESOURCE_EXHAUSTED: "
            "Quota exceeded for metric: "
            "generate_content_free_tier_requests. "
            "Please retry in 9h40m7.032787019s."
        )
    )

    assert isinstance(
        error,
        GeminiQuotaError,
    )

    assert error.status_code == 429
    assert error.retry_after == "9h40m7"


def test_authentication_error_is_classified():
    error = GeminiProvider._translate_exception(
        Exception(
            "401 UNAUTHENTICATED: invalid API key"
        )
    )

    assert isinstance(
        error,
        GeminiAuthenticationError,
    )


def test_model_error_is_classified():
    error = GeminiProvider._translate_exception(
        Exception(
            "404 NOT_FOUND: "
            "model is no longer available"
        )
    )

    assert isinstance(
        error,
        GeminiModelError,
    )


def test_timeout_error_is_classified():
    error = GeminiProvider._translate_exception(
        TimeoutError(
            "request timed out"
        )
    )

    assert isinstance(
        error,
        GeminiTimeoutError,
    )


def test_server_error_is_classified():
    error = GeminiProvider._translate_exception(
        Exception(
            "500 INTERNAL SERVER ERROR"
        )
    )

    assert isinstance(
        error,
        GeminiProviderErrorResponse,
    )


def test_unknown_error_is_wrapped():
    error = GeminiProvider._translate_exception(
        Exception(
            "unexpected provider failure"
        )
    )

    assert isinstance(
        error,
        GeminiProviderError,
    )
