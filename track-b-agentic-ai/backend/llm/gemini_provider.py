import json
import os
import re
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(override=True)


class GeminiProviderError(RuntimeError):
    """Base error for controlled Gemini provider failures."""


class GeminiQuotaError(GeminiProviderError):
    """Raised when Gemini quota or rate limits are exhausted."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        retry_after: Optional[str] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


class GeminiAuthenticationError(GeminiProviderError):
    """Raised when Gemini authentication or authorization fails."""


class GeminiModelError(GeminiProviderError):
    """Raised when the configured Gemini model is unavailable."""


class GeminiTimeoutError(GeminiProviderError):
    """Raised when a Gemini request exceeds the configured timeout."""


class GeminiProviderErrorResponse(GeminiProviderError):
    """Raised for other controlled Gemini API failures."""


class GeminiProvider:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not configured"
            )

        self.model = os.getenv(
            "GEMINI_MODEL",
            "gemini-3.8-flash",
        )

        self.timeout_ms = int(
            os.getenv(
                "GEMINI_TIMEOUT_MS",
                "30000",
            )
        )

        # Disable automatic SDK retries.
        #
        # This is important for an agentic system because a quota
        # failure should fail fast instead of causing the SDK to
        # repeatedly retry the same request.
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=self.timeout_ms,
                retry_options=types.HttpRetryOptions(
                    attempts=1,
                ),
            ),
        )

        self.last_usage: Dict[str, Any] = {}

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        top_p: float = 0.9,
        tools: Optional[list] = None,
    ) -> Dict[str, Any]:

        try:
            interaction = self.client.interactions.create(
                model=self.model,
                input=user_prompt,
                system_instruction=system_prompt,
                generation_config={
                    "temperature": temperature,
                    "top_p": top_p,
                },
            )

            output_text = getattr(
                interaction,
                "output_text",
                None,
            )

            if not output_text:
                raise GeminiProviderError(
                    "Gemini returned no output_text."
                )

            output_text = output_text.strip()

            usage = self._extract_usage(
                interaction,
                system_prompt,
                user_prompt,
                output_text,
            )

            self.last_usage = usage

            try:
                result = json.loads(output_text)

            except json.JSONDecodeError:
                result = {
                    "intent": "general_support",
                    "answer": output_text,
                    "confidence": 0.8,
                    "sources": [],
                    "tool_used": "none",
                }

            result["_usage"] = usage

            return result

        except Exception as exc:
            raise self._translate_exception(exc) from exc

    @staticmethod
    def _translate_exception(
        exc: Exception,
    ) -> Exception:
        """
        Convert raw google-genai exceptions into predictable
        provider-level exceptions.

        The Google SDK exception structure can vary between
        versions, so both HTTP status/code and message text
        are inspected.
        """

        status_code = getattr(
            exc,
            "status_code",
            None,
        )

        if status_code is None:
            status_code = getattr(
                exc,
                "code",
                None,
            )

        try:
            if status_code is not None:
                status_code = int(status_code)
        except (TypeError, ValueError):
            status_code = None

        message = str(exc)
        message_lower = message.lower()

        # ---------------------------------------------------------
        # 429 / RESOURCE_EXHAUSTED
        # ---------------------------------------------------------
        if (
            status_code == 429
            or "resource_exhausted" in message_lower
            or "quota exceeded" in message_lower
            or "rate limit" in message_lower
            or "too many requests" in message_lower
        ):
            retry_after = GeminiProvider._extract_retry_after(
                message
            )

            return GeminiQuotaError(
                "Gemini quota or rate limit was exhausted. "
                "No automatic retry was performed.",
                status_code=429,
                retry_after=retry_after,
            )

        # ---------------------------------------------------------
        # 401 / 403 authentication and authorization
        # ---------------------------------------------------------
        if (
            status_code in {401, 403}
            or "unauthenticated" in message_lower
            or "permission denied" in message_lower
            or "invalid api key" in message_lower
            or "api key" in message_lower
        ):
            return GeminiAuthenticationError(
                "Gemini authentication or authorization failed."
            )

        # ---------------------------------------------------------
        # 404 / unavailable model
        # ---------------------------------------------------------
        #
        # A 404 from the model endpoint is treated as a model
        # configuration/availability problem.
        #
        # We also recognize common model-related error messages
        # even when an explicit HTTP status is unavailable.
        # ---------------------------------------------------------
        if (
            status_code == 404
            or (
                "model" in message_lower
                and (
                    "not found" in message_lower
                    or "not available" in message_lower
                    or "no longer available" in message_lower
                    or "unsupported" in message_lower
                )
            )
        ):
            return GeminiModelError(
                "The configured Gemini model is unavailable: "
                f"{message}"
            )

        # ---------------------------------------------------------
        # Timeout
        # ---------------------------------------------------------
        if (
            isinstance(exc, TimeoutError)
            or "timed out" in message_lower
            or "timeout" in message_lower
            or "readtimeout" in message_lower
            or "connecttimeout" in message_lower
        ):
            return GeminiTimeoutError(
                "Gemini request timed out."
            )

        # ---------------------------------------------------------
        # Other HTTP 5xx errors
        #
        # Some SDK exceptions expose the HTTP status through
        # status_code/code, while others only include it in
        # the exception message.
        message_status_match = re.search(
            r"\b5\d{2}\b",
            message_lower,
        )

        message_status_code = (
            int(message_status_match.group())
            if message_status_match
            else None
        )

        if (
            (
                status_code is not None
                and status_code >= 500
            )
            or (
                message_status_code is not None
                and message_status_code >= 500
            )
        ):
            resolved_status = (
                status_code
                if status_code is not None
                else message_status_code
            )

            return GeminiProviderErrorResponse(
                f"Gemini server error ({resolved_status})."
            )

        # Unknown provider failure
        # ---------------------------------------------------------
        return GeminiProviderError(
            "Gemini request failed: "
            f"{type(exc).__name__}: {message}"
        )

    @staticmethod
    def _extract_retry_after(
        message: str,
    ) -> Optional[str]:
        """
        Extract a human-readable retry duration when Gemini
        includes one.

        Example:
            Please retry in 9h40m7.032787019s.
        """

        marker = "retry in "

        lower_message = message.lower()
        index = lower_message.find(marker)

        if index == -1:
            return None

        value = message[
            index + len(marker):
        ]

        # Remove the fractional seconds portion.
        #
        # Example:
        # 9h40m7.032787019s
        # becomes:
        # 9h40m7
        value = value.split(".", 1)[0].strip()

        # Remove a trailing period if present.
        value = value.rstrip(".")

        return value or None

    def _extract_usage(
        self,
        interaction: Any,
        system_prompt: str,
        user_prompt: str,
        output_text: str,
    ) -> Dict[str, Any]:

        usage = getattr(
            interaction,
            "usage",
            None,
        )

        if usage is not None:

            input_tokens = self._first_int(
                usage,
                [
                    "input_tokens",
                    "prompt_tokens",
                    "input_token_count",
                ],
            )

            output_tokens = self._first_int(
                usage,
                [
                    "output_tokens",
                    "completion_tokens",
                    "output_token_count",
                ],
            )

            total_tokens = self._first_int(
                usage,
                [
                    "total_tokens",
                    "total_token_count",
                ],
            )

            if total_tokens is None:
                if (
                    input_tokens is not None
                    and output_tokens is not None
                ):
                    total_tokens = (
                        input_tokens
                        + output_tokens
                    )

            if total_tokens is not None:

                return {
                    "input_tokens": input_tokens or 0,
                    "output_tokens": output_tokens or 0,
                    "total_tokens": total_tokens,
                    "estimated": False,
                }

        return self._estimate_tokens(
            system_prompt,
            user_prompt,
            output_text,
        )

    @staticmethod
    def _first_int(
        obj: Any,
        names: list,
    ) -> Optional[int]:

        for name in names:
            value = getattr(
                obj,
                name,
                None,
            )

            if isinstance(value, int):
                return value

        return None

    @staticmethod
    def _estimate_tokens(
        system_prompt: str,
        user_prompt: str,
        output_text: str,
    ) -> Dict[str, Any]:

        input_chars = len(
            system_prompt
            + "\n"
            + user_prompt
        )

        output_chars = len(
            output_text
        )

        input_tokens = max(
            1,
            (input_chars + 3) // 4,
        )

        output_tokens = max(
            1,
            (output_chars + 3) // 4,
        )

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": (
                input_tokens
                + output_tokens
            ),
            "estimated": True,
        }

    def _format_timeout(self) -> str:
        return (
            f"{self.timeout_ms / 1000:.1f} seconds"
        )
