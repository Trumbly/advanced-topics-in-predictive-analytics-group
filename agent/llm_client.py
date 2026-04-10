"""OpenAI-compatible LLM client for Ollama (or any other compatible provider).

Ollama exposes an OpenAI-compatible HTTP API at `http://localhost:11434/v1`,
which means we can use the `openai` Python SDK as a drop-in client. Swapping
between `gemma4:e4b`, `qwen3:9b`, `deepseek-r1:8b`, etc. requires only a
parameter change — the agent code stays the same.

This module is intentionally minimal:
- `LLMClient.chat(messages)` — returns the assistant text
- `LLMClient.count_tokens(text)` — rough estimate for context budgeting
- Built-in retry with exponential backoff on transient failures

Token counting note: Ollama does not expose a native tokenizer via the
OpenAI-compatible API. We approximate with `len(text) // 4`, which is
accurate to within ~20% for English/code on most tokenizers. The
ContextHandler uses this approximation to decide what to trim from the
memory; being approximate is fine because we leave a generous margin.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class LLMError(Exception):
    """Base class for all LLM client errors."""


class LLMConnectionError(LLMError):
    """The LLM server is unreachable."""


class LLMTimeoutError(LLMError):
    """The LLM call took longer than the configured timeout."""


class LLMResponseError(LLMError):
    """The LLM returned an error status or invalid response."""


# ---------------------------------------------------------------------------
# Backend protocol (for dependency injection in tests)
# ---------------------------------------------------------------------------


class ChatBackend(Protocol):
    """Minimal interface a backend must implement.

    The real implementation wraps `openai.OpenAI.chat.completions.create`.
    Tests inject a mock that returns canned responses.
    """

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        timeout: float,
    ) -> Any: ...


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


@dataclass
class LLMClient:
    """Thin wrapper around an OpenAI-compatible chat backend.

    The `backend` parameter lets tests inject a mock. In production, omit it
    and the client will lazily construct an `openai.OpenAI` instance pointed
    at Ollama.
    """

    base_url: str = "http://localhost:11434/v1"
    model: str = "gemma4:e4b"
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout_seconds: float = 120.0
    retry_attempts: int = 3
    retry_backoff_seconds: float = 5.0
    api_key: str = "ollama"  # Ollama ignores the key but the SDK requires one
    backend: ChatBackend | None = None

    def __post_init__(self) -> None:
        self._backend: ChatBackend | None = self.backend

    # -- backend bootstrap --------------------------------------------------

    def _get_backend(self) -> ChatBackend:
        """Return the backend, lazily constructing the real OpenAI client."""
        if self._backend is not None:
            return self._backend

        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError as e:
            raise LLMError(
                "The `openai` package is required. Install with "
                "`pip install openai>=1.0`."
            ) from e

        client = OpenAI(base_url=self.base_url, api_key=self.api_key)

        class _OpenAIAdapter:
            def create(
                self,
                *,
                model: str,
                messages: list[dict[str, str]],
                temperature: float,
                max_tokens: int,
                timeout: float,
            ) -> Any:
                return client.chat.completions.create(
                    model=model,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                )

        self._backend = _OpenAIAdapter()
        return self._backend

    # -- public API ---------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send a chat completion and return the assistant text.

        Retries on transient errors up to `retry_attempts` times with
        exponential backoff. Raises `LLMError` subclasses on permanent failure.
        """
        if not messages:
            raise ValueError("messages must not be empty")

        backend = self._get_backend()
        attempt_model = model or self.model
        attempt_temp = temperature if temperature is not None else self.temperature
        attempt_max_tokens = max_tokens or self.max_tokens

        last_exc: Exception | None = None
        for attempt in range(1, self.retry_attempts + 1):
            try:
                response = backend.create(
                    model=attempt_model,
                    messages=messages,
                    temperature=attempt_temp,
                    max_tokens=attempt_max_tokens,
                    timeout=self.timeout_seconds,
                )
                return self._extract_text(response)
            except LLMError:
                # Deliberately not retried — raise immediately
                raise
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt >= self.retry_attempts:
                    break
                delay = self.retry_backoff_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s. Retrying in %.1fs",
                    attempt,
                    self.retry_attempts,
                    exc,
                    delay,
                )
                time.sleep(delay)

        raise LLMConnectionError(
            f"LLM call failed after {self.retry_attempts} attempts: {last_exc}"
        ) from last_exc

    def count_tokens(self, text: str) -> int:
        """Rough token count for context window budgeting.

        Uses the industry-standard approximation of ~4 chars per token.
        This is within ~20% of true token counts for English/code on
        most tokenizers, which is enough precision for our use case
        (trimming experiment memory to fit a token budget).
        """
        if not text:
            return 0
        return max(1, len(text) // 4)

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Extract the assistant message from an OpenAI-shaped response.

        Works for both the real OpenAI SDK objects and the dict-shaped
        mock responses used in tests.
        """
        # Dict-shaped mock
        if isinstance(response, dict):
            try:
                return response["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as e:
                raise LLMResponseError(f"Malformed response: {response}") from e

        # OpenAI SDK object
        try:
            choice = response.choices[0]
            message = choice.message
            content = message.content
        except (AttributeError, IndexError) as e:
            raise LLMResponseError(f"Malformed response: {response}") from e

        if content is None:
            raise LLMResponseError("LLM returned empty content")
        return content


__all__ = [
    "LLMClient",
    "LLMError",
    "LLMConnectionError",
    "LLMTimeoutError",
    "LLMResponseError",
    "ChatBackend",
]
