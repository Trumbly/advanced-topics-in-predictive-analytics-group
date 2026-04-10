"""Unit tests for `agent.llm_client.LLMClient`.

Uses a mock backend (no network) so these tests run in any environment.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent.llm_client import (
    LLMClient,
    LLMConnectionError,
    LLMResponseError,
)


# ---------------------------------------------------------------------------
# Mock backend
# ---------------------------------------------------------------------------


class MockBackend:
    """Canned-response backend for testing LLMClient without a real LLM."""

    def __init__(
        self,
        *,
        response_text: str = "OK",
        raise_on_call: Exception | None = None,
        raise_times: int = 0,
    ) -> None:
        self.response_text = response_text
        self.raise_on_call = raise_on_call
        self.raise_times = raise_times
        self.call_count = 0
        self.calls: list[dict[str, Any]] = []

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        timeout: float,
    ) -> Any:
        self.call_count += 1
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout": timeout,
            }
        )
        if self.raise_on_call and self.call_count <= self.raise_times:
            raise self.raise_on_call
        # Return a dict-shaped response (LLMClient handles both dict and SDK objects)
        return {
            "choices": [
                {"message": {"content": self.response_text, "role": "assistant"}}
            ]
        }


# ---------------------------------------------------------------------------
# Basic behavior
# ---------------------------------------------------------------------------


class TestLLMClientBasics:
    def test_chat_returns_content(self) -> None:
        backend = MockBackend(response_text="hello world")
        client = LLMClient(backend=backend, retry_backoff_seconds=0.0)
        result = client.chat([{"role": "user", "content": "hi"}])
        assert result == "hello world"
        assert backend.call_count == 1

    def test_chat_passes_model_params(self) -> None:
        backend = MockBackend()
        client = LLMClient(
            backend=backend,
            model="gemma4:e4b",
            temperature=0.5,
            max_tokens=2000,
            retry_backoff_seconds=0.0,
        )
        client.chat([{"role": "user", "content": "x"}])
        call = backend.calls[0]
        assert call["model"] == "gemma4:e4b"
        assert call["temperature"] == 0.5
        assert call["max_tokens"] == 2000

    def test_chat_overrides_model_per_call(self) -> None:
        backend = MockBackend()
        client = LLMClient(
            backend=backend, model="default", retry_backoff_seconds=0.0
        )
        client.chat(
            [{"role": "user", "content": "x"}],
            model="override",
            temperature=0.9,
        )
        assert backend.calls[0]["model"] == "override"
        assert backend.calls[0]["temperature"] == 0.9

    def test_empty_messages_raises(self) -> None:
        backend = MockBackend()
        client = LLMClient(backend=backend, retry_backoff_seconds=0.0)
        with pytest.raises(ValueError, match="empty"):
            client.chat([])


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------


class TestLLMClientRetries:
    def test_retries_then_succeeds(self) -> None:
        backend = MockBackend(
            response_text="finally",
            raise_on_call=ConnectionError("transient"),
            raise_times=2,
        )
        client = LLMClient(
            backend=backend,
            retry_attempts=3,
            retry_backoff_seconds=0.0,
        )
        result = client.chat([{"role": "user", "content": "x"}])
        assert result == "finally"
        assert backend.call_count == 3

    def test_retries_exhausted_raises_connection_error(self) -> None:
        backend = MockBackend(
            raise_on_call=ConnectionError("permanent"),
            raise_times=10,
        )
        client = LLMClient(
            backend=backend,
            retry_attempts=2,
            retry_backoff_seconds=0.0,
        )
        with pytest.raises(LLMConnectionError):
            client.chat([{"role": "user", "content": "x"}])
        assert backend.call_count == 2


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


class TestResponseParsing:
    def test_malformed_dict_response_raises(self) -> None:
        class BadBackend:
            def create(self, **kwargs: Any) -> Any:
                return {"broken": True}

        client = LLMClient(backend=BadBackend(), retry_backoff_seconds=0.0)
        with pytest.raises(LLMResponseError):
            client.chat([{"role": "user", "content": "x"}])

    def test_sdk_style_response(self) -> None:
        class Msg:
            def __init__(self, content: str) -> None:
                self.content = content

        class Choice:
            def __init__(self, msg: Msg) -> None:
                self.message = msg

        class Response:
            def __init__(self, content: str) -> None:
                self.choices = [Choice(Msg(content))]

        class SDKBackend:
            def create(self, **kwargs: Any) -> Any:
                return Response("sdk_shaped")

        client = LLMClient(backend=SDKBackend(), retry_backoff_seconds=0.0)
        assert client.chat([{"role": "user", "content": "x"}]) == "sdk_shaped"


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


class TestTokenCounting:
    def test_empty_string_is_zero(self) -> None:
        client = LLMClient(backend=MockBackend())
        assert client.count_tokens("") == 0

    def test_short_text_rounds_up_to_one(self) -> None:
        client = LLMClient(backend=MockBackend())
        assert client.count_tokens("abc") >= 1

    def test_roughly_four_chars_per_token(self) -> None:
        client = LLMClient(backend=MockBackend())
        text = "a" * 400
        # 400 / 4 = 100 tokens
        assert client.count_tokens(text) == 100
