"""LLMClient wire format tests.

No network: we intercept ``_post_with_retry`` and assert the shape of the
payload + headers that the client sends, and the response the client
produces after parsing. The three providers differ enough that broken
wiring would silently select the wrong endpoint or drop the system
prompt — these tests catch that.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from lab.core.llm import LLMClient, LLMPermanentError, format_messages


def _client(provider: str) -> LLMClient:
    return LLMClient(
        base_url="https://example.invalid/v1",
        model="m",
        api_key="secret",
        provider=provider,
        retry_attempts=1,
    )


def test_openai_wire_sends_chat_completions(monkeypatch):
    c = _client("openai")
    captured = {}

    def fake(url, payload, headers):
        captured["url"] = url
        captured["payload"] = payload
        captured["headers"] = headers
        return {"choices": [{"message": {"content": "hi"}}]}

    monkeypatch.setattr(c, "_post_with_retry", fake)
    result = c.chat(format_messages("you are helpful", "hello"))
    assert result == "hi"
    assert captured["url"].endswith("/chat/completions")
    assert captured["headers"]["Authorization"] == "Bearer secret"
    # OpenAI keeps the system role inline in messages
    assert captured["payload"]["messages"][0]["role"] == "system"


def test_ollama_uses_same_openai_format(monkeypatch):
    c = _client("ollama")
    captured = {}
    monkeypatch.setattr(c, "_post_with_retry",
        lambda u, p, h: (captured.update({"url": u, "headers": h}),
                         {"choices": [{"message": {"content": "ok"}}]})[-1])
    c.chat(format_messages("sys", "user"))
    assert captured["url"].endswith("/chat/completions")


def test_anthropic_wire_extracts_system_and_uses_messages_endpoint(monkeypatch):
    c = _client("anthropic")
    captured = {}

    def fake(url, payload, headers):
        captured["url"] = url
        captured["payload"] = payload
        captured["headers"] = headers
        return {"content": [{"type": "text", "text": "hello from claude"}]}

    monkeypatch.setattr(c, "_post_with_retry", fake)
    result = c.chat(format_messages("you are helpful", "hello"))
    assert result == "hello from claude"
    assert captured["url"].endswith("/messages")
    assert captured["headers"]["x-api-key"] == "secret"
    assert captured["headers"]["anthropic-version"]
    # Crucially: the system message is hoisted out of the messages array.
    assert "system" in captured["payload"]
    assert captured["payload"]["system"] == "you are helpful"
    roles = {m["role"] for m in captured["payload"]["messages"]}
    assert roles == {"user"}


def test_anthropic_concatenates_multiple_system_messages(monkeypatch):
    c = _client("anthropic")
    captured = {}
    monkeypatch.setattr(c, "_post_with_retry",
        lambda u, p, h: (captured.update({"p": p}),
                         {"content": [{"type": "text", "text": "x"}]})[-1])
    c.chat([
        {"role": "system", "content": "first"},
        {"role": "system", "content": "second"},
        {"role": "user", "content": "go"},
    ])
    assert captured["p"]["system"] == "first\n\nsecond"


def test_anthropic_rejects_malformed_response(monkeypatch):
    c = _client("anthropic")
    monkeypatch.setattr(c, "_post_with_retry",
        lambda u, p, h: {"content": []})  # empty text blocks
    with pytest.raises(LLMPermanentError):
        c.chat(format_messages("s", "u"))


def test_openai_rejects_malformed_response(monkeypatch):
    c = _client("openai")
    monkeypatch.setattr(c, "_post_with_retry", lambda u, p, h: {"wrong": "shape"})
    with pytest.raises(LLMPermanentError):
        c.chat(format_messages("s", "u"))
