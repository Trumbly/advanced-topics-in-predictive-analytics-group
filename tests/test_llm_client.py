"""I-03 acceptance: LLMClient provider switch + retry semantics."""

from __future__ import annotations

import json
import logging
from typing import Callable

import pytest

from lab.config import LLMConfig
from lab.core.llm import (
    LLMClient,
    LLMPermanentError,
    LLMTransientError,
)


class FakePoster:
    """Programmable HTTPPoster: scripted response queue per call."""

    def __init__(self, scripted: list[tuple[int, dict | bytes]]):
        self.calls: list[tuple[str, bytes, dict[str, str]]] = []
        self._scripted = list(scripted)

    def post(self, url: str, body: bytes, headers: dict[str, str]):
        self.calls.append((url, body, dict(headers)))
        status, payload = self._scripted.pop(0)
        if isinstance(payload, dict):
            payload = json.dumps(payload).encode("utf-8")
        return status, payload


def _ollama_cfg() -> LLMConfig:
    return LLMConfig(
        provider="ollama",
        base_url="http://localhost:11434",
        model="gemma4:e4b",
        temperature=0.0,
        max_tokens=128,
        retry_attempts=2,
        retry_backoff_seconds=0.0,
    )


def _openai_cfg() -> LLMConfig:
    return LLMConfig(
        provider="openai",
        base_url="https://api.openai.com",
        model="gpt-4",
        temperature=0.0,
        max_tokens=128,
        retry_attempts=2,
        retry_backoff_seconds=0.0,
        api_key="sk-test",
    )


def _anthropic_cfg() -> LLMConfig:
    return LLMConfig(
        provider="anthropic",
        base_url="https://api.anthropic.com",
        model="claude-opus-4",
        temperature=0.0,
        max_tokens=128,
        retry_attempts=2,
        retry_backoff_seconds=0.0,
        api_key="ant-test",
    )


_OPENAI_OK = {"choices": [{"message": {"content": "hi"}}]}
_ANTHROPIC_OK = {"content": [{"type": "text", "text": "hi"}]}


# -------- success first try --------

def test_first_call_succeeds_with_ollama():
    poster = FakePoster([(200, _OPENAI_OK)])
    client = LLMClient(_ollama_cfg(), http=poster, sleep=lambda _: None)
    out = client.chat([{"role": "user", "content": "hello"}])
    assert out == "hi"
    assert len(poster.calls) == 1
    url, body, headers = poster.calls[0]
    assert url.endswith("/v1/chat/completions")
    assert "Authorization" not in headers  # ollama has no api key
    payload = json.loads(body)
    assert payload["model"] == "gemma4:e4b"
    assert payload["messages"][0]["content"] == "hello"


# -------- retry-then-succeed --------

def test_500_then_200_retries_and_succeeds():
    sleeps: list[float] = []
    poster = FakePoster([(500, b"oops"), (200, _OPENAI_OK)])
    client = LLMClient(_ollama_cfg(), http=poster, sleep=sleeps.append)
    out = client.chat([{"role": "user", "content": "x"}])
    assert out == "hi"
    assert len(poster.calls) == 2
    assert sleeps == [0.0]  # one backoff between attempts


# -------- permanent fail on 4xx (non-429) --------

def test_400_is_permanent_no_retry():
    poster = FakePoster([(400, b"bad request")])
    client = LLMClient(_ollama_cfg(), http=poster, sleep=lambda _: None)
    with pytest.raises(LLMPermanentError):
        client.chat([{"role": "user", "content": "x"}])
    assert len(poster.calls) == 1


def test_401_is_permanent():
    poster = FakePoster([(401, b"forbidden")])
    client = LLMClient(_openai_cfg(), http=poster, sleep=lambda _: None)
    with pytest.raises(LLMPermanentError):
        client.chat([{"role": "user", "content": "x"}])


# -------- 429 is transient --------

def test_429_is_transient_and_retries():
    poster = FakePoster([(429, b""), (429, b""), (200, _OPENAI_OK)])
    client = LLMClient(_ollama_cfg(), http=poster, sleep=lambda _: None)
    assert client.chat([{"role": "user", "content": "x"}]) == "hi"
    assert len(poster.calls) == 3


# -------- exhausted retries --------

def test_transient_exhausted_raises():
    poster = FakePoster([(503, b"down"), (503, b"down"), (503, b"down")])
    client = LLMClient(_ollama_cfg(), http=poster, sleep=lambda _: None)
    with pytest.raises(LLMTransientError):
        client.chat([{"role": "user", "content": "x"}])
    assert len(poster.calls) == 3  # retry_attempts=2 → 3 total


# -------- network error path --------

def test_network_error_propagates_after_retries():
    class FailingPoster:
        def __init__(self):
            self.calls = 0

        def post(self, url, body, headers):
            self.calls += 1
            raise LLMTransientError("DNS down")

    poster = FailingPoster()
    client = LLMClient(_ollama_cfg(), http=poster, sleep=lambda _: None)
    with pytest.raises(LLMTransientError):
        client.chat([{"role": "user", "content": "x"}])
    assert poster.calls == 3


# -------- provider switch --------

def test_provider_switch_openai_url_and_auth():
    poster = FakePoster([(200, _OPENAI_OK)])
    client = LLMClient(_openai_cfg(), http=poster, sleep=lambda _: None)
    client.chat([{"role": "user", "content": "x"}])
    url, _, headers = poster.calls[0]
    assert url == "https://api.openai.com/v1/chat/completions"
    assert headers["Authorization"] == "Bearer sk-test"


def test_provider_switch_anthropic_url_extract_system():
    poster = FakePoster([(200, _ANTHROPIC_OK)])
    client = LLMClient(_anthropic_cfg(), http=poster, sleep=lambda _: None)
    out = client.chat(
        [
            {"role": "system", "content": "You are X."},
            {"role": "user", "content": "hello"},
        ]
    )
    assert out == "hi"
    url, body, headers = poster.calls[0]
    assert url == "https://api.anthropic.com/v1/messages"
    assert headers["x-api-key"] == "ant-test"
    assert headers["anthropic-version"] == "2023-06-01"
    payload = json.loads(body)
    assert payload["system"] == "You are X."
    assert payload["messages"] == [{"role": "user", "content": "hello"}]


# -------- response parsing failures --------

def test_invalid_json_is_permanent():
    poster = FakePoster([(200, b"not-json")])
    client = LLMClient(_ollama_cfg(), http=poster, sleep=lambda _: None)
    with pytest.raises(LLMPermanentError):
        client.chat([{"role": "user", "content": "x"}])


def test_unexpected_shape_is_permanent():
    poster = FakePoster([(200, {"unexpected": "shape"})])
    client = LLMClient(_ollama_cfg(), http=poster, sleep=lambda _: None)
    with pytest.raises(LLMPermanentError):
        client.chat([{"role": "user", "content": "x"}])


# -------- structured log emitted --------

def test_structured_log_event_emitted(caplog):
    poster = FakePoster([(200, _OPENAI_OK)])
    client = LLMClient(_ollama_cfg(), http=poster, sleep=lambda _: None)
    with caplog.at_level(logging.INFO, logger="lab.llm"):
        client.chat([{"role": "user", "content": "x"}])

    matched = [r for r in caplog.records if getattr(r, "event", None) == "llm_call"]
    assert matched, "expected at least one llm_call log record"
    rec = matched[0]
    assert rec.provider == "ollama"
    assert isinstance(rec.duration_ms, int)
    assert rec.status == 200
