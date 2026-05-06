"""LLMClient must snapshot the messages + response of the latest call so the
orchestrator can attach a visible-in-UI 'what the LLM saw / said' record to
the surrounding Task. The capture is opt-in (None until the first call) and
truncates over 8 KB of content per message + response so a long skeleton
excerpt cannot bloat study.json."""

from __future__ import annotations

import json

from lab.config import LLMConfig
from lab.core.llm import LLMClient


class _FakePoster:
    def __init__(self, body: dict):
        self._body = body

    def post(self, url, body, headers):
        return 200, json.dumps(self._body).encode("utf-8")


def _client() -> LLMClient:
    cfg = LLMConfig(
        provider="ollama",
        base_url="http://localhost:11434",
        model="fake:1b",
        temperature=0.2,
        max_tokens=128,
        retry_attempts=0,
        retry_backoff_seconds=0.0,
    )
    return LLMClient(cfg)


def test_last_messages_and_response_start_none():
    c = _client()
    assert c.last_messages is None
    assert c.last_response is None


def test_last_messages_capture_verbatim():
    c = _client()
    c.http = _FakePoster(
        {"choices": [{"message": {"content": "hi back"}}], "usage": {}}
    )
    msgs = [
        {"role": "system", "content": "you are an agent"},
        {"role": "user", "content": "ping"},
    ]
    out = c.chat(msgs)
    assert out == "hi back"
    assert c.last_messages == msgs
    assert c.last_response == "hi back"


def test_last_messages_resets_on_each_call():
    c = _client()
    c.http = _FakePoster(
        {"choices": [{"message": {"content": "first"}}], "usage": {}}
    )
    c.chat([{"role": "user", "content": "one"}])

    c.http = _FakePoster(
        {"choices": [{"message": {"content": "second"}}], "usage": {}}
    )
    c.chat([{"role": "user", "content": "two"}])
    assert c.last_messages == [{"role": "user", "content": "two"}]
    assert c.last_response == "second"


def test_io_dict_truncates_long_content_per_message():
    """``_io_dict`` is the helper the orchestrator uses to stash the
    in/out on the Task. A 12 KB system prompt becomes 8 KB + a clear
    "[truncated …]" marker; nothing in study.json blows past the cap."""
    from dataclasses import dataclass
    from lab.core.experiment import _LLM_IO_MAX_CHARS, _io_dict

    @dataclass
    class _StubCtx:
        client: object

    big = "x" * (_LLM_IO_MAX_CHARS + 4_000)
    client = _client()
    client.last_messages = [
        {"role": "system", "content": "short"},
        {"role": "user", "content": big},
    ]
    client.last_response = big

    out = _io_dict(_StubCtx(client=client))
    assert out is not None
    assert out["messages"][0]["content"] == "short"
    assert "[truncated " in out["messages"][1]["content"]
    assert len(out["messages"][1]["content"]) < len(big)
    assert "[truncated " in out["response"]


def test_io_dict_returns_none_when_no_call_yet():
    from dataclasses import dataclass
    from lab.core.experiment import _io_dict

    @dataclass
    class _StubCtx:
        client: object

    assert _io_dict(_StubCtx(client=_client())) is None
