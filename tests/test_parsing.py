"""I-06 part 1: proposal JSON parsing with one retry (ADR-018)."""

from __future__ import annotations

import json

import pytest

from lab.config import LLMConfig
from lab.core.llm import LLMClient
from lab.core.models import Proposal
from lab.core.parsing import ProposalParseError, parse_proposal


class _ScriptedPoster:
    def __init__(self, payloads: list[str]):
        self.calls = []
        self.payloads = list(payloads)

    def post(self, url, body, headers):
        self.calls.append(body)
        text = self.payloads.pop(0)
        return 200, json.dumps({"choices": [{"message": {"content": text}}]}).encode()


def _client(payloads):
    cfg = LLMConfig(
        provider="ollama",
        base_url="http://localhost:11434",
        model="m",
        temperature=0.0,
        max_tokens=64,
        retry_attempts=0,
        retry_backoff_seconds=0.0,
    )
    poster = _ScriptedPoster(payloads)
    return LLMClient(cfg, http=poster, sleep=lambda _: None), poster


_VALID = json.dumps(
    {
        "architecture_name": "EffNetB0",
        "family": "efficientnet_pretrained",
        "lr": 3e-4,
        "lr_schedule": "cosine",
        "epochs": 3,
    }
)


def test_first_try_valid():
    p = parse_proposal(_VALID)
    assert isinstance(p, Proposal)


def test_strips_code_fences():
    p = parse_proposal(f"```json\n{_VALID}\n```")
    assert p.architecture_name == "EffNetB0"


def test_retry_path_succeeds():
    client, poster = _client([_VALID])
    proposal = parse_proposal(
        "this is not JSON",
        retry_client=client,
        retry_messages=[{"role": "user", "content": "x"}],
    )
    assert proposal.epochs == 3
    assert len(poster.calls) == 1


def test_retry_exhausted_raises():
    client, _ = _client(["still not json"])
    with pytest.raises(ProposalParseError):
        parse_proposal(
            "broken",
            retry_client=client,
            retry_messages=[{"role": "user", "content": "x"}],
        )


def test_no_retry_client_raises_immediately():
    with pytest.raises(ProposalParseError):
        parse_proposal("nope")
