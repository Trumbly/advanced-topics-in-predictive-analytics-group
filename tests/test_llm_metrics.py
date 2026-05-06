"""LLM generation metrics: capture in client + aggregate per study/model."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from lab.config import LLMConfig
from lab.core.llm import LLMCallStats, LLMClient
from lab.core.llm_metrics import (
    aggregate_by_model,
    experiment_summary,
    iter_call_stats,
    study_summary,
    summarise,
)
from lab.core.models import Experiment, Study, Task


# ---------- LLMClient stats capture ----------


class _FakePoster:
    """Stand-in HTTP poster that returns a hand-crafted response payload."""

    def __init__(self, body: dict, status: int = 200, sleep: float = 0.0):
        self._body = body
        self._status = status
        self._sleep = sleep

    def post(self, url, body, headers):
        if self._sleep:
            import time

            time.sleep(self._sleep)
        return self._status, json.dumps(self._body).encode("utf-8")


def _client(provider: str = "ollama", model: str = "fake:1b") -> LLMClient:
    cfg = LLMConfig(
        provider=provider,
        base_url="http://localhost:11434",
        model=model,
        temperature=0.2,
        max_tokens=128,
        retry_attempts=0,
        retry_backoff_seconds=0.0,
    )
    return LLMClient(cfg)


def _study_with(experiments: list[Experiment]) -> Study:
    return Study(
        id="study_x",
        task_name="track_b",
        status="COMPLETED",
        personality="exploratory",
        agent_memory_enabled=False,
        experiments=experiments,
        created_at=datetime(2026, 5, 6, tzinfo=timezone.utc),
    )


def test_openai_compat_response_parses_token_counts():
    body = {
        "choices": [{"message": {"content": "hi"}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 7},
    }
    client = _client()
    client.http = _FakePoster(body, sleep=0.05)

    out = client.chat([{"role": "user", "content": "x"}])
    assert out == "hi"
    s = client.last_stats
    assert isinstance(s, LLMCallStats)
    assert s.prompt_tokens == 12
    assert s.completion_tokens == 7
    assert s.total_seconds > 0
    assert s.tps is not None
    assert s.ttft_seconds is None


def test_ollama_prompt_eval_duration_becomes_ttft():
    body = {
        "choices": [{"message": {"content": "hi"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        "prompt_eval_duration": 250_000_000,  # 250 ms in ns
    }
    client = _client()
    client.http = _FakePoster(body, sleep=0.0)

    client.chat([{"role": "user", "content": "x"}])
    assert client.last_stats.ttft_seconds == pytest.approx(0.25, rel=1e-3)


def test_anthropic_response_parses_input_output_tokens():
    body = {
        "content": [{"type": "text", "text": "hi"}],
        "usage": {"input_tokens": 4, "output_tokens": 2},
    }
    client = _client(provider="anthropic", model="claude-fake")
    client.http = _FakePoster(body, sleep=0.01)

    client.chat([{"role": "user", "content": "x"}])
    s = client.last_stats
    assert s.prompt_tokens == 4
    assert s.completion_tokens == 2
    assert s.provider == "anthropic"
    assert s.model == "claude-fake"


def test_zero_completion_tokens_yields_none_tps():
    body = {
        "choices": [{"message": {"content": ""}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 0},
    }
    client = _client()
    client.http = _FakePoster(body, sleep=0.01)
    client.chat([{"role": "user", "content": "x"}])
    assert client.last_stats.tps is None


def test_last_stats_starts_none():
    client = _client()
    assert client.last_stats is None


# ---------- iter_call_stats ----------


def _stats_dict(model: str = "m1", tps: float = 10.0, ttft: float | None = 0.1) -> dict:
    return {
        "provider": "ollama",
        "model": model,
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_seconds": 5.0,
        "tps": tps,
        "ttft_seconds": ttft,
    }


def _exp_with_stats(stat_dicts: list[dict]) -> Experiment:
    return Experiment(
        id="exp_test",
        index=0,
        status="JUDGED",
        primary_metric="roc_auc_macro",
        tasks=[
            Task(
                name="propose",
                status="SUCCEEDED",
                input={},
                output={"llm_stats": s},
            )
            for s in stat_dicts
        ],
    )


def test_iter_call_stats_skips_tasks_without_llm_stats():
    exp = Experiment(
        id="exp_test",
        index=0,
        status="JUDGED",
        primary_metric="roc_auc_macro",
        tasks=[
            Task(name="propose", status="SUCCEEDED", input={}, output={}),
            Task(
                name="generate",
                status="SUCCEEDED",
                input={},
                output={"llm_stats": _stats_dict()},
            ),
        ],
    )
    stats = list(iter_call_stats(exp))
    assert len(stats) == 1
    assert stats[0]["model"] == "m1"


# ---------- summarise ----------


def test_summarise_computes_min_max_avg():
    summary = summarise([
        _stats_dict(tps=10.0, ttft=0.1),
        _stats_dict(tps=20.0, ttft=0.3),
        _stats_dict(tps=15.0, ttft=0.2),
    ])
    assert summary.n_calls == 3
    assert summary.tps_min == 10.0
    assert summary.tps_max == 20.0
    assert summary.tps_avg == pytest.approx(15.0)
    assert summary.ttft_min == pytest.approx(0.1)
    assert summary.ttft_max == pytest.approx(0.3)
    assert summary.total_completion_tokens == 150
    assert summary.total_prompt_tokens == 300


def test_summarise_handles_missing_ttft():
    summary = summarise([
        _stats_dict(tps=10.0, ttft=None),
        _stats_dict(tps=20.0, ttft=None),
    ])
    assert summary.tps_avg == pytest.approx(15.0)
    assert summary.ttft_avg is None
    assert summary.ttft_min is None


def test_summarise_empty_returns_zero_calls():
    s = summarise([])
    assert s.n_calls == 0
    assert s.tps_avg is None
    assert s.ttft_avg is None
    assert s.total_completion_tokens == 0


# ---------- experiment + study summaries ----------


def test_experiment_summary_walks_tasks():
    exp = _exp_with_stats([
        _stats_dict(tps=10.0),
        _stats_dict(tps=20.0),
    ])
    s = experiment_summary(exp)
    assert s.n_calls == 2
    assert s.tps_avg == pytest.approx(15.0)


def test_study_summary_aggregates_across_experiments():
    study = _study_with([
        _exp_with_stats([_stats_dict(tps=10.0)]),
        _exp_with_stats([_stats_dict(tps=30.0), _stats_dict(tps=20.0)]),
    ])
    s = study_summary(study)
    assert s.n_calls == 3
    assert s.tps_min == 10.0
    assert s.tps_max == 30.0


# ---------- aggregate_by_model ----------


def test_aggregate_by_model_groups_by_model_field():
    s_a = _stats_dict(model="ollama-fast", tps=50.0, ttft=0.05)
    s_b = _stats_dict(model="ollama-slow", tps=10.0, ttft=0.5)
    s_c = _stats_dict(model="ollama-fast", tps=60.0, ttft=0.04)

    study = _study_with([_exp_with_stats([s_a, s_b, s_c])])
    result = aggregate_by_model([study])
    by_name = {m.model: m.summary for m in result}
    assert set(by_name) == {"ollama-fast", "ollama-slow"}
    assert by_name["ollama-fast"].n_calls == 2
    assert by_name["ollama-fast"].tps_min == 50.0
    assert by_name["ollama-fast"].tps_max == 60.0
    assert by_name["ollama-slow"].n_calls == 1


def test_aggregate_by_model_sorts_by_call_count_desc():
    s = _study_with([
        _exp_with_stats([
            _stats_dict(model="big-model"),
            _stats_dict(model="small-model"),
            _stats_dict(model="small-model"),
            _stats_dict(model="small-model"),
        ])
    ])
    result = aggregate_by_model([s])
    assert [m.model for m in result] == ["small-model", "big-model"]
