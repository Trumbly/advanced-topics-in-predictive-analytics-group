"""I-19 acceptance: experiment + study judge with strict JSON parsing."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from lab.config import LLMConfig, load_settings
from lab.core.judge import Judge, JudgeError
from lab.core.llm import LLMClient
from lab.core.memory import Memory
from lab.core.models import Experiment, Proposal, Study, Task, TaskError, Verdict
from lab.prompts.engine import PromptEngine
from lab.prompts.registry import PromptRegistry

REPO_ROOT = Path(__file__).resolve().parent.parent
SHIPPED_PROMPTS = REPO_ROOT / "config" / "prompts"


class _ScriptedPoster:
    def __init__(self, scripted: list[str]):
        self.calls: list = []
        self._scripted = list(scripted)

    def post(self, url, body, headers):
        self.calls.append((url, body))
        text = self._scripted.pop(0)
        return 200, json.dumps(
            {"choices": [{"message": {"content": text}}]}
        ).encode("utf-8")


def _proposal() -> Proposal:
    return Proposal(
        architecture_name="EffNetB0",
        family="efficientnet_pretrained",
        lr=3e-4,
        lr_schedule="cosine",
        epochs=3,
    )


def _exp(score: float = 0.55) -> Experiment:
    return Experiment(
        id="exp_0001",
        index=0,
        status="JUDGED",
        proposal=_proposal(),
        primary_metric="roc_auc_macro",
        primary_score=score,
        history=[
            {"epoch": 1, "loss": 0.7, "roc_auc_macro": 0.5},
            {"epoch": 2, "loss": 0.5, "roc_auc_macro": score},
        ],
    )


def _failed_exp() -> Experiment:
    return Experiment(
        id="exp_fail",
        index=1,
        status="FAILED",
        proposal=_proposal(),
        primary_metric="roc_auc_macro",
        tasks=[
            Task(
                name="execute",
                status="FAILED",
                error=TaskError(error_type="ShapeMismatch", message="x"),
            )
        ],
    )


def _study() -> Study:
    return Study(
        id="study_test",
        task_name="track_b",
        status="COMPLETED",
        personality="exploratory",
        agent_memory_enabled=False,
        experiments=[_exp(0.55), _exp(0.7), _failed_exp()],
        created_at=datetime(2026, 5, 4, tzinfo=timezone.utc),
    )


def _make_judge(scripted: list[str]) -> tuple[Judge, _ScriptedPoster]:
    cfg = LLMConfig(
        provider="ollama",
        base_url="http://localhost:11434",
        model="m",
        temperature=0.0,
        max_tokens=256,
        retry_attempts=0,
        retry_backoff_seconds=0.0,
    )
    poster = _ScriptedPoster(scripted)
    client = LLMClient(cfg, http=poster, sleep=lambda _: None)
    engine = PromptEngine(PromptRegistry(SHIPPED_PROMPTS))
    return Judge(client, engine), poster


# ---------- experiment ----------

def test_judge_experiment_returns_verdict(tmp_path):
    payload = json.dumps(
        {
            "verdict": "promote",
            "score": 0.7,
            "rationale": "score is the highest so far.",
            "suggested_next": None,
        }
    )
    judge, _ = _make_judge([payload])
    memory = Memory(top_k=3, recent_failures=3, path=tmp_path / "m.json")
    v = judge.judge_experiment(_exp(0.7), memory)
    assert isinstance(v, Verdict)
    assert v.verdict == "promote"


def test_judge_experiment_strips_code_fences(tmp_path):
    payload = (
        "```json\n"
        + json.dumps({"verdict": "keep", "score": 0.5, "rationale": "fine."})
        + "\n```"
    )
    judge, _ = _make_judge([payload])
    memory = Memory(top_k=3, recent_failures=3, path=tmp_path / "m.json")
    v = judge.judge_experiment(_exp(), memory)
    assert v.verdict == "keep"


def test_judge_experiment_retries_on_invalid_json(tmp_path):
    valid = json.dumps(
        {"verdict": "discard", "score": 0.1, "rationale": "shape error"}
    )
    judge, poster = _make_judge(["NOT JSON", valid])
    memory = Memory(top_k=3, recent_failures=3, path=tmp_path / "m.json")
    v = judge.judge_experiment(_failed_exp(), memory)
    assert v.verdict == "discard"
    assert len(poster.calls) == 2


def test_judge_experiment_raises_after_two_failures(tmp_path):
    judge, _ = _make_judge(["broken", "still broken"])
    memory = Memory(top_k=3, recent_failures=3, path=tmp_path / "m.json")
    with pytest.raises(JudgeError):
        judge.judge_experiment(_exp(), memory)


def test_rationale_cap_enforced_via_pydantic(tmp_path):
    payload = json.dumps(
        {"verdict": "keep", "score": 0.5, "rationale": "x" * 600}  # too long
    )
    valid = json.dumps(
        {"verdict": "keep", "score": 0.5, "rationale": "ok"}
    )
    judge, poster = _make_judge([payload, valid])
    memory = Memory(top_k=3, recent_failures=3, path=tmp_path / "m.json")
    v = judge.judge_experiment(_exp(), memory)
    assert v.rationale == "ok"
    assert len(poster.calls) == 2  # first retry triggered


# ---------- study ----------

def test_judge_study_includes_top_experiments_table():
    valid = json.dumps(
        {"verdict": "abort_study", "score": 0.5, "rationale": "stop"}
    )
    judge, poster = _make_judge([valid])
    v = judge.judge_study(_study(), budget_used=3, budget_total=20)
    assert v.verdict == "abort_study"
    body = poster.calls[0][1].decode("utf-8")
    # table headers + best score (0.7) should appear in the prompt
    assert "0.7000" in body
    assert "EffNetB0" in body
    assert "ShapeMismatch" in body  # failure breakdown
