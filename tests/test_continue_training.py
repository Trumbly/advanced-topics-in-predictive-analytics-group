"""Continue-training path: a Proposal with continue_from_experiment_id
must skip propose+generate+validate, reuse the source experiment's
stored ``code``, and surface the previous experiment's checkpoint via
AGENT_CHECKPOINT_IN."""

from __future__ import annotations

from pathlib import Path

import pytest

from lab.core.experiment import _resolve_warm_start, _resume_code_from
from lab.core.memory import Memory
from lab.core.models import Experiment, Proposal


def _proposal(*, init_from: str | None = None, cont_from: str | None = None) -> Proposal:
    return Proposal(
        architecture_name="EffNetB0",
        family="efficientnet_pretrained",
        lr=1e-4,
        lr_schedule="cosine",
        epochs=3,
        init_from_experiment_id=init_from,
        continue_from_experiment_id=cont_from,
    )


def _source_exp(eid: str = "exp_seed", *, code: str | None = "PRINT_HELLO_CODE", checkpoint_path: str | None = "ckpts/seed.pt") -> Experiment:
    return Experiment(
        id=eid,
        index=0,
        status="JUDGED",
        proposal=_proposal(),
        primary_metric="roc_auc_macro",
        primary_score=0.78,
        code=code,
        checkpoint_path=checkpoint_path,
    )


# ---------- proposal schema ----------


def test_proposal_default_continue_from_is_none():
    p = _proposal()
    assert p.continue_from_experiment_id is None


def test_proposal_continue_from_round_trips_through_dump():
    p = _proposal(cont_from="exp_seed")
    payload = p.model_dump()
    assert payload["continue_from_experiment_id"] == "exp_seed"
    again = Proposal.model_validate(payload)
    assert again.continue_from_experiment_id == "exp_seed"


# ---------- warm start resolution ----------


class _FakeSettings:
    """Minimal stand-in for Settings used by _resolve_warm_start."""

    class _Agent:
        def __init__(self, memory_enabled: bool):
            self.memory_enabled = memory_enabled

    def __init__(self, memory_enabled: bool):
        self.agent = self._Agent(memory_enabled)


class _FakeCtx:
    def __init__(self, *, memory: Memory, memory_enabled: bool):
        self.settings = _FakeSettings(memory_enabled)
        self.memory = memory


def test_continue_from_overrides_memory_disabled_gate(tmp_path: Path):
    """``continue_from_experiment_id`` is an explicit "I want to resume
    THIS run" signal -- it must work even when the agent-memory toggle
    is OFF (which gates the transfer-learning ``init_from`` path)."""
    m = Memory(top_k=3, recent_failures=3, path=tmp_path / "m.json")
    m.add(_source_exp("exp_seed"))
    ctx = _FakeCtx(memory=m, memory_enabled=False)
    proposal = _proposal(cont_from="exp_seed")
    out = _resolve_warm_start(ctx, proposal)
    assert out == "ckpts/seed.pt"


def test_init_from_still_requires_memory_enabled(tmp_path: Path):
    m = Memory(top_k=3, recent_failures=3, path=tmp_path / "m.json")
    m.add(_source_exp("exp_seed"))
    ctx = _FakeCtx(memory=m, memory_enabled=False)
    proposal = _proposal(init_from="exp_seed")  # NOT a continuation
    assert _resolve_warm_start(ctx, proposal) is None


def test_continue_from_returns_none_when_source_missing(tmp_path: Path):
    m = Memory(top_k=3, recent_failures=3, path=tmp_path / "m.json")
    ctx = _FakeCtx(memory=m, memory_enabled=True)
    proposal = _proposal(cont_from="exp_does_not_exist")
    assert _resolve_warm_start(ctx, proposal) is None


def test_continue_from_returns_none_when_source_has_no_checkpoint(tmp_path: Path):
    m = Memory(top_k=3, recent_failures=3, path=tmp_path / "m.json")
    m.add(_source_exp("exp_seed", checkpoint_path=None))
    ctx = _FakeCtx(memory=m, memory_enabled=True)
    proposal = _proposal(cont_from="exp_seed")
    assert _resolve_warm_start(ctx, proposal) is None


# ---------- resume code reuse ----------


def _exp_under_test() -> Experiment:
    return Experiment(
        id="exp_continuation",
        index=1,
        status="PROPOSED",
        primary_metric="roc_auc_macro",
    )


class _CtxWithProgress(_FakeCtx):
    """Adds the ``on_progress`` field experiment.py reads via _progress."""

    on_progress = None


def test_resume_code_reuses_source_code(tmp_path: Path):
    m = Memory(top_k=3, recent_failures=3, path=tmp_path / "m.json")
    m.add(_source_exp("exp_seed", code="DEFINITIVE_BUILD_MODEL_BLOCK"))
    ctx = _CtxWithProgress(memory=m, memory_enabled=True)
    exp = _exp_under_test()

    code = _resume_code_from(ctx, exp, _proposal(cont_from="exp_seed"))
    assert code == "DEFINITIVE_BUILD_MODEL_BLOCK"

    # A `generate` task is appended so the UI step history visibly
    # records "we reused exp_seed's code instead of asking the LLM".
    assert any(
        t.name == "generate" and t.input.get("continued_from") == "exp_seed"
        for t in exp.tasks
    )


def test_resume_code_raises_hard_failure_when_source_missing(tmp_path: Path):
    from lab.core.experiment import _HardFailure

    m = Memory(top_k=3, recent_failures=3, path=tmp_path / "m.json")
    ctx = _CtxWithProgress(memory=m, memory_enabled=True)
    exp = _exp_under_test()

    with pytest.raises(_HardFailure) as exc_info:
        _resume_code_from(ctx, exp, _proposal(cont_from="not_there"))
    assert exc_info.value.task_name == "generate"
    assert "not_there" in exc_info.value.error.message


def test_resume_code_raises_when_source_has_no_stored_code(tmp_path: Path):
    from lab.core.experiment import _HardFailure

    m = Memory(top_k=3, recent_failures=3, path=tmp_path / "m.json")
    m.add(_source_exp("exp_seed", code=None))
    ctx = _CtxWithProgress(memory=m, memory_enabled=True)
    exp = _exp_under_test()

    with pytest.raises(_HardFailure):
        _resume_code_from(ctx, exp, _proposal(cont_from="exp_seed"))
