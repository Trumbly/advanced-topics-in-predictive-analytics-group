"""I-20 acceptance: agent memory toggle + warm-start gate (ADR-007)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from lab.config import load_settings
from lab.core.experiment import RunContext, _resolve_warm_start
from lab.core.memory import Memory
from lab.core.models import Experiment, Proposal, Study

REPO_ROOT = Path(__file__).resolve().parent.parent


def _settings(memory_enabled: bool, tmp_path: Path):
    s = load_settings("track_b", repo_root=REPO_ROOT)
    new_paths = s.paths.model_copy(update={"experiments_dir": str(tmp_path / "studies")})
    new_agent = s.agent.model_copy(update={"memory_enabled": memory_enabled})
    return s.model_copy(update={"paths": new_paths, "agent": new_agent})


def _proposal(init_from: str | None = None) -> Proposal:
    return Proposal(
        architecture_name="X",
        family="cnn_scratch",
        lr=1e-3,
        lr_schedule="cosine",
        epochs=2,
        init_from_experiment_id=init_from,
    )


def _success_with_checkpoint(idx: int, score: float, checkpoint: Path | None) -> Experiment:
    e = Experiment(
        id=f"exp_{idx:04d}",
        index=idx,
        status="JUDGED",
        proposal=_proposal(),
        primary_metric="roc_auc_macro",
        primary_score=score,
    )
    if checkpoint is not None:
        e.checkpoint_path = str(checkpoint)
    return e


def _study(experiments, sid="study_first"):
    return Study(
        id=sid,
        task_name="track_b",
        status="COMPLETED",
        personality="exploratory",
        agent_memory_enabled=False,
        experiments=experiments,
        created_at=datetime(2026, 5, 4, tzinfo=timezone.utc),
    )


def _make_ctx(settings, memory):
    # _resolve_warm_start only needs settings + memory; stub the rest with None.
    return RunContext(
        settings=settings,
        adapter=None,  # type: ignore[arg-type]
        client=None,  # type: ignore[arg-type]
        engine=None,  # type: ignore[arg-type]
        memory=memory,
        validator=None,  # type: ignore[arg-type]
        executor=None,  # type: ignore[arg-type]
        recovery=None,  # type: ignore[arg-type]
        judge=None,  # type: ignore[arg-type]
    )


def test_warm_start_disabled_when_memory_off(tmp_path):
    s = _settings(memory_enabled=False, tmp_path=tmp_path)
    ck = tmp_path / "ckpt.pt"
    ck.write_text("x")
    memory = Memory(top_k=5, recent_failures=5, path=tmp_path / "m.json")
    memory.add(_success_with_checkpoint(0, 0.7, ck))

    ctx = _make_ctx(s, memory)
    out = _resolve_warm_start(ctx, _proposal(init_from="exp_0000"))
    assert out is None  # gated by memory_enabled=False


def test_warm_start_enabled_when_memory_on_and_checkpoint_exists(tmp_path):
    s = _settings(memory_enabled=True, tmp_path=tmp_path)
    ck = tmp_path / "ckpt.pt"
    ck.write_text("x")
    memory = Memory(top_k=5, recent_failures=5, path=tmp_path / "m.json")
    memory.add(_success_with_checkpoint(0, 0.7, ck))

    ctx = _make_ctx(s, memory)
    out = _resolve_warm_start(ctx, _proposal(init_from="exp_0000"))
    assert out == str(ck)


def test_warm_start_returns_none_when_no_init_field(tmp_path):
    s = _settings(memory_enabled=True, tmp_path=tmp_path)
    memory = Memory(top_k=5, recent_failures=5, path=tmp_path / "m.json")
    ctx = _make_ctx(s, memory)
    out = _resolve_warm_start(ctx, _proposal(init_from=None))
    assert out is None


def test_warm_start_warns_when_checkpoint_missing(tmp_path, capsys):
    """memory_enabled=True + proposal references an experiment without checkpoint."""
    from lab.core import telemetry

    s = _settings(memory_enabled=True, tmp_path=tmp_path)
    telemetry._reset_for_tests()
    telemetry.configure(s, study_id="study_test_xx")

    memory = Memory(top_k=5, recent_failures=5, path=tmp_path / "m.json")
    memory.add(_success_with_checkpoint(0, 0.7, checkpoint=None))

    ctx = _make_ctx(s, memory)
    out = _resolve_warm_start(ctx, _proposal(init_from="exp_0000"))
    assert out is None
    captured = capsys.readouterr().out
    assert "checkpoint missing" in captured.lower()


def test_seed_from_agent_memory_runs_when_enabled(tmp_path):
    studies_root = tmp_path / "studies"
    studies_root.mkdir()
    s1 = _study([_success_with_checkpoint(0, 0.8, None)], sid="study_s1")
    s1.save(studies_root)

    memory = Memory(top_k=5, recent_failures=5, path=tmp_path / "m.json")
    memory.seed_from_agent_memory(studies_root, task="track_b")
    assert any(e.primary_score == 0.8 for e in memory.wins)
