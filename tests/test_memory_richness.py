"""Memory render carries enough hyperparam + trajectory + verdict detail
for the LLM to (a) avoid re-trying things that already failed and (b)
spot good runs that are still climbing and worth a `continue_from`."""

from __future__ import annotations

from pathlib import Path

import pytest

from lab.core.memory import Memory, _recent_epochs
from lab.core.models import Experiment, Proposal, Task, TaskError, Verdict


def _exp_full(
    *,
    eid: str = "exp_a",
    score: float = 0.81,
    arch: str = "EffNetB0",
    family: str = "efficientnet_pretrained",
    lr: float = 3e-4,
    epochs: int = 5,
    weight_decay: float | None = 1e-4,
    verdict: str = "keep",
    rationale: str = "still rising at epoch 5",
    checkpoint_path: str | None = "sandbox/study_x/exp_a/ckpt.pt",
    history: list[dict] | None = None,
) -> Experiment:
    return Experiment(
        id=eid,
        index=0,
        status="JUDGED",
        proposal=Proposal(
            architecture_name=arch,
            family=family,
            lr=lr,
            lr_schedule="cosine",
            epochs=epochs,
            weight_decay=weight_decay,
        ),
        primary_metric="roc_auc_macro",
        primary_score=score,
        history=history or [
            {"epoch": 1, "loss": 0.6, "roc_auc_macro": 0.55},
            {"epoch": 2, "loss": 0.5, "roc_auc_macro": 0.65},
            {"epoch": 3, "loss": 0.42, "roc_auc_macro": 0.74},
            {"epoch": 4, "loss": 0.36, "roc_auc_macro": 0.78},
            {"epoch": 5, "loss": 0.32, "roc_auc_macro": score},
        ],
        verdict=Verdict(verdict=verdict, score=0.6, rationale=rationale),
        checkpoint_path=checkpoint_path,
    )


# ---------- _recent_epochs ----------


def test_recent_epochs_renders_last_n_only():
    history = [{"epoch": i, "loss": 1.0 - 0.1 * i, "roc_auc_macro": 0.1 * i} for i in range(1, 8)]
    out = _recent_epochs(history, "roc_auc_macro", n=3)
    assert "e5:" in out and "e6:" in out and "e7:" in out
    assert "e1:" not in out and "e4:" not in out


def test_recent_epochs_handles_missing_metric_field():
    history = [{"epoch": 1, "loss": 0.5}, {"epoch": 2, "loss": 0.4}]
    out = _recent_epochs(history, "roc_auc_macro")
    # No metric values -> no metric in render, but loss still there
    assert "loss=0.500" in out
    assert "roc_auc_macro=" not in out


def test_recent_epochs_empty_history():
    assert _recent_epochs([], "roc_auc_macro") == ""


# ---------- per-win richness ----------


def test_win_block_includes_hyperparams(tmp_path: Path):
    m = Memory(top_k=3, recent_failures=3, path=tmp_path / "mem.json")
    m.add(_exp_full(lr=3e-4, epochs=5, weight_decay=1e-4))
    md = m.to_markdown()
    assert "lr=3e-04" in md
    assert "sched=cosine" in md
    assert "epochs=5" in md
    assert "wd=1e-04" in md


def test_win_block_includes_trajectory_and_recent_epochs(tmp_path: Path):
    m = Memory(top_k=3, recent_failures=3, path=tmp_path / "mem.json")
    m.add(_exp_full())
    md = m.to_markdown()
    assert "trajectory:" in md
    assert "recent epochs:" in md
    # the most recent epoch must appear with both loss and metric values
    assert "e5:" in md
    assert "roc_auc_macro=" in md


def test_win_block_includes_verdict_rationale(tmp_path: Path):
    m = Memory(top_k=3, recent_failures=3, path=tmp_path / "mem.json")
    m.add(_exp_full(rationale="trajectory still climbing — extend training"))
    md = m.to_markdown()
    assert "verdict:" in md
    assert "trajectory still climbing" in md


def test_win_block_advertises_checkpoint_for_continue(tmp_path: Path):
    """The LLM needs an explicit "checkpoint available → continue_from: X"
    string to discover that resuming is even possible. Without this hint
    the model never picks the continue path."""
    m = Memory(top_k=3, recent_failures=3, path=tmp_path / "mem.json")
    m.add(_exp_full(eid="exp_continue_me"))
    md = m.to_markdown()
    assert "checkpoint available" in md
    assert "continue_from`: exp_continue_me" in md


def test_win_block_omits_checkpoint_hint_when_none(tmp_path: Path):
    m = Memory(top_k=3, recent_failures=3, path=tmp_path / "mem.json")
    m.add(_exp_full(checkpoint_path=None))
    md = m.to_markdown()
    assert "checkpoint available" not in md


# ---------- failure richness ----------


def test_fail_block_carries_recover_attempt_count(tmp_path: Path):
    """A failure with three recover attempts must say so — the LLM should
    not propose the same architecture that has already chewed through its
    full recovery budget."""
    exp = Experiment(
        id="exp_bad",
        index=0,
        status="FAILED",
        proposal=Proposal(
            architecture_name="EffNetB0",
            family="efficientnet_pretrained",
            lr=1e-3,
            lr_schedule="cosine",
            epochs=3,
        ),
        primary_metric="roc_auc_macro",
        primary_score=None,
        tasks=[
            Task(name="recover", status="SUCCEEDED", input={"attempt": 0}),
            Task(name="recover", status="SUCCEEDED", input={"attempt": 1}),
            Task(name="recover", status="SUCCEEDED", input={"attempt": 2}),
            Task(
                name="execute",
                status="FAILED",
                error=TaskError(error_type="ShapeMismatch", message="3 vs 1 channel"),
            ),
        ],
    )
    m = Memory(top_k=3, recent_failures=3, path=tmp_path / "mem.json")
    m.add(exp)
    md = m.to_markdown()
    assert "3 recovery attempt(s)" in md
    assert "ShapeMismatch" in md
