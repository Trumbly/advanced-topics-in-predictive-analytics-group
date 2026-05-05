"""Live-progress: study.json reflects in-flight experiments."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from lab.core.experiment import Hooks, RunContext, run_experiment
from lab.core.models import (
    Experiment,
    Proposal,
    Study,
    Task,
    new_study_id,
)


class _StubAdapter:
    name = "track_b"
    kind = "audio_multilabel"
    primary_metric = "roc_auc_macro"

    def profile(self):  # pragma: no cover - not exercised
        return None

    def prompt_slots(self):
        return {}

    def model_block_signature(self):
        return ("build_model", "num_classes")

    def spawn_triggering_calls(self):
        return ()

    def build_submission(self, *args, **kwargs):  # pragma: no cover
        return Path()


def test_progress_callback_fires_per_task_append(tmp_path):
    """Each Task.append inside run_experiment should call ctx.on_progress."""
    e = Experiment(
        id="exp_0000",
        index=0,
        status="PROPOSED",
        primary_metric="roc_auc_macro",
    )

    fired: list[int] = []

    # We craft a minimal RunContext that fakes through to a HardFailure on
    # propose so the test stays fast. The callback must fire either way; here
    # we install a synthetic propose Task manually and verify the helper.
    from lab.core import experiment as exp_mod

    ctx = RunContext.__new__(RunContext)  # bypass dataclass init
    ctx.on_progress = lambda: fired.append(len(e.tasks))

    # Simulate the helper used inside _propose / _validate_with_retry / etc.
    e.tasks.append(Task(name="propose", status="SUCCEEDED"))
    exp_mod._progress(ctx)
    e.tasks.append(Task(name="validate", status="FAILED"))
    exp_mod._progress(ctx)

    assert fired == [1, 2]


def test_progress_callback_optional(tmp_path):
    """Missing callback is allowed and does not raise."""
    from lab.core import experiment as exp_mod

    ctx = RunContext.__new__(RunContext)
    ctx.on_progress = None
    exp_mod._progress(ctx)  # no-op, must not raise


def test_progress_callback_swallows_exceptions(tmp_path):
    """A failing callback never breaks the loop."""
    from lab.core import experiment as exp_mod

    ctx = RunContext.__new__(RunContext)

    def boom():
        raise RuntimeError("hook explodes")

    ctx.on_progress = boom
    exp_mod._progress(ctx)  # must not raise


def test_lifecycle_appends_experiment_before_run(tmp_path):
    """After lifecycle starts an experiment, study.json should already list it."""
    # We emulate the lifecycle's pre-run snapshot directly.
    studies_root = tmp_path / "studies"
    studies_root.mkdir()
    sid = new_study_id()
    study = Study(
        id=sid,
        task_name="track_b",
        status="RUNNING",
        personality="exploratory",
        agent_memory_enabled=False,
        experiments=[],
        created_at=datetime(2026, 5, 5, tzinfo=timezone.utc),
    )
    study.save(studies_root)

    # Mimic lifecycle.run() behaviour: append + save BEFORE run_experiment.
    pre_exp = Experiment(
        id="exp_0000",
        index=0,
        status="PROPOSED",
        primary_metric="roc_auc_macro",
    )
    study.experiments.append(pre_exp)
    study.save(studies_root)

    on_disk = json.loads(
        (studies_root / sid / "study.json").read_text()
    )
    assert len(on_disk["experiments"]) == 1
    assert on_disk["experiments"][0]["id"] == "exp_0000"
    assert on_disk["experiments"][0]["status"] == "PROPOSED"
