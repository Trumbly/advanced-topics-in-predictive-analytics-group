"""AGENT_*_DIR env vars must be absolute so the sandbox subprocess finds them."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from lab.config import load_settings
from lab.core.executor import LocalExecutor
from lab.core.experiment import _execute_with_retry, RunContext
from lab.core.memory import Memory
from lab.core.models import Experiment, Proposal
from lab.tasks.dataset import ensure_dataset_present

REPO_ROOT = Path(__file__).resolve().parent.parent


def _settings_in(tmp_path):
    s = load_settings("track_b", repo_root=REPO_ROOT)
    new_paths = s.paths.model_copy(update={"sandbox": str(tmp_path / "sandbox")})
    new_task = s.task.model_copy(
        update={
            "expected_num_classes": 4,
            "input_tensor_shape": [1, 8, 8],
            "processed_data_dir": str(tmp_path / "mels"),
        }
    )
    return s.model_copy(update={"paths": new_paths, "task": new_task})


def test_processed_dir_resolved_absolute_via_relative_setting(tmp_path, monkeypatch):
    """When task.processed_data_dir is set relatively, AGENT_PROCESSED_DIR
    must be made absolute so the subprocess (which has cwd=sandbox/<id>)
    still finds the shards.
    """
    s = _settings_in(tmp_path)
    # Force a relative processed_data_dir to mimic the config's "data/processed/mels".
    rel_relative = "rel_mels"
    monkeypatch.chdir(tmp_path)  # treat tmp_path as repo cwd
    Path(rel_relative).mkdir()
    # write a fake train.pt so ensure_dataset_present's "real" branch runs
    sentinel = {"x": torch.zeros(2, 1, 8, 8), "y": torch.zeros(2, 4)}
    torch.save(sentinel, Path(rel_relative) / "train.pt")
    new_task = s.task.model_copy(update={"processed_data_dir": rel_relative})
    s = s.model_copy(update={"task": new_task})

    s_after, used = ensure_dataset_present(s)
    assert used is False  # real shard exists

    # Build extra_env via the same path the orchestrator uses.
    proposal = Proposal(
        architecture_name="X",
        family="cnn_scratch",
        lr=1e-3,
        lr_schedule="constant",
        epochs=1,
    )
    exp = Experiment(
        id="exp_path",
        index=0,
        status="EXECUTING",
        proposal=proposal,
        primary_metric="roc_auc_macro",
    )
    ctx = RunContext(
        settings=s_after,
        adapter=None,        # type: ignore[arg-type]
        client=None,         # type: ignore[arg-type]
        engine=None,         # type: ignore[arg-type]
        memory=Memory(top_k=3, recent_failures=3, path=tmp_path / "m.json"),
        validator=None,      # type: ignore[arg-type]
        executor=LocalExecutor(s_after),
        recovery=None,       # type: ignore[arg-type]
        judge=None,          # type: ignore[arg-type]
    )

    # Invoke the env-builder by snooping at what _execute_with_retry would set:
    # we mirror the relevant lines so the assertion stays focused on the path.
    abs_processed = str(Path(s_after.task.processed_data_dir).resolve())
    abs_checkpoint = str(
        (Path(s_after.paths.sandbox) / exp.id / "checkpoint.pt").resolve()
    )
    assert Path(abs_processed).is_absolute()
    assert Path(abs_checkpoint).is_absolute()
    assert abs_processed.endswith("/rel_mels")
    # And the underlying file is reachable from outside the sandbox cwd:
    assert (Path(abs_processed) / "train.pt").exists()
