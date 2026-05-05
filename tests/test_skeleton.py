"""I-11 acceptance: render template + run on synthetic shard with identity model."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from lab.config import load_settings
from lab.tasks.skeleton import render_skeleton, splice_build_model

REPO_ROOT = Path(__file__).resolve().parent.parent

# Tiny synthetic shape so the smoke run finishes in a few seconds.
NUM_CLASSES = 8
INPUT_SHAPE = (1, 16, 16)


@pytest.fixture
def settings(tmp_path):
    s = load_settings("track_b", repo_root=REPO_ROOT)
    new_task = s.task.model_copy(
        update={
            "expected_num_classes": NUM_CLASSES,
            "input_tensor_shape": list(INPUT_SHAPE),
            "processed_data_dir": str(tmp_path / "mels"),
        }
    )
    return s.model_copy(update={"task": new_task})


def _make_synthetic_shard(processed_dir: Path, n_train: int = 100, n_val: int = 20):
    processed_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    def make(n):
        x = torch.randn(n, *INPUT_SHAPE)
        y = torch.zeros(n, NUM_CLASSES)
        idx = torch.randint(0, NUM_CLASSES, (n,))
        y[torch.arange(n), idx] = 1.0
        return {"x": x, "y": y}

    torch.save(make(n_train), processed_dir / "train.pt")
    torch.save(make(n_val), processed_dir / "val.pt")


_IDENTITY_BLOCK = """\
def build_model(num_classes: int) -> nn.Module:
    in_chan = INPUT_SHAPE[0]
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_chan * INPUT_SHAPE[1] * INPUT_SHAPE[2], num_classes),
    )
"""


def test_rendered_skeleton_has_markers(settings):
    rendered = render_skeleton(settings)
    assert "# --- AGENT_BUILD_MODEL_START ---" in rendered
    assert "# --- AGENT_BUILD_MODEL_END ---" in rendered
    assert f"NUM_CLASSES = {NUM_CLASSES}" in rendered
    assert f"INPUT_SHAPE = tuple({list(INPUT_SHAPE)})" in rendered


def test_splice_replaces_block(settings):
    rendered = render_skeleton(settings)
    spliced = splice_build_model(rendered, _IDENTITY_BLOCK)
    # the spliced block content should appear, default Conv2d/AdaptiveAvgPool2d should not
    assert "Linear(in_chan * INPUT_SHAPE[1] * INPUT_SHAPE[2]" in spliced
    assert "AdaptiveAvgPool2d" not in spliced


def test_run_on_synthetic_shard_writes_results(tmp_path, settings):
    processed_dir = Path(settings.task.processed_data_dir)
    _make_synthetic_shard(processed_dir, n_train=100, n_val=20)

    rendered = splice_build_model(render_skeleton(settings), _IDENTITY_BLOCK)
    sandbox = tmp_path / "run"
    sandbox.mkdir()
    code_path = sandbox / "code.py"
    code_path.write_text(rendered, encoding="utf-8")

    env = {
        "AGENT_DEVICE": "cpu",
        "AGENT_BATCH_SIZE": "16",
        "AGENT_EPOCHS": "1",
        "AGENT_PROCESSED_DIR": str(processed_dir),
        "AGENT_CHECKPOINT_OUT": str(sandbox / "ckpt.pt"),
        "AGENT_LR": "1e-3",
        "AGENT_LR_SCHEDULE": "constant",
        "AGENT_SEED": "0",
        "PATH": __import__("os").environ.get("PATH", ""),
    }

    completed = subprocess.run(
        [sys.executable, "code.py"],
        cwd=sandbox,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr

    results = json.loads((sandbox / "results.json").read_text())
    assert results["primary_metric"] == "roc_auc_macro"
    assert isinstance(results["primary_score"], float)
    assert results["history"]
    assert (sandbox / "ckpt.pt").exists()


def test_warm_start_lowers_first_epoch_loss(tmp_path, settings):
    processed_dir = Path(settings.task.processed_data_dir)
    _make_synthetic_shard(processed_dir, n_train=200, n_val=40)

    rendered = splice_build_model(render_skeleton(settings), _IDENTITY_BLOCK)
    sandbox_a = tmp_path / "cold"
    sandbox_a.mkdir()
    (sandbox_a / "code.py").write_text(rendered)

    base_env = {
        "AGENT_DEVICE": "cpu",
        "AGENT_BATCH_SIZE": "32",
        "AGENT_EPOCHS": "2",
        "AGENT_PROCESSED_DIR": str(processed_dir),
        "AGENT_LR": "5e-3",
        "AGENT_LR_SCHEDULE": "constant",
        "AGENT_SEED": "0",
        "PATH": __import__("os").environ.get("PATH", ""),
    }

    cold_env = {**base_env, "AGENT_CHECKPOINT_OUT": str(sandbox_a / "ckpt.pt")}
    subprocess.run(
        [sys.executable, "code.py"], cwd=sandbox_a, env=cold_env, check=True, timeout=60
    )

    cold_first_loss = json.loads((sandbox_a / "results.json").read_text())["history"][0]["loss"]

    sandbox_b = tmp_path / "warm"
    sandbox_b.mkdir()
    (sandbox_b / "code.py").write_text(rendered)
    warm_env = {
        **base_env,
        "AGENT_CHECKPOINT_IN": str(sandbox_a / "ckpt.pt"),
        "AGENT_CHECKPOINT_OUT": str(sandbox_b / "ckpt.pt"),
    }
    subprocess.run(
        [sys.executable, "code.py"], cwd=sandbox_b, env=warm_env, check=True, timeout=60
    )

    warm_first_loss = json.loads((sandbox_b / "results.json").read_text())["history"][0]["loss"]
    # warm-start should start with a lower (or equal) loss than cold-start
    assert warm_first_loss <= cold_first_loss + 1e-3, (warm_first_loss, cold_first_loss)
