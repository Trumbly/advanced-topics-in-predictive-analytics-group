"""ensure_dataset_present synthetic fallback."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from lab.config import load_settings
from lab.tasks.dataset import ensure_dataset_present

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def settings(tmp_path):
    s = load_settings("track_b", repo_root=REPO_ROOT)
    new_task = s.task.model_copy(
        update={
            "processed_data_dir": str(tmp_path / "mels"),
            "expected_num_classes": 4,
            "input_tensor_shape": [1, 8, 8],
        }
    )
    return s.model_copy(update={"task": new_task})


def test_writes_synthetic_when_missing(settings):
    out = Path(settings.task.processed_data_dir)
    assert not (out / "train.pt").exists()
    assert ensure_dataset_present(settings, n_train=20, n_val=5) is True
    assert (out / "train.pt").exists()
    assert (out / "val.pt").exists()
    blob = torch.load(out / "train.pt", map_location="cpu", weights_only=False)
    assert blob["x"].shape == (20, 1, 8, 8)
    assert blob["y"].shape == (20, 4)


def test_leaves_real_shards_untouched(settings, tmp_path):
    """If train.pt already exists, do not overwrite."""
    out = Path(settings.task.processed_data_dir)
    out.mkdir(parents=True)
    sentinel = {"x": torch.zeros(3, 1, 8, 8), "y": torch.zeros(3, 4)}
    torch.save(sentinel, out / "train.pt")

    assert ensure_dataset_present(settings) is False
    blob = torch.load(out / "train.pt", map_location="cpu", weights_only=False)
    assert blob["x"].shape == (3, 1, 8, 8)  # sentinel preserved
