"""Real BirdCLEF preprocessing: build train.pt + val.pt from .npy spectrograms."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from lab.config import load_settings
from lab.tasks.real_preprocess import build_real_shards

REPO_ROOT = Path(__file__).resolve().parent.parent


def _settings_in(tmp_path):
    """Create a tmp processed dir + spectrograms + labels.csv layout."""
    s = load_settings("track_b", repo_root=REPO_ROOT)
    processed_dir = tmp_path / "data" / "processed" / "mels"
    new_paths = s.paths.model_copy(update={"sandbox": str(tmp_path / "sandbox")})
    new_task = s.task.model_copy(
        update={
            "processed_data_dir": str(processed_dir),
            "expected_num_classes": 4,
            "input_tensor_shape": [1, 8, 8],
        }
    )
    return s.model_copy(update={"paths": new_paths, "task": new_task})


def _seed_dataset(tmp_path: Path, *, n_per_class: int = 5):
    spec_dir = tmp_path / "data" / "processed" / "spectrograms"
    labels_csv = tmp_path / "data" / "processed" / "labels.csv"
    spec_dir.mkdir(parents=True)

    rng = np.random.default_rng(0)
    rows = ["sample_id,class_id"]
    for cls in ("a", "b", "c", "d"):
        for i in range(n_per_class):
            sid = f"{cls}_{i:03d}"
            arr = rng.standard_normal((1, 8, 8)).astype("float32")
            np.save(spec_dir / f"{sid}.npy", arr)
            rows.append(f"{sid},{cls}")
    labels_csv.write_text("\n".join(rows) + "\n")


def test_build_real_shards_writes_train_and_val(tmp_path):
    s = _settings_in(tmp_path)
    _seed_dataset(tmp_path, n_per_class=10)

    train_path, val_path = build_real_shards(
        s, samples_per_class=10, val_fraction=0.2
    )
    assert train_path.exists() and val_path.exists()

    train = torch.load(train_path, weights_only=False)
    val = torch.load(val_path, weights_only=False)

    assert train["x"].shape[1:] == (1, 8, 8)
    assert train["y"].shape[1] == 4  # 4 distinct classes in seeded csv
    # 10 per class, 20% val → 2 val + 8 train per class, 4 classes
    assert train["x"].shape[0] == 32
    assert val["x"].shape[0] == 8


def test_build_real_shards_handles_2d_spectrograms(tmp_path):
    """Spectrograms saved as (n_mels, n_frames) get a channel dim added."""
    s = _settings_in(tmp_path)
    spec_dir = tmp_path / "data" / "processed" / "spectrograms"
    labels_csv = tmp_path / "data" / "processed" / "labels.csv"
    spec_dir.mkdir(parents=True)
    rows = ["sample_id,class_id"]
    for i in range(5):
        sid = f"x_{i:03d}"
        np.save(spec_dir / f"{sid}.npy", np.zeros((8, 8), dtype="float32"))
        rows.append(f"{sid},only-class")
    labels_csv.write_text("\n".join(rows) + "\n")

    train_path, _ = build_real_shards(s, samples_per_class=5, val_fraction=0.2)
    train = torch.load(train_path, weights_only=False)
    assert train["x"].shape[1:] == (1, 8, 8)


def test_build_real_shards_raises_when_spectrograms_missing(tmp_path):
    s = _settings_in(tmp_path)
    # Note: only labels.csv present, no spectrograms dir
    (tmp_path / "data" / "processed").mkdir(parents=True)
    (tmp_path / "data" / "processed" / "labels.csv").write_text(
        "sample_id,class_id\nx,1\n"
    )
    with pytest.raises(FileNotFoundError):
        build_real_shards(s)


def test_build_real_shards_skip_if_present(tmp_path):
    s = _settings_in(tmp_path)
    _seed_dataset(tmp_path, n_per_class=5)
    train_path, val_path = build_real_shards(s, samples_per_class=5)
    train_mtime = train_path.stat().st_mtime_ns

    # second call without overwrite must NOT rewrite
    build_real_shards(s, samples_per_class=5)
    assert train_path.stat().st_mtime_ns == train_mtime

    # overwrite=True forces a rebuild
    build_real_shards(s, samples_per_class=5, overwrite=True)
    # mtime might be coarse; just verify it still exists + valid
    assert train_path.exists()
    blob = torch.load(train_path, weights_only=False)
    assert blob["x"].shape[0] > 0


def test_cli_run_refuses_when_real_shards_missing(tmp_path, monkeypatch, capsys):
    """`lab run` errors out instead of falling back to synthetic."""
    from lab.cli import cmd_run

    s = load_settings("track_b", repo_root=REPO_ROOT)
    new_task = s.task.model_copy(
        update={"processed_data_dir": str(tmp_path / "no-data")}
    )
    new_settings = s.model_copy(update={"task": new_task})

    monkeypatch.setattr("lab.cli.load_settings", lambda *a, **k: new_settings)

    class _Args:
        task = "track_b"
        predecessor = None
        use_best_prompts = False
        agent_memory = False
        personality = None
        max_experiments = 1
        max_wallclock_min = 1
        llm_model = None

    rc = cmd_run(_Args())
    assert rc == 2
    err = capsys.readouterr().err
    assert "no processed shards" in err.lower() or "preprocess" in err.lower()
