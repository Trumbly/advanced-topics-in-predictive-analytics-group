"""Lazy index path: build_real_shards default + skeleton's LazyMelDataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lab.config import load_settings
from lab.tasks.real_preprocess import build_real_shards

REPO_ROOT = Path(__file__).resolve().parent.parent


def _settings_in(tmp_path):
    s = load_settings("track_b", repo_root=REPO_ROOT)
    processed_dir = tmp_path / "data" / "processed" / "mels"
    new_paths = s.paths.model_copy(update={"sandbox": str(tmp_path / "sandbox")})
    new_task = s.task.model_copy(
        update={
            "processed_data_dir": str(processed_dir),
            "expected_num_classes": 3,
            "input_tensor_shape": [1, 8, 8],
        }
    )
    return s.model_copy(update={"paths": new_paths, "task": new_task})


def _seed(tmp_path: Path, *, n_per_class: int = 5):
    spec_dir = tmp_path / "data" / "processed" / "spectrograms"
    labels_csv = tmp_path / "data" / "processed" / "labels.csv"
    spec_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows = ["sample_id,class_id"]
    for cls in ("a", "b", "c"):
        for i in range(n_per_class):
            sid = f"{cls}_{i:03d}"
            np.save(spec_dir / f"{sid}.npy", rng.standard_normal((1, 8, 8)).astype("float32"))
            rows.append(f"{sid},{cls}")
    labels_csv.write_text("\n".join(rows) + "\n")
    return spec_dir


def test_default_writes_lazy_index(tmp_path):
    s = _settings_in(tmp_path)
    spec_dir = _seed(tmp_path, n_per_class=10)

    train_path, val_path = build_real_shards(s)  # samples_per_class=None
    assert train_path.suffix == ".json"
    assert val_path.suffix == ".json"

    train = json.loads(train_path.read_text())
    val = json.loads(val_path.read_text())
    assert train["num_classes"] == 3
    assert train["spectrograms_dir"] == str(spec_dir.resolve())
    # 10 per class × 3 classes = 30 total, 80/20 → 24 train + 6 val
    assert len(train["samples"]) == 24
    assert len(val["samples"]) == 6
    # No eager .pt should be written in lazy mode
    assert not (Path(s.task.processed_data_dir) / "train.pt").exists()


def test_eager_subsample_writes_pt(tmp_path):
    s = _settings_in(tmp_path)
    _seed(tmp_path, n_per_class=10)

    train_path, val_path = build_real_shards(s, samples_per_class=4)
    assert train_path.suffix == ".pt"
    assert val_path.suffix == ".pt"


def test_lazy_dataset_loads_via_skeleton(tmp_path):
    """Render the skeleton, simulate AGENT_PROCESSED_DIR pointing at the
    lazy index, instantiate LazyMelDataset directly, and verify shapes."""
    s = _settings_in(tmp_path)
    _seed(tmp_path, n_per_class=5)
    train_path, _ = build_real_shards(s)

    # Render skeleton + extract LazyMelDataset class via exec
    from lab.tasks.skeleton import render_skeleton

    code = render_skeleton(s)
    ns: dict = {"__name__": "__sandbox__"}
    exec(compile(code, "<test>", "exec"), ns)
    LazyMelDataset = ns["LazyMelDataset"]

    ds = LazyMelDataset(train_path)
    assert len(ds) > 0
    x, y = ds[0]
    assert tuple(x.shape) == (1, 8, 8)
    assert y.shape == (3,)
    assert y.sum().item() == 1.0  # one-hot
