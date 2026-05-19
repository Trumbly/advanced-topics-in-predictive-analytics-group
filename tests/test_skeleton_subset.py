"""Tests for the AGENT_DATA_SUBSET_PCT subsetting logic in the audio skeleton.

These tests import _maybe_subset and DATA_SUBSET_PCT directly from the rendered
skeleton module. Because the skeleton is a Jinja2 template, we first render it
into a temp file (like the real executor does) and then import it.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
_SKELETON_J2 = REPO_ROOT / "config" / "skeletons" / "audio_multilabel.py.j2"


def _render_skeleton(tmp_path: Path, num_classes: int = 4) -> Path:
    """Render the Jinja skeleton with minimal substitutions and return the path."""
    from jinja2 import Environment

    template_text = _SKELETON_J2.read_text(encoding="utf-8")
    env = Environment()
    template = env.from_string(template_text)
    rendered = template.render(
        num_classes=num_classes,
        input_tensor_shape=[1, 8, 8],
        primary_metric="roc_auc_macro",
    )
    out = tmp_path / "skeleton.py"
    out.write_text(rendered, encoding="utf-8")
    return out


def _import_skeleton(skeleton_path: Path):
    """Dynamically import the rendered skeleton as a module."""
    spec = importlib.util.spec_from_file_location("_skeleton_under_test", skeleton_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_fake_index(tmp_path: Path, n_samples: int, num_classes: int = 4) -> Path:
    """Write a minimal train_index.json and stub .npy files."""
    spec_dir = tmp_path / "spectrograms"
    spec_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    for i in range(n_samples):
        sid = f"sample_{i:04d}"
        npy = spec_dir / f"{sid}.npy"
        import numpy as np
        arr = np.zeros((1, 8, 8), dtype="float32")
        np.save(npy, arr)
        samples.append({"sid": sid, "class_indices": [i % num_classes]})

    index = {
        "spectrograms_dir": str(spec_dir),
        "num_classes": num_classes,
        "samples": samples,
    }
    index_path = tmp_path / "train_index.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    return index_path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def skeleton_30pct(tmp_path, monkeypatch):
    """Import the skeleton with AGENT_DATA_SUBSET_PCT=30."""
    monkeypatch.setenv("AGENT_DATA_SUBSET_PCT", "30")
    monkeypatch.setenv("AGENT_PROCESSED_DIR", str(tmp_path))
    skeleton_path = _render_skeleton(tmp_path)
    return _import_skeleton(skeleton_path)


@pytest.fixture
def skeleton_100pct(tmp_path, monkeypatch):
    """Import the skeleton with AGENT_DATA_SUBSET_PCT=100 (default)."""
    monkeypatch.setenv("AGENT_DATA_SUBSET_PCT", "100")
    monkeypatch.setenv("AGENT_PROCESSED_DIR", str(tmp_path))
    skeleton_path = _render_skeleton(tmp_path)
    return _import_skeleton(skeleton_path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_maybe_subset_with_30pct_reduces_length(skeleton_30pct, tmp_path):
    """_maybe_subset should keep ≈30% of the dataset."""
    n = 100
    _write_fake_index(tmp_path, n_samples=n)
    # Build a simple TensorDataset as a stand-in
    ds = torch.utils.data.TensorDataset(
        torch.zeros(n, 1, 8, 8), torch.zeros(n, 4)
    )
    result = skeleton_30pct._maybe_subset(ds, "train")
    expected = max(1, n * 30 // 100)
    assert len(result) == expected


def test_maybe_subset_with_100pct_returns_original(skeleton_100pct):
    """_maybe_subset at 100% should return the dataset unchanged (not wrapped)."""
    n = 50
    ds = torch.utils.data.TensorDataset(
        torch.zeros(n, 1, 8, 8), torch.zeros(n, 4)
    )
    result = skeleton_100pct._maybe_subset(ds, "train")
    # Should be the same object (not wrapped in Subset)
    assert result is ds


def test_maybe_subset_is_deterministic(skeleton_30pct):
    """Two calls with the same dataset and split return the same indices."""
    n = 100
    ds = torch.utils.data.TensorDataset(
        torch.zeros(n, 1, 8, 8), torch.zeros(n, 4)
    )
    result1 = skeleton_30pct._maybe_subset(ds, "train")
    result2 = skeleton_30pct._maybe_subset(ds, "train")
    assert result1.indices == result2.indices


def test_maybe_subset_train_and_val_get_different_indices(skeleton_30pct):
    """Train and val subsets should have different indices (different seeds)."""
    n = 100
    ds = torch.utils.data.TensorDataset(
        torch.zeros(n, 1, 8, 8), torch.zeros(n, 4)
    )
    train_subset = skeleton_30pct._maybe_subset(ds, "train")
    val_subset = skeleton_30pct._maybe_subset(ds, "val")
    # Both should have indices but they should differ
    assert train_subset.indices != val_subset.indices


def test_maybe_subset_empty_dataset_returns_original(skeleton_30pct):
    """An empty dataset should be returned as-is to avoid division issues."""
    ds = torch.utils.data.TensorDataset(
        torch.zeros(0, 1, 8, 8), torch.zeros(0, 4)
    )
    result = skeleton_30pct._maybe_subset(ds, "train")
    assert result is ds


def test_load_audio_dataset_with_subset_applies_to_lazy_index(skeleton_30pct, tmp_path):
    """load_audio_dataset with 30% subset wraps a LazyMelDataset in Subset."""
    n = 50
    _write_fake_index(tmp_path, n_samples=n)
    result = skeleton_30pct.load_audio_dataset(tmp_path, "train")
    expected = max(1, n * 30 // 100)
    assert len(result) == expected
    assert isinstance(result, torch.utils.data.Subset)
