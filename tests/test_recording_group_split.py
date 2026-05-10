"""Recording-level group split + EDA num_train sanity checks.

The mel pipeline emits sids of the form ``<recording>_w<idx>``. A
sample-level random split lets adjacent windows of one recording leak
across train/val and inflates ROC-AUC. ``_build_lazy_index`` and
``regroup_lazy_indexes`` must keep every recording on a single side of
the split. ``BirdclefAdapter.profile`` must surface a non-zero
``num_train`` derived from the lazy indexes when the optional
metadata.parquet sidecar is absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lab.config import Settings, load_settings
from lab.tasks.real_preprocess import (
    _recording_id,
    regroup_lazy_indexes,
)
from lab.tasks.track_b_birdclef import BirdclefAdapter


def _recording_ids(payload: dict) -> set[str]:
    return {_recording_id(s["sid"]) for s in payload["samples"]}


def _settings(processed_dir: Path, base: Settings) -> Settings:
    """Clone settings with a tmp processed_data_dir so tests don't see
    the developer's real cache."""
    task = base.task.model_copy(update={"processed_data_dir": str(processed_dir)})
    return base.model_copy(update={"task": task})


def _seed_indexes(processed_dir: Path) -> None:
    """Write a deliberately leaked train/val pair: every recording has
    windows w000..w003, alternating train/val."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_samples: list[dict] = []
    val_samples: list[dict] = []
    for rec_idx in range(20):
        rec = f"rec{rec_idx:03d}"
        for w in range(4):
            sid = f"{rec}_w{w:03d}"
            target = train_samples if w % 2 == 0 else val_samples
            target.append({"sid": sid, "class_indices": [rec_idx % 5]})
    (processed_dir / "train_index.json").write_text(
        json.dumps(
            {
                "spectrograms_dir": str(processed_dir),
                "num_classes": 5,
                "samples": train_samples,
            }
        )
    )
    (processed_dir / "val_index.json").write_text(
        json.dumps(
            {
                "spectrograms_dir": str(processed_dir),
                "num_classes": 5,
                "samples": val_samples,
            }
        )
    )


def test_recording_id_strips_window_suffix():
    assert _recording_id("iNat818781_w003") == "iNat818781"
    assert _recording_id("BC2026_Train_0042_w011") == "BC2026_Train_0042"
    # Legacy single-window sids without ``_w<n>`` suffix pass through.
    assert _recording_id("XC1234567") == "XC1234567"


def test_regroup_separates_recordings(tmp_path: Path):
    base = load_settings("track_b")
    processed = tmp_path / "mels"
    _seed_indexes(processed)
    settings = _settings(processed, base)

    # Sanity: the seed has a leaked split — every recording on both sides.
    train_before = json.loads((processed / "train_index.json").read_text())
    val_before = json.loads((processed / "val_index.json").read_text())
    overlap_before = _recording_ids(train_before) & _recording_ids(val_before)
    assert len(overlap_before) > 0, "fixture must start leaked"

    train_path, val_path = regroup_lazy_indexes(settings, val_fraction=0.25)

    train_after = json.loads(train_path.read_text())
    val_after = json.loads(val_path.read_text())

    train_recs = _recording_ids(train_after)
    val_recs = _recording_ids(val_after)

    # The fix: zero overlap.
    assert train_recs.isdisjoint(val_recs)

    # Every original sid is preserved exactly once.
    all_sids = {s["sid"] for s in train_after["samples"]} | {
        s["sid"] for s in val_after["samples"]
    }
    expected_sids = {s["sid"] for s in train_before["samples"]} | {
        s["sid"] for s in val_before["samples"]
    }
    assert all_sids == expected_sids


def test_regroup_raises_when_indexes_missing(tmp_path: Path):
    base = load_settings("track_b")
    settings = _settings(tmp_path / "empty", base)
    with pytest.raises(FileNotFoundError):
        regroup_lazy_indexes(settings)


def test_adapter_profile_reports_lazy_index_count(tmp_path: Path):
    base = load_settings("track_b")
    processed = tmp_path / "mels"
    _seed_indexes(processed)
    settings = _settings(processed, base)

    adapter = BirdclefAdapter(settings)
    profile = adapter.profile()

    # 20 recordings * 4 windows = 80 sids spread across train + val.
    assert profile.num_train == 80
    assert profile.num_classes == settings.task.expected_num_classes
