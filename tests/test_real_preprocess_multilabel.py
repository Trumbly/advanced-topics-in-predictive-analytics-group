"""real_preprocess: multi-label labels.csv + canonical 234-class ordering."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lab.tasks.real_preprocess import (
    _read_labels,
    _read_sample_labels,
    build_real_shards,
)
from lab.config import load_settings

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------- _read_labels accepts both schemas ----------


def test_read_labels_legacy_single_column(tmp_path):
    p = tmp_path / "labels.csv"
    p.write_text(
        "sample_id,class_id\n"
        "iNat1,A\n"
        "iNat2,A\n"
        "iNat3,B\n"
    )
    out = _read_labels(p)
    assert out == {"A": ["iNat1", "iNat2"], "B": ["iNat3"]}


def test_read_labels_multi_column(tmp_path):
    p = tmp_path / "labels.csv"
    p.write_text(
        "sample_id,class_ids\n"
        "iNat1,A\n"
        "iNat2,A;B\n"
        "iNat3,B;C\n"
    )
    out = _read_labels(p)
    assert sorted(out["A"]) == ["iNat1", "iNat2"]
    assert sorted(out["B"]) == ["iNat2", "iNat3"]
    assert out["C"] == ["iNat3"]


def test_read_sample_labels_returns_per_sample_lists(tmp_path):
    p = tmp_path / "labels.csv"
    p.write_text(
        "sample_id,class_ids\n"
        "iNat1,A\n"
        "iNat2,A;B;C\n"
    )
    out = _read_sample_labels(p)
    assert out["iNat1"] == ["A"]
    assert out["iNat2"] == ["A", "B", "C"]


def test_read_labels_rejects_unknown_header(tmp_path):
    p = tmp_path / "labels.csv"
    p.write_text("sample_id,foo\nx,y\n")
    with pytest.raises(ValueError, match="class_id"):
        _read_labels(p)


# ---------- end-to-end lazy index with canonical 234 ordering ----------


def _settings_in(tmp_path: Path):
    """Stage a tmp data layout so build_real_shards uses the canonical
    sample_submission.csv-driven ordering."""
    s = load_settings("track_b", repo_root=REPO_ROOT)
    processed_dir = tmp_path / "data" / "processed" / "mels"
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    new_paths = s.paths.model_copy(update={"sandbox": str(tmp_path / "sandbox")})
    new_task = s.task.model_copy(
        update={
            "processed_data_dir": str(processed_dir),
            "expected_num_classes": 4,
            "input_tensor_shape": [1, 8, 8],
        }
    )
    return s.model_copy(update={"paths": new_paths, "task": new_task})


def _seed_dataset(tmp_path: Path) -> None:
    """4 canonical classes (A,B,C,D); A+B in train.csv, C+D only via
    multi-label soundscape rows. After preprocessing, ALL four must be
    represented in the lazy index (otherwise the no-train-species fix
    has regressed)."""
    raw = tmp_path / "data" / "raw"
    proc = tmp_path / "data" / "processed"
    spec = proc / "spectrograms"
    spec.mkdir(parents=True)

    (raw / "sample_submission.csv").write_text(
        "row_id,A,B,C,D\n_,0,0,0,0\n"
    )

    rng = np.random.default_rng(0)
    for sid in (
        "iNat100", "iNat101", "iNat102", "iNat200", "iNat201",
        "S_w000", "S_w001",
    ):
        np.save(spec / f"{sid}.npy", rng.standard_normal((1, 8, 8)).astype("float32"))

    (proc / "labels.csv").write_text(
        "sample_id,class_ids\n"
        "iNat100,A\n"
        "iNat101,A\n"
        "iNat102,A\n"
        "iNat200,B\n"
        "iNat201,B\n"
        "S_w000,A;C\n"   # multi-label window: known A + previously-missing C
        "S_w001,D\n"     # soundscape-only D
    )


def test_lazy_index_emits_class_indices_and_covers_canonical_classes(tmp_path):
    s = _settings_in(tmp_path)
    _seed_dataset(tmp_path)

    train_path, val_path = build_real_shards(s, val_fraction=0.2)
    train = json.loads(train_path.read_text())
    val = json.loads(val_path.read_text())

    # canonical 4-class ordering from sample_submission.csv (A,B,C,D)
    assert train["num_classes"] == 4

    all_entries = train["samples"] + val["samples"]
    sids = {e["sid"] for e in all_entries}
    assert sids == {"iNat100", "iNat101", "iNat102", "iNat200", "iNat201",
                    "S_w000", "S_w001"}

    # multi-label entry must carry list, not single value
    s_w000 = next(e for e in all_entries if e["sid"] == "S_w000")
    assert s_w000["class_indices"] == [0, 2]   # A=0, C=2 (canonical order)

    # union of class_indices across all entries must include all 4 canonical
    # class indices, proving C and D were recovered from the multi-label path
    seen_indices = {ci for e in all_entries for ci in e.get("class_indices", [])}
    assert seen_indices == {0, 1, 2, 3}


def test_lazy_index_splits_each_sample_only_once(tmp_path):
    """With multi-label samples, an entry must end up in either train or val
    -- never both -- so the same .npy is not seen during evaluation."""
    s = _settings_in(tmp_path)
    _seed_dataset(tmp_path)

    train_path, val_path = build_real_shards(s, val_fraction=0.5)
    train_sids = {e["sid"] for e in json.loads(train_path.read_text())["samples"]}
    val_sids = {e["sid"] for e in json.loads(val_path.read_text())["samples"]}
    assert train_sids.isdisjoint(val_sids)
