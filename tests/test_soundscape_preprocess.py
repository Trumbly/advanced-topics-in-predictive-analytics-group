"""Soundscape ingest: train.csv + train_soundscapes_labels.csv -> unified labels."""

from __future__ import annotations

from pathlib import Path

import pytest

from lab.tasks.soundscape_preprocess import (
    build_unified_labels,
    ingest_soundscape_labels,
    ingest_train_csv,
    load_canonical_class_order,
    merge_label_sources,
    write_multilabel_labels_csv,
)


# ---------- canonical ordering ----------


def test_canonical_class_order_skips_row_id_column(tmp_path):
    sub = tmp_path / "sample_submission.csv"
    sub.write_text(
        "row_id,1161364,116570,1176823\n"
        "BC_x_0001_5,0.0,0.0,0.0\n"
    )
    classes = load_canonical_class_order(sub)
    assert classes == ["1161364", "116570", "1176823"]


def test_canonical_class_order_rejects_unexpected_header(tmp_path):
    sub = tmp_path / "sample_submission.csv"
    sub.write_text("not_row_id,a,b\n0,0,0\n")
    with pytest.raises(ValueError, match="header"):
        load_canonical_class_order(sub)


# ---------- train.csv ingest ----------


def test_ingest_train_csv_strips_extension_and_directory(tmp_path):
    train = tmp_path / "train.csv"
    train.write_text(
        "primary_label,secondary_labels,filename\n"
        "1161364,[],1161364/iNat1114648.ogg\n"
        "1161364,[],1161364/iNat1216197.ogg\n"
        "116570,[],116570/sub/iNat99.wav\n"
    )
    out = ingest_train_csv(train)
    assert list(out.keys()) == ["iNat1114648", "iNat1216197", "iNat99"]
    assert out["iNat1114648"] == ["1161364"]
    assert out["iNat99"] == ["116570"]


def test_ingest_train_csv_skips_blank_primary_label(tmp_path):
    train = tmp_path / "train.csv"
    train.write_text(
        "primary_label,filename\n"
        "1161364,1161364/iNat1.ogg\n"
        ",1161364/iNat2.ogg\n"
    )
    out = ingest_train_csv(train)
    assert "iNat1" in out
    assert "iNat2" not in out


def test_ingest_train_csv_expands_to_window_sids_when_mels_dir_given(tmp_path):
    """When spectrograms_dir is provided, each train.csv row must expand
    into one entry per existing <base>_w<idx>.npy on disk -- the original
    BirdCLEF pipeline slices per-clip audio into 5-second windows, and the
    labels.csv sids have to match those window filenames or the lazy
    DataLoader hits FileNotFoundError at runtime."""
    train = tmp_path / "train.csv"
    train.write_text(
        "primary_label,filename\n"
        "A,A/iNat100.ogg\n"
        "B,B/iNat200.ogg\n"
        "C,C/iNat300.ogg\n"   # has no mels on disk -> dropped
    )
    mels = tmp_path / "spectrograms"
    mels.mkdir()
    for idx in range(3):
        (mels / f"iNat100_w{idx:03d}.npy").touch()
    (mels / "iNat200_w000.npy").touch()
    # iNat300: no mels -> the row should be dropped silently

    out = ingest_train_csv(train, spectrograms_dir=mels)
    assert list(out.keys()) == [
        "iNat100_w000",
        "iNat100_w001",
        "iNat100_w002",
        "iNat200_w000",
    ]
    assert out["iNat100_w000"] == ["A"]
    assert out["iNat200_w000"] == ["B"]


def test_ingest_train_csv_legacy_clip_level_when_no_mels_dir(tmp_path):
    """Backward-compat path: without ``spectrograms_dir`` the function
    still emits one entry per train.csv row keyed by the file stem (used
    by older tests + callers that haven't been updated yet)."""
    train = tmp_path / "train.csv"
    train.write_text(
        "primary_label,filename\nA,A/iNat100.ogg\n"
    )
    out = ingest_train_csv(train)
    assert out == {"iNat100": ["A"]}


def test_build_unified_labels_emits_window_sids_with_spectrograms_dir(tmp_path):
    """End-to-end: build_unified_labels must forward spectrograms_dir to
    ingest_train_csv so the resulting labels.csv has window-level sids
    matching the on-disk mels."""
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "sample_submission.csv").write_text("row_id,A,B\n_,0,0\n")
    (raw / "train.csv").write_text(
        "primary_label,filename\nA,A/iNat100.ogg\n"
    )
    (raw / "train_soundscapes_labels.csv").write_text(
        "filename,start,end,primary_label\n"
        "S.ogg,00:00:00,00:00:05,A;B\n"
    )
    mels = tmp_path / "spectrograms"
    mels.mkdir()
    (mels / "iNat100_w000.npy").touch()
    (mels / "iNat100_w001.npy").touch()
    (mels / "S_w000.npy").touch()

    from lab.tasks.soundscape_preprocess import build_unified_labels

    out_path = tmp_path / "labels.csv"
    merged = build_unified_labels(
        raw_dir=raw, out_path=out_path, spectrograms_dir=mels
    )
    assert "iNat100_w000" in merged and "iNat100_w001" in merged
    assert "iNat100" not in merged   # no clip-level sid leaks through
    assert "S_w000" in merged

    text = out_path.read_text(encoding="utf-8")
    assert "iNat100_w000,A\n" in text
    assert "iNat100_w001,A\n" in text


# ---------- soundscape ingest ----------


def test_ingest_soundscape_labels_assigns_window_index(tmp_path):
    csv_path = tmp_path / "train_soundscapes_labels.csv"
    csv_path.write_text(
        "filename,start,end,primary_label\n"
        "BC2026_Train_0039_S22_20211231_201500.ogg,00:00:00,00:00:05,a;b\n"
        "BC2026_Train_0039_S22_20211231_201500.ogg,00:00:05,00:00:10,b\n"
        "BC2026_Train_0039_S22_20211231_201500.ogg,00:00:55,00:01:00,c;d;e\n"
    )
    out = ingest_soundscape_labels(csv_path)
    base = "BC2026_Train_0039_S22_20211231_201500"
    assert f"{base}_w000" in out
    assert f"{base}_w001" in out
    assert f"{base}_w011" in out
    assert out[f"{base}_w000"] == ["a", "b"]
    assert out[f"{base}_w011"] == ["c", "d", "e"]


def test_ingest_soundscape_labels_dedupes_within_window(tmp_path):
    csv_path = tmp_path / "ss.csv"
    csv_path.write_text(
        "filename,start,end,primary_label\n"
        "BC.ogg,00:00:00,00:00:05,a;b\n"
        "BC.ogg,00:00:00,00:00:05,b;c\n"
    )
    out = ingest_soundscape_labels(csv_path)
    assert out["BC_w000"] == ["a", "b", "c"]


# ---------- merge + filter ----------


def test_merge_drops_labels_not_in_canonical_set(tmp_path):
    src1 = ingest_train_csv(_write(tmp_path / "t.csv", [
        "primary_label,filename",
        "1161364,foo/iNat1.ogg",
        "999999,foo/iNat2.ogg",   # not in canonical
    ]))
    merged = merge_label_sources(src1, canonical_classes=["1161364", "116570"])
    assert "iNat1" in merged
    assert "iNat2" not in merged
    assert merged["iNat1"] == ["1161364"]


def test_merge_unions_overlapping_sample_ids():
    src_a = {"sid_x": ["a", "b"]}
    src_b = {"sid_x": ["b", "c"]}
    out = merge_label_sources(src_a, src_b, canonical_classes=["a", "b", "c"])
    assert out["sid_x"] == ["a", "b", "c"]


# ---------- top-level + recovery ----------


def test_build_unified_labels_recovers_no_train_species(tmp_path):
    """Reproduces the BirdCLEF-2026 split: 234 canonical, 206 in train.csv,
    28 only in soundscape labels. The unified file MUST cover all 234."""
    raw = tmp_path / "raw"
    raw.mkdir()

    # 4 canonical species: A, B (have train rows), C (soundscape-only),
    # D (soundscape-only) -- mirrors the production 206/28 ratio at toy scale.
    (raw / "sample_submission.csv").write_text(
        "row_id,A,B,C,D\nBC_Test_0001_S05_5,0,0,0,0\n"
    )
    (raw / "train.csv").write_text(
        "primary_label,filename\n"
        "A,A/iNat100.ogg\n"
        "A,A/iNat101.ogg\n"
        "B,B/iNat200.ogg\n"
    )
    (raw / "train_soundscapes_labels.csv").write_text(
        "filename,start,end,primary_label\n"
        "BC2026_Train_X.ogg,00:00:00,00:00:05,A;C\n"
        "BC2026_Train_X.ogg,00:00:05,00:00:10,D\n"
    )

    merged = build_unified_labels(
        raw_dir=raw, out_path=tmp_path / "labels.csv"
    )

    seen_classes = {c for v in merged.values() for c in v}
    assert seen_classes == {"A", "B", "C", "D"}, "all 4 canonical species recovered"

    # train.csv samples
    assert merged["iNat100"] == ["A"]
    assert merged["iNat200"] == ["B"]
    # soundscape windows
    assert merged["BC2026_Train_X_w000"] == ["A", "C"]
    assert merged["BC2026_Train_X_w001"] == ["D"]


def test_build_unified_labels_writes_correct_csv_format(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "sample_submission.csv").write_text("row_id,A,B\n_,0,0\n")
    (raw / "train.csv").write_text(
        "primary_label,filename\nA,A/x.ogg\n"
    )
    (raw / "train_soundscapes_labels.csv").write_text(
        "filename,start,end,primary_label\n"
        "S.ogg,00:00:00,00:00:05,A;B\n"
    )

    out_path = tmp_path / "labels_multilabel.csv"
    build_unified_labels(raw_dir=raw, out_path=out_path)

    text = out_path.read_text(encoding="utf-8")
    assert text.startswith("sample_id,class_ids\n")
    assert "x,A\n" in text
    assert "S_w000,A;B\n" in text


def test_write_multilabel_labels_csv_handles_empty_dict(tmp_path):
    p = tmp_path / "empty.csv"
    write_multilabel_labels_csv(p, {})
    assert p.read_text(encoding="utf-8") == "sample_id,class_ids\n"


def test_build_unified_labels_raises_when_no_sources(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "sample_submission.csv").write_text("row_id,A\n_,0\n")
    with pytest.raises(FileNotFoundError):
        build_unified_labels(raw_dir=raw, out_path=tmp_path / "out.csv")


# ---------- helpers ----------


def _write(p: Path, lines: list[str]) -> Path:
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p
