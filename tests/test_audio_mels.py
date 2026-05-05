"""Soundscape mel builder: synthetic 60s WAV -> 12 windows of (128, 313)."""

from __future__ import annotations

from pathlib import Path

import pytest

# Skip the whole module when librosa is not installed -- this is a research
# branch and CI may not have audio libs. The build is exercised by the user
# locally in the .venv where librosa was installed by hand.
librosa = pytest.importorskip("librosa")
soundfile = pytest.importorskip("soundfile")

import numpy as np

from lab.tasks.audio_mels import (
    MelParams,
    build_soundscape_mels,
    mel_spec_db,
    soundscape_sids_for_labels,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _write_sine_wav(path: Path, duration_s: int, sr: int = 32_000, freq: float = 440.0):
    """Synthesise a constant-frequency sine wave -- avoids the OGG codec
    dependency at test time while still feeding librosa real audio."""
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, duration_s, sr * duration_s, endpoint=False, dtype="float32")
    audio = 0.5 * np.sin(2 * np.pi * freq * t).astype("float32")
    soundfile.write(str(path), audio, samplerate=sr, format="WAV")


@pytest.fixture
def raw_dir(tmp_path):
    raw = tmp_path / "raw"
    audio = raw / "train_soundscapes"
    _write_sine_wav(
        audio / "BC2026_Train_0001_S08_20250606_030007.wav", 60
    )
    _write_sine_wav(
        audio / "BC2026_Train_0002_S08_20250606_030007.wav", 30  # only 6 windows
    )
    _write_sine_wav(
        audio / "BC2026_Train_0003_S08_20250606_030007.wav", 4   # too short
    )
    return raw


# ---------------------------------------------------------------------------
# transform
# ---------------------------------------------------------------------------


def test_mel_spec_db_yields_expected_shape():
    params = MelParams()
    audio = np.zeros(params.sr * params.window_seconds, dtype="float32")
    mel = mel_spec_db(audio, params)
    assert mel.shape == (128, 313)


def test_mel_params_documented_constants():
    p = MelParams()
    # If any of these change without a deliberate reason, the existing
    # 233 k mel cache stops being interchangeable with newly-built ones.
    assert (p.sr, p.n_fft, p.hop_length, p.n_mels) == (32_000, 2_048, 512, 128)
    assert p.expected_shape == (128, 313)


# ---------------------------------------------------------------------------
# end-to-end build
# ---------------------------------------------------------------------------


def test_build_soundscape_mels_emits_per_window_npy(raw_dir, tmp_path):
    out = tmp_path / "spectrograms"
    counts = build_soundscape_mels(raw_dir, out)

    # 12 windows from 60s + 6 from 30s + 0 from 4s
    assert counts["built"] == 12 + 6
    assert counts["skipped_short"] == 1

    # naming convention <basename>_w<idx:03d>.npy
    f1 = "BC2026_Train_0001_S08_20250606_030007"
    assert (out / f"{f1}_w000.npy").exists()
    assert (out / f"{f1}_w011.npy").exists()
    assert not (out / f"{f1}_w012.npy").exists()

    # shape matches (128, 313)
    arr = np.load(out / f"{f1}_w000.npy")
    assert arr.shape == (128, 313)
    assert arr.dtype == np.float32


def test_build_soundscape_mels_targets_subset_only(raw_dir, tmp_path):
    out = tmp_path / "spectrograms"
    target = {
        "BC2026_Train_0001_S08_20250606_030007_w005",
        "BC2026_Train_0002_S08_20250606_030007_w001",
    }
    counts = build_soundscape_mels(raw_dir, out, target_sids=target)
    assert counts["built"] == 2
    assert counts["skipped_unrequested"] == (12 + 6) - 2

    files = {p.stem for p in out.iterdir()}
    assert files == target


def test_build_soundscape_mels_idempotent_skip(raw_dir, tmp_path):
    out = tmp_path / "spectrograms"
    build_soundscape_mels(raw_dir, out)
    counts = build_soundscape_mels(raw_dir, out)  # second pass
    assert counts["built"] == 0
    assert counts["skipped_existing"] == 12 + 6


def test_build_soundscape_mels_overwrite_rebuilds(raw_dir, tmp_path):
    out = tmp_path / "spectrograms"
    build_soundscape_mels(raw_dir, out)
    # poison one file
    target = out / "BC2026_Train_0001_S08_20250606_030007_w000.npy"
    target.write_bytes(b"\x00not a real npy")

    counts = build_soundscape_mels(raw_dir, out, overwrite=True)
    assert counts["built"] == 12 + 6
    assert counts["skipped_existing"] == 0
    arr = np.load(target)
    assert arr.shape == (128, 313)


def test_build_raises_on_missing_audio_dir(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing soundscape"):
        build_soundscape_mels(tmp_path, tmp_path / "out")


# ---------------------------------------------------------------------------
# label-driven targeting
# ---------------------------------------------------------------------------


def test_soundscape_sids_for_labels_dedupes_and_uses_window_index(tmp_path):
    csv_path = tmp_path / "ss.csv"
    csv_path.write_text(
        "filename,start,end,primary_label\n"
        "BC2026_Train_X.ogg,00:00:00,00:00:05,a;b\n"
        "BC2026_Train_X.ogg,00:00:05,00:00:10,c\n"
        "BC2026_Train_X.ogg,00:00:00,00:00:05,b;d\n"  # dup row
    )
    sids = soundscape_sids_for_labels(csv_path)
    assert sids == {
        "BC2026_Train_X_w000",
        "BC2026_Train_X_w001",
    }
