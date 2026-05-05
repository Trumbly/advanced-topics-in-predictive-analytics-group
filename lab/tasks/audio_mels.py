"""Mel-spectrogram generation for BirdCLEF soundscape recordings.

The original 233 k single-clip mels under ``data/processed/spectrograms/`` were
produced by an external ``scripts/build_profile.py`` that does not live in this
repo. The soundscape recordings (60 s WAV/OGG under
``data/raw/train_soundscapes/``) were never sliced into mels, so the 28
soundscape-only target species had no training tensors -- this module fills
that gap with the SAME mel parameters used by the original pipeline so the
resulting ``.npy`` files are drop-in compatible::

    sr=32_000, n_fft=2_048, hop_length=512, n_mels=128,
    fmin=20, fmax=16_000, power_to_db(ref=max)

Output shape: ``(128, 313)`` for a 5 s window. The skeleton's
``LazyMelDataset`` adds the channel dim at load time.

`librosa` and `soundfile` are imported lazily so the rest of the lab does not
depend on them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_LOG = logging.getLogger("lab.preprocess.audio_mels")


@dataclass(frozen=True)
class MelParams:
    """Single source of truth for mel-spec generation.

    Defaults match the existing 233 k-file pipeline so newly-built soundscape
    mels are interchangeable with old per-clip mels.
    """

    sr: int = 32_000
    window_seconds: int = 5
    n_fft: int = 2_048
    hop_length: int = 512
    n_mels: int = 128
    fmin: int = 20
    fmax: int = 16_000
    expected_shape: tuple[int, int] = (128, 313)


# ---------------------------------------------------------------------------
# top-level
# ---------------------------------------------------------------------------


def build_soundscape_mels(
    raw_dir: Path,
    out_dir: Path,
    *,
    target_sids: Iterable[str] | None = None,
    params: MelParams = MelParams(),
    overwrite: bool = False,
    soundscapes_subdir: str = "train_soundscapes",
) -> dict[str, int]:
    """Slice soundscape audio into windows + write per-window mels.

    Walks ``raw_dir/soundscapes_subdir`` for audio files. For each, slices
    into consecutive ``params.window_seconds``-second windows and emits
    ``<file_stem>_w<idx:03d>.npy`` into ``out_dir``.

    ``target_sids`` (optional): only build the windows whose sid (e.g.
    ``BC2026_Train_0001_S08_..._w003``) appears in the set. Lets the caller
    skip unlabelled windows when disk pressure is a concern.

    Returns a counter dict with keys ``built`` / ``skipped_existing`` /
    ``skipped_short`` / ``skipped_unrequested`` / ``files_processed``.

    Raises ``ImportError`` (with an actionable hint) when ``librosa`` is not
    installed.
    """
    try:
        import librosa  # type: ignore[import-not-found]
        import numpy as np
        import soundfile as sf  # noqa: F401  # pulled in by librosa, listed explicitly
    except ImportError as exc:
        raise ImportError(
            "soundscape mel generation requires `librosa` + `soundfile`. "
            "Install with `.venv/bin/pip install librosa soundfile` "
            "(or `pip install -e .[dev]` once pyproject deps are picked up)."
        ) from exc

    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_dir = raw_dir / soundscapes_subdir
    if not audio_dir.exists():
        raise FileNotFoundError(
            f"missing soundscape audio dir: {audio_dir}"
        )

    target_set = set(target_sids) if target_sids is not None else None

    counts = {
        "files_processed": 0,
        "built": 0,
        "skipped_existing": 0,
        "skipped_short": 0,
        "skipped_unrequested": 0,
    }

    audio_paths = sorted(_iter_audio_files(audio_dir))
    if not audio_paths:
        _LOG.warning("no audio files under %s", audio_dir)
        return counts

    win_samples = params.sr * params.window_seconds

    for path in audio_paths:
        counts["files_processed"] += 1
        base = path.stem
        try:
            audio, _ = librosa.load(path, sr=params.sr, mono=True)
        except Exception as exc:  # corrupt OGG / unreadable -> skip
            _LOG.warning("failed to load %s: %s", path, exc)
            continue

        n_windows = len(audio) // win_samples
        if n_windows == 0:
            counts["skipped_short"] += 1
            continue

        for w in range(n_windows):
            sid = f"{base}_w{w:03d}"
            if target_set is not None and sid not in target_set:
                counts["skipped_unrequested"] += 1
                continue
            out_path = out_dir / f"{sid}.npy"
            if out_path.exists() and not overwrite:
                counts["skipped_existing"] += 1
                continue

            chunk = audio[w * win_samples : (w + 1) * win_samples]
            mel = mel_spec_db(chunk, params)
            if mel.shape != params.expected_shape:
                _LOG.warning(
                    "%s yielded shape %s, expected %s -- writing anyway",
                    sid,
                    mel.shape,
                    params.expected_shape,
                )
            np.save(out_path, mel.astype("float32"))
            counts["built"] += 1

        if counts["files_processed"] % 10 == 0:
            _LOG.info(
                "processed %d files (built=%d skipped_existing=%d)",
                counts["files_processed"],
                counts["built"],
                counts["skipped_existing"],
            )

    _LOG.info(
        "soundscape mel build done: %d files -> %d new mels "
        "(skipped %d existing, %d unrequested, %d too-short)",
        counts["files_processed"],
        counts["built"],
        counts["skipped_existing"],
        counts["skipped_unrequested"],
        counts["skipped_short"],
    )
    return counts


def mel_spec_db(audio, params: MelParams):
    """Mel-spec -> dB. Public so tests can pin the exact transform."""
    import librosa  # type: ignore[import-not-found]
    import numpy as np

    mel = librosa.feature.melspectrogram(
        y=np.asarray(audio, dtype="float32"),
        sr=params.sr,
        n_fft=params.n_fft,
        hop_length=params.hop_length,
        n_mels=params.n_mels,
        fmin=params.fmin,
        fmax=params.fmax,
    )
    return librosa.power_to_db(mel, ref=np.max)


def soundscape_sids_for_labels(soundscape_labels_path: Path) -> set[str]:
    """Return the labelled-window sid set so callers can target only the
    ~739 windows that actually have a label, skipping the ~10 k unlabelled
    ones."""
    from lab.tasks.soundscape_preprocess import ingest_soundscape_labels

    return set(ingest_soundscape_labels(soundscape_labels_path).keys())


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _iter_audio_files(audio_dir: Path):
    for ext in ("*.ogg", "*.OGG", "*.wav", "*.WAV", "*.flac", "*.mp3"):
        yield from audio_dir.glob(ext)
