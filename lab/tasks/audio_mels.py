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


def build_audio_mels(
    audio_dir: Path,
    out_dir: Path,
    *,
    target_sids: Iterable[str] | None = None,
    params: MelParams = MelParams(),
    overwrite: bool = False,
    recursive: bool = False,
    label: str = "audio",
) -> dict[str, int]:
    """Slice ``audio_dir/*.ogg`` (or recursive glob when ``recursive=True``)
    into consecutive ``params.window_seconds``-second windows and emit
    ``<file_stem>_w<idx:03d>.npy`` into ``out_dir``.

    Two real call sites:

    - per-clip mode (``recursive=True``) for ``data/raw/train_audio/<class>/<sid>.ogg``
      which has class-id subdirectories.
    - soundscape mode (``recursive=False``, default) for the flat
      ``data/raw/train_soundscapes/*.ogg`` layout.

    ``target_sids``: when supplied, only the matching window IDs are written;
    everything else is counted as ``skipped_unrequested``. Useful for
    targeting only the ~739 *labelled* soundscape windows out of the ~127 k
    that exist for the full corpus.

    Returns a counter dict (keys ``files_processed`` / ``built`` /
    ``skipped_existing`` / ``skipped_short`` / ``skipped_unrequested``).

    Raises ``ImportError`` with an actionable hint when ``librosa`` is not
    installed.
    """
    try:
        import librosa  # type: ignore[import-not-found]
        import numpy as np
        import soundfile as sf  # noqa: F401  # pulled in by librosa, listed explicitly
    except ImportError as exc:
        raise ImportError(
            "audio mel generation requires `librosa` + `soundfile`. "
            "Install with `uv pip install -e .` "
            "(or `pip install librosa soundfile`)."
        ) from exc

    audio_dir = Path(audio_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not audio_dir.exists():
        raise FileNotFoundError(f"missing audio dir: {audio_dir}")

    target_set = set(target_sids) if target_sids is not None else None

    counts = {
        "files_processed": 0,
        "built": 0,
        "skipped_existing": 0,
        "skipped_short": 0,
        "skipped_unrequested": 0,
    }

    audio_paths = sorted(_iter_audio_files(audio_dir, recursive=recursive))
    if not audio_paths:
        _LOG.warning("no audio files under %s", audio_dir)
        return counts

    win_samples = params.sr * params.window_seconds

    _LOG.info(
        "%s mel build: %d files in %s -> %s",
        label,
        len(audio_paths),
        audio_dir,
        out_dir,
    )

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

        if counts["files_processed"] % 50 == 0:
            _LOG.info(
                "  %s progress: %d/%d files (built=%d, skipped_existing=%d)",
                label,
                counts["files_processed"],
                len(audio_paths),
                counts["built"],
                counts["skipped_existing"],
            )

    _LOG.info(
        "%s mel build done: %d files -> %d new mels "
        "(skipped %d existing, %d unrequested, %d too-short)",
        label,
        counts["files_processed"],
        counts["built"],
        counts["skipped_existing"],
        counts["skipped_unrequested"],
        counts["skipped_short"],
    )
    return counts


def build_soundscape_mels(
    raw_dir: Path,
    out_dir: Path,
    *,
    target_sids: Iterable[str] | None = None,
    params: MelParams = MelParams(),
    overwrite: bool = False,
    soundscapes_subdir: str = "train_soundscapes",
) -> dict[str, int]:
    """Soundscape-mode wrapper around :func:`build_audio_mels`.

    Kept as a thin alias so existing callers and CLI flags continue to work.
    """
    return build_audio_mels(
        Path(raw_dir) / soundscapes_subdir,
        out_dir,
        target_sids=target_sids,
        params=params,
        overwrite=overwrite,
        recursive=False,
        label="soundscape",
    )


def build_train_audio_mels(
    raw_dir: Path,
    out_dir: Path,
    *,
    params: MelParams = MelParams(),
    overwrite: bool = False,
    train_audio_subdir: str = "train_audio",
) -> dict[str, int]:
    """Per-clip mode: ``data/raw/train_audio/<class_id>/<sid>.ogg`` is sliced
    into 5 s windows just like soundscape audio. Yields the same
    ``<sid>_w<idx>.npy`` filenames the existing 233 k cache uses, so a fresh
    rebuild is interchangeable with the legacy off-repo pipeline.
    """
    return build_audio_mels(
        Path(raw_dir) / train_audio_subdir,
        out_dir,
        target_sids=None,
        params=params,
        overwrite=overwrite,
        recursive=True,
        label="train_audio",
    )


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


def _iter_audio_files(audio_dir: Path, *, recursive: bool = False):
    iter_fn = audio_dir.rglob if recursive else audio_dir.glob
    for ext in ("*.ogg", "*.OGG", "*.wav", "*.WAV", "*.flac", "*.mp3"):
        yield from iter_fn(ext)
