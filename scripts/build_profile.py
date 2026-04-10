"""Preprocess BirdCLEF audio and build the DatasetProfile.

Reads raw audio from `data/raw/`, runs it through the AudioPipeline to
produce mel-spectrograms in `data/processed/spectrograms/`, writes a
labels CSV to `data/processed/labels.csv`, and finally builds a
DatasetProfile at `data/processed/dataset_profile.json`.

This script runs ONCE before any agent experiments. The agent never
re-runs preprocessing — it only reads the precomputed artifacts.

Usage (full dataset):
    python scripts/build_profile.py \
        --raw-dir data/raw \
        --processed-dir data/processed \
        --sample-rate 32000 --n-mels 128

Usage (quick subset for smoke tests):
    python scripts/build_profile.py --sample 100

Input layout (typical BirdCLEF):
    data/raw/
        train_audio/
            <species_code>/
                <audio_id>.ogg
        train.csv (optional — provides canonical sample_id -> class mapping)

Output layout:
    data/processed/
        spectrograms/
            <sample_id>.npy            # one spectrogram per window
        labels.csv                     # columns: sample_id, class_id
        dataset_profile.json
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Make the repo root importable so that `pipelines` and `agent` resolve
# when the script is invoked as `python scripts/build_profile.py`.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
from tqdm import tqdm  # noqa: E402

from pipelines.audio_pipeline import AudioPipeline, AudioPipelineConfig  # noqa: E402
from pipelines.dataset_profile import build_dataset_profile  # noqa: E402


AUDIO_EXTENSIONS = {".ogg", ".wav", ".mp3", ".flac"}


def _discover_audio(raw_dir: Path) -> list[tuple[Path, str]]:
    """Find audio files under `raw_dir` and infer class from parent folder.

    Returns a list of `(audio_path, class_id)` tuples. BirdCLEF organizes
    training audio under `train_audio/<species_code>/*.ogg`, so the
    parent folder name is the class id.
    """
    train_audio = raw_dir / "train_audio"
    if not train_audio.exists():
        train_audio = raw_dir  # allow flat layout for tests
    out: list[tuple[Path, str]] = []
    for path in train_audio.rglob("*"):
        if path.suffix.lower() in AUDIO_EXTENSIONS:
            class_id = path.parent.name
            out.append((path, class_id))
    return sorted(out)


def preprocess_audio(
    raw_dir: Path,
    processed_dir: Path,
    *,
    sample: int | None = None,
    config: AudioPipelineConfig | None = None,
) -> Path:
    """Precompute spectrograms for every audio file.

    Returns the path to the generated labels CSV.
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    spec_dir = processed_dir / "spectrograms"
    spec_dir.mkdir(parents=True, exist_ok=True)

    pipeline = AudioPipeline(config or AudioPipelineConfig())
    audio_files = _discover_audio(raw_dir)
    if sample is not None:
        audio_files = audio_files[:sample]

    if not audio_files:
        raise FileNotFoundError(f"No audio files found under {raw_dir}")

    labels_csv = processed_dir / "labels.csv"
    with labels_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sample_id", "class_id"])

        for audio_path, class_id in tqdm(audio_files, desc="Preprocessing"):
            try:
                specs = pipeline.process_file(audio_path)
            except Exception as e:  # noqa: BLE001 — robust preprocessing
                print(
                    f"WARN: skipping {audio_path} ({type(e).__name__}: {e})",
                    file=sys.stderr,
                )
                continue

            base_id = audio_path.stem
            for window_idx, spec in enumerate(specs):
                sample_id = (
                    f"{base_id}_w{window_idx:03d}"
                    if len(specs) > 1
                    else base_id
                )
                np.save(spec_dir / f"{sample_id}.npy", spec)
                writer.writerow([sample_id, class_id])

    return labels_csv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--processed-dir", type=Path, default=Path("data/processed")
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Process only the first N audio files (for smoke tests)",
    )
    parser.add_argument("--sample-rate", type=int, default=32_000)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--window-seconds", type=float, default=5.0)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)

    args = parser.parse_args()

    config = AudioPipelineConfig(
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        window_seconds=args.window_seconds,
    )

    labels_csv = preprocess_audio(
        args.raw_dir,
        args.processed_dir,
        sample=args.sample,
        config=config,
    )

    print(f"Building dataset profile...")
    profile = build_dataset_profile(
        args.processed_dir,
        labels_csv,
        sample_rate=args.sample_rate,
        split_seed=args.split_seed,
        val_fraction=args.val_fraction,
    )

    out_path = args.processed_dir / "dataset_profile.json"
    profile.to_json_file(out_path)
    print(
        f"\nDone.\n"
        f"  classes: {profile.num_classes}\n"
        f"  samples: {profile.num_samples}\n"
        f"  spectrogram shape: {profile.spectrogram_shape}\n"
        f"  imbalance ratio: {profile.imbalance_ratio:.2f}\n"
        f"  train / val: {len(profile.train_indices)} / {len(profile.val_indices)}\n"
        f"  profile written to: {out_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
