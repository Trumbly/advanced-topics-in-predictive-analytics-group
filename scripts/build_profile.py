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


def _load_secondary_labels(raw_dir: Path) -> dict[str, list[str]]:
    """Parse train.csv to extract secondary labels per filename.

    Returns a dict mapping ``<class_id>/<filename>`` (the ``filename``
    column in train.csv) to a list of secondary species IDs.  Files
    without secondary labels map to an empty list.
    """
    train_csv = raw_dir / "train.csv"
    if not train_csv.exists():
        return {}

    import ast

    secondary: dict[str, list[str]] = {}
    with train_csv.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            fn = row.get("filename", "")
            sl = row.get("secondary_labels", "[]")
            try:
                labels = ast.literal_eval(sl)
            except (ValueError, SyntaxError):
                labels = []
            if isinstance(labels, list) and labels:
                secondary[fn] = [str(l) for l in labels]

    print(f"  secondary labels: {len(secondary)} files have overlapping species")
    return secondary


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

    # Load secondary labels from train.csv (species that co-occur in the same recording)
    secondary_labels = _load_secondary_labels(raw_dir)

    labels_csv = processed_dir / "labels.csv"
    secondary_count = 0
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

            # Look up secondary labels for this file
            # train.csv uses "<class_id>/<filename>" as the filename column
            rel_filename = f"{class_id}/{audio_path.name}"
            sec_labels = secondary_labels.get(rel_filename, [])

            base_id = audio_path.stem
            for window_idx, spec in enumerate(specs):
                sample_id = (
                    f"{base_id}_w{window_idx:03d}"
                    if len(specs) > 1
                    else base_id
                )
                np.save(spec_dir / f"{sample_id}.npy", spec)
                # Primary label
                writer.writerow([sample_id, class_id])
                # Secondary labels (co-occurring species in the same recording)
                for sl in sec_labels:
                    writer.writerow([sample_id, sl])
                    secondary_count += 1

    print(f"  wrote {secondary_count} secondary label entries")

    # Extract labeled segments from train soundscapes
    _preprocess_soundscapes(raw_dir, processed_dir, labels_csv, pipeline)

    return labels_csv


def _parse_time(t: str) -> float:
    """Parse HH:MM:SS to seconds."""
    parts = t.strip().split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


def _preprocess_soundscapes(
    raw_dir: Path,
    processed_dir: Path,
    labels_csv: Path,
    pipeline: AudioPipeline,
) -> None:
    """Extract labeled segments from train_soundscapes_labels.csv.

    Each row in the CSV specifies a soundscape file, a 5-second time
    window (start/end), and one or more species labels (semicolon-
    separated).  We load just that segment, compute its mel spectrogram,
    save it as a .npy, and append label rows to labels.csv.

    This is the ONLY source of training data for the 28 species that
    have no dedicated focal recordings in train_audio/.
    """
    labels_file = raw_dir / "train_soundscapes_labels.csv"
    soundscapes_dir = raw_dir / "train_soundscapes"

    if not labels_file.exists():
        print("  train_soundscapes_labels.csv not found — skipping soundscape extraction")
        return
    if not soundscapes_dir.exists():
        print("  train_soundscapes/ directory not found — skipping soundscape extraction")
        return

    import librosa

    spec_dir = processed_dir / "spectrograms"
    sr = pipeline.config.sample_rate

    # Read all labeled segments
    segments: list[dict[str, str]] = []
    with labels_file.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            segments.append(row)

    print(f"  extracting {len(segments)} labeled soundscape segments...")

    added = 0
    with labels_csv.open("a", newline="") as fh:
        writer = csv.writer(fh)

        for seg in tqdm(segments, desc="Soundscapes"):
            filename = seg["filename"]
            audio_path = soundscapes_dir / filename
            if not audio_path.exists():
                continue

            start_sec = _parse_time(seg["start"])
            end_sec = _parse_time(seg["end"])
            duration = end_sec - start_sec
            species_list = seg["primary_label"].split(";")

            try:
                y, _ = librosa.load(
                    str(audio_path),
                    sr=sr,
                    mono=True,
                    offset=start_sec,
                    duration=duration,
                )
            except Exception as e:
                print(
                    f"WARN: skipping {audio_path} segment {start_sec}-{end_sec} "
                    f"({type(e).__name__}: {e})",
                    file=sys.stderr,
                )
                continue

            if len(y) < sr:  # less than 1 second of audio
                continue

            # Pad to exact window length if needed
            window_samples = int(sr * duration)
            if len(y) < window_samples:
                padded = np.zeros(window_samples, dtype=np.float32)
                padded[: len(y)] = y
                y = padded
            else:
                y = y[:window_samples]

            # Compute mel spectrogram using the same pipeline config
            spec = pipeline.to_mel_spectrogram(y)

            # Save spectrogram with a unique ID
            sample_id = f"sc_{Path(filename).stem}_{int(start_sec):06d}"
            np.save(spec_dir / f"{sample_id}.npy", spec)

            # Write a label row for each species in this segment
            for sp in species_list:
                sp = sp.strip()
                if sp:
                    writer.writerow([sample_id, sp])
                    added += 1

    print(f"  added {added} soundscape label entries from {len(segments)} segments")


def _read_all_species(sample_submission_csv: Path) -> list[str]:
    """Extract the full competition species list from sample_submission.csv.

    The CSV header is ``row_id,species1,species2,...,speciesN``.
    Returns sorted species IDs (everything after ``row_id``).
    """
    with Path(sample_submission_csv).open() as fh:
        header = fh.readline().strip().split(",")
    # First column is row_id; the rest are species codes
    species = [col for col in header[1:] if col]
    print(f"  competition species from sample_submission.csv: {len(species)}")
    return species


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
    parser.add_argument(
        "--sample-submission",
        type=Path,
        default=None,
        help=(
            "Path to sample_submission.csv from the competition. "
            "If provided, the profile will include ALL competition "
            "species (even those with 0 training samples), ensuring "
            "num_classes matches the full submission requirement."
        ),
    )

    args = parser.parse_args()

    # Auto-detect sample_submission.csv if not explicitly provided
    if args.sample_submission is None:
        auto_path = args.raw_dir / "sample_submission.csv"
        if auto_path.exists():
            args.sample_submission = auto_path
            print(f"Auto-detected sample_submission.csv at {auto_path}")

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

    # Read full competition species list if available
    all_classes = None
    if args.sample_submission is not None:
        all_classes = _read_all_species(args.sample_submission)

    print(f"Building dataset profile...")
    profile = build_dataset_profile(
        args.processed_dir,
        labels_csv,
        sample_rate=args.sample_rate,
        split_seed=args.split_seed,
        val_fraction=args.val_fraction,
        all_classes=all_classes,
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
