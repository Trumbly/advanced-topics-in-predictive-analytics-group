#!/usr/bin/env bash
# Full preprocessing pipeline for BirdCLEF+ 2026 (Track B).
#
# Builds the mel-spectrogram cache + canonical 234-class label index from the
# raw Kaggle dataset shipped under ``data/raw/``::
#
#   data/raw/
#     train.csv
#     taxonomy.csv
#     sample_submission.csv
#     train_soundscapes_labels.csv
#     train_audio/<class_id>/<sample_id>.ogg
#     train_soundscapes/<filename>.ogg
#
# Pipeline:
#   1. lab preprocess --train-audio   -- per-clip mels (~35 k clips, ~233 k
#                                        windows, the bulk of training time).
#   2. lab preprocess --soundscapes   -- 739 soundscape window mels covering
#                                        the 28 soundscape-only species.
#   3. lab preprocess --unify-labels  -- merge train.csv + soundscape labels
#                                        into a single multi-label labels.csv
#                                        ordered by sample_submission columns.
#   4. lab preprocess --overwrite     -- rebuild the lazy index from the
#                                        unified labels.csv.
#
# Steps 1+2 produce the 234-class mel cache; steps 3+4 wire it up for the
# training skeleton.
#
# Usage::
#
#   ./scripts/build_all_mels.sh                    # use repo .venv
#   PYTHON=/opt/miniconda3/envs/birdclef/bin/python \
#       ./scripts/build_all_mels.sh                # explicit interpreter
#
# Set SKIP_TRAIN_AUDIO=1 to keep the existing per-clip cache (fast path when
# the legacy 233 k mels are already on disk and you only need the soundscape
# windows). Set OVERWRITE=1 to force rebuild every step.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
SKIP_TRAIN_AUDIO="${SKIP_TRAIN_AUDIO:-0}"
OVERWRITE="${OVERWRITE:-0}"
overwrite_flag=""
[ "$OVERWRITE" = "1" ] && overwrite_flag="--overwrite"

echo "[build_all_mels] repo:   $REPO_ROOT"
echo "[build_all_mels] python: $PYTHON"

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: python not found at $PYTHON. Bootstrap with:" >&2
    echo "    uv venv .venv && uv pip install -e ." >&2
    exit 1
fi

# Create the expected raw-data subdirectories up front so the Python steps
# below get a "dir exists, files missing" error instead of "dir missing".
# The user is still responsible for placing the Kaggle dump into them.
mkdir -p data/raw/train_audio data/raw/train_soundscapes

if ! "$PYTHON" -c "import librosa, soundfile" >/dev/null 2>&1; then
    echo "ERROR: librosa+soundfile missing. Run:" >&2
    echo "    uv pip install -e .   # picks up pyproject deps" >&2
    exit 1
fi

# Sanity-check that the user actually populated the raw dir before we burn
# minutes spinning up the Python entry points.
missing=()
[ -f "data/raw/train.csv" ] || missing+=("data/raw/train.csv")
[ -f "data/raw/sample_submission.csv" ] || missing+=("data/raw/sample_submission.csv")
[ -f "data/raw/train_soundscapes_labels.csv" ] || missing+=("data/raw/train_soundscapes_labels.csv")
if [ -z "$(find data/raw/train_audio -name '*.ogg' -print -quit 2>/dev/null)" ] \
   && [ "$SKIP_TRAIN_AUDIO" != "1" ]; then
    missing+=("data/raw/train_audio/<class_id>/<sid>.ogg  (no .ogg files found)")
fi
if [ -z "$(find data/raw/train_soundscapes -name '*.ogg' -print -quit 2>/dev/null)" ]; then
    missing+=("data/raw/train_soundscapes/<filename>.ogg  (no .ogg files found)")
fi

if [ "${#missing[@]}" -gt 0 ]; then
    echo "ERROR: raw BirdCLEF+ 2026 data is incomplete. Missing:" >&2
    for item in "${missing[@]}"; do echo "  - $item" >&2; done
    echo "" >&2
    echo "Download the competition dump from" >&2
    echo "  https://www.kaggle.com/competitions/birdclef-2026/data" >&2
    echo "and unzip it under data/raw/ so the layout matches the README." >&2
    echo "" >&2
    echo "If you only have the soundscape windows on disk and want to skip" >&2
    echo "the ~35 k per-clip train_audio build, rerun with:" >&2
    echo "  SKIP_TRAIN_AUDIO=1 ./scripts/build_all_mels.sh" >&2
    exit 1
fi

# -----------------------------------------------------------------------------
# 1. per-clip mels (the existing 233 k cache; opt-out for the fast path)
# -----------------------------------------------------------------------------
if [ "$SKIP_TRAIN_AUDIO" = "1" ]; then
    echo "[build_all_mels] step 1/4: --train-audio  SKIPPED (SKIP_TRAIN_AUDIO=1)"
else
    echo "[build_all_mels] step 1/4: --train-audio  (~35 k clips -> ~233 k mels)"
    "$PYTHON" -m lab preprocess --train-audio $overwrite_flag
fi

# -----------------------------------------------------------------------------
# 2. soundscape mels (closes the 28-species gap)
# -----------------------------------------------------------------------------
echo "[build_all_mels] step 2/4: --soundscapes  (~739 mels covering 234 classes)"
"$PYTHON" -m lab preprocess --soundscapes $overwrite_flag

# -----------------------------------------------------------------------------
# 3. unified labels.csv
# -----------------------------------------------------------------------------
echo "[build_all_mels] step 3/4: --unify-labels  (canonical 234-class union)"
"$PYTHON" -m lab preprocess --unify-labels

# -----------------------------------------------------------------------------
# 4. lazy index
# -----------------------------------------------------------------------------
echo "[build_all_mels] step 4/4: --overwrite  (rebuild train_index.json + val_index.json)"
"$PYTHON" -m lab preprocess --overwrite

echo "[build_all_mels] done."
echo "[build_all_mels] sanity-check:"
"$PYTHON" - <<'PY'
import json, csv
from pathlib import Path

labels = Path("data/processed/labels.csv")
train_idx = Path("data/processed/mels/train_index.json")
val_idx = Path("data/processed/mels/val_index.json")

with labels.open() as f:
    rows = list(csv.DictReader(f))
classes = {c for row in rows for c in row["class_ids"].split(";") if c}
print(f"  labels.csv:       {len(rows)} samples, {len(classes)} distinct classes")

for p in (train_idx, val_idx):
    payload = json.loads(p.read_text())
    print(
        f"  {p.name}: {len(payload['samples']):>7} samples, "
        f"num_classes={payload['num_classes']}"
    )
PY
