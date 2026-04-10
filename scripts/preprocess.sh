#!/usr/bin/env bash
# One-shot preprocessing: download BirdCLEF data and build the DatasetProfile.
# Run this ONCE before starting the agent loop.
#
# Usage:
#   bash scripts/preprocess.sh               # full dataset
#   bash scripts/preprocess.sh --sample 100  # subset for smoke tests
#
# Prerequisites:
#   - Python environment with dependencies installed (`pip install -r requirements.txt`)
#   - Kaggle CLI configured (`pip install kaggle` + ~/.kaggle/kaggle.json)
#   - BirdCLEF 2026 competition rules accepted at kaggle.com

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Step 1/2: downloading BirdCLEF 2026 data"
if [ -d "data/raw/train_audio" ] && [ "$(find data/raw/train_audio -type f 2>/dev/null | head -n 1)" != "" ]; then
    echo "    data/raw/train_audio already exists — skipping download"
else
    python scripts/download_data.py --dest data/raw
fi

echo "==> Step 2/2: preprocessing audio + building dataset profile"
python scripts/build_profile.py --raw-dir data/raw --processed-dir data/processed "$@"

echo "==> Done. Dataset profile at: data/processed/dataset_profile.json"
