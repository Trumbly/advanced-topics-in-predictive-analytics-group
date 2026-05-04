#!/usr/bin/env bash
# Smoke run for the autonomous research agent.
# Designed for a fresh clone on macOS / Linux; finishes in ≤5 minutes on CPU.
set -euo pipefail

cd "$(dirname "$0")/.."

# 1. preprocess synthetic shards (skip when real BirdCLEF data already cached)
python -m lab preprocess --task track_b --synthetic

# 2. run a tiny study (2 experiments)
python -m lab run --task track_b --max-experiments 2 --max-wallclock-min 5

# 3. pick the most recent study and render its report + submission
LAST="$(ls -1t experiments/studies | grep '^study_' | head -1)"
test -n "$LAST" || { echo "no study found"; exit 1; }

python -m lab report "$LAST"
python -m lab submit "$LAST" || echo "submit step skipped (best experiment may be unavailable)"

echo "Demo OK — study: $LAST"
