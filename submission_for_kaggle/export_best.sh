#!/usr/bin/env bash
# Export the best experiment from the most recent study as a Kaggle notebook
# and copy it into this folder for easy upload.
#
# Usage:
#   bash submission_for_kaggle/export_best.sh
#   bash submission_for_kaggle/export_best.sh <study_id>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEST_DIR="$SCRIPT_DIR"

cd "$REPO_ROOT"

# Run the submit command (exports notebook to experiments/studies/<study>/submissions/)
if [ $# -ge 1 ]; then
    echo "Exporting best experiment from study: $1"
    python -m agent.main submit "$1"
else
    echo "Exporting best experiment from most recent study..."
    python -m agent.main submit
fi

# Find the most recently created submission notebook
LATEST_NB=$(find experiments/studies -name "*_submission.ipynb" -type f -print0 \
    | xargs -0 ls -t 2>/dev/null | head -1)

if [ -z "$LATEST_NB" ]; then
    echo "ERROR: No submission notebook found. Did the export succeed?"
    exit 1
fi

# Copy to this folder
BASENAME=$(basename "$LATEST_NB")
cp "$LATEST_NB" "$DEST_DIR/$BASENAME"

echo ""
echo "=========================================="
echo "  Kaggle notebook ready!"
echo "=========================================="
echo ""
echo "  File: submission_for_kaggle/$BASENAME"
echo ""
echo "  Next steps:"
echo "    1. Go to https://www.kaggle.com/competitions/birdclef-2026/submit"
echo "    2. Upload: submission_for_kaggle/$BASENAME"
echo "    3. Wait for Kaggle to run it (< 90 min)"
echo "    4. Check your score on the leaderboard"
echo ""
