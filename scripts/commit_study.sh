#!/usr/bin/env bash
# Commit all results for a single study + the prompt config that produced them.
# Usage: ./scripts/commit_study.sh <study_id>
set -e

STUDY_ID="${1:?Usage: commit_study.sh <study_id>}"

if [ ! -d "experiments/studies/$STUDY_ID" ]; then
    echo "Error: experiments/studies/$STUDY_ID does not exist"
    exit 1
fi

git add "experiments/studies/$STUDY_ID/"
git add "config/prompts/"
git commit -m "study: $STUDY_ID results

$(cat experiments/studies/$STUDY_ID/study.json 2>/dev/null | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(f'Name: {d.get(\"name\",\"?\")}'  )
print(f'Status: {d.get(\"status\",\"?\")}'  )
print(f'Experiments: {len(d.get(\"experiment_ids\",[]))}'  )
print(f'Best score: {d.get(\"best_score\",\"n/a\")}'  )
" 2>/dev/null || echo "(study metadata unavailable)")"

echo "Committed study $STUDY_ID"
