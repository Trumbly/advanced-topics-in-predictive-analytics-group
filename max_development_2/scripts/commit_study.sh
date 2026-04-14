#!/usr/bin/env bash
# Commit a single study + the prompt versions that produced it.
#
# Usage: ./scripts/commit_study.sh <study_id>

set -euo pipefail

STUDY_ID="${1:-}"
if [[ -z "${STUDY_ID}" ]]; then
  echo "usage: $0 <study_id>" >&2
  exit 1
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

STUDY_DIR="max_development_2/experiments/studies/${STUDY_ID}"
if [[ ! -d "${STUDY_DIR}" ]]; then
  echo "no such study: ${STUDY_DIR}" >&2
  exit 1
fi

git add "${STUDY_DIR}" "max_development_2/config/prompts/_registry.yaml"

# Extract a one-line metadata summary from study.json for the commit msg.
META=$(python3 -c "
import json, sys, pathlib
p = pathlib.Path('${STUDY_DIR}/study.json')
d = json.loads(p.read_text())
name = d.get('name', '${STUDY_ID}')
task = d.get('task_name', '?')
n = len(d.get('experiments', []))
best = d.get('best_score')
best_s = f'{best:.4f}' if best is not None else '—'
print(f'{name} | task={task} | n={n} | best={best_s}')
")

git commit -m "study: ${META}" -m "ID: ${STUDY_ID}"
echo "committed study ${STUDY_ID}"
