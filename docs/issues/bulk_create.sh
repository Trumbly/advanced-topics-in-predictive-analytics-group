#!/usr/bin/env bash
# Bulk-create GitHub issues from docs/issues/*.md.
# Requires: gh CLI authenticated against Trumbly/advanced-topics-in-predictive-analytics-group.
# Usage:  bash docs/issues/bulk_create.sh
# Safe to re-run: skips titles that already exist as open issues.

set -euo pipefail

REPO="Trumbly/advanced-topics-in-predictive-analytics-group"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

labels=(
  "track-foundation"
  "track-loop"
  "track-task"
  "track-output"
  "track-ui-ops"
  "p0"
  "p1"
  "p2"
)

milestones=(
  "week-1-foundations"
  "week-2-loop-and-task"
  "week-3-integration"
  "week-4-ui-and-demo"
)

echo "==> Ensuring labels exist"
for l in "${labels[@]}"; do
  gh label create "$l" --repo "$REPO" --force >/dev/null 2>&1 || true
done

echo "==> Ensuring milestones exist"
for m in "${milestones[@]}"; do
  gh api "repos/$REPO/milestones" -f title="$m" >/dev/null 2>&1 || true
done

# Map milestone name -> number
declare -A MSNUM
while IFS=$'\t' read -r num title; do
  MSNUM["$title"]=$num
done < <(gh api "repos/$REPO/milestones" --jq '.[] | "\(.number)\t\(.title)"')

existing_titles="$(gh issue list --repo "$REPO" --state all --limit 500 --json title --jq '.[].title')"

for file in "$DIR"/I-*.md; do
  # First line holds the title:  "# I-01 — Settings module (YAML-only)"
  title_line="$(head -n1 "$file" | sed -E 's/^#\s*//')"

  if grep -Fxq "$title_line" <<<"$existing_titles"; then
    echo "skip (exists): $title_line"
    continue
  fi

  body="$(sed '1d' "$file")"   # drop the title line, keep the rest as body

  # Extract labels + milestone from the body via regex (one-line fields).
  lbl_line="$(grep -m1 '^\*\*Labels:\*\*' "$file" || true)"
  ms_line="$(grep -m1 '^\*\*Milestone:\*\*' "$file" || true)"

  # Parse `**Labels:** `a`, `b`` → a,b
  label_args=""
  if [[ -n "$lbl_line" ]]; then
    labels_csv="$(sed -E 's/.*Labels:\*\*\s*//; s/`//g; s/,\s*/,/g' <<<"$lbl_line")"
    IFS=',' read -ra arr <<<"$labels_csv"
    for l in "${arr[@]}"; do label_args+=" --label $l"; done
  fi

  milestone_arg=""
  if [[ -n "$ms_line" ]]; then
    ms_name="$(sed -E 's/.*Milestone:\*\*\s*//' <<<"$ms_line")"
    if [[ -n "${MSNUM[$ms_name]:-}" ]]; then
      milestone_arg="--milestone ${MSNUM[$ms_name]}"
    fi
  fi

  echo "create: $title_line"
  # shellcheck disable=SC2086
  gh issue create --repo "$REPO" \
      --title "$title_line" \
      --body "$body" \
      $label_args $milestone_arg
done

echo "==> Done"
