# I-DEMO — End-to-end smoke + README + video prep

**Labels:** `track-ui-ops`, `p0`
**Milestone:** week-4-ui-and-demo
**Owner:** All devs (pair)

## Context
D1 ("works out of the box") and D3 (video) depend on this. If the reviewer clones and runs, it must complete without intervention.

## Scope
End-to-end smoke script, polished README, 5-min demo plan.

## Deliverables

### 1. Smoke script
`scripts/demo_run.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
python -m lab preprocess --task track_b --synthetic
python -m lab run --task track_b --max-experiments 2
LAST=$(ls -1t experiments/studies/ | head -1)
python -m lab report "$LAST"
python -m lab submit "$LAST"
echo "Demo OK — study: $LAST"
```

### 2. README
Must include:
- One-paragraph description
- Setup: `uv sync`, `ollama pull <model>`, data placement (or `--synthetic` flag)
- Run: `python -m lab run --task track_b`
- UI: `python -m lab ui`
- Link to `docs/ARCHITECTURE.md`, `docs/PRODUCTION_REVIEW.md`
- Sprint / deliverables mapping to grading rubric

### 3. Video plan (5 min)
- 0:00–0:45 architecture diagram (from ARCHITECTURE §3)
- 0:45–2:30 live run — show memory + judge + recovery
- 2:30–3:30 report page + best learning curve + per-class AUC
- 3:30–4:30 prompt dashboard + "use best"
- 4:30–5:00 honest limitations (CPU, no cross-task, no Kaggle auto-push)

## Acceptance
- Fresh clone on a clean macOS laptop: `bash scripts/demo_run.sh` finishes ≤5 min.
- README runs command verbatim.
- Video recorded, linked in README.

## Depends on
All other issues merged.
