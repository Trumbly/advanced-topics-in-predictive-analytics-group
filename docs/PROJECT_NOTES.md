# Project Notes — Internal History

This file captures internal context that does not belong in the user-facing
README but is useful for graders and future maintainers.

## Status

Rewrite implemented across 24 GitHub issues (`docs/issues/I-01..I-21` +
`I-DELETE` + `I-DEMO` + #40 + #41), every component covered by tests.
See `docs/REDESIGN_PLAN.md` for the rewrite blueprint, `docs/TEAM_PLAN.md`
for issue ownership, and the merged PRs on the `feature/rewrite` branch
for the work.

## Grading rubric mapping

| PDF rubric component | Where to look |
|---|---|
| Agent design & implementation (40%) | `lab/core/{lifecycle,experiment,parsing,recovery,judge}.py`, `docs/ARCHITECTURE.md` |
| Model performance (20%) | `experiments/studies/<id>/study.json`, `lab.core.benchmark` |
| Use of course content (15%) | `config/skeletons/audio_multilabel.py.j2`, `docs/REDESIGN_PLAN.md` §11 |
| Report & video (25%) | `lab/reporting/`, `scripts/demo_run.sh`, `docs/issues/I-DEMO-end-to-end.md` |

## Video plan (5 min)

- 0:00–0:45 — architecture diagram (`docs/ARCHITECTURE.md` §3)
- 0:45–2:30 — live `lab run` showing memory, judge, recovery
- 2:30–3:30 — report page: best learning curve + per-class AUC
- 3:30–4:30 — prompt dashboard, "use best prompts" + benchmark page
- 4:30–5:00 — honest limitations (CPU, no Track A, no Kaggle auto-push)

## Team

- Trumbly
- danish-m-qureshi
- Lorry171717
- SebastianMis23

## Related documents

- `docs/REDESIGN_PLAN.md` — rewrite blueprint and design principles
- `docs/PRODUCTION_REVIEW.md` — gap analysis vs the PDF rubric
- `docs/ARCHITECTURE.md` — current architecture and ADRs
- `docs/TEAM_PLAN.md` — issue assignment across team accounts
- `docs/issues/` — per-issue specifications (one PR each)
- `docs/report/report.md` — the D4 deliverable report
