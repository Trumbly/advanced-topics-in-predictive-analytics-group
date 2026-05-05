# Autonomous Research Agent — Track B (BirdCLEF+ 2026)

Advanced Predictive Analytics 2025/2026 — group project.

This package (`lab/`) implements an autonomous research agent that designs,
trains, evaluates, and iterates on deep learning models for the
[BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026) Kaggle
competition, driven by a locally-hosted Large Language Model.

> **Status:** rewrite implemented across 24 GitHub issues
> (`docs/issues/I-01..I-21` + `I-DELETE` + `I-DEMO` + #40 + #41), every
> component covered by tests. See `docs/REDESIGN_PLAN.md` for the rewrite
> blueprint, `docs/TEAM_PLAN.md` for issue ownership, and the merged PRs
> on the `feature/rewrite` branch for the work.

## Setup

```bash
# 1. Clone + create environment
git clone <repo-url> && cd advanced-topics-in-predictive-analytics-group
uv sync                              # or: pip install -e .[dev]

# 2. Local LLM (Ollama recommended)
ollama pull gemma4                   # or any model from docs/REDESIGN_PLAN.md

# 3. Place BirdCLEF data under data/raw/birdclef-2026/
#    (or use --synthetic for the smoke path)
```

## Run

```bash
python -m lab preprocess --task track_b              # one-time mel cache
python -m lab run --task track_b --max-experiments 5 # autonomous study
python -m lab report <study_id>                      # build report
python -m lab submit <study_id>                      # build Kaggle notebook
python -m lab benchmark                              # cross-study leaderboard
python -m lab ui                                     # FastAPI dashboard
```

### One-shot smoke

```bash
bash scripts/demo_run.sh
```

Runs preprocess (synthetic shards) + a 2-experiment study + report + submission
build, end-to-end, in ≤5 minutes on a fresh clone.

Full CLI reference lives in `docs/issues/I-17-cli.md`.

## Grading rubric mapping

| PDF rubric component | Where to look |
|---|---|
| Agent design & implementation (40%) | `lab/core/{lifecycle,experiment,parsing,recovery,judge}.py`, `docs/ARCHITECTURE.md` |
| Model performance (20%) | `experiments/studies/<id>/study.json`, `lab.core.benchmark` |
| Use of course content (15%) | `config/skeletons/audio_multilabel.py.j2`, `docs/REDESIGN_PLAN.md` §11 |
| Report & video (25%) | `lab/reporting/`, `scripts/demo_run.sh`, `docs/issues/I-DEMO-end-to-end.md` |

## Architecture

| Layer | Module | Responsibility |
|---|---|---|
| Config | `lab.config` | Single typed `Settings` from `config/*.yaml` |
| LLM | `lab.core.llm` | Provider-agnostic `chat()` (Ollama/OpenAI/Anthropic) |
| Prompts | `lab.prompts.*` | Versioned registry, slot-filled engine, A/B scoring |
| Memory | `lab.core.memory` | Top-K wins + recent failures, ≤3 kB markdown |
| Tasks | `lab.tasks.*` | `TaskAdapter` ABC + BirdCLEF implementation |
| Validation | `lab.core.validator` | AST checks + smoke forward pass |
| Execution | `lab.core.executor` | Subprocess sandbox + error classification |
| Recovery | `lab.core.recovery` | Auto-fix + LLM re-prompt |
| Loop | `lab.core.lifecycle` | Study runner + experiment orchestration |
| Judge | `lab.core.judge` | Per-experiment + per-study verdicts |
| Reporting | `lab.reporting.*` | Figures + Jinja2 markdown report |
| Submission | `lab.submission.*` | CPU-only Kaggle notebook builder |
| UI | `lab.ui.*` | FastAPI + HTMX dashboard, SSE live log |
| CLI | `lab.cli` | argparse entrypoint |

See `docs/ARCHITECTURE.md` for the full design and ADRs.

## Video plan (5 min)

0:00–0:45 — architecture diagram (`docs/ARCHITECTURE.md` §3)
0:45–2:30 — live `lab run` showing memory, judge, recovery
2:30–3:30 — report page: best learning curve + per-class AUC
3:30–4:30 — prompt dashboard, "use best prompts" + benchmark page
4:30–5:00 — honest limitations (CPU, no Track A, no Kaggle auto-push)

## Project Documents

- `docs/REDESIGN_PLAN.md` — rewrite blueprint and design principles
- `docs/PRODUCTION_REVIEW.md` — gap analysis vs PDF rubric
- `docs/ARCHITECTURE.md` — current architecture and contracts
- `docs/TEAM_PLAN.md` — issue assignment across team accounts
- `docs/issues/` — per-issue specifications (one PR each)

## Team

- Trumbly
- danish-m-qureshi
- Lorry171717
- SebastianMis23

## License

See `LICENSE`.
