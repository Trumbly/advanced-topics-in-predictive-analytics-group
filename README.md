# Autonomous Research Agent — Track B (BirdCLEF+ 2026)

Advanced Predictive Analytics 2025/2026 — group project.

This package (`lab/`) implements an autonomous research agent that designs,
trains, evaluates, and iterates on deep learning models for the
[BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026) Kaggle
competition, driven by a locally-hosted Large Language Model.

> **Status:** ground-up rewrite per spec. The agent is implemented across
> 24 GitHub issues — see `docs/issues/` for per-component specifications and
> `docs/REDESIGN_PLAN.md` for the rewrite blueprint. The current
> `feature/rewrite` branch is the integration target; each issue lands via
> its own `feature/I-XX-*` branch and pull request.

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
python -m lab ui                                     # FastAPI dashboard
```

Full CLI reference lives in `docs/issues/I-17-cli.md`.

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
