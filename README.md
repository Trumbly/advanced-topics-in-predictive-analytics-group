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

### 1. Python environment

The project uses `uv` to manage a venv that lives next to the source tree at
`.venv/`.

```bash
# install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# in the repo root: create the venv + install lab + all deps from pyproject.toml
uv venv .venv
uv pip install -e .

# verify (must report librosa+soundfile too -- needed for mel generation)
.venv/bin/python -c "import torch, librosa, soundfile; print('ok')"
```

If you prefer conda, any 3.10+ env with the `pyproject.toml` deps installed
works. Substitute the python path in `PYTHON=...` for any command below.

### 2. Local LLM (Ollama)

```bash
ollama pull gemma4                # default; any chat-capable Ollama model works
ollama serve                      # leave running in another terminal
```

### 3. BirdCLEF+ 2026 raw data

Download the competition dump from Kaggle and place it under `data/raw/`:

```
data/raw/
  train.csv
  taxonomy.csv
  sample_submission.csv               # canonical 234-class column set
  train_soundscapes_labels.csv         # multi-label per 5s window
  train_audio/<class_id>/<sid>.ogg     # ~35 k single-label clips
  train_soundscapes/<filename>.ogg     # ~10 k 60s soundscape recordings
```

(For the synthetic smoke path you can skip this and run with `--synthetic`.)

### 4. Build the mel cache + label index

The per-clip cache from the off-repo legacy pipeline does **not** include the
739 soundscape windows that cover the 28 species which are absent from
`train.csv` (Insect sonotypes + 3 Amphibia). The script below rebuilds the
full cache so all 234 target species appear in training:

```bash
./scripts/build_all_mels.sh
```

What it runs:

| Step | Command | Output | Time |
|---|---|---|---|
| 1 | `lab preprocess --train-audio` | ~233 k per-clip mels under `data/processed/spectrograms/` | ~2-4 h CPU |
| 2 | `lab preprocess --soundscapes` | 739 soundscape window mels (closes 28-species gap) | ~5-15 min |
| 3 | `lab preprocess --unify-labels` | `data/processed/labels.csv` with canonical 234-class union | ~5 s |
| 4 | `lab preprocess --overwrite` | `train_index.json` + `val_index.json` lazy index | ~30 s |

Useful env vars:

```bash
SKIP_TRAIN_AUDIO=1 ./scripts/build_all_mels.sh   # keep existing per-clip cache, rebuild only soundscapes + labels
OVERWRITE=1 ./scripts/build_all_mels.sh           # force-rewrite every step
PYTHON=/opt/miniconda3/envs/birdclef/bin/python ./scripts/build_all_mels.sh   # different interpreter
```

Mel parameters are pinned in `lab.tasks.audio_mels.MelParams`
(sr=32 kHz, n_fft=2048, hop=512, n_mels=128, fmin=20, fmax=16 kHz, dB scaling)
and produce shape `(128, 313)` for a 5 s window. New mels are interchangeable
with anything the legacy pipeline produced.

## Run

```bash
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
