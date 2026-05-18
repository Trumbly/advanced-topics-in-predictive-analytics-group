# Autonomous Research Agent — BirdCLEF+ 2026

An autonomous research agent that designs, trains, evaluates, and iterates on
deep-learning models for the
[BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026) Kaggle
competition, driven by a locally-hosted Large Language Model (Ollama).
Built for the *Advanced Topics in Predictive Analytics* course, 2025/2026.

The agent runs a closed loop:

1. **Propose** a new architecture (LLM prompt grounded in EDA + memory of past runs).
2. **Generate** the training code by filling a Jinja2 skeleton.
3. **Validate** the code (AST checks + smoke forward pass).
4. **Execute** the experiment in a subprocess sandbox with a watchdog.
5. **Recover** from common failures by auto-fix or LLM re-prompt.
6. **Judge** the result, update memory, repeat.

Reports, Kaggle submission notebooks, and a live FastAPI dashboard fall out the other side.

> **Course report:** the 10-page project report lives at `docs/report/report.md`
> (Word build: `docs/report/report.docx`). It is the D4 deliverable and complements
> this README.

## Quickstart (≤5 minutes, no Kaggle data needed)

```bash
# 1. Install uv + create the venv
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv
uv pip install -e .

# 2. Start a local Ollama server in another terminal
ollama pull gemma4:e4b      # the default; any chat-capable Ollama model works
ollama serve

# 3. End-to-end smoke against synthetic shards
bash scripts/demo_run.sh
```

`demo_run.sh` runs `preprocess --synthetic` + a 2-experiment study + report + submission
build, end-to-end, in ≤5 minutes on a fresh CPU clone. Open
[http://127.0.0.1:8000](http://127.0.0.1:8000) after `python -m lab ui` for the dashboard.

## Setup with real BirdCLEF+ 2026 data

### 1. Python environment

```bash
uv venv .venv
uv pip install -e .

# verify (must report librosa+soundfile too — needed for mel generation)
.venv/bin/python -c "import torch, librosa, soundfile; print('ok')"
```

If you prefer conda, any 3.10+ env with the `pyproject.toml` deps installed
works. Substitute the python path in `PYTHON=...` for the mel build script.

### 2. Local LLM (Ollama)

```bash
ollama pull gemma4:e4b       # default — set in config/config.yaml
ollama serve                 # leave running in another terminal
```

Switch model per-run with `--llm-model <tag>` or globally in
`config/config.yaml:llm.model`.

### 3. Raw data

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

### 4. Build the mel cache + label index

```bash
./scripts/build_all_mels.sh
```

What it runs:

| Step | Command | Output | Time |
|---|---|---|---|
| 1 | `lab preprocess --train-audio` | ~233 k per-clip mels under `data/processed/spectrograms/` | ~2–4 h CPU |
| 2 | `lab preprocess --soundscapes` | 739 soundscape window mels (closes the 28-species gap) | ~5–15 min |
| 3 | `lab preprocess --unify-labels` | `data/processed/labels.csv` with canonical 234-class union | ~5 s |
| 4 | `lab preprocess --overwrite` | `train_index.json` + `val_index.json` lazy index | ~30 s |

Why step 2 matters: the legacy per-clip cache misses 28 species that only
appear in soundscapes. Step 2 closes that gap so all 234 target species
show up in training.

Useful env vars:

```bash
SKIP_TRAIN_AUDIO=1 ./scripts/build_all_mels.sh                                   # rebuild only soundscapes + labels
OVERWRITE=1 ./scripts/build_all_mels.sh                                           # force-rewrite every step
PYTHON=/opt/miniconda3/envs/birdclef/bin/python ./scripts/build_all_mels.sh       # different interpreter
```

Mel parameters are pinned in `lab.tasks.audio_mels.MelParams` (sr=32 kHz,
n_fft=2048, hop=512, n_mels=128, fmin=20, fmax=16 kHz, dB scaling) and
produce shape `(128, 313)` for a 5 s window.

## CLI reference

`python -m lab <subcommand>` (or simply `lab <subcommand>` after `pip install -e .`).

| Subcommand | What it does |
|---|---|
| `lab run --task track_b --max-experiments N` | launch an autonomous study with N experiments |
| `lab report <study_id>` | render the study's Markdown + HTML report under `experiments/studies/<id>/` |
| `lab submit <study_id> [--experiment-id <eid>]` | build the Kaggle submission notebook for the study's best (or a chosen) experiment |
| `lab prompts list` | list every prompt task and its active version |
| `lab prompts activate <task> <version>` | set the active version for a prompt task |
| `lab prompts new <task> --system-file SYS --user-file USR` | save a new prompt version |
| `lab preprocess [flags…]` | one-off mel + label-index build (see `--help` for every flag) |
| `lab benchmark` | print a cross-study leaderboard grouped by (architecture family, name) |
| `lab ui [--host H] [--port P]` | start the FastAPI dashboard |

Run any subcommand with `--help` for the full flag set.

### Common `lab run` flags

| Flag | Default | Purpose |
|---|---|---|
| `--task` | `track_b` | which `config/tasks/*.yaml` to load |
| `--predecessor <id>` | — | seed memory from a previous study |
| `--use-best-prompts` | off | pick the highest-scoring prompt version per task |
| `--agent-memory` | off | enable top-K wins + recent-failures memory |
| `--personality` | `exploratory` | `exploratory` vs `conservative` proposal style |
| `--max-experiments` | from config | hard cap on the experiment count |
| `--max-wallclock-min` | from config | hard cap on wall-clock minutes |
| `--llm-model <tag>` | from config | override the Ollama model for this run only |

## UI tour

`python -m lab ui` starts a FastAPI + HTMX dashboard on
[http://127.0.0.1:8000](http://127.0.0.1:8000). Key pages:

| Path | What it shows |
|---|---|
| `/` and `/studies` | every study under `experiments/studies/`, newest first |
| `/studies/<id>` | study detail: per-experiment cards, learning curves, judge verdicts |
| `/experiments/<study_id>/<exp_id>` | full experiment detail incl. live stdout tail |
| `/prompts` | prompt registry: list versions per task, set active, view A/B scores |
| `/prompts/<task>/<version>` | edit a prompt version |
| `/benchmark` | cross-study (family, architecture) leaderboard |
| `/dashboard` | KPIs over all studies (success rate, mean score, etc.) |
| `/settings` | edit `config/config.yaml` from the browser |
| `/new` | launch form: pick task, personality, prompt versions, model |
| `/live/<study_id>` | Server-Sent Events stream tailing `run.log.jsonl` |
| `/studies/<id>/submission` (POST) | build the Kaggle notebook + local CSV + weights for the study's best experiment |

The UI is read-mostly over the JSON study state on disk, so the same data is
also exposed under `/api/*` for scripted access.

## Architecture (one-screen overview)

| Layer | Module | Responsibility |
|---|---|---|
| Config | `lab.config` | single typed `Settings` from `config/*.yaml` |
| LLM | `lab.core.llm` | provider-agnostic `chat()` (Ollama/OpenAI/Anthropic) |
| Prompts | `lab.prompts.*` | versioned registry, slot-filled engine, A/B scoring |
| Memory | `lab.core.memory` | top-K wins + recent failures, ≤3 kB markdown |
| Tasks | `lab.tasks.*` | `TaskAdapter` ABC + BirdCLEF implementation |
| Validation | `lab.core.validator` | AST checks + smoke forward pass |
| Execution | `lab.core.executor` | subprocess sandbox + error classification |
| Recovery | `lab.core.recovery` | auto-fix + LLM re-prompt |
| Loop | `lab.core.lifecycle` | study runner + experiment orchestration |
| Judge | `lab.core.judge` | per-experiment + per-study verdicts |
| Reporting | `lab.reporting.*` | figures + Jinja2 markdown report |
| Submission | `lab.submission.*` | CPU-only Kaggle notebook builder |
| UI | `lab.ui.*` | FastAPI + HTMX dashboard, SSE live log |
| CLI | `lab.cli` | argparse entry point |

Full design rationale: `docs/ARCHITECTURE.md`. Per-component contracts and
ADRs are in the same file.

## Tests

```bash
.venv/bin/python -m pytest -q
```

The suite covers every layer above. ~400 tests, runs in under a minute.

## Repository layout

```
config/           # YAML configs: global, per-task, prompt registry, code skeletons
data/             # raw/ (gitignored) and processed/ (gitignored) data caches
docs/             # ARCHITECTURE.md, REDESIGN_PLAN.md, report/, issues/, PROJECT_NOTES.md
experiments/      # study artefacts (study.json, reports, submissions) — tracked
lab/              # the package: cli.py, config.py, core/, prompts/, tasks/, reporting/, submission/, ui/
sandbox/          # gitignored subprocess scratch (per-experiment stdout, weights, etc.)
scripts/          # build_all_mels.sh, demo_run.sh, plus the project-report helper scripts
tests/            # the test suite
```

## Further reading

- `docs/ARCHITECTURE.md` — design, ADRs, every contract
- `docs/report/report.md` — the 10-page project report (D4 deliverable)
- `docs/PROJECT_NOTES.md` — internal status, rubric mapping, team table
- `docs/issues/` — per-issue specifications, one PR per issue

## License

See `LICENSE`.
