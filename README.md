# `lab` — task-agnostic LLM research agent (v2)

_Ground-up rewrite of `max_development/`. Same agent behaviour, no
BirdCLEF-specific hardcoding. Runs both Track A (Disaster Tweets) and
Track B (BirdCLEF+) out of the box._

> See [`docs/REDESIGN_PLAN.md`](docs/REDESIGN_PLAN.md) for the full design
> rationale and what changed vs. `max_development/`.

## TL;DR

```bash
# 0) Install
pip install -e ".[dev]"

# 1) Pick a task (default is track_b)
python -m lab tasks

# 2) Drop raw data in the expected folders
#    Track A: data/raw/track_a/{train,test}.csv
#    Track B: data/raw/track_b/{train_audio,train_metadata.csv,test_soundscapes}

# 3) Build the profile
python scripts/build_profile.py --task track_a

# 4) Run a study
python -m lab --task track_a run --name smoke --tags baseline --report

# 5) Open the dashboard
python -m lab ui   # → http://127.0.0.1:8765
```

## What makes v2 different

| v1 (`max_development/`) | v2 (`lab/`) |
|---|---|
| Audio-specific (`BIRDCLEF_*` env vars, `(1,128,313)` input hardcoded) | Fully config-driven via `config/config.yaml` and `config/tasks/<track>.yaml` |
| One metric name hardcoded in 6+ files | Primary metric comes from the task YAML; every axis label, validator, and reporter reads it |
| One submission format (Kaggle notebook CPU) | Per-task adapter decides: `csv` for Track A, `kaggle_notebook_cpu` for Track B |
| 3 phase-pipelines (only 1 used) | 1 loop, explicit |
| Promotion phase | Removed |
| psutil heartbeat | Removed |
| LLM writes 700-line reports | Jinja2 template writes the report; LLM writes only the 1-paragraph executive summary |
| No selective study sharing | `scripts/upload_studies.py` + per-study `publish` flag |

## Directory map

```
max_development_2/
├── config/
│   ├── config.yaml              # global defaults — single source of truth
│   ├── tasks/<track>.yaml       # everything task-specific (paths, metric, ...)
│   ├── prompts/                 # immutable versioned prompts + _registry.yaml
│   └── skeletons/               # code skeletons the LLM fills in
├── registry/<track>_models.yaml # candidate architectures per track
├── lab/
│   ├── core/                    # orchestrator, executor, validator, memory, llm
│   ├── prompts/                 # engine + registry + A/B scoring
│   ├── tasks/                   # base + track_a + track_b adapters
│   ├── reporting/               # figures + Jinja2 template
│   ├── submission/              # dispatch to adapter.build_submission
│   └── ui/                      # FastAPI + HTMX dashboard + SSE live tail
├── scripts/
│   ├── build_profile.py
│   ├── upload_studies.py        # selective publish
│   └── commit_study.sh
├── experiments/studies/         # one folder per study
├── sandbox/                     # one folder per experiment
└── tests/
```

## CLI

```
lab run       — run a study
lab list      — list all studies on disk
lab tasks     — list available tasks
lab report    — render a study report (markdown + figures)
lab submit    — build the submission artifact for the best experiment
lab ui        — serve the web dashboard
lab config    — print the merged configuration as JSON
lab validate  — validate a .py file against the validator rules (useful in CI)
```

### Study continuation

Abort a study with Ctrl+C (SIGINT — the current experiment finishes, then
the study is persisted as `aborted`). Resume later with:

```bash
python -m lab --task track_b run --resume study_20260412_140000_a1b2
```

The new study's memory is seeded from the predecessor's top-K so the
agent sees what was already tried.

## Selective study upload

```bash
# Mark the studies you want to share in the UI (the "publish" toggle),
# then:
python scripts/upload_studies.py --published-only

# Or combine filters:
python scripts/upload_studies.py --tags baseline --best 3
python scripts/upload_studies.py --ids study_20260412_140000_a1b2
python scripts/upload_studies.py --task track_a --format git --branch upload-track-a
```

`--format tar` (default) writes `reports/bundles/studies_bundle_<ts>.tar.gz`;
`--format git` creates a dedicated branch with only those studies committed.

## Prompt A/B testing

The prompt engine keeps versioned, immutable prompt files under
`config/prompts/<task>/vN.yaml`. The UI (`/prompts`) shows:

- mean and best score per (task, version) across all studies
- a side-by-side view of any two versions
- a "Draft new version" form that writes `vN+1.yaml` atomically

Each study records the active prompt version for each prompt task in its
`study.json`, so every experiment score is attributable back to the
prompts that produced it.

## Dashboard

```bash
python -m lab ui
# → http://127.0.0.1:8765
```

- **Studies list** with sort + tag filter + publish toggle
- **Study detail** with live progress and per-experiment drill-down
- **Experiment detail** with code, metrics, and live-streaming stdout
  (Server-Sent Events tail of `sandbox/<exp_id>/stdout.log`)
- **Prompts** A/B dashboard
- **New study** form with a **task dropdown**, tag input, and predecessor
  selector for study continuation
- **Report viewer** with the markdown rendered client-side

## Adding a new track

1. Write `config/tasks/track_c.yaml` (data paths, metric, prompt slots).
2. Implement `class TrackCAdapter(TaskAdapter):` in `lab/tasks/track_c.py`.
3. Point `adapter:` in the task YAML at it.
4. Add `registry/track_c_models.yaml` with a few candidate architectures.
5. Add a `config/skeletons/<kind>.py.j2` training skeleton.

That's it — no `lab/core/*` changes needed.

## What is preserved verbatim from `max_development/`

- `_check_torch_nn_attributes` — reflects against `dir(torch.nn)` to
  reject hallucinated layer names.
- `_check_main_guard` — blocks PyTorch spawn-storm on macOS by forcing
  DataLoader-triggering calls inside `if __name__ == "__main__":`.
- Subprocess sandbox with process-group SIGTERM/SIGKILL tree-kill.
- Error classification table (OOM / SyntaxError / ShapeMismatch / ...).
- Two-layer retry (codegen vs runtime error recovery).
- Immutable-versioned prompt registry with mean-score aggregation.
- Top-K + recent-failures memory markdown for LLM context.

## Tests

```bash
pytest
```

The test suite covers the validator (including both AST checks), the
executor, memory, prompt engine, config loader, task registry, and the
selective-upload filter logic.

> **Note on AI assistance.** Some of the tests in this repo were
> scaffolded with AI assistance and then reviewed and pared down to what
> actually guards against regressions. The test content is our own —
> each assertion targets a specific behaviour we care about.

## Runtime topology

```
┌──────────────┐        ┌───────────────┐
│  config.yaml │◀──────▶│  task/<T>.yaml│
└──────┬───────┘        └───────┬───────┘
       │                        │
       ▼                        ▼
┌──────────────────────────────────────┐
│              Settings                │  (single source of truth at runtime)
└──────┬────────────────────┬──────────┘
       ▼                    ▼
┌─────────────┐      ┌─────────────┐
│Orchestrator │◀────▶│TaskAdapter  │
│  - memory   │      │ - build_prof│
│  - context  │      │ - env_vars  │
│  - executor │      │ - submit    │
│  - validator│      │ - slots     │
└──────┬──────┘      └─────────────┘
       ▼
┌─────────────┐
│    LLM      │   (Ollama / OpenAI via a tiny urllib client)
└─────────────┘
```

## Troubleshooting

- **No studies show up in the dashboard.** The UI reads
  `experiments/studies/*/study.json`. Make sure you ran `lab run` at
  least once; check the CLI output for a study ID.
- **`lab run` hangs at `propose_architecture`.** Your LLM endpoint
  (`llm.base_url` in `config.yaml`) is unreachable. Verify with
  `curl http://localhost:11434/v1/models`.
- **Validator rejects `nn.Conv2x2d`.** That's the point — the LLM
  hallucinated a layer. A new generation will usually fix it on its own.
- **Live SSE stdout stops updating.** The subprocess finished. Reload
  the page; the static `live_stdout` panel shows the final content.
