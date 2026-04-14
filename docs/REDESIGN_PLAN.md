# `max_development_2` — Redesign Plan

_Status: blueprint for the ground-up rewrite. Supersedes `max_development/`._

## 1. Why a rewrite

`max_development/` worked, but it was welded to BirdCLEF. Audio shapes, mel-spectrogram
preprocessing, ROC-AUC/F1 metric names, BirdCLEF env-var prefixes, and a Kaggle-notebook
submission format were hardcoded across ~15 files. Switching to Track A (Kaggle *Disaster
Tweets*, binary text classification) would mean editing half the codebase and risking
regressions on the BirdCLEF path.

The goal of `max_development_2` is a single, professional, progressive codebase that runs
**both tracks out of the box** and is extensible to further tracks by dropping in a new
task adapter and a new task YAML — no core changes.

## 2. Design principles

1. **`config.yaml` is the single source of truth.** Every runtime knob lives there.
   Code reads settings through a typed `Settings` object. No hardcoded paths, metric
   names, shapes, epoch counts, or env-var names anywhere in `lab/core/*`.
2. **Task adapters isolate task-specific code.** A task adapter provides the data
   profile, the data loader import line, the submission artifact builder, and a set of
   prompt slot values. The agent loop is task-agnostic.
3. **Prompts still flow through the prompt engine.** The engineering module (engine +
   versioned registry + A/B scoring) stays. Task descriptions are injected as prompt
   slots rather than written into the prompt itself — this lets us A/B prompt wording
   across tasks.
4. **Keep the good, delete the dead.** We preserve the validator AST checks, the
   subprocess sandbox with process-group kill, the two-layer retry, the memory top-K
   markdown, the prompt engine, and the UI. We delete the promotion phase, the phase
   pipelines, the psutil heartbeat, and the LLM-authored full report.
5. **Study continuation is a first-class feature.** You can abort any study and resume
   it later; the new study inherits the predecessor's memory.
6. **Selective study publishing.** Each study has a `publish` flag and tags.
   `scripts/upload_studies.py` bundles only selected studies for sharing.

## 3. Directory layout

```
max_development_2/
├── README.md
├── pyproject.toml
├── .gitattributes
├── docs/REDESIGN_PLAN.md
├── config/
│   ├── config.yaml                  # single source of truth
│   ├── tasks/
│   │   ├── track_a.yaml             # Disaster Tweets
│   │   └── track_b.yaml             # BirdCLEF+ 2026
│   ├── prompts/
│   │   ├── _registry.yaml
│   │   ├── propose_architecture/v1.yaml
│   │   ├── generate_code/v1.yaml
│   │   ├── analyze_result/v1.yaml
│   │   ├── recover_from_error/v1.yaml
│   │   └── executive_summary/v1.yaml
│   └── skeletons/
│       ├── text_binary.py.j2
│       └── audio_multilabel.py.j2
├── registry/
│   ├── track_a_models.yaml
│   └── track_b_models.yaml
├── lab/                             # the agent package
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── config.py                    # pydantic Settings
│   ├── core/
│   │   ├── models.py                # pydantic data model
│   │   ├── orchestrator.py          # loop + two-layer retry
│   │   ├── executor.py              # subprocess sandbox (no psutil)
│   │   ├── validator.py             # AST checks (nn-reflection, main guard)
│   │   ├── memory.py                # top-K + predecessor seeding
│   │   ├── context.py               # prompt slot filler
│   │   ├── llm.py                   # Ollama OpenAI-compatible client
│   │   └── telemetry.py             # structured logging
│   ├── prompts/
│   │   ├── engine.py
│   │   ├── registry.py
│   │   └── scoring.py
│   ├── tasks/
│   │   ├── base.py                  # TaskAdapter ABC
│   │   ├── registry.py
│   │   ├── track_a_disaster_tweets.py
│   │   └── track_b_birdclef.py
│   ├── reporting/
│   │   ├── figures.py
│   │   ├── generator.py
│   │   └── templates/study_report.md.j2
│   ├── submission/
│   │   └── builder.py
│   └── ui/                          # FastAPI + HTMX dashboard
│       ├── app.py
│       ├── loaders.py
│       ├── live.py
│       ├── routes/{studies,experiments,prompts,tasks,reports,api}.py
│       ├── static/{styles.css,app.js}
│       └── templates/{base,index,study,experiment,new_study,prompts,report}.html
├── scripts/
│   ├── build_profile.py             # dispatches to task adapter
│   ├── preprocess.py
│   ├── commit_study.sh
│   └── upload_studies.py            # selective publishing
├── experiments/studies/
├── sandbox/
└── tests/
```

## 4. What we preserve verbatim from `max_development/`

| Component | From | Reason |
|---|---|---|
| `_check_main_guard` (AST) | `agent/handlers/validate_code.py:435-450` | Prevents spawn-storm on macOS PyTorch DataLoader workers. Critical. |
| `_check_torch_nn_attributes` (AST reflection) | `agent/handlers/validate_code.py:509-543` | Catches hallucinated layer names like `nn.Conv2x2d`. Cheap, high-value. |
| `FORBIDDEN_IMPORTS`, `FORBIDDEN_PATTERNS` | `agent/handlers/validate_code.py:38-79` | Always-on safety baseline. |
| Process-group subprocess + SIGTERM/SIGKILL tree-kill | `agent/executor.py:251-323,549-564` | Reliable timeout on multi-worker DataLoaders. |
| Live stdout streaming (buffer + live file + logger) | `agent/executor.py:382-423` | Enables the UI's live-tail panel. |
| Error classification table | `agent/executor.py:69-113` | LLM reacts differently to OOM vs Shape vs Syntax vs Timeout. |
| Two-layer retry (codegen + error recovery) | `agent/orchestrator.py` | The actual agent behavior. |
| Memory top-K markdown + predecessor seeding | `agent/memory.py` | Fits ~5 experiments in <1 KB. |
| Prompt engine slot filling + immutable versioned registry | `agent/prompt_engine.py`, `agent/prompt_registry.py` | Backbone of A/B. |
| `PromptScoreStats` aggregation | `agent/prompt_scoring.py` | Evidence for v1-vs-v2. |
| Ollama OpenAI-compatible LLM client w/ retry | `agent/llm_client.py` | Works, keep. |
| `.gitattributes` merge=ours on per-study files | `.gitattributes` | Lets parallel studies merge cleanly. |

## 5. What we delete

| Component | Why |
|---|---|
| Promotion phase (`orchestrator.py:214-397`) | Dead — default `promoted_epochs=1`. |
| Phase pipelines (`default`/`exploration`/`exploitation`) | Only `default` was used; `exploitation` referenced a non-existent config knob. |
| psutil heartbeat | Operator doesn't need CPU% — orchestrator already logs progress. |
| LLM-authored full report (`report.py`) | 701 LOC of brittle template sprawl. Replaced with Jinja2 markdown + (optional) LLM-written executive summary paragraph. |
| Hardcoded `BIRDCLEF_*` env-var names | Replaced with `AGENT_*` and a task-owned env-var prefix option. |
| Duplicate "task-config" systems | Only `config/tasks/*.yaml` survives. |
| Hardcoded metric names (`f1_macro`, `roc_auc_macro`) in 6+ files | All reads flow from task config's `metrics.primary`. |

## 6. Config-first architecture

**Precedence at startup** (highest wins):
```
CLI flag  >  AGENT_* env var  >  config/tasks/<track>.yaml  >  config/config.yaml  >  dataclass defaults
```

**`config/config.yaml`** holds global defaults: LLM provider, paths, compute budget,
logging, UI port, env-var prefix, default task name. Everything that appears nowhere else.

**`config/tasks/<track>.yaml`** holds everything task-specific: data paths, preprocessing,
feature type, label type, metrics, model registry path, submission spec, prompt slot
values (task description, valid architecture families, code skeleton template).

A typed pydantic `Settings` merges them and is the ONLY way code reads configuration.
No `os.environ.get("BIRDCLEF_EPOCHS", ...)` anywhere in `lab/core/*`.

## 7. TaskAdapter contract

```python
class TaskAdapter(ABC):
    name: str
    primary_metric: str
    feature_kind: str          # "text" | "audio_spectrogram" | ...
    output_head: str           # "sigmoid_1" | "sigmoid_N" | "softmax_N"

    def build_profile(self, raw_dir, processed_dir) -> DatasetProfile: ...
    def data_loader_import(self) -> str: ...
    def build_submission(self, code, exp_id, out_path) -> Path: ...
    def validate_training_output(self, results_json) -> list[str]: ...
    def prompt_slot_values(self) -> dict[str, str]: ...
    def env_vars(self) -> dict[str, str]: ...
```

- `track_b_birdclef.py` implements all audio preprocessing, spectrogram DataLoader,
  Kaggle-notebook submission.
- `track_a_disaster_tweets.py` implements CSV → token tensors, binary classification
  head, direct CSV submission.
- Adding Track C would mean: write one adapter class + one `track_c.yaml`. No core change.

## 8. Selective study publishing

Three orthogonal mechanisms so any combination works:

1. **Per-study `publish: bool`** written into `study.json`. UI offers a toggle.
2. **Tags** on each study (`tags: ["good-baseline", "cnn-family"]`). UI offers filters.
3. **CLI selector** `scripts/upload_studies.py --ids ID1,ID2 --tags foo --published-only
   --best N`. Produces either:
   - a `studies_bundle_<timestamp>.tar.gz` for out-of-band sharing, or
   - a `git commit` containing only the selected study directories + their prompts.

Each mechanism works alone. The combined form — "published-only **and** tagged with
`baseline`" — is what team leads will use at presentation time.

## 9. Dashboard

The UI is a first-class deliverable (professor awards points). Stack:

- FastAPI + HTMX + Jinja2 (no SPA framework — keeps the repo lean).
- Dark theme, single-page look per route (no heavy CSS framework; just tailored CSS
  variables). Chart.js loaded from a local static file for client-side charts.
- Server-Sent Events (`/live/<experiment_id>`) for the live stdout tail — faster and
  simpler than polling.
- Routes:
  - `GET /` — studies list (sortable, filterable, publish toggle)
  - `GET /studies/{id}` — study detail with live progress, experiment grid
  - `GET /studies/{id}/report` — rendered markdown report (templated, not LLM)
  - `GET /experiments/{id}` — code/stdout/stderr/metric tabs
  - `GET /prompts/` — A/B dashboard with side-by-side versions and score histogram
  - `GET /prompts/{task}/{version}` — editor
  - `GET /new` — **new study form: task dropdown + prompt versions + predecessor**
  - `POST /studies/{id}/abort`, `POST /studies/{id}/publish`, `POST /studies/{id}/tags`

## 10. Prompt A/B

Preserved. The prompt-registry contract from `max_development/` is kept intact:
immutable version files, mutable `_registry.yaml`, scored via
`PromptScoreStats.mean_score`. The UI shows a dashboard with a bar chart per task ×
version. New versions are drafted in a form and written as `v2.yaml`, `v3.yaml`, …

## 11. Reporting

- `lab/reporting/figures.py` renders matplotlib PNGs: score progression, error
  breakdown, learning curve of best experiment, per-architecture-family boxplot.
  Metrics are parameterized (primary metric comes from the task config — no
  hardcoded "ROC-AUC" axis labels).
- `lab/reporting/templates/study_report.md.j2` is a Jinja2 markdown template with
  sections: overview, best experiment, metric trajectory, failure analysis, prompt
  versions used. Charts are embedded via relative PNG links.
- `lab/reporting/generator.py` optionally calls the LLM for the **executive summary
  paragraph only** (not the whole report). If the LLM is down, the paragraph is
  replaced with a deterministic fallback.

## 12. What bleeds through from max_development

| Idea | Status |
|---|---|
| Two-layer retry | Kept |
| Validator AST checks | Kept |
| Process-group kill | Kept |
| Live stdout streaming | Kept |
| Memory top-K markdown | Kept |
| Predecessor seeding | Kept & promoted to first-class |
| Prompt engine + registry + scoring | Kept |
| Git-transportable studies | Kept + extended with selective publish |
| UI dashboard | Rewritten, enhanced |
| Report | Template-first, LLM optional |
| Promotion phase | Deleted |
| Phase pipelines | Deleted |
| psutil heartbeat | Deleted |
| LLM full-report generation | Deleted (exec summary only) |
| Hardcoded metric/audio fields | Deleted — flows from task config |

## 13. Delivery checklist

- [ ] `lab run --task track_a` completes a 3-experiment smoke study end-to-end on CPU.
- [ ] `lab run --task track_b` completes a 3-experiment smoke study on the existing
      spectrogram cache.
- [ ] `lab ui` serves the dashboard; new-study form has a task dropdown.
- [ ] `scripts/upload_studies.py --published-only` produces a tarball.
- [ ] Validator + executor + memory + prompt engine have unit tests.
- [ ] README documents both tracks with copy-paste commands.
