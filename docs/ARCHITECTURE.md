# Architecture — Autonomous Research Agent (Track B: BirdCLEF)

> Companion to `docs/PRODUCTION_REVIEW.md`. This file is the **single source of truth for design decisions**. Every ADR is numbered and immutable once merged; supersede by adding a new ADR that references the old one.

---

## 1. Goals

1. **Autonomous ML research loop** for BirdCLEF (234-class multi-label audio, macro ROC-AUC, CPU-only 90-min Kaggle submission) driven by a **locally-hosted LLM**.
2. **Understandable** — every component justified by a grading criterion or a PDF requirement. No speculative code.
3. **Reproducible** — clone repo, `ollama pull`, run one command, reviewer gets the same study JSON + report.
4. **Parallelisable** — 4 devs can work on 18 issues without blocking each other after foundations (I-01..I-04) land.

## 2. Non-goals

- Kaggle/Modal/cloud executor dispatch.
- Online study sync / web service.
- Experiment-within-experiment, agent persona framework, live CLI builder, live YAML editor.
- Cross-task transfer (only same-task continuation).

## 3. Agent loop (PDF Section 2.2 mapping)

```
[1 load+explore]  ──▶  EDA (once per study)
                        │
[2 propose]       ──▶  LLM → Proposal JSON
                        │
[3 generate]      ──▶  LLM → build_model(num_classes) block
                        │
                    Validator (static + smoke)
                        │    ▲
                    fail?   │ re-prompt up to max_codegen_retries
                        └───┘
[4 execute]       ──▶  LocalExecutor subprocess
                        │    ▲
                    fail?   │ classify → autofix or LLM re-prompt
                        └───┘ up to max_recovery_attempts
[5 capture]       ──▶  results.json → history + metrics
[6 feedback]      ──▶  Judge verdict (per-experiment)
[7 iterate]       ──▶  Memory.add(exp) → next Proposal
[8 submit]        ──▶  Judge verdict (study end)
                        → Submission builder → validator → .ipynb
```

## 4. Component boundaries

```
lab/
├── config.py              # Settings loader (YAML-only)
├── cli.py                 # `python -m lab run ...`
├── __main__.py
├── core/
│   ├── models.py          # Pydantic: Study, Experiment, Task, Verdict, Proposal
│   ├── loaders.py         # Disk I/O for studies
│   ├── llm.py             # LLMClient.chat(messages) -> str
│   ├── memory.py          # top-K + failures; agent-memory seed
│   ├── telemetry.py       # log_event / log_human; JSONL + human stream
│   ├── validator.py       # static AST + smoke forward pass
│   ├── executor.py        # LocalExecutor subprocess + error classification
│   ├── recovery.py        # try_autofix(code, finding) | ask_llm(code, err)
│   ├── judge.py           # LLM-as-judge per experiment + per study
│   ├── parsing.py         # parse_proposal (strict JSON + one retry)
│   ├── experiment.py      # run_experiment(exp, ctx)
│   └── lifecycle.py       # StudyRunner.run()
├── prompts/
│   ├── registry.py        # immutable versioned prompts + pointer file
│   ├── engine.py          # slot filling
│   └── scoring.py         # aggregate stats across studies, pick best version
├── tasks/
│   ├── base.py            # TaskAdapter ABC
│   ├── registry.py        # dynamic adapter loader
│   ├── track_b_birdclef.py
│   └── eda.py             # once-per-study EDA
├── reporting/
│   ├── generator.py       # MD + HTML report
│   ├── figures.py         # plotly charts (incl. per-class AUC)
│   └── templates/report.md.j2
├── submission/
│   ├── builder.py         # validator-before-build, writes .ipynb
│   └── notebook_template.ipynb.j2
└── ui/
    ├── app.py             # FastAPI factory
    ├── routes/            # studies, experiments, prompts, reports, run, live
    └── templates/, static/
config/
├── config.yaml            # global settings (single source of truth)
├── tasks/track_b.yaml     # task-specific
├── skeletons/audio_multilabel.py.j2
└── prompts/
    ├── _registry.yaml
    ├── propose_architecture/v1.yaml ...
    ├── generate_code/v1.yaml ...
    ├── recover_from_error/v1.yaml ...
    ├── analyze_result/v1.yaml ...
    ├── judge_experiment/v1.yaml ...
    └── judge_study/v1.yaml ...
experiments/
├── studies/{study_id}/
│   ├── study.json
│   ├── memory.json
│   ├── run.log.jsonl
│   ├── run.log
│   └── report/
└── checkpoints/{experiment_id}/
sandbox/{experiment_id}/
├── code.py
├── results.json
├── stdout.log
└── stderr.log
data/
├── raw/               # raw audio (gitignored)
├── processed/         # precomputed mels (gitignored)
└── offline_weights/   # HF_HOME/TORCH_HOME/TIMM_HOME target
tests/
```

## 5. Data model (locked)

```python
class TaskStatus(str, Enum): PENDING=..., RUNNING=..., COMPLETED=..., FAILED=..., ABORTED=...
class Task(BaseModel):
    name: Literal["propose","generate","validate","execute","recover","analyze","judge"]
    status: TaskStatus
    input: dict; output: dict
    prompt_paths: dict[str, Path] = {}
    error: TaskError | None = None

class Proposal(BaseModel):
    architecture_name: str; family: str
    lr: float; lr_schedule: Literal["constant","cosine","onecycle"]
    epochs: int
    init_from_experiment_id: str | None = None   # honored only if agent.memory_enabled

class Verdict(BaseModel):
    verdict: Literal["promote","keep","discard","abort_study"]
    score: float                  # 0..1
    rationale: str                # ≤500 chars
    suggested_next: str | None = None

class Experiment(BaseModel):
    id: str; index: int
    status: ExperimentStatus
    proposal: Proposal | None
    code: str | None
    primary_metric: str
    primary_score: float | None
    metrics: dict[str, float] = {}
    history: list[dict] = []               # per-epoch
    tasks: list[Task] = []
    verdict: Verdict | None = None
    duration_seconds: float | None
    sandbox_path: str | None
    checkpoint_path: str | None

class Study(BaseModel):
    id: str
    task_name: str
    status: StudyStatus
    personality: Literal["exploratory","conservative"]
    agent_memory_enabled: bool
    predecessor_id: str | None
    prompt_template_paths: dict[str, Path]
    experiments: list[Experiment] = []
    best_experiment_id: str | None
    best_score: float | None
    study_verdict: Verdict | None = None
    created_at: datetime
    finished_at: datetime | None = None
```

## 6. Prompt contract

Slots injected into every prompt template:

| Slot | Source | Required in |
|---|---|---|
| `{task_description}` | adapter.prompt_slots | all |
| `{primary_metric}` | adapter.primary_metric | all |
| `{num_classes}` | adapter.profile | propose, generate, recover |
| `{input_tensor_shape}` | adapter.profile | generate, recover |
| `{eda_summary}` | EDAReport.markdown | propose |
| `{experiment_memory}` | Memory.to_markdown() | propose, recover, judge |
| `{personality}` | settings.agent.personality | propose, judge |
| `{allow_transfer_learning}` | settings.agent.memory_enabled | propose |
| `{valid_architecture_families}` | task yaml | propose |
| `{code_skeleton_content}` | adapter.code_skeleton | generate, recover |
| `{architecture_proposal}` | prior step output | generate |
| `{broken_code}`, `{error_type}`, `{error_message}`, `{error_traceback}` | recovery ctx | recover |

## 7. Config contract (single source of truth)

Only `config/config.yaml` (global) + `config/tasks/{task}.yaml` (task-specific) are read. No env-var overrides. No runtime dict overrides. One loader:

```python
def load_settings(task: str = "track_b") -> Settings: ...
```

See `docs/PRODUCTION_REVIEW.md` §11.7 for full YAML shape.

## 8. Logging contract

One init per process. Identical streams whether called from CLI or UI:

- `experiments/studies/{study_id}/run.log.jsonl` — one JSON object per line, schema below.
- `experiments/studies/{study_id}/run.log` — human-readable pretty form of same events.
- `stdout` — human form; tqdm-style epoch progress during `execute`.

JSONL schema:
```json
{"ts": "ISO8601", "study_id": "...", "experiment_id": "... | null",
 "event": "study_start|experiment_start|llm_call|validate|execute|epoch|recover|judge|experiment_end|study_end|error",
 "level": "info|warn|error", "fields": { "arbitrary": "payload" }}
```

## 9. ADRs

Each ADR: **Context → Decision → Consequences**. Immutable after merge; add ADR-NNN to supersede.

---

### ADR-001 — Single `LLMClient.chat(messages) -> str` for all providers

**Context.** PDF requires model-agnostic agent; local LLMs expose OpenAI-compatible API. Project must also support Anthropic + direct Ollama.

**Decision.** One class, one method, provider selected by config. Uses stdlib `urllib` — no extra deps. Classifies errors into `LLMTransientError` (retry) / `LLMPermanentError` (fail fast). Retry with exponential backoff.

**Consequences.** Swapping model is a YAML change. Tests can fake HTTP. Cost: provider-specific features (tool use, streaming) are not exposed. We do not need them for this loop.

---

### ADR-002 — Config from YAML only. No env-var overrides. No runtime Python overrides.

**Context.** Prototype had YAML + `AGENT__X__Y` env + Python dict overrides. Three sources of truth created confusion (env precedence non-obvious, tests mutating dicts leaked).

**Decision.** `load_settings(task)` reads `config/config.yaml` merged with `config/tasks/{task}.yaml`. That is the only path. Tests construct `Settings` directly via pydantic.

**Consequences.** Clear precedence. One file to audit when a run misbehaves. Cost: cannot override with a shell var; must edit YAML or construct Settings in code.

---

### ADR-003 — Task-specific knowledge in a `TaskAdapter`. Training skeleton fixed. LLM writes only `build_model(num_classes)`.

**Context.** PDF warns against overengineering and requires understanding. LLM-generated full training loops fail frequently (the "fast almost everything fails" note from the user).

**Decision.** Skeleton is a committed Jinja2 template (`config/skeletons/audio_multilabel.py.j2`) containing data loaders, optimizer, loss, training loop, eval, checkpointing, results.json write. LLM must emit only a `build_model(num_classes: int) -> nn.Module` block, which the skeleton imports.

**Consequences.** Massive drop in failure rate. LLM focus is on architecture choice, which is the course content. Cost: the set of architectures the skeleton supports is bounded by what the skeleton's training loop can run (e.g., pure-module outputs, no custom training). Acceptable for Track B.

---

### ADR-004 — Immutable versioned prompts via pointer file

**Context.** PDF asks for "mechanisms to test prompt robustness across models". Studies must be reproducible even if prompts are later edited.

**Decision.** `config/prompts/_registry.yaml` maps `task → active_version`. Each version is a separate `vN.yaml` file, never mutated once written. New version = new file. `Study.prompt_template_paths` snapshots the exact paths used.

**Consequences.** Prompts travel via git. A/B testing is clean. Cost: prompt churn leaves stale `vN.yaml` files on disk — fine, cheap, and the scoring module can ignore them after `min_runs`.

---

### ADR-005 — Two-layer self-repair: validator before run, error-classified retry after run

**Context.** PDF: "how do you handle code that fails to execute". LLM code frequently has (a) syntactic / attribute errors catchable statically, (b) runtime errors (shape mismatch, OOM).

**Decision.**
- **Layer 1:** validator runs before subprocess. AST checks + smoke forward-pass. On fail: deterministic `try_autofix` first (e.g., `nn.Conv2x2d` → `nn.Conv2d`), else LLM `recover_from_error` prompt. Up to `max_codegen_retries`.
- **Layer 2:** executor runs code in subprocess. On fail: classify error (`OOM`, `Timeout`, `ShapeMismatch`, etc.). Hard failures (`Timeout`, `OOM`, `FileNotFound`) abort retries. Recoverable failures re-prompt via `recover_from_error`. Up to `max_recovery_attempts`.

**Consequences.** Experiments that can run, do run. Experiments that can't are classified with a useful error on the study report. Cost: two retry loops add complexity — mitigated by extracting `Recovery` into its own module.

---

### ADR-006 — Memory is compact markdown: top-K wins + N recent failures

**Context.** Local LLMs have limited context. Memory must fit in every prompt.

**Decision.** `Memory.to_markdown()` emits ≤3 kB markdown: top-K experiments by primary_score with one-line curve summary (`"loss 0.5→0.3→0.25 | f1 0.1→0.3 | trend improving"`) + last N failures with one-line error classification.

**Consequences.** Memory is injected into propose + recover + judge prompts. Stays under budget at 20+ experiments. Cost: bounded detail — the full history is on disk for reports, not in prompts.

---

### ADR-007 — Agent memory (cross-study) is opt-in. Transfer learning gated by same flag.

**Context.** PDF scope is one loop; teacher notes avoid overengineering. User asked for optional cross-study learning without making it default.

**Decision.** `agent.memory_enabled` (bool, default `false`) in config. When `false`: Memory starts empty each study; `Proposal.init_from_experiment_id` is ignored; no warm-start checkpoint. When `true`: `Memory.seed_from_agent_memory(studies_root, task)` pulls top-K across all prior studies of same task; `{allow_transfer_learning}` slot set to `"true"` so LLM may request warm-start via `init_from_experiment_id`; executor wires `AGENT_CHECKPOINT_IN` to the referenced experiment's `checkpoint_path`.

**Consequences.** Clean default matches teacher's "understand what you built". Power users can switch on and the system upgrades coherently (memory + transfer together). Cost: the code path must be tested in both modes; two CI tests required.

---

### ADR-008 — Agent personality is a single string slot

**Context.** User floated "agent personality"; the cheap reading (one string) is enough.

**Decision.** `agent.personality: Literal["exploratory","conservative"]`. Injected as `{personality}` into `propose_architecture` and `judge_*` prompts. Prompts define the meaning: exploratory = "try architectures you haven't tried"; conservative = "tune the current best family".

**Consequences.** No `Agent` class, no persona framework. Changing default is a YAML edit. Cost: more personalities require prompt edits — acceptable, prompts are versioned.

---

### ADR-009 — Judge role runs after each experiment and at study end

**Context.** User request. Also fits PDF "feed results back for analysis". Keeps the loop's feedback step explicit.

**Decision.** Separate prompts `judge_experiment/v1.yaml` and `judge_study/v1.yaml`. Separate `Judge.judge_experiment(exp, memory) -> Verdict` and `Judge.judge_study(study) -> Verdict`. `Verdict.verdict ∈ {promote, keep, discard, abort_study}`. `abort_study` short-circuits the loop.

**Consequences.** Explicit discard/promote reasoning on every run, in the report, and available for the demo video. Cost: one extra LLM call per experiment. Acceptable; local LLM, a few seconds.

---

### ADR-010 — Only LocalExecutor. Kaggle + Modal executors removed.

**Context.** User wants to avoid overengineering. Kaggle executor is 52 kB, rarely used, and the 90-min-CPU submission constraint is met at submission build time, not at every training run.

**Decision.** One executor class: `LocalExecutor`. Builds sandbox dir, subprocesses, classifies errors, returns `ExecutionResult`. Submission is a built `.ipynb` the user uploads manually to Kaggle.

**Consequences.** Thousands of LOC deleted. Demo simpler. Cost: we do not measure Kaggle-CPU runtime automatically. Mitigated: submission builder runs a 60-s CPU smoke on the notebook.

---

### ADR-011 — Submission builder validates code before writing notebook

**Context.** Multiple prototype submissions had forbidden imports / bad signatures slip through.

**Decision.** `build_submission_for_study` calls `_validate_for_submission(code, adapter, settings)` first. Reuses the same `Validator` used in the loop, with stricter checks (no `requests`/`urllib`, no `subprocess`, offline-only weights paths, signature must match adapter). On fail: raises `SubmissionValidationError` with remediation steps.

**Consequences.** "Should have worked" submissions stop being submitted. Cost: one more validation pass. Negligible.

---

### ADR-012 — Structured JSON logs + human stream, identical from CLI and UI

**Context.** Prototype printed to stdout and had SSE for UI; CLI runs lacked per-epoch visibility.

**Decision.** Single telemetry init. Writes `run.log.jsonl` (structured) and `run.log` (human) per study. Stdout mirrors `run.log`. UI SSE tails the JSONL.

**Consequences.** `tail -f run.log` works whether or not UI is running. Reports are built from JSONL. Cost: writing two files per run. Disk-cheap.

---

### ADR-013 — Studies persisted as flat JSON under `experiments/studies/{id}/` and shared via git

**Context.** Team needs to share study results. PDF: "experiment logs generated by the agent" required in repo.

**Decision.** One directory per study. `study.json` is the pydantic dump. Reports, memory, logs are sibling files. Users commit under a user-scoped branch or push to `max_development_2` with a subfolder.

**Consequences.** Simple, git-diffable, reviewer inspects without running code. Cost: repo size grows; add `.gitignore` rules for `sandbox/`, `data/raw/`, `data/processed/`, `experiments/checkpoints/` (only `best_epoch.pt` kept when small).

---

### ADR-014 — Mel spectrograms precomputed once, outside the loop

**Context.** Recomputing mels per experiment blows the compute budget.

**Decision.** `scripts/preprocess_track_b.py` runs once per hardware. Writes `data/processed/mels/*.npy`. `DatasetProfile.input_tensor_shape` reflects the saved shape. Agent does not generate mels; agent does not call an audio tool.

**Consequences.** Experiment iteration is fast. Cost: preprocessing hyperparameters (n_mels, hop_length) are not part of the agent's search space. Acceptable for grade; document choice in report.

---

### ADR-015 — Pydantic models for every on-disk artifact

**Context.** Loosely-typed dicts in the prototype caused the "234 vs 206 classes" drift and the "first-run broken JSON" bug.

**Decision.** `Proposal`, `Verdict`, `Experiment`, `Study`, `Task`, `EDAReport`, `DatasetProfile`, `ExecutionResult`, `ValidationResult` — all pydantic. Parsing LLM outputs goes through pydantic with one retry on failure.

**Consequences.** Schema violations caught at boundaries, not 30 minutes into a training run. Cost: stricter prompts. Acceptable.

---

### ADR-016 — All bounds live in `ComputeBudget`

**Context.** Prototype had `_LR_MIN`, `_LR_MAX`, `_MAX_PROPOSAL_EPOCHS` as module-level constants in `orchestrator.py`.

**Decision.** Every numerical limit in `ComputeBudget` (see config YAML): `max_experiments`, `max_wallclock_minutes`, `max_experiment_seconds`, `max_epochs_per_run`, `max_codegen_retries`, `max_recovery_attempts`, `lr_min`, `lr_max`.

**Consequences.** One file to tune. Cost: none.

---

### ADR-017 — EDA runs once per study, not per experiment

**Context.** Step 1 of PDF loop, should inform every subsequent propose prompt.

**Decision.** `run_eda(adapter, settings) -> EDAReport` executed by `StudyRunner.run()` before the experiment loop. Report is saved to `experiments/studies/{id}/eda.md` and injected as `{eda_summary}` slot for the whole study.

**Consequences.** Every propose sees class imbalance / spectrogram shape / data stats. Cost: none (precomputed data loads quickly).

---

### ADR-018 — `parse_proposal` is strict JSON with one retry

**Context.** "First run always broken JSON" noted by user.

**Decision.** `parse_proposal(text) -> Proposal`:
1. Strip fences; locate the outermost JSON object.
2. Attempt `Proposal.model_validate_json(...)`.
3. On ValidationError: one LLM retry with the exact error and the instruction to re-emit valid JSON. No schema drift.
4. On second fail: raise `ProposalParseError` → experiment fails with a clear error, loop continues.

**Consequences.** First-run bug eliminated. Cost: one extra LLM round on bad outputs. Cheap.

---

## 10. Non-functional properties

- **Determinism.** Skeleton sets `torch.manual_seed(seed)` from `AGENT_SEED`. Seed recorded in `Experiment`. Two runs of the same `study.json` + same seed produce equal `primary_score` ± fp noise.
- **Offline.** All HF/TORCH/TIMM homes point at `data/offline_weights/`. Submission notebook declares offline. Preprocessor is the only code allowed to read `data/raw/`.
- **Time-budget.** Every experiment subject to `max_experiment_seconds`. Every study subject to `max_wallclock_minutes`. Orchestrator breaks the loop when wallclock exceeded.
- **SIGINT.** `StudyRunner` traps SIGINT, marks current experiment `ABORTED`, finalises study as `ABORTED`, flushes logs.

## 11. Test strategy

- **Unit:** each module has its own test file. Target ≥80% line coverage on `lab/core/*` and `lab/tasks/*`.
- **Integration:** one end-to-end test with stubbed `LLMClient` returning a trivial `build_model` the skeleton can run over synthetic mels for 1 epoch. Asserts:
  - `study.json` valid + loadable;
  - `run.log.jsonl` contains all expected event types;
  - `report.md` renders with at least one learning curve;
  - submission build succeeds and notebook validates.
- **Determinism test:** same seed, same code → same score.
- **Recovery test:** broken `nn.Conv2x2d` code → autofix succeeds; shape-mismatched code → LLM recovery succeeds.
- **Agent-memory off/on:** two runs, second with flag on → warm-start path exercised.

## 12. Out-of-scope register

| Thing | Why out |
|---|---|
| KaggleExecutor | ADR-010 |
| ModalExecutor | Dead code, no use |
| Experiment-in-experiment | Not in PDF |
| Live CLI builder UI | Not in PDF, not wired |
| Live YAML editor UI | ADR-002 footgun |
| Online study sync | User call, overengineering |
| Agent persona framework | ADR-008 |
| Cross-task transfer | Out of scope — same-task only |
| Tool-calling audio pipeline | ADR-014 |

---

*ADRs merged on `design-prep` branch 2026-04-23. Supersede with ADR-NNN referencing the old one.*
