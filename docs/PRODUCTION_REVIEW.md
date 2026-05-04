# Production Review — Autonomous Research Agent (Track B: BirdCLEF)

**Status:** review only, no code changes yet.
**Scope:** evaluate prototype in `lab/` and `max_development_2/`, map against `Project_Handout.pdf` and personal notes, propose a leaner production scope that 4 people can parallelise.

---

## 1. TL;DR

- Prototype is **functionally complete for the core loop** (load → propose → code → execute → capture → recover → iterate → submit). Grade-critical items in the PDF (D1 "works out of the box", D4 "agent decides what to try next", error handling, prompt engineering) are already covered in some form.
- It is **heavily overengineered** relative to the PDF warning "Avoid to overengineer a solution that you will not understand". ~10 kLOC across lab/core + lab/ui + tests, 1.3 kLOC orchestrator, 1.0 kLOC validator, a Modal executor that is dead code, empty `agent/handlers/` dirs, 52 kB Kaggle executor we cannot defend in 5 min of video.
- **Production scope should shrink, not grow.** Keep the good abstractions (Adapter, PromptRegistry, Memory, two-layer self-repair), drop the speculative ones (Modal, nested-experiments, CLI builder in UI, config editor writing to YAML live), finish the items that are half-done and matter for the grade (validate-before-submit, progress logging, predecessor-as-prestudy, per-class metrics for macro ROC-AUC).
- **Deliver as ~18 parallelisable GitHub issues** (§7). Each has an interface contract so two people can build against it without stepping on each other.

---

## 2. Method

1. Extracted the full PDF requirements (Section 2.2 agent loop, Section 4 deliverables, Section 5 grading, Section 6 "small-scale first", Track B 90-min CPU submission constraint).
2. Mapped the repo architecture: orchestrator, executors, validator, prompt registry, memory, task adapters, UI, reporting, submission, config. (See §3.)
3. Scored the 48 items from the notes against actual code (file:line evidence). (See §4.)
4. Compared the two against each other and the PDF grading rubric, then produced a keep / rework / drop table (§5) and an issue backlog with interfaces (§7).

---

## 3. Current architecture map (summary)

| Layer | Module | Key file | Role |
|---|---|---|---|
| Agent loop | `lab.core.orchestrator` | `orchestrator.py` (1363 LOC) | Study lifecycle, propose→validate→execute→recover→capture |
| LLM | `lab.core.llm` | `llm.py` | Ollama / OpenAI / Anthropic, retry + backoff |
| Prompts | `lab.prompts.registry` + `engine` + `scoring` | `registry.py`, `engine.py` | Immutable versioned prompts, slot filling, A/B aggregation |
| Memory | `lab.core.memory` | `memory.py` | Top-K wins + recent failures, markdown-rendered |
| Validation | `lab.core.validator` | `validator.py` (1024 LOC) | AST checks + smoke forward-pass |
| Execution | `lab.core.executor`, `kaggle_executor`, `modal_executor` | `executor.py` | Subprocess sandbox + Kaggle kernel push; Modal is dead |
| Tasks | `lab.tasks.base` + `track_b_birdclef` + `eda` | `base.py`, `track_b_birdclef.py`, `eda.py` | Adapter ABC, Track B specifics, EDA summary |
| Data | `lab.core.models` | `models.py` | Pydantic Study/Experiment/Task |
| Config | `lab.config` + `config/*.yaml` | `config.py` (291) | Nested pydantic settings, YAML + env overrides |
| Reports | `lab.reporting.generator` + `figures` | `generator.py`, `figures.py` | Learning curves, score progression, failure breakdown |
| Submission | `lab.submission.builder` + adapter-provided notebook builder | `builder.py` | CPU-only Kaggle notebook for Track B |
| UI | `lab.ui.app` + routes | `app.py`, `routes/*.py` | FastAPI: studies, experiments, prompts, reports, config editor, live SSE, run, launches |
| CLI | `lab.cli` | `cli.py` | argparse entrypoint |

Loop is already the PDF's 8 steps. Contract for each phase exists and is strong enough to keep.

---

## 4. Implementation completeness (48 notes items)

Condensed from the explorer audit. See `notes.md` for source phrases.

### Fully implemented (26)
swappable models, prompt-compliance retry, per-run benchmark/duration, prompt A/B dashboard (aggregation done), logging module (`lab.core.telemetry`), structured export (Study.save), hyperparameter proposal via env, Study hierarchy (no `agent_id` field yet), Experiment with dynamic prompt templating, per-experiment submission (best-exp path), fixed `num_classes` in validator smoke test, `max_epochs_per_run` enforced, timeout handling, learning-curve figures, report UI, prompt editor UI (create-only immutable vN), active-version selection per task, prompts transported via git (checked-in YAML), EDA step, LR bounds + schedule enforced, model comparison via score progression, architecture-only generation (skeletons fix the rest), offline weights path + HF_HOME etc, loss-curve in report, prompts-per-task selection in launcher form, studies dump to disk.

### Partial (16)
- **ContextHandler as mergable MD** — memory is a single markdown blob, not a mergable document per run history.
- **Terminal/Input module** — only argparse CLI, no live terminal UX during runs.
- **Mel/audio pipeline as tool-calling** — mels are precomputed, LLM does not call a tool.
- **Study memory persists only within a study** — cross-study learning via `predecessor_id` exists (`orchestrator.py:127-128`) but no "resume from experiment N+1" — the predecessor only seeds memory, baseline is re-run.
- **Transfer learning across studies** — infra ready (`INIT_FROM_EXPERIMENT_ID` env var, `checkpoint_path` on Experiment), but agent has no autonomous "continue previous model" decision; user picks.
- **Validate before Kaggle submission** — validator is before execution, **not** before the submission notebook is built (`submission/builder.py` goes straight from best_experiment.code to notebook).
- **Progress indicator during runs (tqdm / "where we are now")** — only final metrics logged; per-epoch progress visible only via SSE if UI is open.
- **Multi-metric comparison beyond primary** — metrics stored as free-form dict, but no per-class AUC / PR curves rendered; Track B is macro ROC-AUC so per-class view matters a lot.
- **"Evaluation framework / judge role"** — LLM sees memory but there is no distinct judge/critic step.
- **Prompts-per-task selection + "use best"** — per-task override exists; "use best" button not wired.
- **Online version (push results online)** — `scripts/upload_studies.py` tars + commits manually; no in-app sync.
- **Error handling in runs** — retry/skip works, but "skip and continue to next experiment" vs "abort study" policy is hard-coded by error type; not configurable.
- **Code validation with LLM retry** — works, but auto-fix + re-prompt flow is inside `orchestrator.py` rather than the validator; coupled.
- **Centralized upper limits** — most are in `ComputeBudget`, a few bounds (`_LR_MIN`, `_LR_MAX`, `_MAX_PROPOSAL_EPOCHS`) are module-level constants in `orchestrator.py`.
- **Study-as-prestudy** — `--resume` loads predecessor, but no "start at experiment N+1" or "skip baseline" control.
- **LR experimentation** — bounds enforced, but no dedicated LR sweep.

### Not implemented (5)
- **Experiment-within-experiment** — intentionally not needed; can stay unbuilt.
- **Agent personality** — no `Agent` class with persona; no `agent_id` on Study.
- **Live CLI command builder** in the new-study UI form — not present.
- **Default prompt per-task selected from UI "best"** button — partial; no auto-select action.
- **Loss curves across studies (cross-study report)** — per study only.

### Unclear / bug
- **Internal server error on a completed study detail page** — not reproduced in code read; needs to be repro'd against disk state. Likely `KeyError` on missing metric or missing `eda_summary` slot. Ticket in §7.

---

## 5. PDF rubric alignment (where the points come from)

| Rubric (weight) | What graders look for | Gap to close |
|---|---|---|
| Agent design & impl (40%) | Loop quality, prompt engineering, experiment memory, error handling, **runs out of the box** | Single `make run` / `python -m lab run` path, docs README, unified requirements.txt, handle "first run → broken JSON" bug |
| Model performance (20%) | Best leaderboard score, multiple improving subs | Validate-before-submit; warm-start between best experiments; CPU-90min submission timing |
| Use of course content (15%) | Architectures from course explored, reasoning in report | Prompt library covers: 2D CNN from scratch, transfer learning (EfficientNet/MobileNet/YAMNet), augmentations, multi-label head |
| Report & video (25%) | Loop diagram, iteration analysis, per-iteration scores, honest limitations | Report with study diagram + learning curves + per-architecture comparison; video shows live iteration |

The PDF also asks explicitly:
- "How prevent LLM from repeating failed experiments?" → Memory module covers; keep it.
- "How handle code that fails to execute?" → two-layer self-repair covers; keep.
- "Prompt robustness across models" → PromptRegistry + A/B aggregation covers; finish the `scoring.py` aggregation + "use-best" action.
- "Mechanisms to test prompt robustness across models" → add a simple script that runs the same prompt against 2 models on the same (tiny) dataset and reports delta.

---

## 6. Architecture decisions — keep / rework / drop

### 6.1 Keep (load-bearing, well-scoped)

| Decision | Why keep | Notes |
|---|---|---|
| `LLMClient` as single `chat()` interface with provider switch (`ollama`/`openai`/`anthropic`) | Directly satisfies PDF "abstract the LLM provider". ~187 LOC, uses stdlib `urllib` → no extra deps. | Keep as-is. |
| `TaskAdapter` ABC injecting everything task-specific | Adding Track A later (if needed) is ~100 LOC; skeletons + prompts + metric all routed through adapter. | Keep. |
| Immutable `PromptRegistry` (pointer file + `vN.yaml` files) | Prompts travel via git (required), versions are never overwritten (clean A/B). | Keep. Document the pointer-file convention in README. |
| Two-layer self-repair: **AST validator → smoke test → re-prompt** then **execute → classify error → re-prompt** | Exactly the PDF's failing-code question. Empirically what makes the first run not crash. | Keep; decouple auto-fix (see 6.2). |
| Pydantic `Study`/`Experiment`/`Task` models with JSON save/load | Reviewer can inspect any run from disk; drives UI + reports. | Keep. |
| `Memory` top-K + recent failures rendered as ≤3 kB markdown | Keeps prompts under the local-LLM context window; answers "how prevent repeating failed experiments". | Keep. |
| `ComputeBudget` (max_experiments, max_wallclock_minutes, max_experiment_seconds) | Directly implements "set a compute budget per run" tip. | Keep. |
| CPU-only Kaggle-notebook submission builder | Required by Track B (90 min CPU). | Keep; add `validate_before_build()` (see §7 I-14). |
| Offline weights path (`TORCH_HOME`/`HF_HOME`/`TIMM_HOME`) | Kaggle kernel submission must be offline. | Keep. |
| EDA step at study start | PDF step 1 "load and explore" + keeps first prompt well-grounded. | Keep, extend output to include class imbalance stats. |
| Reports with learning curve + score progression + failure breakdown | D4 report needs these figures. | Keep. |
| Prompt overrides per study, prompt_template_paths snapshotted onto Study | Makes study reproducible. | Keep. |
| Studies persisted under `experiments/studies/{study_id}/study.json` | Reviewers can clone + inspect without running code. | Keep. |

### 6.2 Rework / simplify

| Decision | Why rework | Proposal |
|---|---|---|
| `orchestrator.py` 1363 LOC | Too big to defend in video, too big to test, too many constants embedded. | Split into `lifecycle.py` (study loop), `experiment.py` (one iteration), `recovery.py` (auto-fix + re-prompt), `parsing.py` (proposal JSON parsing). Move magic numbers to `ComputeBudget`. Target ≤400 LOC each. |
| `validator.py` 1024 LOC | Many checks we cannot justify ("AST forbidden patterns" list is long). | Keep 3 checks: forbidden imports, `build_model(num_classes)` signature, `nn.*` attribute reflection. Move smoke-test to its own file. Target ≤400 LOC. |
| `kaggle_executor.py` 52 kB | Heavy, rarely used on dev laptop. | Keep but gate behind `--executor kaggle` flag; default to local. Do not demo it if local works — keeps demo simple. |
| Config via YAML **and** `AGENT__SECTION__KEY` env **and** Python dict overrides | Three sources of truth, confusing precedence. | Keep YAML + env only; drop Python dict override (callers can mutate the loaded Settings). |
| `config_editor` UI route that writes YAML live | Footgun: a running study can see config mutate. | Rework: UI is read-only view of current effective config. Editing stays in Git / editor. |
| Live CLI builder in new-study form | Not wired. | Drop from scope; CLI usage documented in README suffices. |
| `PromptScoring` half-wired | Aggregation exists, UI "use best" does not. | Finish: add `get_best_version(task) -> str`, expose as default in study launcher with a "use best per task" checkbox. |
| Auto-fix embedded in orchestrator | Coupling; hard to test auto-fix alone. | Extract `lab.core.recovery.try_autofix(code, finding) -> code | None`. Orchestrator calls it before going to LLM. |
| Memory as single markdown blob | Mergable-MD note implies the user wanted per-run append. | Keep the markdown shape but persist as `memory/{experiment_id}.md` so cross-study / cross-branch merges become trivial with git. |
| Bare-module constants `_LR_MIN`, `_LR_MAX`, `_MAX_PROPOSAL_EPOCHS` in orchestrator | Should be settings. | Move to `ComputeBudget`. |

### 6.3 Drop (dead / speculative)

| Thing | Why drop |
|---|---|
| `lab/core/modal_executor.py` (10 kB) | No call sites outside tests. Not needed for grade. |
| Empty `agent/handlers/`, `agent/ui/` directories | Dead scaffolding. Confuses reviewers. |
| "Experiment-within-experiment" concept | Never implemented; adds nothing; PDF doesn't need it. |
| "Agent personality" framing | Marketing, no payoff in grading. If wanted, one `persona: str` field on `Study` is enough. |
| Live CLI command builder in UI | Nice-to-have, not wired, not worth 4 people's time. |
| Dual logging (stdout + file both plain text) | Keep one structured logger (JSON lines) + one human logger for UI console. |

---

## 7. Parallelisable issue backlog (for 4 devs)

Each issue lists: **Goal → Files → Interface/contract → Acceptance → Depends on**. Interfaces are deliberately precise so two people can build to the same shape in parallel. Issue IDs are `I-01`..`I-18`.

### Track 0 — Foundations (must land first; 1 owner)

#### I-01 — Settings & paths module
- **Goal:** single `Settings` pydantic model, YAML + env merge, no Python dict source.
- **Files:** `lab/config.py`, `config/config.yaml`, `config/tasks/track_b.yaml`.
- **Interface:**
  ```python
  def load_settings(task: str = "track_b", *, overrides_yaml: Path | None = None) -> Settings: ...
  class Settings(BaseModel):
      repo_root: Path; task_name: str
      llm: LLMConfig; compute_budget: ComputeBudget
      paths: PathsConfig; logging: LoggingConfig
      executor: ExecutorConfig; ui: UIConfig
      task: TaskConfig           # adapter class path + task YAML content
  class ComputeBudget(BaseModel):
      max_experiments: int; max_wallclock_minutes: int
      max_experiment_seconds: int; max_epochs_per_run: int
      max_codegen_retries: int; max_recovery_attempts: int
      lr_min: float; lr_max: float
  ```
- **Accept:** `pytest tests/test_config.py -q` green; `AGENT__COMPUTE_BUDGET__MAX_EXPERIMENTS=3` overrides YAML.
- **Depends on:** —

#### I-02 — Core data models (Study / Experiment / Task)
- **Goal:** pydantic models + disk persistence under `experiments/studies/{study_id}/study.json`.
- **Files:** `lab/core/models.py`, helpers in `lab/core/loaders.py`.
- **Interface:**
  ```python
  class Task(BaseModel):
      name: Literal["propose","generate","validate","execute","recover","analyze"]
      status: TaskStatus; input: dict; output: dict
      prompt_paths: dict[str, Path] = {}
      error: TaskError | None = None
  class Experiment(BaseModel):
      id: str; index: int; status: ExperimentStatus
      architecture_name: str | None; architecture_family: str | None
      code: str | None; primary_metric: str
      primary_score: float | None; metrics: dict[str, float] = {}
      history: list[dict] = []                  # per-epoch
      tasks: list[Task] = []; duration_seconds: float | None
      sandbox_path: str | None; checkpoint_path: str | None
  class Study(BaseModel):
      id: str; task_name: str; status: StudyStatus
      experiments: list[Experiment] = []
      memory_path: Path
      best_experiment_id: str | None; best_score: float | None
      predecessor_id: str | None
      prompt_template_paths: dict[str, Path] = {}
      executor_backend: str
      def save(self, root: Path) -> None: ...
      @classmethod
      def load(cls, root: Path, study_id: str) -> "Study": ...
  ```
- **Accept:** round-trip save/load test; forward-compat unknown fields ignored.
- **Depends on:** I-01.

#### I-03 — LLM client
- **Goal:** single `LLMClient.chat(messages) -> str` for `ollama`/`openai`/`anthropic`.
- **Files:** `lab/core/llm.py`.
- **Interface:**
  ```python
  class LLMClient:
      def __init__(self, cfg: LLMConfig): ...
      def chat(self, messages: list[dict[str,str]], *, temperature: float | None = None) -> str: ...
  class LLMTransientError(Exception): ...
  class LLMPermanentError(Exception): ...
  ```
- **Accept:** `tests/test_llm_client.py` covers retry-on-429, fail-on-400, provider switch by fake HTTP.
- **Depends on:** I-01.

#### I-04 — Prompt registry + engine
- **Goal:** immutable versioned prompt store + slot filling. Prompts live in `config/prompts/{task}/v{N}.yaml`. Pointer file `_registry.yaml`.
- **Files:** `lab/prompts/registry.py`, `lab/prompts/engine.py`, `config/prompts/_registry.yaml`.
- **Interface:**
  ```python
  class PromptRegistry:
      def active_version(self, task: PromptTask) -> str: ...
      def set_active(self, task: PromptTask, version: str) -> None: ...
      def load(self, task: PromptTask, version: str | None = None) -> PromptTemplate: ...
      def save_new_version(self, task: PromptTask, system: str, user: str) -> str: ...  # returns new v
      def list_versions(self, task: PromptTask) -> list[str]: ...

  class PromptEngine:
      def __init__(self, registry: PromptRegistry): ...
      def render(self, task: PromptTask, slots: dict[str, str], version: str | None = None
                ) -> tuple[str, str]: ...  # (system, user)
      def extract_slots(self, template: str) -> set[str]: ...
  ```
  `PromptTask = Literal["propose_architecture","generate_code","recover_from_error","analyze_result"]`
- **Accept:** missing-slot raises `MissingSlotError` with the slot name; extra slots ignored.
- **Depends on:** —

---

### Track 1 — The agent loop (1 owner; hardest)

#### I-05 — Memory
- **Goal:** top-K successes + last N failures, render to ≤3 kB markdown.
- **Files:** `lab/core/memory.py`.
- **Interface:**
  ```python
  class Memory:
      def __init__(self, top_k: int, recent_failures: int, path: Path): ...
      def add(self, exp: Experiment) -> None: ...
      def to_markdown(self) -> str: ...          # used in prompt slot {experiment_memory}
      def seed_from_predecessor(self, other: "Memory") -> None: ...
      def save(self) -> None: ...
      @classmethod
      def load(cls, path: Path) -> "Memory": ...
  ```
- **Accept:** curve summary `"loss 0.5→0.3→0.25 | f1 0.1→0.3 | trend improving"`; output <3 kB under 20 exps.
- **Depends on:** I-02.

#### I-06 — Orchestrator (lifecycle + experiment)
- **Goal:** split the current 1363-LOC orchestrator into 3 files, keep the same external behaviour.
- **Files:** `lab/core/lifecycle.py`, `lab/core/experiment.py`, `lab/core/parsing.py`.
- **Interface:**
  ```python
  class StudyRunner:
      def __init__(self, settings: Settings, adapter: TaskAdapter, client: LLMClient,
                   engine: PromptEngine, memory: Memory, validator: Validator,
                   executor: Executor, recovery: Recovery, hooks: Hooks | None = None): ...
      def run(self, *, predecessor: Study | None = None) -> Study: ...
      def abort(self) -> None: ...

  def run_experiment(exp: Experiment, *, ctx: RunContext) -> Experiment: ...
  def parse_proposal(text: str) -> Proposal: ...
  class Proposal(BaseModel):
      architecture_name: str; family: str
      lr: float; lr_schedule: Literal["constant","cosine","onecycle"]
      epochs: int; init_from_experiment_id: str | None
  ```
- **Accept:** same e2e output as legacy orchestrator on a frozen study fixture.
- **Depends on:** I-01..I-05, I-07, I-09, I-10.

#### I-07 — Validator (static + smoke)
- **Goal:** slim validator; 3 static checks + smoke forward pass.
- **Files:** `lab/core/validator.py`.
- **Interface:**
  ```python
  @dataclass
  class ValidationResult:
      ok: bool
      error_type: str | None           # "ForbiddenImport" | "BadSignature" | "UnknownTorchNN" | "SmokeFailed" | "Syntax"
      message: str | None
      findings: list[str] = field(default_factory=list)
      autofix_hint: str | None = None

  class Validator:
      def __init__(self, settings: Settings): ...
      def validate(self, code: str, *, signature: tuple[str,str],
                   smoke_input_shape: tuple[int,...], smoke_num_classes: int
                  ) -> ValidationResult: ...
  ```
- **Accept:** existing test_validator cases green; new `nn.Conv2x2d` rejection test.
- **Depends on:** I-01.

#### I-08 — Executor (local)
- **Goal:** subprocess sandboxed run, captures `results.json`, classifies errors.
- **Files:** `lab/core/executor.py`.
- **Interface:**
  ```python
  @dataclass
  class ExecutionResult:
      succeeded: bool
      primary_score: float | None
      metrics: dict[str, float]
      history: list[dict]
      duration_seconds: float
      stdout: str; stderr: str
      error: TaskError | None            # error_type in {OOM, Timeout, ShapeMismatch, ValueError, RuntimeError, Other}

  class LocalExecutor:
      def __init__(self, settings: Settings): ...
      def run(self, code: str, *, experiment_id: str,
              extra_env: dict[str,str], timeout_s: int) -> ExecutionResult: ...
  ```
- **Accept:** OOM/Timeout/ShapeMismatch classified correctly on fixtures; results.json schema enforced.
- **Depends on:** I-01.

#### I-09 — Recovery (auto-fix + LLM re-prompt)
- **Goal:** attempt deterministic auto-fix first (e.g., rename `nn.Conv2x2d` → `nn.Conv2d`), else ask LLM via `recover_from_error` prompt.
- **Files:** `lab/core/recovery.py`.
- **Interface:**
  ```python
  class Recovery:
      def __init__(self, client: LLMClient, engine: PromptEngine, settings: Settings): ...
      def try_autofix(self, code: str, result: ValidationResult) -> str | None: ...
      def ask_llm(self, code: str, error: TaskError | ValidationResult, *, slots: dict[str,str]
                  ) -> str: ...
  _HARD_FAILURES: set[str] = {"Timeout","OOM","FileNotFound"}
  ```
- **Accept:** given a stubbed `nn.Conv2x2d` code, autofix returns fixed code; hard failures short-circuit retries.
- **Depends on:** I-03, I-04, I-07, I-08.

---

### Track 2 — Task surface (Track B) (1 owner)

#### I-10 — Task adapter + BirdCLEF adapter
- **Goal:** finalise the adapter contract; land Track B adapter + profile.
- **Files:** `lab/tasks/base.py`, `lab/tasks/track_b_birdclef.py`, `config/tasks/track_b.yaml`, `config/skeletons/audio_multilabel.py.j2`.
- **Interface:**
  ```python
  class TaskAdapter(ABC):
      name: str; kind: str; primary_metric: str
      @abstractmethod
      def profile(self) -> DatasetProfile: ...
      @abstractmethod
      def prompt_slots(self) -> dict[str, str]: ...
      @abstractmethod
      def model_block_signature(self) -> tuple[str, str]: ...   # ("build_model","num_classes")
      @abstractmethod
      def spawn_triggering_calls(self) -> Iterable[str]: ...
      @abstractmethod
      def build_submission(self, code: str, experiment_id: str, out_dir: Path) -> Path: ...
  class DatasetProfile(BaseModel):
      num_classes: int; num_train: int
      input_tensor_shape: tuple[int,...]        # e.g., (1,128,313)
      class_imbalance: dict[str,int] | None
  ```
- **Accept:** `profile().num_classes == 234`; smoke forward pass of skeleton against `build_model(234)` works.
- **Depends on:** I-01, I-02.

#### I-11 — Training skeleton (Jinja2 template)
- **Goal:** fixed training loop, LLM writes only `build_model(num_classes)`.
- **Files:** `config/skeletons/audio_multilabel.py.j2`.
- **Contract:** skeleton reads `AGENT_DEVICE`, `AGENT_BATCH_SIZE`, `AGENT_EPOCHS`, `AGENT_PROCESSED_DIR`, `AGENT_CHECKPOINT_IN`, writes `results.json` with:
  ```json
  {"primary_score": 0.41, "metrics": {"f1_macro":0.41, "roc_auc_macro":0.86},
   "history": [{"epoch":1,"loss":0.7,"roc_auc_macro":0.70}], "stopped_early": false}
  ```
- **Accept:** runs end-to-end on a tiny synthetic shard in ≤60 s on CPU; warm-start via `AGENT_CHECKPOINT_IN`.
- **Depends on:** I-10.

#### I-12 — EDA step
- **Goal:** once per study, produce markdown summary consumed by prompts.
- **Files:** `lab/tasks/eda.py`.
- **Interface:**
  ```python
  def run_eda(adapter: TaskAdapter, settings: Settings) -> EDAReport
  class EDAReport(BaseModel):
      markdown: str
      num_classes: int; num_train: int
      imbalance_ratio: float; notes: list[str]
  ```
- **Accept:** `markdown` ≤4 kB, contains class imbalance + mel shape; slot `{eda_summary}` populated for `propose_architecture`.
- **Depends on:** I-10.

---

### Track 3 — Output surface (1 owner)

#### I-13 — Reporting
- **Goal:** per-study report (MD + HTML) with loop diagram, learning curves, per-arch comparison, failure breakdown.
- **Files:** `lab/reporting/generator.py`, `lab/reporting/figures.py`, `lab/reporting/templates/report.md.j2`.
- **Interface:**
  ```python
  def generate_report(study: Study, settings: Settings) -> Path: ...
  def render_figures(study: Study, out_dir: Path) -> dict[str, Path]: ...
  # figure keys: "score_progression","best_learning_curve","failure_breakdown","family_performance","per_class_auc"
  ```
- **Accept:** all 5 figures present on a sample study with ≥3 experiments; report contains the study's loop diagram.
- **Depends on:** I-02.

#### I-14 — Submission builder (validate-before-build)
- **Goal:** build CPU-only Kaggle notebook from best experiment; **re-validate** before writing.
- **Files:** `lab/submission/builder.py`, `lab/submission/notebook_template.ipynb.j2`.
- **Interface:**
  ```python
  def build_submission_for_study(study: Study, settings: Settings) -> Path: ...
  def _validate_for_submission(code: str, adapter: TaskAdapter, settings: Settings) -> None:
      # raises SubmissionValidationError with remediation steps
      ...
  ```
- **Accept:** rejects code with network imports; emits `.ipynb` that runs ≤90 min on Kaggle CPU; submission.csv schema matches competition template.
- **Depends on:** I-02, I-07, I-10.

#### I-15 — Logging / progress
- **Goal:** structured JSON log + per-epoch progress line readable in console and SSE'd to UI.
- **Files:** `lab/core/telemetry.py`, skeleton hook emits `{"event":"epoch","epoch":i,"loss":..,"metric":..}`.
- **Interface:**
  ```python
  def configure(settings: Settings) -> None: ...
  def log_event(**fields) -> None: ...        # structured JSON line
  ```
- **Accept:** `lab/experiments/**/run.log.jsonl` exists per experiment; tail-f shows progress.
- **Depends on:** I-01.

---

### Track 4 — UI + Ops (1 owner, parallel to Tracks 1–3)

#### I-16 — FastAPI UI (studies + experiments + reports + prompts)
- **Goal:** read-mostly UI. Studies list, study detail, experiment detail, prompt dashboard with A/B mean scores, report viewer, run-form to launch a study.
- **Files:** `lab/ui/app.py`, `lab/ui/routes/{studies,experiments,prompts,reports,run,live}.py`, `lab/ui/templates/*.html`.
- **Contract (routes):**
  - `GET /studies` → list
  - `GET /studies/{id}` → detail + figures
  - `GET /experiments/{study_id}/{exp_id}` → detail
  - `GET /prompts` → dashboard `{task: {version: {mean, stdev, best, count}}}`
  - `POST /prompts/{task}/activate?version=vN` → swap pointer
  - `POST /prompts/{task}/new` → body `{system, user}` → returns new version
  - `GET /reports/{study_id}` → HTML
  - `POST /run` → body `{task, predecessor_id?, prompt_overrides?, use_best_prompts?:bool}` → `{study_id}`
  - `GET /live/{study_id}` → SSE stream of telemetry events
- **Accept:** opening a completed study does **not** 500 (closes the unclear bug from notes); launching a run from form reaches orchestrator.
- **Depends on:** I-02, I-04, I-13, I-18.
- **Drop:** live CLI command builder, live YAML config editor (read-only view OK).

#### I-17 — CLI
- **Goal:** `python -m lab run --task track_b [--predecessor STUDY_ID] [--use-best-prompts] [--executor local|kaggle]`.
- **Files:** `lab/cli.py`, `lab/__main__.py`.
- **Accept:** single command from clean clone + README-documented `ollama pull` starts a run.
- **Depends on:** I-06.

#### I-18 — Prompt scoring + "use best" action
- **Goal:** finish aggregation; expose `get_best_version(task) -> str`.
- **Files:** `lab/prompts/scoring.py`.
- **Interface:**
  ```python
  def aggregate_prompt_scores(experiments_dir: Path
                              ) -> dict[str, dict[str, PromptScoreStats]]: ...
  def get_best_version(task: str, experiments_dir: Path, *, min_runs: int = 3
                       ) -> str | None: ...
  ```
- **Accept:** given fixture with 2 versions × 5 runs each, picks version with higher mean primary_score.
- **Depends on:** I-02.

---

### Cross-cutting

- **Tests (all owners):** each issue lands unit tests; add one end-to-end smoke test that runs a stubbed LLM (returns a trivial `build_model` that the skeleton can execute) over 2 iterations and asserts the Study JSON and a report file exist.
- **README (I-16 owner writes the top-level, each issue appends a short "how to run this piece" section).**
- **Requirements.txt / pyproject.toml:** one owner locks versions for CPU-only torch + librosa.

---

## 8. Named risks

1. **90-min CPU kaggle constraint** — easy to violate with a transfer-learning model; I-14 must measure notebook run time on a Kaggle CPU session before declaring done.
2. **First run returns broken JSON** (noted) — harden `parse_proposal` with a strict JSON schema + one retry asking the LLM to re-emit valid JSON.
3. **"234 vs 206 classes" discrepancy** (noted) — fix in `DatasetProfile`; validator must reject `build_model(206)` if profile says 234.
4. **Overengineering penalty** — every issue that adds code needs a one-line justification in the commit message tied to a PDF requirement.
5. **Reproducibility for graders** — README with `ollama pull`, dataset placement, and `python -m lab run` tested on a clean macOS and a clean Linux VM.

---

## 9. Suggested ownership (4 devs)

- **Dev A (foundations):** I-01, I-02, I-03, I-04.
- **Dev B (loop):** I-05, I-06, I-07, I-08, I-09.
- **Dev C (task/output):** I-10, I-11, I-12, I-13, I-14.
- **Dev D (ui/ops):** I-15, I-16, I-17, I-18.

Dependency order: A unblocks B, C, D. B and C unblock each other only via I-10/I-11. D depends on A + artefacts from B and C but can scaffold routes/templates against stubs.

---

## 10. Out of scope for production

- Modal executor, Google Cloud provisioning automation, nested-experiment concept, live YAML editor, CLI command builder UI, agent personas, cross-task transfer learning, distributed runs.

---

*End of review. Open issues from §7 only after the team agrees on drops in §6.3 and ownership in §9.*

---

## 11. Decisions locked (owner feedback, 2026-04-23)

Overrides earlier draft where conflicting. Source of truth for issue creation.

### 11.1 Scope

- **Precomputation of mels stays as-is.** No tool-calling for audio pipeline. EDA/preprocess runs once per study start, not per experiment.
- **Within-study memory persists.** Unchanged from §6.1.
- **Agent memory (cross-study) is optional.** Default **off**. When on, memory seeds from prior studies of the same task. When on, **transfer learning is allowed** (warm-start from prior study's best checkpoint via `INIT_FROM_EXPERIMENT_ID`). When off, each study is a clean slate.
- **Validator runs before Kaggle submission.** Locked; see I-14.
- **Judge role added.** Runs at end of each experiment and at end of study. See I-19.
- **Agent personality = one string.** Literal `"exploratory"` | `"conservative"`. Injected as prompt slot `{personality}`. No class, no framework.
- **No online version.** Dropped. Studies sync via plain git commit of `experiments/` dir when a user chooses to push.
- **Kaggle executor removed.** Overengineered for this project. Local executor only. Submission is a built `.ipynb` the user uploads manually to Kaggle — no programmatic push.
- **Single source of truth for config.** YAML only. Drop env-var overrides and Python-dict overrides. `load_settings(task)` reads `config/config.yaml` + `config/tasks/{task}.yaml`, merges, returns `Settings`. That is the only path.

### 11.2 Delta vs §6

| Section | Change |
|---|---|
| §6.1 Keep | Add: judge role (I-19), agent personality string, agent-memory toggle. |
| §6.2 Rework | Config sources: YAML-only (was YAML+env). Remove env-var override path. |
| §6.3 Drop | Add: `lab/core/kaggle_executor.py` (full file). Add: online/sync-online feature. Keep drop: Modal, empty agent handlers, nested experiments, live CLI builder, live config editor. |

### 11.3 New / revised issues

#### I-19 — Judge role (new)
- **Goal:** LLM-as-judge evaluation after each experiment and at study end. Separate prompt + separate memory entry.
- **Files:** `lab/core/judge.py`, `config/prompts/judge_experiment/v1.yaml`, `config/prompts/judge_study/v1.yaml`.
- **Interface:**
  ```python
  class Judge:
      def __init__(self, client: LLMClient, engine: PromptEngine): ...
      def judge_experiment(self, exp: Experiment, memory: Memory) -> Verdict: ...
      def judge_study(self, study: Study) -> Verdict: ...
  class Verdict(BaseModel):
      verdict: Literal["promote","keep","discard","abort_study"]
      score: float                   # 0..1 confidence
      rationale: str                 # ≤500 chars
      suggested_next: str | None     # free-form hint fed to next proposal
  ```
- **Wiring:** `StudyRunner` calls `judge_experiment` after `_capture_metrics`; writes `Verdict` onto `Experiment.output["verdict"]`. Calls `judge_study` after the loop ends; writes `Verdict` onto `Study.metadata["verdict"]`.
- **Accept:** verdict attached to every experiment JSON; `abort_study` short-circuits the loop; rationale appears in the report.
- **Depends on:** I-02, I-03, I-04, I-05.

#### I-20 — Agent memory toggle + transfer learning gate (new)
- **Goal:** optional cross-study memory. Off by default. When on, memory seeds from same-task prior studies and transfer learning via warm-start checkpoint is allowed.
- **Files:** `lab/core/memory.py` (extend), `lab/core/lifecycle.py` (wire toggle), `config/config.yaml` (`agent.memory_enabled: false`, `agent.personality: "exploratory"`).
- **Interface:**
  ```python
  class AgentConfig(BaseModel):
      memory_enabled: bool = False
      personality: Literal["exploratory","conservative"] = "exploratory"

  class Memory:
      def seed_from_agent_memory(self, studies_root: Path, task: str) -> None: ...
      # walks experiments/studies/*/study.json, pulls top-K across all prior studies
      # only called when settings.agent.memory_enabled is True
  ```
- **Wiring:** `StudyRunner` seeds memory from agent-memory when enabled. `Recovery`/`propose_architecture` prompt slot `{allow_transfer_learning}` set to `"true"` only when `memory_enabled`. Skeleton honors `AGENT_CHECKPOINT_IN` regardless but proposal's `init_from_experiment_id` is ignored when memory disabled.
- **Personality slot:** `{personality}` injected into `propose_architecture` system prompt. Exploratory = "try architectures you haven't tried"; conservative = "tune the current best family".
- **Accept:** off-by-default confirmed in config; enabling flag produces warm-start on exp 2+ if a prior study has a checkpoint; disabling ignores prior checkpoints.
- **Depends on:** I-01, I-02, I-05, I-06.

#### I-21 — Logging everywhere (revises I-15)
- **Goal:** structured + human logs work identically whether running from CLI or UI. All subsystems (LLM, validator, executor, recovery, judge, reporting, submission) emit structured events to one logfile per study.
- **Files:** `lab/core/telemetry.py`, plus one-line `log_event` calls in each subsystem.
- **Interface:**
  ```python
  def configure(settings: Settings, *, study_id: str | None = None) -> None: ...
  def log_event(event: str, **fields) -> None:
      # event in {"study_start","study_end","experiment_start","experiment_end",
      # "validate","execute","recover","llm_call","judge","submission","error","epoch"}
  def log_human(msg: str, *, level: str = "info") -> None: ...
  ```
- **Output:**
  - `experiments/studies/{study_id}/run.log.jsonl` — structured per-event lines.
  - `experiments/studies/{study_id}/run.log` — human-readable tail, same content pretty-printed.
  - stdout — human-readable, tqdm-style per-epoch progress during `execute`.
- **Accept:** identical log content whether invoked via `python -m lab run` or via UI `POST /run`. `tail -f run.log` shows progress even without UI.
- **Depends on:** I-01.

### 11.4 Issues removed / revised

- **I-01 Settings:** YAML-only merge. Drop env-var path, drop Python-dict override. Single callsite: `load_settings(task: str) -> Settings`.
- **I-08 Executor:** `LocalExecutor` only. Remove `KaggleExecutor`.
- **I-14 Submission:** notebook is **built and written to disk**, user uploads manually. Keep validator-before-build.
- **Removed:** any issue around "online sync", "live CLI builder", "config editor UI writing YAML". `scripts/upload_studies.py` kept as a thin helper (git add/commit/push the `experiments/` dir); no in-app sync.

### 11.5 Revised ownership (unchanged tracks, new issues assigned)

- **Dev A:** I-01..I-04.
- **Dev B:** I-05..I-09, I-20 (agent memory + transfer toggle).
- **Dev C:** I-10..I-14, I-19 (judge).
- **Dev D:** I-15→I-21 (logging-everywhere), I-16..I-18.

### 11.6 Files to delete

- `lab/core/kaggle_executor.py`
- `lab/core/modal_executor.py`
- Empty dirs under `agent/handlers/`, `agent/ui/`
- `lab/ui/routes/config_editor.py` (replace with read-only view if desired)
- Any "live CLI builder" template/JS fragment

### 11.7 Config shape (locked)

```yaml
# config/config.yaml (global)
project: lab
version: 2.0.0
default_task: track_b
llm:
  provider: ollama
  base_url: http://localhost:11434
  model: gemma4:e4b
  temperature: 0.2
  max_tokens: 4096
  retry_attempts: 3
  retry_backoff_seconds: 5.0
compute_budget:
  max_experiments: 20
  max_wallclock_minutes: 240
  max_experiment_seconds: 1800
  max_epochs_per_run: 10
  max_codegen_retries: 5
  max_recovery_attempts: 5
  lr_min: 1.0e-5
  lr_max: 1.0e-2
agent:
  memory_enabled: false
  personality: exploratory
context:
  max_prompt_tokens: 6000
  memory_top_k: 5
  recent_failures: 5
logging:
  level: INFO
  structured: true
paths:
  data_root: data
  experiments_dir: experiments/studies
  sandbox: sandbox
  prompts_dir: config/prompts
  offline_weights: data/offline_weights
ui:
  host: 127.0.0.1
  port: 8000
```

Task YAML adds only task-specific fields (adapter path, metric names, skeleton path, dataset paths, prompt_slots). No env overrides. No Python-dict overrides. No runtime YAML edits.

