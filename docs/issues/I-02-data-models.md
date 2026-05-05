# I-02 — Core data models

**Labels:** `track-foundation`, `p0`
**Milestone:** week-1-foundations
**Owner:** Dev A

## Context
Every on-disk artifact must be typed. Prototype dicts caused class-count drift and broken-JSON bugs. ADR-015.

## Scope
Pydantic models for `Task`, `Proposal`, `Verdict`, `Experiment`, `Study`, `DatasetProfile`, `EDAReport`, `ExecutionResult`, `ValidationResult`, `TaskError`. Disk persistence for `Study`.

## Interface
```python
class Proposal(BaseModel):
    architecture_name: str; family: str
    lr: float; lr_schedule: Literal["constant","cosine","onecycle"]
    epochs: int
    init_from_experiment_id: str | None = None

class Verdict(BaseModel):
    verdict: Literal["promote","keep","discard","abort_study"]
    score: float; rationale: str
    suggested_next: str | None = None

class Task(BaseModel):
    name: Literal["propose","generate","validate","execute","recover","analyze","judge"]
    status: TaskStatus; input: dict; output: dict
    prompt_paths: dict[str, Path] = {}
    error: TaskError | None = None

class Experiment(BaseModel):
    id: str; index: int; status: ExperimentStatus
    proposal: Proposal | None
    code: str | None
    primary_metric: str
    primary_score: float | None
    metrics: dict[str, float] = {}
    history: list[dict] = []
    tasks: list[Task] = []
    verdict: Verdict | None = None
    duration_seconds: float | None = None
    sandbox_path: str | None = None
    checkpoint_path: str | None = None

class Study(BaseModel):
    id: str; task_name: str; status: StudyStatus
    personality: Literal["exploratory","conservative"]
    agent_memory_enabled: bool
    predecessor_id: str | None = None
    prompt_template_paths: dict[str, Path] = {}
    experiments: list[Experiment] = []
    best_experiment_id: str | None = None
    best_score: float | None = None
    study_verdict: Verdict | None = None
    created_at: datetime
    finished_at: datetime | None = None

    def save(self, root: Path) -> None: ...
    @classmethod
    def load(cls, root: Path, study_id: str) -> "Study": ...
```

## Files
- Create `lab/core/models.py`
- Create `lab/core/loaders.py` (helpers: list_studies, load_many)
- Create `tests/test_models.py`

## Acceptance
- Round-trip save/load preserves all fields incl. nested `Task`/`Verdict`.
- Forward-compat: unknown fields in JSON ignored.
- Loading a non-existent study raises `StudyNotFoundError`.
- Study IDs follow pattern `study_YYYYMMDD_HHMMSS_xxxx` (random 4-char suffix).

## Depends on
I-01.
