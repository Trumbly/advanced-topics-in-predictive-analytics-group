# I-01 — Settings module (YAML-only)

**Labels:** `track-foundation`, `p0`
**Milestone:** week-1-foundations
**Owner:** Dev A

## Context
Prototype merged YAML + `AGENT__*` env + Python dict overrides — three sources of truth. ADR-002 locks a single YAML-only loader.

## Scope
Implement `load_settings(task)` and the full `Settings` pydantic tree. Read `config/config.yaml` + `config/tasks/{task}.yaml`. No env overrides, no runtime dict overrides.

## Interface
```python
def load_settings(task: str = "track_b", *, repo_root: Path | None = None) -> Settings

class Settings(BaseModel):
    project: str; default_task: str; task_name: str
    llm: LLMConfig
    compute_budget: ComputeBudget
    agent: AgentConfig
    context: ContextConfig
    logging: LoggingConfig
    paths: PathsConfig
    ui: UIConfig
    task: TaskConfig

class ComputeBudget(BaseModel):
    max_experiments: int; max_wallclock_minutes: int
    max_experiment_seconds: int; max_epochs_per_run: int
    max_codegen_retries: int; max_recovery_attempts: int
    lr_min: float; lr_max: float

class AgentConfig(BaseModel):
    memory_enabled: bool = False
    personality: Literal["exploratory","conservative"] = "exploratory"
```

Full ComputeBudget + PathsConfig + LLMConfig shapes in `docs/ARCHITECTURE.md` §7 and `docs/PRODUCTION_REVIEW.md` §11.7.

## Files
- Create `lab/config.py`
- Create `config/config.yaml` (locked shape, review §11.7)
- Create `config/tasks/track_b.yaml`
- Create `tests/test_config.py`

## Acceptance criteria
- `pytest tests/test_config.py -q` green.
- Test confirms `AGENT__COMPUTE_BUDGET__MAX_EXPERIMENTS=999` env var is **ignored**.
- Passing an unknown task name raises `FileNotFoundError` with the missing path.
- Missing required key raises pydantic `ValidationError` at load time, not at use time.

## Out of scope
- Env-var overrides, Python-dict overrides, runtime YAML editing.

## Depends on
None.
