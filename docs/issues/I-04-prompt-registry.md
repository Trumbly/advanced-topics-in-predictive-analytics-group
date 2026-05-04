# I-04 — Prompt registry + engine

**Labels:** `track-foundation`, `p0`
**Milestone:** week-1-foundations
**Owner:** Dev A

## Context
ADR-004. Immutable versioned prompts, shared via git.

## Scope
`PromptRegistry` manages `config/prompts/` layout. `PromptEngine` renders prompts by filling `{slot}` placeholders. Author the v1 prompts for all 6 tasks.

## Interface
```python
PromptTask = Literal[
    "propose_architecture","generate_code","recover_from_error",
    "analyze_result","judge_experiment","judge_study"
]

class PromptRegistry:
    def __init__(self, root: Path): ...
    def active_version(self, task: PromptTask) -> str: ...
    def set_active(self, task: PromptTask, version: str) -> None: ...
    def load(self, task: PromptTask, version: str | None = None) -> PromptTemplate: ...
    def save_new_version(self, task: PromptTask, system: str, user: str) -> str: ...  # returns "vN"
    def list_versions(self, task: PromptTask) -> list[str]: ...

class PromptTemplate(BaseModel):
    system: str; user: str
    task: str; version: str; path: Path

class PromptEngine:
    def __init__(self, registry: PromptRegistry): ...
    def render(self, task: PromptTask, slots: dict[str, str],
               version: str | None = None) -> tuple[str, str]: ...
    def extract_slots(self, template: str) -> set[str]: ...

class MissingSlotError(Exception): ...
```

Disk layout (fully per ADR-004):
```
config/prompts/
  _registry.yaml           # {propose_architecture: v3, generate_code: v2, ...}
  propose_architecture/v1.yaml v2.yaml ...
  generate_code/v1.yaml ...
  recover_from_error/v1.yaml ...
  analyze_result/v1.yaml ...
  judge_experiment/v1.yaml ...
  judge_study/v1.yaml ...
```

YAML shape per version file:
```yaml
system: |
  You are ...
user: |
  Task: {task_description}
  ...
```

## Files
- Keep/audit: `lab/prompts/registry.py`, `lab/prompts/engine.py`
- Create: `config/prompts/{propose_architecture,generate_code,recover_from_error,analyze_result,judge_experiment,judge_study}/v1.yaml`
- Create: `tests/test_prompts.py`

## Acceptance
- Missing slot raises `MissingSlotError` naming the slot.
- `save_new_version` increments `vN`; never overwrites.
- `set_active` rewrites `_registry.yaml` atomically (write-to-temp + rename).
- Unit tests cover render + missing-slot + version swap.

## Depends on
None.
