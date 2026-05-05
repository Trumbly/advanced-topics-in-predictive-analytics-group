# I-05 — Memory (top-K + failures, agent-memory seed)

**Labels:** `track-loop`, `p0`
**Milestone:** week-2-loop-and-task
**Owner:** Dev B

## Context
ADR-006 (compact markdown memory) + ADR-007 (optional cross-study agent memory).

## Scope
Hold top-K successful experiments + N recent failures. Render to ≤3 kB markdown for prompts. Seed from a predecessor study; seed from agent memory (all prior studies of same task).

## Interface
```python
class Memory:
    def __init__(self, top_k: int, recent_failures: int, path: Path): ...
    def add(self, exp: Experiment) -> None: ...
    def to_markdown(self) -> str: ...              # used as {experiment_memory} slot
    def seed_from_predecessor(self, predecessor: Study) -> None: ...
    def seed_from_agent_memory(self, studies_root: Path, task: str) -> None: ...
    def save(self) -> None: ...
    @classmethod
    def load(cls, path: Path, *, top_k: int, recent_failures: int) -> "Memory": ...
    def has_checkpoint_for(self, experiment_id: str) -> bool: ...
```

Markdown shape (≤3 kB total):
```md
## Top-5 experiments
1. exp_abcd1234 · EfficientNetB0 (transfer) · f1_macro=0.62 · loss 0.60→0.42→0.38 | trend improving
2. ...

## Recent failures (last 5)
- exp_efgh5678 · ResNet50-scratch · ShapeMismatch: expected (1,128,313) got (3,128,313)
- ...
```

Curve summary helper: `summarize_curve(history, metric) -> "loss X→Y→Z | metric A→B→C | trend improving|flat|regressing"`.

## Files
- Create `lab/core/memory.py`
- Create `tests/test_memory.py` with fixtures under `tests/fixtures/studies/`

## Acceptance
- `to_markdown()` ≤3 kB with 20 experiments.
- `seed_from_agent_memory` walks `experiments/studies/*/study.json`, filters by task, picks global top-K by `primary_score`.
- Trend classification correct for monotonic + flat + regressing fixtures.

## Depends on
I-02.
