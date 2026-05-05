# I-12 — EDA step

**Labels:** `track-task`, `p1`
**Milestone:** week-2-loop-and-task
**Owner:** Dev C

## Context
ADR-017. Runs once per study at start. Feeds every `propose_architecture` prompt.

## Scope
Read the task's processed dataset, compute a compact summary, cache on disk.

## Interface
```python
class EDAReport(BaseModel):
    markdown: str
    num_classes: int; num_train: int
    imbalance_ratio: float                 # max/min class count
    input_tensor_shape: tuple[int, ...]
    notes: list[str] = []

def run_eda(adapter: TaskAdapter, settings: Settings) -> EDAReport: ...
```

Markdown shape (≤4 kB):
```md
# EDA — {task}
- Train samples: 15 320
- Classes: 234
- Class imbalance (max/min): 37.1x  (top class: 412 samples; bottom: 11)
- Input tensor shape: (1, 128, 313)
- Notable: 18 classes have <25 samples; consider augmentation + class-weighted loss.
```

Cache: `experiments/studies/{study_id}/eda.md`. `StudyRunner.run()` calls `run_eda` once and injects result as `{eda_summary}` for every propose prompt.

## Files
- Rewrite `lab/tasks/eda.py`
- Create `tests/test_eda.py`

## Acceptance
- Report cap ≤4 kB.
- Imbalance ratio correct on fixture with 2 classes (5 vs 50 → 10.0).
- Notes list mentions long-tail classes when `<25 samples` in any class.

## Depends on
I-10.
