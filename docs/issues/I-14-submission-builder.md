# I-14 — Submission builder (validate-before-build)

**Labels:** `track-output`, `p0`
**Milestone:** week-3-integration
**Owner:** Dev C

## Context
ADR-011. Track B Kaggle submission is a CPU-only notebook with 90-min runtime limit. Built from best experiment. Validator runs first.

## Scope
Reuses `Validator` in submission mode. Writes a notebook from Jinja template.

## Interface
```python
class SubmissionValidationError(Exception):
    def __init__(self, message: str, remediation: list[str]): ...

def build_submission_for_study(study: Study, settings: Settings) -> Path: ...
def _validate_for_submission(code: str, adapter: TaskAdapter, settings: Settings) -> None: ...
```

Behavior:
1. Load best experiment by `study.best_experiment_id`.
2. `_validate_for_submission(best.code, adapter, settings)` → raises on fail with remediation list.
3. Render `notebook_template.ipynb.j2` with cells:
   - Cell 1: `import os; os.environ.update({AGENT_DEVICE:"cpu", ...})` + offline weight paths.
   - Cell 2: paste `build_model` block only (LLM's contribution).
   - Cell 3: paste the skeleton content minus the training loop (inference only).
   - Cell 4: load test spectrograms, run inference, write `submission.csv`.
4. Write to `experiments/studies/{study_id}/submission.ipynb`.
5. Optional: run a 60-s CPU smoke test on the rendered notebook's inference cell.

## Files
- Rewrite `lab/submission/builder.py`
- Create `lab/submission/notebook_template.ipynb.j2`
- Create `tests/test_submission.py`

## Acceptance
- Rejects code with `urllib`/`requests`/`subprocess` (submission mode).
- Rejects code that references non-existent `build_model` or wrong signature.
- Produced notebook is a valid `.ipynb` JSON (parses via `nbformat.read`).
- CSV has `row_id,species_id,probability` columns matching competition template.

## Depends on
I-02, I-07, I-10.
