# I-09 — Recovery (autofix + LLM re-prompt)

**Labels:** `track-loop`, `p0`
**Milestone:** week-2-loop-and-task
**Owner:** Dev B

## Context
ADR-005. Keep the loop running. Try cheap deterministic fixes first; only re-prompt LLM when autofix doesn't apply.

## Scope
`Recovery` exposes both paths. Called from `run_experiment` after validator fail or executor fail.

## Interface
```python
class Recovery:
    def __init__(self, client: LLMClient, engine: PromptEngine,
                 settings: Settings): ...
    def try_autofix(self, code: str, finding: ValidationResult) -> str | None: ...
    def ask_llm(self, code: str,
                error: TaskError | ValidationResult,
                *, slots: dict[str, str]) -> str: ...

_HARD_FAILURE_ERROR_TYPES: set[str] = {"Timeout","OOM","FileNotFound"}
```

Autofix patterns:
- `UnknownTorchNN` with `autofix_hint=="rename nn.ConvXxYd→nn.Conv2d"` → regex replace.
- `BadSignature` and LLM emitted `build_model()` without arg → inject `num_classes` param.
- Over-cap epochs in Proposal → clamp to `max_epochs_per_run` in env, not in code.

LLM path: renders `recover_from_error` prompt with slots `{broken_code}`, `{error_type}`, `{error_message}`, `{error_traceback}`, plus the usual memory/profile slots.

## Files
- Create `lab/core/recovery.py`
- Create `tests/test_recovery.py`

## Acceptance
- Autofix test: `nn.Conv2x2d` → `nn.Conv2d` via deterministic path (no LLM call).
- LLM path exercised with fake `LLMClient` on `ShapeMismatch`.
- Hard failures never trigger retries in `StudyRunner` (consumer enforces, but `_HARD_FAILURE_ERROR_TYPES` is exported).

## Depends on
I-03, I-04, I-07, I-08.
