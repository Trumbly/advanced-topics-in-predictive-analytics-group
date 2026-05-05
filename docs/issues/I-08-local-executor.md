# I-08 — LocalExecutor

**Labels:** `track-loop`, `p0`
**Milestone:** week-2-loop-and-task
**Owner:** Dev B

## Context
ADR-010. Only local execution. Subprocess sandbox under `sandbox/{experiment_id}/`.

## Scope
Run generated code in a subprocess with env-var injection, enforce timeout, capture `results.json`, classify errors.

## Interface
```python
@dataclass
class ExecutionResult:
    succeeded: bool
    primary_score: float | None
    metrics: dict[str, float]
    history: list[dict]
    duration_seconds: float
    stdout: str; stderr: str
    error: TaskError | None     # None iff succeeded

class LocalExecutor:
    def __init__(self, settings: Settings): ...
    def run(self, code: str, *, experiment_id: str,
            extra_env: dict[str, str], timeout_s: int) -> ExecutionResult: ...

# error_type values:
_ERROR_TYPES = {"OOM","Timeout","ShapeMismatch","ValueError","RuntimeError",
                "FileNotFound","ImportError","Other"}
_HARD_FAILURES = {"Timeout","OOM","FileNotFound"}   # exposed for Recovery
```

Behavior:
- Create `sandbox/{experiment_id}/code.py`, write code there.
- `subprocess.Popen([sys.executable, "code.py"], cwd=sandbox_dir, env=merged_env, ...)`.
- Monitor with `wait(timeout=timeout_s)`. On timeout: kill + `error_type="Timeout"`.
- Parse `sandbox/{experiment_id}/results.json` if present.
- Classify stderr via regex table when results.json absent.
- Always write `stdout.log` and `stderr.log`.

## Files
- Rewrite `lab/core/executor.py`
- Create `tests/test_executor.py` with fixture scripts under `tests/fixtures/runs/`

## Acceptance
- Fixtures classified: OOM, Timeout, ShapeMismatch, ValueError, success.
- results.json schema enforced (missing fields → error).
- Sandbox directory exists after run with logs + code.

## Depends on
I-01.
