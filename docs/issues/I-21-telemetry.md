# I-21 — Telemetry (structured + human, CLI + UI identical)

**Labels:** `track-ui-ops`, `p0`
**Milestone:** week-3-integration
**Owner:** Dev D

## Context
ADR-012. Logs identical whether run from CLI or UI. Per-epoch progress visible to tail -f.

## Scope
Single init. Two sinks (JSONL + human). Stdout mirrors human sink.

## Interface
```python
def configure(settings: Settings, *, study_id: str | None = None) -> None: ...

def log_event(event: Literal[
    "study_start","study_end","experiment_start","experiment_end",
    "llm_call","validate","execute","recover","judge","submission",
    "epoch","error"
], level: Literal["info","warn","error"] = "info", **fields) -> None: ...

def log_human(msg: str, *, level: str = "info") -> None: ...
```

JSONL schema per ADR-012 / ARCHITECTURE.md §8.

Instrumentation required from:
- `LLMClient.chat` → `llm_call` events with duration + provider
- `Validator.validate` → `validate` event with error_type
- `LocalExecutor.run` → `execute` + `epoch` events; tqdm-style on stdout
- `Recovery.try_autofix`, `Recovery.ask_llm` → `recover` event
- `Judge.judge_*` → `judge` event
- `StudyRunner` → `study_start`, `experiment_start`, `experiment_end`, `study_end`
- `SubmissionBuilder` → `submission` event

UI SSE route (`/live/{study_id}`) reads `run.log.jsonl` and streams new lines.

## Files
- Rewrite `lab/core/telemetry.py`
- Instrument the modules listed above (one-line `log_event` calls)
- Create `tests/test_telemetry.py`

## Acceptance
- Running same study twice (once from CLI, once from `POST /run`) produces identical event schemas.
- `tail -f experiments/studies/{id}/run.log` shows per-epoch progress.
- No double-writes when UI is also open.

## Depends on
I-01.
