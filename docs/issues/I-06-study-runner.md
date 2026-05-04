# I-06 — StudyRunner + experiment + parsing

**Labels:** `track-loop`, `p0`
**Milestone:** week-3-integration
**Owner:** Dev B

## Context
ADR-005 (two-layer self-repair) + ADR-018 (strict proposal JSON + one retry). The agent loop lives here.

## Scope
Split the 1363-LOC orchestrator prototype into three focused files: lifecycle (study loop), experiment (one iteration), parsing (proposal JSON).

## Interface
```python
# lab/core/parsing.py
def parse_proposal(text: str, *, retry_client: LLMClient | None = None,
                   retry_messages: list[dict] | None = None) -> Proposal: ...
class ProposalParseError(Exception): ...

# lab/core/experiment.py
@dataclass
class RunContext:
    settings: Settings; adapter: TaskAdapter; client: LLMClient
    engine: PromptEngine; memory: Memory
    validator: Validator; executor: LocalExecutor
    recovery: Recovery; judge: Judge
    hooks: Hooks | None = None

def run_experiment(exp: Experiment, ctx: RunContext) -> Experiment: ...

# lab/core/lifecycle.py
class StudyRunner:
    def __init__(self, ctx: RunContext): ...
    def run(self, *, predecessor: Study | None = None) -> Study: ...
    def abort(self) -> None: ...     # SIGINT-safe
```

`run_experiment` stages:
1. `_propose(ctx, exp)` → `Proposal`  (one retry on JSON parse fail)
2. `_generate(ctx, exp, proposal)` → code
3. `_validate_and_retry(ctx, exp, code)` → validated code (ADR-005 layer 1)
4. `_execute_and_retry(ctx, exp, code)` → `ExecutionResult` (ADR-005 layer 2)
5. `_capture(exp, exec_result)`
6. `ctx.judge.judge_experiment(exp, ctx.memory)` → sets `exp.verdict`
7. `ctx.memory.add(exp)`

Loop control:
- `max_experiments`, `max_wallclock_minutes` enforced in `StudyRunner.run`.
- If `exp.verdict.verdict == "abort_study"` → stop loop.
- On SIGINT: current exp → `ABORTED`, study → `ABORTED`, flush logs.

## Files
- Create `lab/core/parsing.py`, `lab/core/experiment.py`, `lab/core/lifecycle.py`
- Create `tests/test_parsing.py`, `tests/test_experiment.py`, `tests/test_lifecycle.py`
- Create `tests/test_integration_e2e.py` — stubbed LLMClient, synthetic mel fixture, 2 experiments

## Acceptance
- E2E test produces valid `study.json` with two completed experiments.
- `abort_study` verdict halts loop at experiment index 1 of 5.
- SIGINT leaves a valid `study.json` with `status=ABORTED`.
- `parse_proposal` JSON-retry test: first LLM response is invalid JSON, second is valid → returns `Proposal`.

## Depends on
I-01..I-05, I-07, I-08, I-09, I-10, I-11, I-19.
