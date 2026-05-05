# I-19 — Judge role

**Labels:** `track-loop`, `p0`
**Milestone:** week-3-integration
**Owner:** Dev C

## Context
ADR-009. Per-experiment + per-study LLM judgment. Feeds demo video.

## Scope
Two prompts. Two methods. One pydantic `Verdict`.

## Interface
```python
class Judge:
    def __init__(self, client: LLMClient, engine: PromptEngine): ...
    def judge_experiment(self, exp: Experiment, memory: Memory) -> Verdict: ...
    def judge_study(self, study: Study) -> Verdict: ...
```

Prompt slots for `judge_experiment`:
- `{architecture_name}`, `{architecture_family}`, `{primary_metric}`, `{primary_score}`
- `{curve_summary}`, `{error}` (if any)
- `{experiment_memory}`, `{personality}`

Prompt slots for `judge_study`:
- `{top_experiments_table}` (top 5 markdown)
- `{failure_breakdown}`
- `{budget_used}`, `{budget_total}`

`Verdict.verdict` values meaning:
- `promote`: candidate for submission
- `keep`: keep in memory, do not promote
- `discard`: do not seed next experiments from this
- `abort_study`: stop the loop now

LLM response is strict JSON; on parse fail retry once with the validation error (same pattern as ADR-018).

## Files
- Create `lab/core/judge.py`
- Create `config/prompts/judge_experiment/v1.yaml`
- Create `config/prompts/judge_study/v1.yaml`
- Create `tests/test_judge.py`

## Acceptance
- `abort_study` verdict short-circuits `StudyRunner` loop (regression test).
- `Verdict` attached to every experiment JSON after I-06 integration.
- Rationale ≤500 chars enforced via pydantic.

## Depends on
I-02, I-03, I-04, I-05.
