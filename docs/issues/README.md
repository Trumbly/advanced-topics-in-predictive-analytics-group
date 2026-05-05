# GitHub Issue Pack

One file per issue. Each body is the final issue body — ready for `gh issue create --body-file`.

Bulk-create all at once:

```bash
bash docs/issues/bulk_create.sh
```

Requires `gh` authenticated against `Trumbly/advanced-topics-in-predictive-analytics-group`.

## Index

| # | Title | Owner | Week | Depends | ADR |
|---|---|---|---|---|---|
| I-01 | Settings module (YAML-only) | A | 1 | — | ADR-002 |
| I-02 | Core data models | A | 1 | I-01 | ADR-015 |
| I-03 | LLMClient (provider switch + retries) | A | 1 | I-01 | ADR-001 |
| I-04 | Prompt registry + engine | A | 1 | — | ADR-004 |
| I-05 | Memory (top-K + failures, agent-memory seed) | B | 2 | I-02 | ADR-006, ADR-007 |
| I-06 | StudyRunner + experiment + parsing | B | 3 | I-01..I-05, I-07..I-09 | ADR-005, ADR-018 |
| I-07 | Validator (static + smoke) | B | 2 | I-01 | ADR-005 |
| I-08 | LocalExecutor (subprocess + error classification) | B | 2 | I-01 | ADR-010 |
| I-09 | Recovery (autofix + LLM re-prompt) | B | 2 | I-03, I-04, I-07, I-08 | ADR-005 |
| I-10 | TaskAdapter + BirdCLEF adapter | C | 2 | I-01, I-02 | ADR-003 |
| I-11 | Training skeleton (Jinja2) | C | 2 | I-10 | ADR-003, ADR-014 |
| I-12 | EDA step | C | 2 | I-10 | ADR-017 |
| I-13 | Reporting (figures + generator) | C | 3 | I-02 | — |
| I-14 | Submission builder (validate-before-build) | C | 3 | I-02, I-07, I-10 | ADR-011 |
| I-16 | FastAPI UI | D | 4 | I-02, I-04, I-13, I-18, I-21 | — |
| I-17 | CLI | D | 4 | I-06 | — |
| I-18 | Prompt scoring + use-best | D | 3 | I-02, I-04 | ADR-004 |
| I-19 | Judge role | C | 3 | I-02, I-03, I-04, I-05 | ADR-009 |
| I-20 | Agent memory toggle + transfer-learning gate | B | 3 | I-05, I-06 | ADR-007 |
| I-21 | Telemetry (structured + human, CLI + UI identical) | D | 3 | I-01 | ADR-012 |
| I-DELETE | Remove Kaggle/Modal executors + dead dirs | any | 4 | I-16 green | ADR-010 |
| I-DEMO | End-to-end smoke + README + video | all | 4 | all | — |

## Labels

All issues carry one "track" label and one "priority" label:

- **track-foundation**, **track-loop**, **track-task**, **track-output**, **track-ui-ops**
- **p0** (blocker), **p1** (critical path), **p2** (nice-to-close)

## Milestones

- `week-1-foundations`
- `week-2-loop-and-task`
- `week-3-integration`
- `week-4-ui-and-demo`
