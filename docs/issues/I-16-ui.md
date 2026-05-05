# I-16 — FastAPI UI

**Labels:** `track-ui-ops`, `p1`
**Milestone:** week-4-ui-and-demo
**Owner:** Dev D

## Context
Graders will poke around the UI to see studies, prompts, reports. UI is read-mostly.

## Scope
Studies list/detail, experiment detail, prompt dashboard with A/B aggregation, report viewer, run form with "use best prompts" checkbox. Live SSE stream of telemetry per study.

## Routes
| Method | Path | Body / Query | Returns |
|---|---|---|---|
| GET | `/studies` | — | HTML list |
| GET | `/studies/{id}` | — | HTML detail + figures |
| GET | `/experiments/{study_id}/{exp_id}` | — | HTML detail |
| GET | `/prompts` | — | HTML dashboard |
| POST | `/prompts/{task}/activate` | `version=vN` | 303 → `/prompts` |
| POST | `/prompts/{task}/new` | `{system, user}` | `{version: "vN"}` |
| GET | `/reports/{study_id}` | — | HTML |
| POST | `/run` | `{task, predecessor_id?, prompt_overrides?, use_best_prompts?: bool, personality?, agent_memory?: bool}` | `{study_id}` |
| GET | `/live/{study_id}` | — | text/event-stream |
| GET | `/api/studies` | — | JSON list |
| GET | `/api/studies/{id}` | — | JSON detail |

SSE events mirror telemetry JSONL schema.

## Files
- Rewrite `lab/ui/app.py`
- Rewrite `lab/ui/routes/*.py` (remove `config_editor`)
- Rewrite `lab/ui/templates/*.html`
- Create `tests/test_ui.py` (httpx + TestClient)

## Acceptance
- All routes respond; 500 on completed study detail is eliminated (regression test on fixture study that triggered the bug).
- "use best prompts" checkbox calls `PromptScoring.get_best_version(task)` for each task.
- SSE streams at least one `experiment_start` event when a run is launched.

## Out of scope
- Live YAML editor, live CLI builder (both dropped).

## Depends on
I-02, I-04, I-13, I-18, I-21.
