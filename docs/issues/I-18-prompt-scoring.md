# I-18 — Prompt scoring + use-best

**Labels:** `track-ui-ops`, `p1`
**Milestone:** week-3-integration
**Owner:** Dev D

## Context
Finish what prototype started. PDF asks for prompt robustness. Dashboard + "use best" power that story in the video.

## Scope
Aggregate mean/stdev/best/count of primary_score per (task, version) across all studies.

## Interface
```python
class PromptScoreStats(BaseModel):
    version: str
    mean: float; stdev: float; best: float
    count: int

def aggregate_prompt_scores(experiments_dir: Path
                            ) -> dict[str, dict[str, PromptScoreStats]]: ...

def get_best_version(task: str, experiments_dir: Path, *,
                     min_runs: int = 3) -> str | None: ...
```

Rules:
- Iterate `experiments_dir/study_*/study.json`.
- For each experiment with a known prompt version in `Study.prompt_template_paths[task]`, record `primary_score`.
- Drop versions with `count < min_runs` from the "best" selection (prevents single lucky run).
- Tie-break on `count` desc.

## Files
- Finish `lab/prompts/scoring.py`
- Create `tests/test_scoring.py` with 2 versions × 4 runs each fixture

## Acceptance
- Fixture with version A (mean 0.3) and B (mean 0.5) → `get_best_version` returns `B`.
- Filter `min_runs=3` excludes a 1-run version even if its score is higher.
- Dashboard JSON endpoint returns stats in the structure above.

## Depends on
I-02, I-04.
