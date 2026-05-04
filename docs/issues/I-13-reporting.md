# I-13 — Reporting (figures + generator)

**Labels:** `track-output`, `p1`
**Milestone:** week-3-integration
**Owner:** Dev C

## Context
D4 report needs diagrams + iteration analysis + failure distribution. ROC-AUC is Track B primary → need per-class view.

## Scope
Figures + markdown/HTML report.

## Interface
```python
def render_figures(study: Study, out_dir: Path) -> dict[str, Path]:
    """
    Keys: "score_progression","best_learning_curve","failure_breakdown",
          "family_performance","per_class_auc"
    """
def generate_report(study: Study, settings: Settings) -> Path: ...
```

Figures (plotly):
- **score_progression** — X: experiment index; Y: primary_score; annotated with family.
- **best_learning_curve** — two-axis: loss + primary metric per epoch of best exp.
- **failure_breakdown** — pie of `error_type` across failed experiments.
- **family_performance** — boxplot of scores per `architecture_family`.
- **per_class_auc** — sorted bar chart of per-class ROC-AUC from best exp (expects key `per_class_auc` in metrics; if absent, skip).

Report template (`report.md.j2`):
- Executive summary (LLM-generated optional; fallback to a deterministic stat summary)
- Study metadata: personality, agent_memory_enabled, predecessor
- Top 5 experiments table
- Failure summary
- Judge verdicts per experiment + study-level verdict
- Embedded figures

HTML: run `markdown-it` on the MD; save alongside `report.md`.

## Files
- Rewrite `lab/reporting/generator.py`, `lab/reporting/figures.py`
- Rewrite `lab/reporting/templates/report.md.j2`
- Create `tests/test_reporting.py` + study fixture

## Acceptance
- All 5 figures present on fixture study with ≥3 experiments (skip `per_class_auc` allowed if metric missing).
- Report renders without Jinja undefined errors even if optional fields absent.
- HTML file valid (parses under `html.parser`).

## Depends on
I-02.
