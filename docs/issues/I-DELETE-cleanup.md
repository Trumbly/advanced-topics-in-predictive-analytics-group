# I-DELETE — Remove dead code

**Labels:** `track-ui-ops`, `p2`
**Milestone:** week-4-ui-and-demo
**Owner:** any dev after I-16 green

## Context
Rebuild is done. Clean up what was explicitly dropped in ADR-010 and review §11.

## Scope
Delete these files/dirs. **Do not delete before I-16 is green on main.** Deletions in one atomic PR.

## Targets
- `lab/core/kaggle_executor.py` (ADR-010)
- `lab/core/modal_executor.py`
- `lab/ui/routes/config_editor.py`
- Empty dirs: `agent/handlers/`, `agent/ui/`
- Any `kaggle`/`modal` imports in `lab/core/*` after above deletions
- Legacy prompts under `config/prompts/*/v*.yaml` that are no longer referenced (keep those that produced good study results — gated by `I-18 get_best_version`)

## Acceptance
- `rg "kaggle_executor|modal_executor|config_editor"` returns no hits in `lab/`.
- All tests still green.
- No `ImportError` on `python -m lab --help`.

## Depends on
I-16 merged and green.
