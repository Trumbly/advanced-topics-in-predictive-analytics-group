# I-17 — CLI

**Labels:** `track-ui-ops`, `p0`
**Milestone:** week-4-ui-and-demo
**Owner:** Dev D

## Context
Graders run the agent from the command line. Single command must work out of the box.

## Scope
```bash
python -m lab run --task track_b \
    [--predecessor STUDY_ID] \
    [--use-best-prompts] \
    [--agent-memory] \
    [--personality exploratory|conservative] \
    [--max-experiments N] [--max-wallclock-min N]
python -m lab report <study_id>
python -m lab submit <study_id>
python -m lab prompts list|activate|new ...
python -m lab ui [--host H --port P]
python -m lab preprocess --task track_b
```

## Interface
```python
# lab/cli.py
def main(argv: list[str] | None = None) -> int: ...
```

Uses `argparse` (stdlib). Each subcommand is a function in `lab/cli.py`:
- `cmd_run(args) -> int`
- `cmd_report(args) -> int`
- `cmd_submit(args) -> int`
- `cmd_prompts(args) -> int`
- `cmd_ui(args) -> int`
- `cmd_preprocess(args) -> int`

## Files
- Rewrite `lab/cli.py`
- Keep `lab/__main__.py` → `from lab.cli import main; sys.exit(main())`
- Create `tests/test_cli.py`

## Acceptance
- `python -m lab run --help` prints all flags.
- `python -m lab run --task track_b --max-experiments 1` end-to-end on synthetic data finishes ≤3 min.
- Invalid flag combinations fail fast with a helpful error.

## Depends on
I-06.
