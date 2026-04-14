"""Pure-helper tests for the orchestrator (no LLM, no subprocess).

Targets the two hardenings that prevent prose-as-code crashes:
  * ``_strip_fences`` extracts the FIRST fenced ```python``` block
    when the LLM wraps its prose around code.
  * Bare prose (no fence) goes through unchanged so the validator
    rejects it cleanly with SyntaxError on the next round-trip.
"""
from __future__ import annotations

from lab.core.models import Experiment, ExperimentStatus, StudyStatus
from lab.core.orchestrator import Orchestrator, _strip_fences


def test_strip_fences_extracts_python_block_among_prose():
    raw = (
        "Looking at the error, the issue is …\n"
        "Here's the fix:\n"
        "```python\n"
        "import torch\n"
        "print('hi')\n"
        "```\n"
        "Hope this helps!"
    )
    out = _strip_fences(raw)
    assert out == "import torch\nprint('hi')"


def test_strip_fences_extracts_first_block_when_multiple():
    raw = "```python\nx = 1\n```\n\nand\n\n```python\ny = 2\n```"
    assert _strip_fences(raw) == "x = 1"


def test_strip_fences_handles_bare_outer_fence():
    raw = "```\nimport torch\n```"
    assert _strip_fences(raw) == "import torch"


def test_strip_fences_passthrough_when_no_fences():
    raw = "import torch\nprint('hi')\n"
    assert _strip_fences(raw).startswith("import torch")


def test_strip_fences_handles_pythonless_fence_marker():
    raw = "```py\nx = 42\n```"
    assert _strip_fences(raw) == "x = 42"


def test_strip_fences_strips_pure_prose_to_itself():
    """Bare prose has no fence — return as-is so the validator rejects
    it cleanly with a SyntaxError (don't try to be clever and "fix" it
    here; the recovery loop should re-prompt the LLM)."""
    raw = "Looking at the error, I think we should just give up."
    out = _strip_fences(raw)
    assert "Looking at the error" in out


# ---------------------------------------------------------------------------
# Final study status derivation
# ---------------------------------------------------------------------------


def _exp(status: ExperimentStatus) -> Experiment:
    return Experiment(study_id="s", primary_metric="x", status=status)


def test_final_status_aborted_wins_even_with_a_success():
    derive = Orchestrator._derive_final_status
    exps = [_exp(ExperimentStatus.COMPLETED), _exp(ExperimentStatus.FAILED)]
    assert derive(exps, aborted=True) == StudyStatus.ABORTED


def test_final_status_completed_when_at_least_one_succeeded():
    derive = Orchestrator._derive_final_status
    exps = [_exp(ExperimentStatus.FAILED), _exp(ExperimentStatus.COMPLETED), _exp(ExperimentStatus.FAILED)]
    assert derive(exps, aborted=False) == StudyStatus.COMPLETED


def test_final_status_failed_when_all_experiments_failed():
    derive = Orchestrator._derive_final_status
    exps = [_exp(ExperimentStatus.FAILED), _exp(ExperimentStatus.FAILED)]
    assert derive(exps, aborted=False) == StudyStatus.FAILED


def test_final_status_failed_when_no_experiments_ran():
    derive = Orchestrator._derive_final_status
    assert derive([], aborted=False) == StudyStatus.FAILED
