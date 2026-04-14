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


# ---------------------------------------------------------------------------
# Hard-failure short-circuit
# ---------------------------------------------------------------------------


def test_hard_failure_set_includes_environmental_errors():
    """Recovery should NOT be attempted on errors the LLM can't fix
    from inside the script: missing data, OOM, timeouts."""
    from lab.core.orchestrator import _HARD_FAILURE_ERROR_TYPES
    assert "Timeout" in _HARD_FAILURE_ERROR_TYPES
    assert "OOM" in _HARD_FAILURE_ERROR_TYPES
    assert "FileNotFound" in _HARD_FAILURE_ERROR_TYPES
    # Code-level errors stay recoverable.
    assert "SyntaxError" not in _HARD_FAILURE_ERROR_TYPES
    assert "ShapeMismatch" not in _HARD_FAILURE_ERROR_TYPES
    assert "RuntimeError" not in _HARD_FAILURE_ERROR_TYPES


# ---------------------------------------------------------------------------
# In-band error field in results.json must be honoured
# ---------------------------------------------------------------------------


def test_capture_metrics_returns_error_field_in_raw(tmp_path):
    """_capture_metrics surfaces the raw results dict so the caller can
    inspect an in-band error field even when the script exited 0."""
    import json
    from lab.core.executor import ExecutionResult
    from lab.core.orchestrator import Orchestrator

    results = tmp_path / "results.json"
    results.write_text(json.dumps({
        "primary_metric": "f1_macro",
        "primary_score": 0.0,
        "history": [],
        "final": {"f1_macro": 0.0},
        "best": {"f1_macro": 0.0},
        "error": "FileNotFoundError: missing labels.csv",
    }))
    exec_result = ExecutionResult(
        exit_code=0,
        stdout="",
        stderr="",
        duration_seconds=1.0,
        workdir=tmp_path,
        results_json_path=results,
        error=None,
        timed_out=False,
    )
    # Build a stub orchestrator just to call _capture_metrics — we don't
    # need the full LLM/executor wiring for this, only the adapter shim.
    class _StubAdapter:
        primary_metric = "f1_macro"

    orch = Orchestrator.__new__(Orchestrator)
    orch.adapter = _StubAdapter()
    out = orch._capture_metrics(exec_result)
    assert out["raw"]["error"].startswith("FileNotFoundError")
