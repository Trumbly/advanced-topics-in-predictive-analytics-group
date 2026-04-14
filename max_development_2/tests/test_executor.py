"""Executor smoke tests — verifies subprocess sandbox basics without torch."""
from __future__ import annotations

from pathlib import Path

from lab.core.executor import CodeExecutor


def test_executor_runs_simple_script_and_collects_results(tmp_path):
    script = (
        "import json, os\n"
        "from pathlib import Path\n"
        "Path('results.json').write_text(json.dumps({'primary_metric': 'acc', 'primary_score': 1.0}))\n"
        "print('ok')\n"
    )
    ex = CodeExecutor(sandbox_root=tmp_path, timeout_seconds=10, stream_output=False)
    r = ex.run(script, experiment_id="exp_smoke")
    assert r.succeeded, r.error
    assert "ok" in r.stdout


def test_executor_flags_no_results_when_script_exits_clean_but_skipped_write(tmp_path):
    script = "print('hello but no results file')\n"
    ex = CodeExecutor(sandbox_root=tmp_path, timeout_seconds=10, stream_output=False)
    r = ex.run(script, experiment_id="exp_noresult")
    assert not r.succeeded
    assert r.error is not None
    assert r.error.error_type == "NoResultsFile"


def test_executor_classifies_syntax_error(tmp_path):
    script = "this is not python at all\n"
    ex = CodeExecutor(sandbox_root=tmp_path, timeout_seconds=10, stream_output=False)
    r = ex.run(script, experiment_id="exp_syn")
    assert not r.succeeded
    assert r.error is not None
    assert r.error.error_type in {"SyntaxError", "UnknownError"}


def test_executor_respects_timeout(tmp_path):
    script = "import time\ntime.sleep(10)\n"
    ex = CodeExecutor(sandbox_root=tmp_path, timeout_seconds=1, stream_output=False)
    r = ex.run(script, experiment_id="exp_slow")
    assert r.timed_out
    assert r.error is not None
    assert r.error.error_type == "Timeout"
