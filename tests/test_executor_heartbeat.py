"""LocalExecutor heartbeat: emits progress callbacks while subprocess runs."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from lab.config import load_settings
from lab.core.executor import LocalExecutor

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def executor(tmp_path):
    s = load_settings("track_b", repo_root=REPO_ROOT)
    new_paths = s.paths.model_copy(update={"sandbox": str(tmp_path / "sandbox")})
    s = s.model_copy(update={"paths": new_paths})
    return LocalExecutor(s)


def test_heartbeat_fires_during_long_run(executor):
    """A subprocess that runs for >2s with heartbeat_interval=0.5s should call back."""
    code = (
        "import time, json\n"
        "from pathlib import Path\n"
        "for _ in range(10):\n"
        "    time.sleep(0.3)\n"
        "Path('results.json').write_text(json.dumps({"
        "'primary_score': 0.5, 'primary_metric': 'roc_auc_macro',"
        "'metrics': {}, 'history': []}))\n"
    )

    calls: list[tuple[float, int]] = []

    def cb(elapsed: float, stdout_bytes: int) -> None:
        calls.append((elapsed, stdout_bytes))

    result = executor.run(
        code,
        experiment_id="exp_hb",
        extra_env={},
        timeout_s=30,
        heartbeat_interval_s=0.5,
        on_heartbeat=cb,
    )
    assert result.succeeded, result.error
    assert calls, "expected at least one heartbeat call"
    # Heartbeats are roughly monotone in elapsed time
    elapsed_values = [c[0] for c in calls]
    assert elapsed_values == sorted(elapsed_values)


def test_heartbeat_skipped_when_callback_none(executor):
    """No callback => no crash, behaves like before."""
    code = (
        "import json\nfrom pathlib import Path\n"
        "Path('results.json').write_text(json.dumps({"
        "'primary_score': 0.5, 'primary_metric': 'roc_auc_macro',"
        "'metrics': {}, 'history': []}))\n"
    )
    result = executor.run(
        code, experiment_id="exp_no_hb", extra_env={}, timeout_s=10
    )
    assert result.succeeded


def test_heartbeat_callback_exception_does_not_break_run(executor):
    code = (
        "import time, json\nfrom pathlib import Path\n"
        "time.sleep(1.0)\n"
        "Path('results.json').write_text(json.dumps({"
        "'primary_score': 0.5, 'primary_metric': 'roc_auc_macro',"
        "'metrics': {}, 'history': []}))\n"
    )

    def boom(elapsed: float, stdout_bytes: int) -> None:
        raise RuntimeError("hook explodes")

    result = executor.run(
        code,
        experiment_id="exp_bad_hb",
        extra_env={},
        timeout_s=10,
        heartbeat_interval_s=0.3,
        on_heartbeat=boom,
    )
    assert result.succeeded


def test_timeout_still_kills_subprocess(executor):
    code = "import time\ntime.sleep(60)\n"
    t0 = time.monotonic()
    result = executor.run(
        code,
        experiment_id="exp_to",
        extra_env={},
        timeout_s=1,
        heartbeat_interval_s=10,
    )
    elapsed = time.monotonic() - t0
    assert not result.succeeded
    assert result.error.error_type == "Timeout"
    assert elapsed < 8  # killed quickly
