"""I-08 acceptance: subprocess sandbox executor."""

from __future__ import annotations

from pathlib import Path

import pytest

from lab.config import load_settings
from lab.core.executor import LocalExecutor

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS = Path(__file__).parent / "fixtures" / "runs"


@pytest.fixture
def executor(tmp_path, monkeypatch):
    settings = load_settings("track_b", repo_root=REPO_ROOT)
    # Override sandbox path in-place via attribute mutation; pydantic model is
    # frozen-ish, so we rebuild a Settings-like proxy via copy.
    new_paths = settings.paths.model_copy(update={"sandbox": str(tmp_path / "sandbox")})
    new_settings = settings.model_copy(update={"paths": new_paths})
    return LocalExecutor(new_settings)


def _read(name: str) -> str:
    return (RUNS / name).read_text(encoding="utf-8")


def test_success_run_returns_metrics(executor, tmp_path):
    code = _read("success.py")
    result = executor.run(
        code, experiment_id="exp_succ", extra_env={}, timeout_s=10
    )
    assert result.succeeded is True
    assert result.primary_score == pytest.approx(0.42)
    assert "roc_auc_macro" in result.metrics
    assert len(result.history) == 2
    assert result.error is None


def test_sandbox_files_written(executor, tmp_path):
    executor.run(_read("success.py"), experiment_id="exp_files", extra_env={}, timeout_s=10)
    sb = Path(executor.sandbox_root) / "exp_files"
    assert (sb / "code.py").exists()
    assert (sb / "stdout.log").exists()
    assert (sb / "stderr.log").exists()
    assert (sb / "results.json").exists()


def test_value_error_classified(executor):
    result = executor.run(
        _read("value_error.py"), experiment_id="exp_val", extra_env={}, timeout_s=10
    )
    assert not result.succeeded
    assert result.error is not None
    assert result.error.error_type == "ValueError"


def test_shape_mismatch_classified(executor):
    result = executor.run(
        _read("shape_mismatch.py"), experiment_id="exp_shape", extra_env={}, timeout_s=30
    )
    assert not result.succeeded
    assert result.error is not None
    assert result.error.error_type == "ShapeMismatch"


def test_oom_classified(executor):
    result = executor.run(
        _read("oom.py"), experiment_id="exp_oom", extra_env={}, timeout_s=10
    )
    assert not result.succeeded
    assert result.error.error_type == "OOM"


def test_timeout_killed_and_classified(executor):
    result = executor.run(
        _read("timeout.py"), experiment_id="exp_to", extra_env={}, timeout_s=1
    )
    assert not result.succeeded
    assert result.error.error_type == "Timeout"
    assert result.duration_seconds < 5  # killed quickly, not full 60s


def test_missing_required_results_keys_is_failure(executor):
    result = executor.run(
        _read("missing_required_keys.py"),
        experiment_id="exp_bad_results",
        extra_env={},
        timeout_s=10,
    )
    assert not result.succeeded
    assert result.error is not None
    assert "primary_score" in result.error.message


def test_extra_env_is_injected(executor, tmp_path):
    code = (
        "import os, json\n"
        "from pathlib import Path\n"
        "Path('results.json').write_text(json.dumps({"
        "'primary_score': 0.5, 'primary_metric': 'roc_auc_macro',"
        "'metrics': {}, 'history': []}))\n"
        "Path('out.txt').write_text(os.environ.get('AGENT_TEST_KEY', 'missing'))\n"
    )
    result = executor.run(
        code,
        experiment_id="exp_env",
        extra_env={"AGENT_TEST_KEY": "abc-123"},
        timeout_s=10,
    )
    assert result.succeeded
    out_path = Path(executor.sandbox_root) / "exp_env" / "out.txt"
    assert out_path.read_text() == "abc-123"
