"""KaggleExecutor tests — exercise the wire without touching Kaggle.

We don't want the tests to require a real ``kaggle`` CLI or real
credentials, so we monkey-patch the subprocess boundary:

  * ``shutil.which("kaggle")`` is stubbed to return a fake path.
  * ``KaggleExecutor._run_kaggle`` is monkey-patched per test to return
    canned ``subprocess.CompletedProcess`` instances.

That lets us verify metadata generation, slug sanitisation, push+poll
sequencing, and the ExecutionResult shape across the success, error,
and cli-unavailable paths.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lab.core.kaggle_executor import KaggleCLIUnavailable, KaggleExecutor


def _completed(stdout: str = "", stderr: str = "", returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["kaggle"], returncode=returncode, stdout=stdout, stderr=stderr,
    )


def _make(tmp_path: Path) -> KaggleExecutor:
    return KaggleExecutor(
        username="maxuser",
        kernel_prefix="test",
        enable_gpu=True,
        poll_interval_seconds=0,          # no real sleeping
        sandbox_root=tmp_path,
        repo_root=tmp_path,
    )


def test_run_fails_cleanly_when_cli_missing(monkeypatch, tmp_path):
    monkeypatch.setattr("lab.core.kaggle_executor.shutil.which", lambda _: None)
    ex = _make(tmp_path)
    result = ex.run("print('hi')", experiment_id="exp_1")
    assert not result.succeeded
    assert result.error is not None
    assert result.error.error_type == "SpawnError"
    assert "kaggle CLI not found" in result.error.message


def test_run_happy_path_pushes_polls_and_fetches(monkeypatch, tmp_path):
    monkeypatch.setattr("lab.core.kaggle_executor.shutil.which", lambda _: "/usr/bin/kaggle")
    ex = _make(tmp_path)

    calls: list[list[str]] = []

    def fake_run(self, cli, argv, *, timeout):
        calls.append(argv)
        if argv[0:2] == ["kernels", "push"]:
            return _completed(stdout="Kernel pushed.\n")
        if argv[0:2] == ["kernels", "status"]:
            return _completed(stdout='Kernel has status "complete".\n')
        if argv[0:2] == ["kernels", "output"]:
            # Emulate the kernel having written results.json
            workdir = tmp_path / "exp_ok"
            workdir.mkdir(exist_ok=True)
            (workdir / "results.json").write_text(json.dumps({
                "primary_metric": "f1_macro",
                "primary_score": 0.77,
                "history": [],
                "final": {"f1_macro": 0.77},
                "best": {"f1_macro": 0.77},
            }))
            return _completed(stdout="epoch 1 ok\n")
        return _completed(stdout="")

    monkeypatch.setattr(KaggleExecutor, "_run_kaggle", fake_run)
    result = ex.run("print('hi')", experiment_id="exp_ok")

    assert result.succeeded, result.error
    assert result.exit_code == 0
    assert result.results_json_path is not None
    # Push happened, poll happened at least once, output was fetched.
    assert any(c[:2] == ["kernels", "push"] for c in calls)
    assert any(c[:2] == ["kernels", "status"] for c in calls)
    assert any(c[:2] == ["kernels", "output"] for c in calls)


def test_run_surfaces_kernel_error_as_task_error(monkeypatch, tmp_path):
    monkeypatch.setattr("lab.core.kaggle_executor.shutil.which", lambda _: "/usr/bin/kaggle")
    ex = _make(tmp_path)

    def fake_run(self, cli, argv, *, timeout):
        if argv[0:2] == ["kernels", "push"]:
            return _completed()
        if argv[0:2] == ["kernels", "status"]:
            return _completed(stdout='Kernel has status "error".\n')
        if argv[0:2] == ["kernels", "output"]:
            # Typical OOM traceback landing in the run log
            return _completed(stdout=(
                "Training started…\n"
                "Traceback (most recent call last):\n"
                "RuntimeError: CUDA out of memory. Tried to allocate …\n"
            ))
        return _completed()

    monkeypatch.setattr(KaggleExecutor, "_run_kaggle", fake_run)
    result = ex.run("print('hi')", experiment_id="exp_err")
    assert not result.succeeded
    assert result.error is not None
    # classify_error should have bucketed CUDA-OOM into OOM.
    assert result.error.error_type == "OOM"


def test_kernel_slug_sanitises_experiment_id(tmp_path):
    ex = _make(tmp_path)
    assert ex._kernel_slug("exp_AB_12") == "maxuser/test-exp-ab-12"


def test_infrastructure_reports_gpu_and_sources(tmp_path):
    ex = KaggleExecutor(
        username="u",
        enable_gpu=True,
        competition_sources=["birdclef-2026"],
        sandbox_root=tmp_path,
        repo_root=tmp_path,
    )
    infra = ex.infrastructure()
    assert infra["backend"] == "kaggle"
    assert infra["enable_gpu"] is True
    assert infra["competition_sources"] == ["birdclef-2026"]
