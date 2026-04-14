"""Unit tests for `agent.executor.CodeExecutor`."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from agent.executor import CodeExecutor, ExecutionResult, classify_error


# ---------------------------------------------------------------------------
# classify_error — pure function, no subprocess
# ---------------------------------------------------------------------------


class TestClassifyError:
    def test_timeout(self) -> None:
        err = classify_error("", timed_out=True)
        assert err is not None
        assert err.error_type == "Timeout"

    def test_no_stderr_no_error(self) -> None:
        assert classify_error("", timed_out=False) is None

    def test_syntax_error(self) -> None:
        stderr = "  File 'x.py', line 1\n    1 +\n       ^\nSyntaxError: invalid syntax"
        err = classify_error(stderr, timed_out=False)
        assert err is not None
        assert err.error_type == "SyntaxError"

    def test_oom(self) -> None:
        err = classify_error("MemoryError: out of memory", timed_out=False)
        assert err is not None
        assert err.error_type == "OOM"

    def test_shape_mismatch(self) -> None:
        err = classify_error(
            "RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x128 and 256x234)",
            timed_out=False,
        )
        assert err is not None
        assert err.error_type == "ShapeMismatch"

    def test_import_error(self) -> None:
        err = classify_error("ModuleNotFoundError: No module named 'foo'", timed_out=False)
        assert err is not None
        assert err.error_type == "ImportError"

    def test_unknown_error(self) -> None:
        err = classify_error("Something weird happened", timed_out=False)
        assert err is not None
        assert err.error_type == "UnknownError"


# ---------------------------------------------------------------------------
# CodeExecutor — real subprocess
# ---------------------------------------------------------------------------


class TestCodeExecutor:
    @pytest.fixture
    def executor(self, tmp_path: Path) -> CodeExecutor:
        return CodeExecutor(
            sandbox_root=tmp_path / "sandbox",
            timeout_seconds=10,
            python_executable=sys.executable,
        )

    def test_successful_run_writes_results_json(
        self, executor: CodeExecutor
    ) -> None:
        code = """
import json, pathlib
pathlib.Path('results.json').write_text(json.dumps({'metrics': {'f1_macro': 0.5, 'roc_auc_macro': 0.5, 'loss': 1.0}}))
print('done')
"""
        result = executor.run(code, experiment_id="exp_001")
        assert result.succeeded
        assert result.exit_code == 0
        assert "done" in result.stdout
        assert result.results_json_path is not None
        assert result.results_json_path.exists()
        assert result.error is None

    def test_syntax_error_is_classified(self, executor: CodeExecutor) -> None:
        code = "def broken(:\n    pass\n"
        result = executor.run(code, experiment_id="exp_syntax")
        assert not result.succeeded
        assert result.error is not None
        assert result.error.error_type == "SyntaxError"
        assert result.results_json_path is None

    def test_runtime_error_is_classified(self, executor: CodeExecutor) -> None:
        code = "raise ValueError('oops')\n"
        result = executor.run(code, experiment_id="exp_runtime")
        assert not result.succeeded
        assert result.error is not None
        assert result.error.error_type == "ValueError"

    def test_timeout_is_classified(self, tmp_path: Path) -> None:
        executor = CodeExecutor(
            sandbox_root=tmp_path / "sandbox",
            timeout_seconds=1,
            python_executable=sys.executable,
        )
        code = "import time\nwhile True:\n    time.sleep(0.1)\n"
        result = executor.run(code, experiment_id="exp_timeout")
        assert result.timed_out
        assert result.error is not None
        assert result.error.error_type == "Timeout"

    def test_workdir_is_reused_and_cleaned(
        self, executor: CodeExecutor
    ) -> None:
        """Rerunning the same experiment should start fresh, not see stale files."""
        code_1 = "import json, pathlib; pathlib.Path('results.json').write_text(json.dumps({'metrics': {'roc_auc_macro': 0.1, 'loss': 2.0}}))"
        code_2 = "raise RuntimeError('fail')"

        r1 = executor.run(code_1, experiment_id="exp_reuse")
        assert r1.succeeded
        assert r1.results_json_path is not None and r1.results_json_path.exists()

        r2 = executor.run(code_2, experiment_id="exp_reuse")
        assert not r2.succeeded
        # Stale results.json from the first run must be gone
        assert r2.results_json_path is None

    def test_stdout_and_stderr_files_written(
        self, executor: CodeExecutor
    ) -> None:
        code = "import sys; print('hi'); print('bye', file=sys.stderr)"
        result = executor.run(code, experiment_id="exp_io")
        stdout_file = result.workdir / "stdout.log"
        stderr_file = result.workdir / "stderr.log"
        assert stdout_file.exists()
        assert stderr_file.exists()
        assert "hi" in stdout_file.read_text()
        assert "bye" in stderr_file.read_text()

    def test_exit_zero_but_no_results_is_failure(
        self, executor: CodeExecutor
    ) -> None:
        """Regression: if a script exits 0 but does not produce results.json,
        the run must be marked as failed with a clear NoResultsFile error,
        not silently reported as succeeded. Previously this caused confusing
        downstream failures in capture_metrics."""
        code = "print('I did some work but forgot to write results.json')\n"
        result = executor.run(code, experiment_id="exp_silent")
        assert result.exit_code == 0
        assert result.results_json_path is None
        assert not result.succeeded
        assert result.error is not None
        assert result.error.error_type == "NoResultsFile"
        assert "results.json" in result.error.message

    def test_env_vars_for_data_paths_are_set(
        self, tmp_path: Path
    ) -> None:
        """The executor must export BIRDCLEF_DATASET_PROFILE,
        BIRDCLEF_SPECTROGRAMS_DIR, and BIRDCLEF_LABELS_CSV as ABSOLUTE paths
        so sandboxed code can read data paths regardless of its cwd."""
        executor = CodeExecutor(
            sandbox_root=tmp_path / "sandbox",
            timeout_seconds=10,
            python_executable=sys.executable,
            repo_root=tmp_path / "fake_repo",
        )
        code = (
            "import os, json, pathlib\n"
            "paths = {\n"
            "    'profile': os.environ.get('BIRDCLEF_DATASET_PROFILE'),\n"
            "    'spectrograms': os.environ.get('BIRDCLEF_SPECTROGRAMS_DIR'),\n"
            "    'labels': os.environ.get('BIRDCLEF_LABELS_CSV'),\n"
            "}\n"
            "pathlib.Path('results.json').write_text(json.dumps({\n"
            "    'metrics': {'f1_macro': 0.5, 'roc_auc_macro': 0.5, 'loss': 1.0},\n"
            "    'env': paths,\n"
            "}))\n"
        )
        result = executor.run(code, experiment_id="exp_env")
        assert result.succeeded, f"stderr: {result.stderr}"
        import json
        data = json.loads(result.results_json_path.read_text())  # type: ignore[union-attr]
        env = data["env"]
        assert env["profile"] is not None and env["profile"].endswith(
            "dataset_profile.json"
        )
        assert env["spectrograms"] is not None and env["spectrograms"].endswith(
            "spectrograms"
        )
        assert env["labels"] is not None and env["labels"].endswith("labels.csv")
        # All three must be absolute paths
        for key, val in env.items():
            assert val.startswith("/"), f"{key} is not absolute: {val}"

    def test_training_env_is_forwarded_to_subprocess(self, tmp_path: Path) -> None:
        """The `training_env` dict on CodeExecutor must be exported into
        the subprocess environment. This is how config.yaml's `training:`
        section reaches `pipelines.data_loader` inside the sandbox."""
        executor = CodeExecutor(
            sandbox_root=tmp_path / "sandbox",
            timeout_seconds=10,
            python_executable=sys.executable,
            training_env={
                "BIRDCLEF_DEVICE": "mps",
                "BIRDCLEF_BATCH_SIZE": "256",
                "BIRDCLEF_NUM_WORKERS": "10",
                "BIRDCLEF_PERSISTENT_WORKERS": "true",
                "BIRDCLEF_PREFETCH_FACTOR": "8",
            },
        )
        code = (
            "import os, json, pathlib\n"
            "vals = {\n"
            "    k: os.environ.get(k) for k in (\n"
            "        'BIRDCLEF_DEVICE', 'BIRDCLEF_BATCH_SIZE',\n"
            "        'BIRDCLEF_NUM_WORKERS', 'BIRDCLEF_PERSISTENT_WORKERS',\n"
            "        'BIRDCLEF_PREFETCH_FACTOR',\n"
            "    )\n"
            "}\n"
            "pathlib.Path('results.json').write_text(json.dumps({\n"
            "    'metrics': {'f1_macro': 0.5, 'roc_auc_macro': 0.5, 'loss': 1.0},\n"
            "    'training_env': vals,\n"
            "}))\n"
        )
        result = executor.run(code, experiment_id="exp_training_env")
        assert result.succeeded, f"stderr: {result.stderr}"
        import json
        data = json.loads(result.results_json_path.read_text())  # type: ignore[union-attr]
        env = data["training_env"]
        assert env["BIRDCLEF_DEVICE"] == "mps"
        assert env["BIRDCLEF_BATCH_SIZE"] == "256"
        assert env["BIRDCLEF_NUM_WORKERS"] == "10"
        assert env["BIRDCLEF_PERSISTENT_WORKERS"] == "true"
        assert env["BIRDCLEF_PREFETCH_FACTOR"] == "8"

    def test_thread_env_vars_are_set(self, tmp_path: Path) -> None:
        """The executor must export OMP_NUM_THREADS, MKL_NUM_THREADS etc.
        so PyTorch matrix ops don't silently run single-threaded on
        conda/macOS default setups."""
        executor = CodeExecutor(
            sandbox_root=tmp_path / "sandbox",
            timeout_seconds=10,
            python_executable=sys.executable,
        )
        code = (
            "import os, json, pathlib\n"
            "vals = {\n"
            "    k: os.environ.get(k) for k in (\n"
            "        'OMP_NUM_THREADS', 'MKL_NUM_THREADS',\n"
            "        'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',\n"
            "        'NUMEXPR_NUM_THREADS',\n"
            "    )\n"
            "}\n"
            "pathlib.Path('results.json').write_text(json.dumps({\n"
            "    'metrics': {'f1_macro': 0.5, 'roc_auc_macro': 0.5, 'loss': 1.0},\n"
            "    'thread_vars': vals,\n"
            "}))\n"
        )
        result = executor.run(code, experiment_id="exp_threads")
        assert result.succeeded, f"stderr: {result.stderr}"
        import json, os as host_os
        data = json.loads(result.results_json_path.read_text())  # type: ignore[union-attr]
        vars_ = data["thread_vars"]
        expected = str(host_os.cpu_count() or 1)
        for k in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            assert vars_[k] == expected, (
                f"{k} should equal cpu_count ({expected}), got {vars_[k]!r}"
            )

    def test_relative_sandbox_root_does_not_double_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: if sandbox_root is relative (as used by the CLI),
        the subprocess must not see a duplicated path. Previously we passed
        `str(code_path)` which was re-resolved against the cwd=workdir,
        producing `sandbox/x/sandbox/x/code.py`.

        Reproduce by chdir'ing into a scratch dir and creating an executor
        with a *relative* sandbox_root. The run must succeed.
        """
        monkeypatch.chdir(tmp_path)
        executor = CodeExecutor(
            sandbox_root=Path("sandbox"),  # deliberately relative
            timeout_seconds=10,
            python_executable=sys.executable,
        )
        code = (
            "import json, pathlib\n"
            "pathlib.Path('results.json').write_text("
            "json.dumps({'metrics': {'f1_macro': 0.5, 'roc_auc_macro': 0.5, 'loss': 1.0}}))\n"
            "print('ok')\n"
        )
        result = executor.run(code, experiment_id="exp_rel")
        assert result.succeeded, (
            f"Run failed with stderr:\n{result.stderr}\n"
            f"exit_code={result.exit_code}"
        )
        assert "ok" in result.stdout
        assert result.results_json_path is not None
        assert result.results_json_path.exists()
        # The key signal of the original bug was that the subprocess reported
        # "can't open file '.../sandbox/exp_rel/sandbox/exp_rel/code.py'".
        # With the fix, stderr is empty and exit_code is 0.
        assert result.exit_code == 0
        assert "sandbox/exp_rel/sandbox/exp_rel" not in result.stderr

    def test_stdout_is_streamed_to_logger(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Regression for "stuck on execute_training": the subprocess's
        stdout must be forwarded to the `agent.executor` logger in real
        time, not buffered until the process exits. We can't easily assert
        timing in a unit test, but we can assert the lines DO reach the
        logger by the time .run() returns."""
        executor = CodeExecutor(
            sandbox_root=tmp_path / "sandbox",
            timeout_seconds=10,
            python_executable=sys.executable,
            heartbeat_seconds=0,  # off for deterministic output
        )
        code = (
            "import json, pathlib, sys\n"
            "print('epoch 1: loss=0.9')\n"
            "sys.stdout.flush()\n"
            "print('epoch 2: loss=0.5')\n"
            "sys.stdout.flush()\n"
            "pathlib.Path('results.json').write_text("
            "json.dumps({'metrics': {'f1_macro': 0.5, 'roc_auc_macro': 0.5, 'loss': 1.0}}))\n"
        )
        with caplog.at_level("INFO", logger="agent.executor"):
            result = executor.run(code, experiment_id="exp_stream")
        assert result.succeeded
        # Both progress lines must have made it into the logger
        all_log_text = "\n".join(r.getMessage() for r in caplog.records)
        assert "epoch 1: loss=0.9" in all_log_text
        assert "epoch 2: loss=0.5" in all_log_text

    def test_stderr_is_streamed_as_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """stderr lines from the subprocess should surface as WARNING
        on the agent.executor logger so they stand out in the terminal."""
        executor = CodeExecutor(
            sandbox_root=tmp_path / "sandbox",
            timeout_seconds=10,
            python_executable=sys.executable,
            heartbeat_seconds=0,
        )
        code = (
            "import sys, json, pathlib\n"
            "print('WARN: class 42 has no positives', file=sys.stderr)\n"
            "sys.stderr.flush()\n"
            "pathlib.Path('results.json').write_text("
            "json.dumps({'metrics': {'f1_macro': 0.5, 'roc_auc_macro': 0.5, 'loss': 1.0}}))\n"
        )
        with caplog.at_level("INFO", logger="agent.executor"):
            result = executor.run(code, experiment_id="exp_stderr_stream")
        assert result.succeeded
        warnings = [
            r for r in caplog.records
            if r.name == "agent.executor" and r.levelname == "WARNING"
        ]
        assert any("WARN: class 42" in r.getMessage() for r in warnings)

    def test_stream_output_can_be_disabled(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """With stream_output=False the subprocess output is still captured
        into the result, but nothing is forwarded to the logger."""
        executor = CodeExecutor(
            sandbox_root=tmp_path / "sandbox",
            timeout_seconds=10,
            python_executable=sys.executable,
            stream_output=False,
            heartbeat_seconds=0,
        )
        code = (
            "import json, pathlib\n"
            "print('silent output')\n"
            "pathlib.Path('results.json').write_text("
            "json.dumps({'metrics': {'f1_macro': 0.5, 'roc_auc_macro': 0.5, 'loss': 1.0}}))\n"
        )
        with caplog.at_level("INFO", logger="agent.executor"):
            result = executor.run(code, experiment_id="exp_silent_stream")
        assert result.succeeded
        assert "silent output" in result.stdout  # captured in result
        forwarded = [
            r for r in caplog.records
            if r.name == "agent.executor" and "silent output" in r.getMessage()
        ]
        assert not forwarded  # but NOT forwarded to the logger
