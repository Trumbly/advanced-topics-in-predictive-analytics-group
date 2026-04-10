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
pathlib.Path('results.json').write_text(json.dumps({'metrics': {'roc_auc_macro': 0.5, 'loss': 1.0}}))
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
            "json.dumps({'metrics': {'roc_auc_macro': 0.5, 'loss': 1.0}}))\n"
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
