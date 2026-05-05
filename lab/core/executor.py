"""Subprocess sandbox executor (ADR-010).

Runs generated code under ``sandbox/{experiment_id}/`` with env injection and a
hard wallclock timeout. Reads ``results.json`` written by the training script;
falls back to stderr regex classification when execution failed before the
script could write its result file.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

from lab.config import Settings
from lab.core.models import ErrorType, ExecutionResult, TaskError

_ERROR_TYPES: frozenset[str] = frozenset(
    {
        "OOM",
        "Timeout",
        "ShapeMismatch",
        "ValueError",
        "RuntimeError",
        "FileNotFound",
        "ImportError",
        "Other",
    }
)

_HARD_FAILURES: frozenset[str] = frozenset({"Timeout", "OOM", "FileNotFound"})

# Order matters: most specific first.
_STDERR_PATTERNS: tuple[tuple[ErrorType, re.Pattern[str]], ...] = (
    ("OOM", re.compile(r"(out of memory|CUDA out of memory|MemoryError)", re.I)),
    (
        "ShapeMismatch",
        re.compile(
            r"(size mismatch|shape '\[.*\]' is invalid|RuntimeError:.*expected.*got|"
            r"mat1 and mat2 shapes cannot be multiplied|Given.*expected.*to have)",
            re.I,
        ),
    ),
    ("FileNotFound", re.compile(r"FileNotFoundError|No such file or directory", re.I)),
    ("ImportError", re.compile(r"(ImportError|ModuleNotFoundError)", re.I)),
    ("ValueError", re.compile(r"\bValueError\b")),
    ("RuntimeError", re.compile(r"\bRuntimeError\b")),
)

_REQUIRED_RESULT_KEYS: tuple[str, ...] = ("primary_score", "primary_metric")


class LocalExecutor:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.sandbox_root = Path(settings.paths.sandbox)

    def run(
        self,
        code: str,
        *,
        experiment_id: str,
        extra_env: dict[str, str],
        timeout_s: int,
    ) -> ExecutionResult:
        sandbox_dir = self.sandbox_root / experiment_id
        sandbox_dir.mkdir(parents=True, exist_ok=True)

        code_path = sandbox_dir / "code.py"
        code_path.write_text(code, encoding="utf-8")
        results_path = sandbox_dir / "results.json"
        stdout_path = sandbox_dir / "stdout.log"
        stderr_path = sandbox_dir / "stderr.log"

        env = dict(os.environ)
        env.update(extra_env)
        env.setdefault("PYTHONUNBUFFERED", "1")

        t0 = time.monotonic()
        timed_out = False
        with stdout_path.open("wb") as out_f, stderr_path.open("wb") as err_f:
            process = subprocess.Popen(
                [sys.executable, "code.py"],
                cwd=str(sandbox_dir),
                env=env,
                stdout=out_f,
                stderr=err_f,
                start_new_session=True,
            )
            try:
                returncode = process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                _kill_process_group(process)
                returncode = process.wait()
        duration = time.monotonic() - t0

        stdout = stdout_path.read_text(errors="replace")
        stderr = stderr_path.read_text(errors="replace")

        if timed_out:
            return _failed(
                duration,
                stdout,
                stderr,
                error=TaskError(
                    error_type="Timeout",
                    message=f"process exceeded timeout of {timeout_s}s",
                ),
            )

        if results_path.exists():
            try:
                results = json.loads(results_path.read_text())
            except json.JSONDecodeError as exc:
                return _failed(
                    duration,
                    stdout,
                    stderr,
                    error=TaskError(
                        error_type="Other",
                        message=f"results.json is not valid JSON: {exc}",
                    ),
                )
            schema_err = _validate_results_schema(results)
            if schema_err is not None:
                return _failed(duration, stdout, stderr, error=schema_err)

            if returncode != 0:
                return ExecutionResult(
                    succeeded=False,
                    primary_score=results.get("primary_score"),
                    metrics=results.get("metrics", {}) or {},
                    history=results.get("history", []) or [],
                    duration_seconds=duration,
                    stdout=stdout,
                    stderr=stderr,
                    error=_classify_stderr(stderr)
                    or TaskError(
                        error_type="Other",
                        message=f"non-zero exit ({returncode}) despite results.json",
                    ),
                )

            return ExecutionResult(
                succeeded=True,
                primary_score=float(results["primary_score"]),
                metrics={k: float(v) for k, v in (results.get("metrics") or {}).items()},
                history=list(results.get("history") or []),
                duration_seconds=duration,
                stdout=stdout,
                stderr=stderr,
                error=None,
            )

        # No results.json: classify stderr.
        return _failed(duration, stdout, stderr, error=_classify_stderr(stderr))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _failed(
    duration: float, stdout: str, stderr: str, *, error: TaskError
) -> ExecutionResult:
    return ExecutionResult(
        succeeded=False,
        primary_score=None,
        metrics={},
        history=[],
        duration_seconds=duration,
        stdout=stdout,
        stderr=stderr,
        error=error,
    )


def _validate_results_schema(results: object) -> TaskError | None:
    if not isinstance(results, dict):
        return TaskError(
            error_type="Other",
            message=f"results.json must be a JSON object, got {type(results).__name__}",
        )
    missing = [k for k in _REQUIRED_RESULT_KEYS if k not in results]
    if missing:
        return TaskError(
            error_type="Other",
            message=f"results.json missing required keys: {missing}",
        )
    return None


def _classify_stderr(stderr: str) -> TaskError:
    if not stderr.strip():
        return TaskError(error_type="Other", message="empty stderr; cause unknown")
    for kind, pattern in _STDERR_PATTERNS:
        m = pattern.search(stderr)
        if m:
            return TaskError(
                error_type=kind,
                message=_excerpt(stderr, m.start(), m.end()),
                traceback=stderr[-2000:],
            )
    return TaskError(error_type="Other", message=stderr.splitlines()[-1][:500])


def _excerpt(text: str, start: int, end: int, *, window: int = 200) -> str:
    s = max(0, start - window)
    e = min(len(text), end + window)
    return text[s:e].strip().splitlines()[-1][:500] if text[s:e].strip() else text[s:e][:500]


def _kill_process_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        time.sleep(0.5)
        if process.poll() is None:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):  # pragma: no cover
        pass
