"""Sandboxed code executor for LLM-generated training scripts.

The CodeExecutor writes generated Python code to a per-experiment workdir
under `sandbox/`, runs it in a subprocess with a timeout, captures
stdout/stderr, and classifies any error it encounters into one of the
known categories (so the LLM can react differently to different failure
modes).

Design
------
- Each run gets its own directory: `sandbox/<experiment_id>/` containing
  `code.py`, `stdout.log`, `stderr.log`, and (on success) `results.json`.
- The subprocess runs with `cwd = repo_root` so the generated code can
  import `pipelines.data_loader` etc. We inject the repo root into
  `PYTHONPATH` to be robust.
- A timeout kills the whole process group — not just the subprocess —
  so child processes (e.g. torch DataLoader workers) are also reaped.
- Error classification is done by scanning stderr for well-known substrings.
  This is deliberately simple: the LLM does not need pixel-perfect error
  types, it just needs to know "OOM" vs "SyntaxError" vs "Timeout" vs
  "ShapeMismatch" vs "Other".

Security
--------
This is NOT a hardened sandbox. We run LLM-generated code as a normal
subprocess with the same user permissions as the agent. We rely on the
`validate_code` handler (Phase 3, below) to reject imports of `subprocess`,
`os.system`, `urllib.request`, etc. before the code ever reaches the
executor. A proper sandbox (containers, seccomp) is out of scope for a
course project but is noted as future work in the plan.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from agent.models import TaskError


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------


# Patterns used to classify a failed run. First match wins — order matters.
# Each entry is (error_type, list of substrings to look for in stderr).
ERROR_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("OOM", ("MemoryError", "OutOfMemoryError", "CUDA out of memory", "Killed")),
    ("SyntaxError", ("SyntaxError", "IndentationError", "TabError")),
    (
        "ShapeMismatch",
        (
            "size mismatch",
            "shape mismatch",
            "Expected input batch_size",
            "Expected size",
            "RuntimeError: mat1 and mat2 shapes cannot be multiplied",
            "RuntimeError: Given groups",
            "RuntimeError: expected input",
        ),
    ),
    ("ImportError", ("ModuleNotFoundError", "ImportError")),
    ("FileNotFound", ("FileNotFoundError",)),
    ("ValueError", ("ValueError",)),
    ("RuntimeError", ("RuntimeError",)),
]


def classify_error(stderr: str, *, timed_out: bool) -> TaskError | None:
    """Return a TaskError describing the failure, or None if there is none."""
    if timed_out:
        return TaskError(
            error_type="Timeout",
            message="Process exceeded configured timeout",
            traceback=_tail(stderr, 20),
        )
    if not stderr.strip():
        return None
    for error_type, patterns in ERROR_PATTERNS:
        if any(p in stderr for p in patterns):
            return TaskError(
                error_type=error_type,
                message=_extract_error_line(stderr, patterns),
                traceback=_tail(stderr, 30),
            )
    # Unknown error — return a catch-all
    return TaskError(
        error_type="UnknownError",
        message=_extract_error_line(stderr, ()) or "Unknown error",
        traceback=_tail(stderr, 30),
    )


def _extract_error_line(stderr: str, patterns: tuple[str, ...]) -> str:
    """Return the first line containing one of the patterns, or the last line."""
    lines = [line for line in stderr.splitlines() if line.strip()]
    for line in lines:
        if any(p in line for p in patterns):
            return line.strip()
    return lines[-1].strip() if lines else ""


def _tail(text: str, n: int) -> str:
    """Return the last n lines of `text`."""
    lines = text.splitlines()
    return "\n".join(lines[-n:])


# ---------------------------------------------------------------------------
# Execution result
# ---------------------------------------------------------------------------


@dataclass
class ExecutionResult:
    """Structured outcome of a subprocess run."""

    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    workdir: Path
    results_json_path: Path | None
    error: TaskError | None
    timed_out: bool = False

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0 and self.error is None


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


@dataclass
class CodeExecutor:
    """Writes code to a sandbox dir and runs it as a subprocess."""

    sandbox_root: Path = Path("sandbox")
    timeout_seconds: int = 600
    python_executable: str = field(default_factory=lambda: sys.executable)
    repo_root: Path = field(default_factory=lambda: Path.cwd())

    def run(
        self,
        code: str,
        *,
        experiment_id: str,
        extra_env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        """Write `code` to disk and execute it in an isolated directory.

        Returns an `ExecutionResult` describing what happened. Never raises
        for subprocess failures — every failure is captured in the result.
        """
        workdir = self._prepare_workdir(experiment_id).resolve()
        code_path = workdir / "code.py"
        code_path.write_text(code)

        env = self._build_env(extra_env)

        stdout_path = workdir / "stdout.log"
        stderr_path = workdir / "stderr.log"

        start = time.monotonic()
        timed_out = False
        try:
            # Run in its own process group so we can kill the whole tree on timeout.
            # On Windows, `preexec_fn=os.setsid` is not available — fall back to
            # default behavior there (the project mainly targets macOS/Linux).
            #
            # IMPORTANT: we pass just "code.py" (not the joined path) because
            # cwd is already set to workdir. Passing a joined relative path
            # would be re-resolved relative to cwd by the subprocess,
            # producing the classic duplicated-path bug:
            #   sandbox/study_x/exp_1/sandbox/study_x/exp_1/code.py
            popen_kwargs: dict[str, object] = {
                "cwd": str(workdir),
                "env": env,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "text": True,
            }
            if os.name != "nt":
                popen_kwargs["preexec_fn"] = os.setsid  # type: ignore[assignment]

            process = subprocess.Popen(
                [self.python_executable, "code.py"],
                **popen_kwargs,  # type: ignore[arg-type]
            )
            try:
                stdout, stderr = process.communicate(timeout=self.timeout_seconds)
                exit_code = process.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
                self._kill_process_tree(process)
                stdout, stderr = process.communicate()
                exit_code = -1
        except Exception as exc:  # noqa: BLE001 — any spawn error is captured
            duration = time.monotonic() - start
            err = TaskError(
                error_type="SpawnError",
                message=f"Failed to start subprocess: {exc}",
            )
            return ExecutionResult(
                exit_code=-1,
                stdout="",
                stderr=str(exc),
                duration_seconds=duration,
                workdir=workdir,
                results_json_path=None,
                error=err,
                timed_out=False,
            )

        duration = time.monotonic() - start

        stdout_path.write_text(stdout or "")
        stderr_path.write_text(stderr or "")

        results_json = workdir / "results.json"
        results_json_path = results_json if results_json.exists() else None

        error = (
            classify_error(stderr or "", timed_out=timed_out)
            if (exit_code != 0 or timed_out)
            else None
        )

        return ExecutionResult(
            exit_code=exit_code,
            stdout=stdout or "",
            stderr=stderr or "",
            duration_seconds=duration,
            workdir=workdir,
            results_json_path=results_json_path,
            error=error,
            timed_out=timed_out,
        )

    # -- helpers ------------------------------------------------------------

    def _prepare_workdir(self, experiment_id: str) -> Path:
        workdir = self.sandbox_root / experiment_id
        workdir.mkdir(parents=True, exist_ok=True)
        # Clean previous artifacts so a rerun doesn't see stale files
        for name in ("results.json", "stdout.log", "stderr.log", "code.py"):
            stale = workdir / name
            if stale.exists():
                stale.unlink()
        return workdir

    def _build_env(self, extra: dict[str, str] | None) -> dict[str, str]:
        env = os.environ.copy()
        # Make sure the generated code can import the repo's packages
        pypath = env.get("PYTHONPATH", "")
        repo_root_str = str(self.repo_root.resolve())
        env["PYTHONPATH"] = (
            f"{repo_root_str}{os.pathsep}{pypath}" if pypath else repo_root_str
        )
        # Disable CUDA by default — BirdCLEF submission must be CPU-only
        env.setdefault("CUDA_VISIBLE_DEVICES", "")
        if extra:
            env.update(extra)
        return env

    @staticmethod
    def _kill_process_tree(process: subprocess.Popen[str]) -> None:
        """Kill the subprocess and all its children."""
        if os.name == "nt":
            process.kill()
            return
        try:
            pgid = os.getpgid(process.pid)
            os.killpg(pgid, signal.SIGTERM)
            # Give it a moment to shut down, then SIGKILL if still alive
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass  # Already dead


__all__ = [
    "CodeExecutor",
    "ExecutionResult",
    "classify_error",
]
