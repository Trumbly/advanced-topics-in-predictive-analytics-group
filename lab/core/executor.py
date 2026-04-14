"""Executor abstraction + local subprocess sandbox.

Two concrete implementations live in this package:

  * ``LocalExecutor`` (this file) runs the generated training script as a
    detached subprocess on the same machine. Keeps process-group kill,
    error classification, and live stdout streaming from max_development.

  * ``KaggleExecutor`` (``lab.core.kaggle_executor``) pushes the script
    as a Kaggle kernel, polls until done, and fetches the output.

Both implement the lightweight ``Executor`` protocol below, and return
the same ``ExecutionResult``, so the orchestrator never has to branch.
"""
from __future__ import annotations

import logging
import os
import platform
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Protocol

from lab.core.models import TaskError


logger = logging.getLogger("lab.executor")


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------


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


def _tail(text: str, n: int) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-n:])


def _extract_error_line(stderr: str, patterns: tuple[str, ...]) -> str:
    lines = [line for line in stderr.splitlines() if line.strip()]
    for line in lines:
        if any(p in line for p in patterns):
            return line.strip()
    return lines[-1].strip() if lines else ""


def classify_error(stderr: str, *, timed_out: bool) -> TaskError | None:
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
    return TaskError(
        error_type="UnknownError",
        message=_extract_error_line(stderr, ()) or "Unknown error",
        traceback=_tail(stderr, 30),
    )


# ---------------------------------------------------------------------------
# Execution result
# ---------------------------------------------------------------------------


@dataclass
class ExecutionResult:
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
        return (
            self.exit_code == 0
            and self.error is None
            and self.results_json_path is not None
            and self.results_json_path.exists()
        )


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class Executor(Protocol):
    """The interface every backend implements.

    All methods return a fully-populated ``ExecutionResult`` — failures
    are captured in the result rather than raised, so the orchestrator
    can classify them uniformly and feed the error into the recovery
    prompt.
    """

    backend: str  # e.g. "local" | "kaggle"

    def run(
        self, code: str, *, experiment_id: str,
        extra_env: dict[str, str] | None = None,
    ) -> ExecutionResult: ...

    def infrastructure(self) -> dict[str, Any]:
        """Return a small JSON-serialisable dict describing where the
        code will actually run. Recorded once per study so the UI can
        show "ran on local CPU" vs "ran on Kaggle T4" forever after."""


@dataclass
class LocalExecutor:
    """Write a training script to a sandbox dir and run it as a subprocess."""

    sandbox_root: Path = Path("sandbox")
    timeout_seconds: int = 1800
    python_executable: str = field(default_factory=lambda: sys.executable)
    repo_root: Path = field(default_factory=lambda: Path.cwd())
    stream_output: bool = True
    log_line_prefix: str = "      │ "
    training_env: dict[str, str] = field(default_factory=dict)

    backend: str = field(default="local", init=False)

    def run(
        self,
        code: str,
        *,
        experiment_id: str,
        extra_env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        workdir = self._prepare_workdir(experiment_id).resolve()
        code_path = workdir / "code.py"
        code_path.write_text(code)

        env = self._build_env(extra_env)

        stdout_path = workdir / "stdout.log"
        stderr_path = workdir / "stderr.log"
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []

        start = time.monotonic()
        timed_out = False
        process: subprocess.Popen[str] | None = None

        try:
            popen_kwargs: dict[str, object] = {
                "cwd": str(workdir),
                "env": env,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "text": True,
                "bufsize": 1,
            }
            if os.name != "nt":
                popen_kwargs["preexec_fn"] = os.setsid  # type: ignore[assignment]
            process = subprocess.Popen(
                [self.python_executable, "-u", "code.py"],
                **popen_kwargs,  # type: ignore[arg-type]
            )
        except Exception as exc:  # noqa: BLE001
            duration = time.monotonic() - start
            return ExecutionResult(
                exit_code=-1,
                stdout="",
                stderr=str(exc),
                duration_seconds=duration,
                workdir=workdir,
                results_json_path=None,
                error=TaskError(error_type="SpawnError", message=f"Failed to start subprocess: {exc}"),
                timed_out=False,
            )

        assert process is not None and process.stdout is not None and process.stderr is not None

        stdout_live_fh = stdout_path.open("w")
        stderr_live_fh = stderr_path.open("w")

        stop_event = threading.Event()
        stdout_reader = threading.Thread(
            target=self._stream_reader,
            args=(process.stdout, stdout_lines, "stdout", stop_event, stdout_live_fh),
            daemon=True,
        )
        stderr_reader = threading.Thread(
            target=self._stream_reader,
            args=(process.stderr, stderr_lines, "stderr", stop_event, stderr_live_fh),
            daemon=True,
        )
        stdout_reader.start()
        stderr_reader.start()

        try:
            process.wait(timeout=self.timeout_seconds)
            exit_code = process.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            self._kill_process_tree(process)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            exit_code = -1

        stop_event.set()
        stdout_reader.join(timeout=5)
        stderr_reader.join(timeout=5)
        for fh in (stdout_live_fh, stderr_live_fh):
            try:
                fh.close()
            except Exception:  # noqa: BLE001
                pass

        duration = time.monotonic() - start
        stdout_text = "".join(stdout_lines)
        stderr_text = "".join(stderr_lines)

        results_json = workdir / "results.json"
        results_json_path = results_json if results_json.exists() else None

        if exit_code != 0 or timed_out:
            error = classify_error(stderr_text, timed_out=timed_out)
        elif results_json_path is None:
            error = TaskError(
                error_type="NoResultsFile",
                message=(
                    "Script exited 0 but did not produce a results.json. Make "
                    "sure your code writes `results.json` at the end of training."
                ),
                traceback=_tail(stdout_text, 10) or None,
            )
        else:
            error = None

        return ExecutionResult(
            exit_code=exit_code,
            stdout=stdout_text,
            stderr=stderr_text,
            duration_seconds=duration,
            workdir=workdir,
            results_json_path=results_json_path,
            error=error,
            timed_out=timed_out,
        )

    # -- streaming ----------------------------------------------------------

    def _stream_reader(
        self,
        stream: IO[str],
        buffer: list[str],
        label: str,
        stop_event: threading.Event,
        live_file: IO[str] | None,
    ) -> None:
        try:
            for line in iter(stream.readline, ""):
                buffer.append(line)
                if live_file is not None:
                    try:
                        live_file.write(line)
                        live_file.flush()
                    except Exception:  # noqa: BLE001
                        pass
                if self.stream_output:
                    text = line.rstrip("\n")
                    if text:
                        (logger.warning if label == "stderr" else logger.info)(
                            "%s%s", self.log_line_prefix, text
                        )
                if stop_event.is_set():
                    break
        except Exception:  # noqa: BLE001
            pass
        finally:
            try:
                stream.close()
            except Exception:  # noqa: BLE001
                pass

    # -- helpers ------------------------------------------------------------

    def _prepare_workdir(self, experiment_id: str) -> Path:
        workdir = self.sandbox_root / experiment_id
        workdir.mkdir(parents=True, exist_ok=True)
        for name in ("results.json", "stdout.log", "stderr.log", "code.py"):
            stale = workdir / name
            if stale.exists():
                stale.unlink()
        return workdir

    def _build_env(self, extra: dict[str, str] | None) -> dict[str, str]:
        env = os.environ.copy()
        pypath = env.get("PYTHONPATH", "")
        repo_root = self.repo_root.resolve()
        repo_root_str = str(repo_root)
        env["PYTHONPATH"] = f"{repo_root_str}{os.pathsep}{pypath}" if pypath else repo_root_str
        env.setdefault("CUDA_VISIBLE_DEVICES", "")
        cpu_count = str(os.cpu_count() or 1)
        env.setdefault("OMP_NUM_THREADS", cpu_count)
        env.setdefault("MKL_NUM_THREADS", cpu_count)
        env.setdefault("OPENBLAS_NUM_THREADS", cpu_count)
        env.setdefault("NUMEXPR_NUM_THREADS", cpu_count)
        for k, v in self.training_env.items():
            env.setdefault(k, v)
        if extra:
            env.update(extra)
        return env

    @staticmethod
    def _kill_process_tree(process: subprocess.Popen[str]) -> None:
        if os.name == "nt":
            process.kill()
            return
        try:
            pgid = os.getpgid(process.pid)
            os.killpg(pgid, signal.SIGTERM)
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def infrastructure(self) -> dict[str, Any]:
        """Describe the local box the generated code will run on."""
        device = self.training_env.get(f"{_env_prefix()}_DEVICE", "cpu")
        return {
            "backend": "local",
            "device": device,
            "cpu_count": os.cpu_count() or 0,
            "python_version": ".".join(map(str, sys.version_info[:3])),
            "platform": platform.platform(terse=True),
            "hostname": platform.node(),
        }


def _env_prefix() -> str:
    # Cheap helper — avoids importing Settings here just to read one str.
    return os.environ.get("AGENT_ENV_PREFIX", "AGENT")


# Backward-compat alias. External tools may still import ``CodeExecutor``.
CodeExecutor = LocalExecutor


__all__ = [
    "Executor", "LocalExecutor", "CodeExecutor",
    "ExecutionResult", "classify_error",
]
