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

import logging
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Callable

from agent.models import TaskError

logger = logging.getLogger(__name__)

# psutil is optional — if it's not installed, the heartbeat simply omits
# the CPU percentage from its message. We do NOT hard-fail on the import,
# because the rest of the executor must work without it.
try:
    import psutil  # type: ignore[import-not-found]

    _HAS_PSUTIL = True
except ImportError:  # pragma: no cover — tested manually
    psutil = None  # type: ignore[assignment]
    _HAS_PSUTIL = False


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
        """A run only counts as successful if it produced a results.json.

        Without this check, the orchestrator could call an experiment
        "completed" even when the generated code silently exited 0 without
        writing anything, and the real failure would only surface later in
        capture_metrics with a misleading "NoResults" message. Requiring
        results.json here gives the LLM a crisp, actionable error at the
        point where it makes sense.
        """
        return (
            self.exit_code == 0
            and self.error is None
            and self.results_json_path is not None
            and self.results_json_path.exists()
        )


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


@dataclass
class CodeExecutor:
    """Writes code to a sandbox dir and runs it as a subprocess.

    Streams stdout/stderr live into log files AND into the agent's logger
    so the operator can watch what the generated training script is doing
    in real time. A heartbeat thread prints a "still running" line every
    few seconds so the operator can distinguish a slow-but-alive process
    from a hung one.
    """

    sandbox_root: Path = Path("sandbox")
    timeout_seconds: int = 600
    python_executable: str = field(default_factory=lambda: sys.executable)
    repo_root: Path = field(default_factory=lambda: Path.cwd())
    stream_output: bool = True
    """If True, live-stream subprocess stdout/stderr into the logger."""
    heartbeat_seconds: float = 10.0
    """Interval for the 'still running...' heartbeat. 0 disables."""
    log_line_prefix: str = "      │ "
    """Prefix for every streamed line (indented under the task marker)."""
    training_env: dict[str, str] = field(default_factory=dict)
    """Extra env vars (typically BIRDCLEF_BATCH_SIZE, BIRDCLEF_NUM_WORKERS, ...)
    injected into every subprocess. Populated by the CLI from the
    `training:` section of config/config.yaml."""

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

        # Line buffers that the reader threads append into. Use plain lists
        # (thread-safe enough for append from a single producer per list).
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []

        start = time.monotonic()
        timed_out = False
        process: subprocess.Popen[str] | None = None

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
            #
            # bufsize=1 + text=True gives line-buffered output so the reader
            # threads see each print() as soon as the child flushes.
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

        assert process is not None
        assert process.stdout is not None
        assert process.stderr is not None

        # Start the live reader threads. Each reader drains one pipe into
        # its corresponding buffer and (optionally) forwards lines to the
        # logger so the operator can watch progress in real time.
        stop_event = threading.Event()
        stdout_reader = threading.Thread(
            target=self._stream_reader,
            args=(process.stdout, stdout_lines, "stdout", stop_event),
            daemon=True,
        )
        stderr_reader = threading.Thread(
            target=self._stream_reader,
            args=(process.stderr, stderr_lines, "stderr", stop_event),
            daemon=True,
        )
        stdout_reader.start()
        stderr_reader.start()

        # Heartbeat thread — ticks every `heartbeat_seconds` and prints how
        # long the subprocess has been running. Disabled if heartbeat_seconds <= 0.
        heartbeat = None
        if self.heartbeat_seconds > 0:
            heartbeat = threading.Thread(
                target=self._heartbeat,
                args=(process, start, experiment_id, stop_event),
                daemon=True,
            )
            heartbeat.start()

        # Wait for the child to finish, honoring our timeout.
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

        # Tell threads to stop and join them. Reader threads exit naturally
        # once the pipes close, but we still want bounded-time joins.
        stop_event.set()
        stdout_reader.join(timeout=5)
        stderr_reader.join(timeout=5)
        if heartbeat is not None:
            heartbeat.join(timeout=2)

        duration = time.monotonic() - start

        stdout_text = "".join(stdout_lines)
        stderr_text = "".join(stderr_lines)

        stdout_path.write_text(stdout_text)
        stderr_path.write_text(stderr_text)

        results_json = workdir / "results.json"
        results_json_path = results_json if results_json.exists() else None

        if exit_code != 0 or timed_out:
            error = classify_error(stderr_text, timed_out=timed_out)
        elif results_json_path is None:
            # Exit 0 but the script never produced a results.json — this is
            # a silent failure. Make it LOUD so the LLM learns to always
            # write the results file, and the orchestrator surfaces a
            # crisp NoResultsFile error instead of a confusing downstream one.
            error = TaskError(
                error_type="NoResultsFile",
                message=(
                    "Script exited with status 0 but did not produce a "
                    "results.json file in the working directory. Make sure "
                    "your code writes `results.json` at the end of training."
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

    # -- streaming helpers --------------------------------------------------

    def _stream_reader(
        self,
        stream: IO[str],
        buffer: list[str],
        label: str,
        stop_event: threading.Event,
    ) -> None:
        """Read lines from `stream` into `buffer` and optionally the logger."""
        try:
            for line in iter(stream.readline, ""):
                buffer.append(line)
                if self.stream_output:
                    # Strip trailing newline for logger, but keep it in buffer
                    text = line.rstrip("\n")
                    if text:
                        if label == "stderr":
                            logger.warning("%s%s", self.log_line_prefix, text)
                        else:
                            logger.info("%s%s", self.log_line_prefix, text)
                if stop_event.is_set():
                    break
        except Exception:  # noqa: BLE001 — reader must not crash
            pass
        finally:
            try:
                stream.close()
            except Exception:  # noqa: BLE001
                pass

    def _heartbeat(
        self,
        process: subprocess.Popen[str],
        start: float,
        experiment_id: str,
        stop_event: threading.Event,
    ) -> None:
        """Print a 'still running' line at a fixed interval.

        When `psutil` is available, also reports SYSTEM-wide CPU utilization.
        This answers the "is my training actually using all the CPUs"
        question without needing a separate htop window. We deliberately
        use system CPU% (not per-process) because:

          1. It is simple and always accurate — no priming, no child
             tracking, no race conditions.
          2. Max is running the agent specifically to train models, so
             if his system CPU is at 10% during execute_training, his
             subprocess is not saturating the cores no matter what the
             internal thread count says.
          3. On macOS, psutil per-process CPU% has known issues with
             multi-threaded processes (it often under-reports).
        """
        interval = max(1.0, self.heartbeat_seconds)
        n_cores = os.cpu_count() or 1
        if _HAS_PSUTIL:
            try:
                n_cores = psutil.cpu_count(logical=True) or n_cores  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001
                pass

        while not stop_event.wait(interval):
            if process.poll() is not None:
                return
            elapsed = time.monotonic() - start
            cpu_info = self._read_cpu_info(n_cores)
            logger.info(
                "%s... still running (%.0fs elapsed, timeout at %ds)%s",
                self.log_line_prefix,
                elapsed,
                self.timeout_seconds,
                cpu_info,
            )

    @staticmethod
    def _read_cpu_info(n_cores: int) -> str:
        """Return a ' | CPU: x.x/N cores (y%)' suffix, or empty string.

        Uses a 0.1s blocking measurement window which is short enough
        to not noticeably delay the heartbeat but long enough to give
        a real reading across all threads / processes on the system.
        """
        if not _HAS_PSUTIL:
            return ""
        try:
            # System-wide CPU% over a 0.1s sample window
            pct = psutil.cpu_percent(interval=0.1)  # type: ignore[union-attr]
            cores_used = pct * n_cores / 100.0
            return f" | CPU: {cores_used:.1f}/{n_cores} cores ({pct:.0f}%)"
        except Exception:  # noqa: BLE001
            return ""

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
        repo_root = self.repo_root.resolve()
        repo_root_str = str(repo_root)
        env["PYTHONPATH"] = (
            f"{repo_root_str}{os.pathsep}{pypath}" if pypath else repo_root_str
        )
        # Disable CUDA by default — BirdCLEF submission must be CPU-only
        env.setdefault("CUDA_VISIBLE_DEVICES", "")

        # Saturate CPU cores during training. Without these, some conda /
        # macOS setups silently default to OMP_NUM_THREADS=1 and PyTorch
        # runs matrix ops on a single thread — the user sees their 10-core
        # machine at ~10% utilization. Setting these ensures PyTorch's
        # intra-op parallelism uses every available core.
        cpu_count = str(os.cpu_count() or 1)
        env.setdefault("OMP_NUM_THREADS", cpu_count)
        env.setdefault("MKL_NUM_THREADS", cpu_count)
        env.setdefault("OPENBLAS_NUM_THREADS", cpu_count)
        env.setdefault("VECLIB_MAXIMUM_THREADS", cpu_count)  # macOS Accelerate
        env.setdefault("NUMEXPR_NUM_THREADS", cpu_count)

        # Expose absolute data paths so the sandboxed code (which runs with a
        # different cwd) can find them. `load_precomputed_dataset` reads these
        # env vars as defaults when its path arguments are not supplied.
        env.setdefault(
            "BIRDCLEF_DATASET_PROFILE",
            str(repo_root / "data" / "processed" / "dataset_profile.json"),
        )
        env.setdefault(
            "BIRDCLEF_SPECTROGRAMS_DIR",
            str(repo_root / "data" / "processed" / "spectrograms"),
        )
        env.setdefault(
            "BIRDCLEF_LABELS_CSV",
            str(repo_root / "data" / "processed" / "labels.csv"),
        )

        # Training knobs forwarded from config/config.yaml. Only set env vars
        # for values that were actually configured — missing entries let
        # `pipelines.data_loader` fall back to its own hardcoded defaults.
        for key, value in self.training_env.items():
            env.setdefault(key, value)

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
