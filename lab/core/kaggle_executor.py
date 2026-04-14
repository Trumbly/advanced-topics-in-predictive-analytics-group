"""Kaggle Kernels executor.

Runs a generated training script as a Kaggle notebook kernel via the
``kaggle`` CLI — no hard dependency on the kaggle Python package. We
only need ``kaggle`` on PATH and valid credentials in
``~/.kaggle/kaggle.json``.

Workflow per experiment
-----------------------
1. Create a per-experiment temp dir under ``sandbox/<exp_id>/``.
2. Write ``kernel-metadata.json`` with the kernel slug, enable_gpu,
   dataset/competition sources, etc.
3. Write ``code.py`` (the LLM-generated script) plus a small
   ``_kaggle_bootstrap.py`` that maps Kaggle input paths and any extra
   env vars onto ``AGENT_*`` before running the real training.
4. Copy the ``lab/`` package into the kernel dir so the generated code's
   ``from lab.tasks.* import …`` imports resolve inside the kernel.
5. ``kaggle kernels push`` to create/update the kernel.
6. Poll ``kaggle kernels status`` every ``poll_interval_seconds`` until
   the kernel leaves ``queued``/``running``.
7. ``kaggle kernels output`` to fetch the kernel output (including
   ``results.json``, ``stdout.log``, ``stderr.log`` if the code wrote
   them).
8. Return an :class:`ExecutionResult` shaped identically to the local
   executor's so the orchestrator never branches.

Failure modes are translated into ``TaskError`` via ``classify_error``
— same taxonomy the local executor uses (OOM, ShapeMismatch, etc.) so
the recovery prompt can react uniformly.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lab.core.executor import ExecutionResult, classify_error
from lab.core.models import TaskError


logger = logging.getLogger("lab.kaggle_executor")


class KaggleCLIUnavailable(RuntimeError):
    """kaggle CLI isn't installed / not on PATH."""


def _require_kaggle_cli() -> str:
    path = shutil.which("kaggle")
    if not path:
        raise KaggleCLIUnavailable(
            "kaggle CLI not found on PATH. Install with `pip install kaggle` "
            "and place credentials in ~/.kaggle/kaggle.json."
        )
    return path


@dataclass
class KaggleExecutor:
    """Run generated code as a Kaggle kernel."""

    username: str = ""
    kernel_prefix: str = "lab-exp"
    enable_gpu: bool = True
    enable_internet: bool = False
    poll_interval_seconds: int = 30
    poll_timeout_seconds: int = 36_000       # 10 h safety net
    dataset_sources: list[str] = field(default_factory=list)
    competition_sources: list[str] = field(default_factory=list)
    sandbox_root: Path = Path("sandbox")
    repo_root: Path = field(default_factory=lambda: Path.cwd())
    training_env: dict[str, str] = field(default_factory=dict)

    backend: str = field(default="kaggle", init=False)

    # ------------------------------------------------------------------
    # Executor protocol
    # ------------------------------------------------------------------

    def run(
        self,
        code: str,
        *,
        experiment_id: str,
        extra_env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        start = time.monotonic()
        workdir = (self.sandbox_root / experiment_id).resolve()
        workdir.mkdir(parents=True, exist_ok=True)

        try:
            cli = _require_kaggle_cli()
        except KaggleCLIUnavailable as exc:
            return self._failed(
                workdir, start,
                TaskError(error_type="SpawnError", message=str(exc)),
            )

        slug = self._kernel_slug(experiment_id)
        kernel_dir = workdir / "kernel"
        kernel_dir.mkdir(parents=True, exist_ok=True)

        # Write kernel source + metadata
        full_env = {**self.training_env, **(extra_env or {})}
        (kernel_dir / "code.py").write_text(
            _KAGGLE_BOOTSTRAP.format(env_json=json.dumps(full_env)) + "\n" + code
        )
        (kernel_dir / "kernel-metadata.json").write_text(json.dumps({
            "id": slug,
            "title": slug.split("/", 1)[-1],
            "code_file": "code.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true" if self.enable_gpu else "false",
            "enable_internet": "true" if self.enable_internet else "false",
            "dataset_sources": list(self.dataset_sources),
            "competition_sources": list(self.competition_sources),
            "kernel_sources": [],
        }, indent=2))

        # Bundle the lab package so `from lab.tasks.* import ...` resolves
        # inside the kernel. We copy the source tree verbatim — lab is
        # small enough that this stays well under the kernel-source limit.
        lab_src = self.repo_root / "lab"
        if lab_src.exists():
            dst = kernel_dir / "lab"
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(lab_src, dst, ignore=shutil.ignore_patterns("__pycache__", "ui"))

        # Push
        logger.info("Pushing kernel %s from %s", slug, kernel_dir)
        push = self._run_kaggle(
            cli, ["kernels", "push", "-p", str(kernel_dir)], timeout=300,
        )
        if push.returncode != 0:
            return self._failed(
                workdir, start,
                TaskError(
                    error_type="SpawnError",
                    message=f"kaggle kernels push failed: {push.stderr.strip()[:200]}",
                    traceback=push.stderr.strip()[-600:],
                ),
            )

        # Poll
        status, logs = self._poll_until_done(cli, slug)
        # Stream logs to workdir for UI consistency
        (workdir / "stdout.log").write_text(logs or "")
        (workdir / "stderr.log").write_text("")

        # Fetch output (results.json etc.)
        out = self._run_kaggle(
            cli, ["kernels", "output", slug, "-p", str(workdir)], timeout=300,
        )
        if out.returncode != 0:
            logger.warning("kaggle kernels output failed: %s", out.stderr.strip()[:200])

        results_json = workdir / "results.json"
        results_path = results_json if results_json.exists() else None

        duration = time.monotonic() - start

        if status == "complete" and results_path:
            return ExecutionResult(
                exit_code=0,
                stdout=logs,
                stderr="",
                duration_seconds=duration,
                workdir=workdir,
                results_json_path=results_path,
                error=None,
                timed_out=False,
            )

        # Error path — let the shared classifier read the kernel log.
        timed_out = status == "timeout"
        error = classify_error(logs, timed_out=timed_out) or TaskError(
            error_type="UnknownError",
            message=f"kernel status={status} and no results.json",
            traceback=(logs or "")[-600:],
        )
        return ExecutionResult(
            exit_code=-1,
            stdout=logs,
            stderr="",
            duration_seconds=duration,
            workdir=workdir,
            results_json_path=results_path,
            error=error,
            timed_out=timed_out,
        )

    def infrastructure(self) -> dict[str, Any]:
        return {
            "backend": "kaggle",
            "username": self.username or "<unset>",
            "enable_gpu": self.enable_gpu,
            "enable_internet": self.enable_internet,
            "dataset_sources": list(self.dataset_sources),
            "competition_sources": list(self.competition_sources),
            "kernel_prefix": self.kernel_prefix,
        }

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _kernel_slug(self, experiment_id: str) -> str:
        # Kaggle slugs: [a-z0-9-] only, lowercased. Prefix with username/.
        safe_exp = experiment_id.lower().replace("_", "-")
        name = f"{self.kernel_prefix}-{safe_exp}".strip("-")
        return f"{self.username}/{name}" if self.username else name

    def _run_kaggle(self, cli: str, argv: list[str], *, timeout: int) -> subprocess.CompletedProcess:
        return subprocess.run(
            [cli, *argv], capture_output=True, text=True, timeout=timeout,
        )

    def _poll_until_done(self, cli: str, slug: str) -> tuple[str, str]:
        """Return (final_status, log_text). ``final_status`` is one of
        'complete', 'error', 'cancelled', 'timeout'."""
        deadline = time.monotonic() + self.poll_timeout_seconds
        last_status = "unknown"
        while time.monotonic() < deadline:
            status_proc = self._run_kaggle(
                cli, ["kernels", "status", slug], timeout=60,
            )
            raw = (status_proc.stdout + "\n" + status_proc.stderr).lower()
            if "has status \"complete\"" in raw or "status: complete" in raw:
                last_status = "complete"
                break
            if "has status \"error\"" in raw or "status: error" in raw:
                last_status = "error"
                break
            if "has status \"cancelled\"" in raw or "status: cancelled" in raw:
                last_status = "cancelled"
                break
            last_status = "running"
            logger.info("Kaggle kernel %s status=running (sleeping %ds)",
                        slug, self.poll_interval_seconds)
            time.sleep(self.poll_interval_seconds)
        else:
            last_status = "timeout"

        # Fetch run log
        log_proc = self._run_kaggle(
            cli, ["kernels", "output", slug, "-w", "-p", "-"], timeout=120,
        )
        log_text = log_proc.stdout or ""
        return last_status, log_text

    def _failed(self, workdir: Path, start: float, err: TaskError) -> ExecutionResult:
        return ExecutionResult(
            exit_code=-1,
            stdout="",
            stderr=err.message,
            duration_seconds=time.monotonic() - start,
            workdir=workdir,
            results_json_path=None,
            error=err,
            timed_out=False,
        )


# Prepended to the generated code inside the kernel. Sets up AGENT_* env
# vars from a serialised dict + remaps Kaggle input paths if the task
# adapter configured dataset_sources.
_KAGGLE_BOOTSTRAP = '''\
# --- lab Kaggle bootstrap (auto-generated, do not edit) ---------------
import json as _json
import os as _os
_env = _json.loads(r"""{env_json}""")
for _k, _v in _env.items():
    _os.environ.setdefault(_k, str(_v))
# Map the first /kaggle/input/* subdir onto AGENT_PROCESSED_DIR when the
# env var is not already set — lets vanilla scripts find Kaggle data.
_input = "/kaggle/input"
if _os.path.isdir(_input) and "AGENT_PROCESSED_DIR" not in _os.environ:
    _subdirs = [p for p in _os.listdir(_input) if _os.path.isdir(_os.path.join(_input, p))]
    if _subdirs:
        _os.environ["AGENT_PROCESSED_DIR"] = _os.path.join(_input, _subdirs[0])
# --- end bootstrap ---------------------------------------------------
'''


__all__ = ["KaggleExecutor", "KaggleCLIUnavailable"]
