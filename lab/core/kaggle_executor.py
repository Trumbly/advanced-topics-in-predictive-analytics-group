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

import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
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
    # Flipped to True by kill_running() so the poll loop returns early
    # instead of waiting out a full poll_timeout_seconds.
    _aborted: bool = field(default=False, init=False, repr=False)

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

        # Ship the lab/ source tree as a Kaggle dataset the first time
        # we push from this machine, then attach it to every kernel so
        # the generated code's `from lab.tasks.* import …` resolves.
        # Failures here are non-fatal — the kernel will surface a clear
        # ImportError which the recovery prompt can react to.
        lab_slug = self._ensure_lab_dataset(cli)

        # Write kernel source + metadata
        full_env = {**self.training_env, **(extra_env or {})}
        (kernel_dir / "code.py").write_text(
            _wrap_with_bootstrap(code, full_env, lab_dataset_slug=lab_slug)
        )
        dataset_sources = list(self.dataset_sources)
        if lab_slug:
            dataset_sources.append(lab_slug)

        (kernel_dir / "kernel-metadata.json").write_text(json.dumps({
            "id": slug,
            "title": slug.split("/", 1)[-1],
            "code_file": "code.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true" if self.enable_gpu else "false",
            "enable_internet": "true" if self.enable_internet else "false",
            "dataset_sources": dataset_sources,
            "competition_sources": list(self.competition_sources),
            "kernel_sources": [],
        }, indent=2))

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
        if status == "aborted":
            error = TaskError(error_type="Aborted", message="User requested abort")
        else:
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

    def kill_running(self) -> bool:
        """Cooperative abort: the poll loop checks this flag each tick.

        We don't cancel the Kaggle kernel server-side — the CLI doesn't
        expose a cancel verb that's universally available, and the
        kernel will time out or complete on its own regardless. The
        important thing is that the orchestrator stops *waiting* on it
        so the agent can shut down.
        """
        self._aborted = True
        return True

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
        # Avoid ``lab-exp-exp-…`` if the prefix already contains the
        # token ``exp`` and the id starts with it too. Just strip the
        # leading 'exp-' from the id in that case.
        prefix = self.kernel_prefix.rstrip("-")
        if prefix.endswith("exp") and safe_exp.startswith("exp-"):
            safe_exp = safe_exp[len("exp-"):]
        name = f"{prefix}-{safe_exp}".strip("-")
        return f"{self.username}/{name}" if self.username else name

    def _run_kaggle(self, cli: str, argv: list[str], *, timeout: int) -> subprocess.CompletedProcess:
        return subprocess.run(
            [cli, *argv], capture_output=True, text=True, timeout=timeout,
        )

    def _poll_until_done(self, cli: str, slug: str) -> tuple[str, str]:
        """Return (final_status, log_text). ``final_status`` is one of
        'complete', 'error', 'cancelled', 'timeout', 'aborted'."""
        deadline = time.monotonic() + self.poll_timeout_seconds
        last_status = "unknown"
        while time.monotonic() < deadline:
            if self._aborted:
                last_status = "aborted"
                break
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
            # Sleep in small slices so kill_running() cuts in fast.
            end_sleep = time.monotonic() + self.poll_interval_seconds
            while time.monotonic() < end_sleep:
                if self._aborted:
                    break
                time.sleep(min(0.5, end_sleep - time.monotonic()))
        else:
            last_status = "timeout"

        # Fetch run log
        log_proc = self._run_kaggle(
            cli, ["kernels", "output", slug, "-w", "-p", "-"], timeout=120,
        )
        log_text = log_proc.stdout or ""
        return last_status, log_text

    def _ensure_lab_dataset(self, cli: str) -> str | None:
        """Ensure a ``<username>/lab-agent-src`` dataset exists with the
        current ``lab/`` source, and return its slug.

        The first call on a fresh machine creates the dataset; subsequent
        calls skip via a content-hash cache unless ``lab/`` changed. When
        the content has changed we push a new dataset version so running
        kernels always see the matching source.

        Returns None when: no username is configured, no local ``lab/``
        dir is present, or the upload failed. Callers treat None as
        "don't attach, let the kernel ImportError surface itself".
        """
        slug, err = self.ensure_lab_dataset_verbose(cli)
        if err:
            logger.warning("lab-src upload failed: %s", err)
        return slug

    def ensure_lab_dataset_verbose(self, cli: str | None = None) -> tuple[str | None, str | None]:
        """Same contract as ``_ensure_lab_dataset`` but also returns a
        human-readable error string so the CLI can surface it.

        Returns (slug, error). On success: (slug, None). On skip for a
        clean reason: (None, None). On failure: (None, error_string).
        """
        if not self.username:
            return None, "no kaggle username configured (executor.kaggle.username)"
        lab_src = self.repo_root / "lab"
        if not lab_src.exists():
            return None, f"no local lab/ directory at {lab_src}"
        if cli is None:
            try:
                cli = _require_kaggle_cli()
            except KaggleCLIUnavailable as exc:
                return None, str(exc)

        slug = f"{self.username}/lab-agent-src"
        digest = _hash_directory(lab_src)
        cache_file = self.sandbox_root / ".lab_dataset_cache"
        if cache_file.exists():
            try:
                cached = json.loads(cache_file.read_text())
                if cached.get("slug") == slug and cached.get("hash") == digest:
                    return slug, None
            except json.JSONDecodeError:
                pass

        logger.info("Uploading lab/ as Kaggle dataset %s (hash %s)", slug, digest[:8])

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            shutil.copytree(
                lab_src, tmp_path / "lab",
                ignore=shutil.ignore_patterns("__pycache__", "ui", "*.pyc"),
            )
            (tmp_path / "dataset-metadata.json").write_text(json.dumps({
                "title": "lab-agent-src",
                "id": slug,
                "licenses": [{"name": "CC0-1.0"}],
            }, indent=2))

            create = self._run_kaggle(
                cli, ["datasets", "create", "-p", str(tmp_path)], timeout=600,
            )
            if create.returncode == 0:
                logger.info("lab-src dataset created: %s", slug)
            else:
                # Likely already exists — push a new version.
                msg = f"lab src {digest[:8]}"
                version = self._run_kaggle(
                    cli,
                    ["datasets", "version", "-p", str(tmp_path), "-m", msg],
                    timeout=600,
                )
                if version.returncode != 0:
                    combined = (
                        f"`datasets create` exit {create.returncode}: "
                        f"{(create.stderr or create.stdout or '').strip()[:300]}\n"
                        f"`datasets version` exit {version.returncode}: "
                        f"{(version.stderr or version.stdout or '').strip()[:300]}"
                    )
                    return None, combined
                logger.info("lab-src dataset versioned: %s", slug)

        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps({"slug": slug, "hash": digest}))
        except OSError:
            pass
        return slug, None

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


# Prepended to the generated code inside the kernel. Sets up:
#   - AGENT_* env vars from a serialised dict
#   - /kaggle/input/<lab-agent-src>/  on sys.path so `from lab.tasks.*` works
#   - AGENT_PROCESSED_DIR pointing at the first non-lab /kaggle/input subdir,
#     so generic scripts find competition/user datasets without task-specific
#     knowledge
#
# Must be injected *after* any leading ``from __future__ import …`` and
# the module docstring — Python requires future imports at the very top.
_KAGGLE_BOOTSTRAP = '''\
# --- lab Kaggle bootstrap (auto-generated, do not edit) ---------------
import json as _json
import os as _os
import sys as _sys
_env = _json.loads(r"""{env_json}""")
for _k, _v in _env.items():
    _os.environ.setdefault(_k, str(_v))
_input = "/kaggle/input"
_lab_dir_name = {lab_dir_name!r}
_lab_mount = _os.path.join(_input, _lab_dir_name) if _lab_dir_name else ""
# Put the lab-agent-src dataset on sys.path so `import lab.*` resolves
# inside the kernel. If it's expected but not mounted, dump the
# /kaggle/input tree so the error log explains WHY the import will
# fail — users can verify their executor.kaggle.username / dataset
# presence at a glance.
if _lab_dir_name:
    if _os.path.isdir(_lab_mount):
        _sys.path.insert(0, _lab_mount)
    else:
        print(
            "[lab bootstrap] expected lab source dataset at " + _lab_mount
            + " but it isn't mounted. Kaggle /kaggle/input contains: "
            + (", ".join(sorted(_os.listdir(_input))) if _os.path.isdir(_input) else "<no input dir>")
        )
        print(
            "[lab bootstrap] attach the dataset in executor.kaggle.dataset_sources "
            "or run `python -m lab kaggle sync-lab` once to upload it."
        )
# Map the first non-lab /kaggle/input subdir onto AGENT_PROCESSED_DIR.
if _os.path.isdir(_input) and "AGENT_PROCESSED_DIR" not in _os.environ:
    _subdirs = sorted(
        p for p in _os.listdir(_input)
        if _os.path.isdir(_os.path.join(_input, p)) and p != _lab_dir_name
    )
    if _subdirs:
        _os.environ["AGENT_PROCESSED_DIR"] = _os.path.join(_input, _subdirs[0])
# --- end bootstrap ---------------------------------------------------
'''


def _hash_directory(path: Path) -> str:
    """Stable hex hash of a directory's .py file contents. Used to decide
    whether to push a new version of the lab-agent-src dataset."""
    h = hashlib.sha256()
    for py in sorted(path.rglob("*.py")):
        if "__pycache__" in py.parts or "/ui/" in str(py) or "\\ui\\" in str(py):
            continue
        rel = py.relative_to(path).as_posix().encode()
        h.update(rel)
        h.update(b"\0")
        h.update(py.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def _split_header(code: str) -> tuple[str, str]:
    """Split `code` into (header, rest).

    `header` contains everything that Python requires at the very top of
    a module — shebang, encoding declaration, module docstring, blank
    lines, comments, and any `from __future__ import …` statements.
    Everything after the first "real" statement goes into `rest`.

    This lets us inject our bootstrap between the header and the rest
    without tripping Python's "__future__ imports must occur at the
    beginning of the file" rule.
    """
    lines = code.splitlines(keepends=True)
    prefix: list[str] = []
    i = 0
    in_docstring = False
    docstring_delim: str | None = None

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if in_docstring:
            prefix.append(line)
            if docstring_delim and docstring_delim in stripped:
                in_docstring = False
            i += 1
            continue

        if stripped == "" or stripped.startswith("#"):
            prefix.append(line)
            i += 1
            continue

        # Start of a module-level docstring
        if stripped.startswith(('"""', "'''")):
            docstring_delim = stripped[:3]
            prefix.append(line)
            # Single-line docstring closes on the same line.
            if stripped.count(docstring_delim) >= 2 and stripped != docstring_delim:
                in_docstring = False
            else:
                in_docstring = True
            i += 1
            continue

        if stripped.startswith("from __future__ import"):
            prefix.append(line)
            i += 1
            continue

        break

    return "".join(prefix), "".join(lines[i:])


def _wrap_with_bootstrap(
    code: str,
    env: dict[str, str],
    *,
    lab_dataset_slug: str | None = None,
) -> str:
    header, rest = _split_header(code)
    # The Kaggle mount path is the dataset name without the username/
    # prefix. For "max/lab-agent-src" the mount is /kaggle/input/lab-agent-src.
    lab_dir_name = ""
    if lab_dataset_slug:
        lab_dir_name = lab_dataset_slug.split("/", 1)[-1]
    bootstrap = _KAGGLE_BOOTSTRAP.format(
        env_json=json.dumps(env),
        lab_dir_name=lab_dir_name,
    )
    # Ensure a blank line between pieces so line numbers stay readable
    # in Kaggle's traceback output.
    out = header
    if header and not header.endswith("\n"):
        out += "\n"
    out += bootstrap
    if not out.endswith("\n"):
        out += "\n"
    out += rest
    return out


__all__ = ["KaggleExecutor", "KaggleCLIUnavailable"]
