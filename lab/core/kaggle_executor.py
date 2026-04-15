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
        full_env = _kaggle_safe_env({**self.training_env, **(extra_env or {})})
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
        try:
            return subprocess.run(
                [cli, *argv], capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            # Shape a synthetic CompletedProcess so callers never have
            # to branch on timeout — keeps the poll loop alive.
            return subprocess.CompletedProcess(
                args=list(exc.cmd or []), returncode=-1,
                stdout=(exc.stdout or b"").decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
                stderr=f"kaggle CLI timeout after {timeout}s",
            )

    def _poll_until_done(self, cli: str, slug: str) -> tuple[str, str]:
        """Return (final_status, log_text). ``final_status`` is one of
        'complete', 'error', 'cancelled', 'timeout', 'aborted'."""
        deadline = time.monotonic() + self.poll_timeout_seconds
        last_status = "unknown"
        poll_count = 0
        while time.monotonic() < deadline:
            if self._aborted:
                last_status = "aborted"
                break
            status_proc = self._run_kaggle(
                cli, ["kernels", "status", slug], timeout=60,
            )
            raw = (status_proc.stdout + "\n" + status_proc.stderr)
            classified = _classify_kaggle_status(raw)
            poll_count += 1
            # Log the actual CLI output every ~10 polls so the user can
            # see what Kaggle said when we're not matching. Always log
            # the first poll too so there's immediate signal.
            if poll_count == 1 or poll_count % 10 == 0:
                sample = raw.strip().splitlines()
                logger.info(
                    "Kaggle kernel %s poll #%d → %s. Raw: %s",
                    slug, poll_count, classified,
                    " | ".join(sample[-3:]) if sample else "<empty>",
                )
            if classified in ("complete", "error", "cancelled"):
                last_status = classified
                break
            last_status = "running"
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
        # Bump the cache-schema token whenever the upload shape changes —
        # invalidates old caches so subsequent runs push the new content.
        digest = _hash_directory(lab_src) + ":zip"
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

            # `--dir-mode zip` is critical: Kaggle's default `skip` mode
            # silently drops subdirectories (= our entire lab/ tree).
            # `zip` uploads subdirs as zip files which the kernel
            # bootstrap extracts at runtime.
            create = self._run_kaggle(
                cli,
                ["datasets", "create", "-p", str(tmp_path), "--dir-mode", "zip"],
                timeout=600,
            )
            if create.returncode == 0:
                logger.info("lab-src dataset created: %s", slug)
            else:
                # Likely already exists — push a new version.
                msg = f"lab src {digest[:8]}"
                version = self._run_kaggle(
                    cli,
                    ["datasets", "version", "-p", str(tmp_path), "-m", msg,
                     "--dir-mode", "zip"],
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
_lab_owner = {lab_owner!r}


def _lab_bootstrap_find_lab_root():
    """Return a directory containing a usable ``lab`` package, or None.

    Kaggle mounts datasets under a few possible layouts:
      /kaggle/input/<dataset-name>/lab/…              (older flat)
      /kaggle/input/datasets/<owner>/<dataset>/lab/…  (newer nested)

    AND: `kaggle datasets create --dir-mode zip` uploads subdirectories
    as zip files; Kaggle often auto-extracts them and in the process
    flattens the wrapper directory. A dataset made from ``tmp/lab/*``
    can end up on the kernel as ``<mount>/__init__.py`` +
    ``<mount>/core/…`` — the ``lab/`` wrapper gone. We detect that
    "lab signature" (dataset root has __init__.py + core/ + tasks/)
    and stage the tree into ``/tmp/.../lab/`` on the fly.

    AND: if ``lab.zip`` is present uncompressed, we extract it.
    """
    import shutil as _shutil
    import tempfile as _tempfile
    import zipfile as _zipfile

    def _has_lab_pkg(d):
        return _os.path.isfile(_os.path.join(d, "lab", "__init__.py"))

    def _looks_like_lab(d):
        """Heuristic: this dir IS the lab package, just not named 'lab'."""
        if not _os.path.isfile(_os.path.join(d, "__init__.py")):
            return False
        # Our lab package always ships core/ and tasks/ subpackages.
        for sub in ("core", "tasks"):
            if not _os.path.isdir(_os.path.join(d, sub)):
                return False
        return True

    def _stage_as_lab(src):
        dest = _tempfile.mkdtemp(prefix="lab-staged-")
        _shutil.copytree(src, _os.path.join(dest, "lab"))
        return dest

    def _maybe_extract_zip(candidate):
        zp = _os.path.join(candidate, "lab.zip")
        if not _os.path.isfile(zp):
            return None
        dest = _tempfile.mkdtemp(prefix="lab-extracted-")
        try:
            with _zipfile.ZipFile(zp) as z:
                z.extractall(dest)
        except _zipfile.BadZipFile:
            return None
        if _has_lab_pkg(dest):
            return dest
        # Zip written with the subdir as top-level entry
        for entry in _os.listdir(dest):
            inner = _os.path.join(dest, entry)
            if _os.path.isdir(inner) and _has_lab_pkg(inner):
                return inner
        if _looks_like_lab(dest):
            return _stage_as_lab(dest)
        return None

    hints = []
    if _lab_dir_name:
        hints.extend([
            _os.path.join(_input, _lab_dir_name),
            _os.path.join(_input, "datasets", _lab_owner, _lab_dir_name),
        ])
    for hint in hints:
        if _has_lab_pkg(hint):
            return hint
        if _looks_like_lab(hint):
            staged = _stage_as_lab(hint)
            print("[lab bootstrap] staged flattened lab package from " + hint)
            return staged
        extracted = _maybe_extract_zip(hint)
        if extracted:
            print("[lab bootstrap] extracted lab.zip from " + hint)
            return extracted

    # Bounded walk — catches any layout we haven't hinted.
    if not _os.path.isdir(_input):
        return None
    base_depth = _input.count(_os.sep)
    for root, dirs, files in _os.walk(_input):
        if root.count(_os.sep) - base_depth > 4:
            dirs[:] = []
            continue
        if _has_lab_pkg(root):
            return root
        if _looks_like_lab(root):
            staged = _stage_as_lab(root)
            print("[lab bootstrap] staged flattened lab package from " + root)
            return staged
        if "lab.zip" in files:
            extracted = _maybe_extract_zip(root)
            if extracted:
                print("[lab bootstrap] extracted lab.zip from " + root)
                return extracted
    return None


_lab_root = _lab_bootstrap_find_lab_root()
if _lab_root:
    _sys.path.insert(0, _lab_root)
    print("[lab bootstrap] lab source on sys.path from " + _lab_root)
else:
    print(
        "[lab bootstrap] could not find lab/ under " + _input + "."
        " Dumping input tree for diagnosis:"
    )
    if _os.path.isdir(_input):
        _base = _input.count(_os.sep)
        for _root, _dirs, _files in _os.walk(_input):
            _depth = _root.count(_os.sep) - _base
            if _depth > 4:
                _dirs[:] = []
                continue
            _indent = "  " * _depth
            print(_indent + _os.path.basename(_root) + "/")
            for _f in sorted(_files)[:5]:
                print(_indent + "  " + _f)
            if len(_files) > 5:
                print(_indent + "  ... (" + str(len(_files) - 5) + " more)")
    print(
        "[lab bootstrap] expected "
        + (_lab_owner + "/" + _lab_dir_name if _lab_dir_name else "lab-agent-src")
        + ". If it's in the tree above under a different path, that's a"
        " Kaggle layout change and you should file an issue."
        " Otherwise check `kaggle datasets list -m` shows this dataset as"
        " `ready` (not `processing`) and retry."
    )


def _lab_bootstrap_find_processed_dir():
    """Pick a non-lab /kaggle/input subtree that looks like training data.

    We look for labels.csv up to depth 3 under each candidate — the
    team's preprocessed dataset has it under ``<dataset>/processed/``,
    not at the dataset root."""
    if not _os.path.isdir(_input):
        return None
    skip = {{_lab_dir_name}} if _lab_dir_name else set()
    candidates = []
    # Top-level entries first
    for entry in sorted(_os.listdir(_input)):
        if entry in skip:
            continue
        p = _os.path.join(_input, entry)
        if _os.path.isdir(p):
            candidates.append(p)
    # Also peek one or two levels deeper for the nested layout
    for parent in ("datasets", "competitions"):
        parent_dir = _os.path.join(_input, parent)
        if not _os.path.isdir(parent_dir):
            continue
        for owner in sorted(_os.listdir(parent_dir)):
            owner_dir = _os.path.join(parent_dir, owner)
            if not _os.path.isdir(owner_dir):
                continue
            for name in sorted(_os.listdir(owner_dir)):
                if name == _lab_dir_name:
                    continue
                d = _os.path.join(owner_dir, name)
                if _os.path.isdir(d):
                    candidates.append(d)

    # Bounded search for labels.csv within each candidate.
    def _find_labels_csv(root, max_depth=3):
        base = root.count(_os.sep)
        for r, _dirs, _files in _os.walk(root):
            if r.count(_os.sep) - base > max_depth:
                _dirs[:] = []
                continue
            if "labels.csv" in _files:
                return r
        return None

    for c in candidates:
        hit = _find_labels_csv(c)
        if hit:
            return hit
    # Fallback: first non-lab directory we found
    return candidates[0] if candidates else None


if "AGENT_PROCESSED_DIR" not in _os.environ:
    _d = _lab_bootstrap_find_processed_dir()
    if _d:
        _os.environ["AGENT_PROCESSED_DIR"] = _d
        print("[lab bootstrap] AGENT_PROCESSED_DIR = " + _d)
    else:
        print("[lab bootstrap] no data directory found under " + _input)
# --- end bootstrap ---------------------------------------------------
'''


def _classify_kaggle_status(raw: str) -> str:
    """Map raw ``kaggle kernels status`` CLI output to a coarse state.

    Kaggle has shipped several output formats over the years and none
    are documented as stable — we match substrings liberally rather
    than hard-coding one specific phrasing, to avoid the "poll forever
    because we missed ONE character" failure mode.

    Returns: 'complete', 'error', 'cancelled', or 'running'.
    """
    t = (raw or "").lower()

    def _word_hit(phrases: tuple[str, ...]) -> bool:
        return any(p in t for p in phrases)

    # Order matters: check failure states first so "complete" doesn't
    # win over "completed with error" hypothetical combos.
    if _word_hit((
        'has status "error"',
        'status: error',
        'status=error',
        '"error"',
        'has failed',
        'failed to run',
    )):
        return "error"
    if _word_hit((
        'has status "cancelled"',
        'has status "canceled"',
        'status: cancelled',
        'status: canceled',
        '"cancelled"',
        '"canceled"',
    )):
        return "cancelled"
    if _word_hit((
        'has status "complete"',
        'has status "succeeded"',
        'status: complete',
        'status: succeeded',
        'status=complete',
        '"complete"',
        '"succeeded"',
    )):
        return "complete"
    return "running"


def _kaggle_safe_env(env: dict[str, str]) -> dict[str, str]:
    """Strip env vars whose values point at the host filesystem.

    The orchestrator's `_training_env` populates things like
    ``AGENT_PROCESSED_DIR = /Users/max/.../data/processed`` so local
    subprocesses find their data. Forwarding those absolute host paths
    into a Kaggle kernel is worse than useless — the kernel tries to
    open a non-existent path and fails before the bootstrap's own
    `/kaggle/input/*`-based AGENT_PROCESSED_DIR mapping can help.

    Heuristic: keep every var whose value is NOT an absolute filesystem
    path, plus the ones that are already Kaggle-rooted. Drop everything
    else — the bootstrap re-derives the data paths from /kaggle/input.
    """
    safe: dict[str, str] = {}
    for k, v in env.items():
        if not isinstance(v, str):
            continue
        # Keep POSIX relative paths, flags, numbers, etc.
        looks_absolute_posix = v.startswith("/") and v != "/"
        looks_absolute_windows = len(v) >= 3 and v[1:3] == ":\\"
        if not (looks_absolute_posix or looks_absolute_windows):
            safe[k] = v
            continue
        # Keep Kaggle-rooted absolute paths (shouldn't really occur here
        # but leaves the door open for explicit /kaggle/working overrides).
        if v.startswith("/kaggle/"):
            safe[k] = v
    return safe


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
    # The Kaggle mount path depends on the kernel-metadata version:
    #   older: /kaggle/input/<dataset-name>/
    #   newer: /kaggle/input/datasets/<owner>/<dataset-name>/
    # We pass both parts so the bootstrap can check both layouts at runtime.
    lab_owner = ""
    lab_dir_name = ""
    if lab_dataset_slug:
        if "/" in lab_dataset_slug:
            lab_owner, lab_dir_name = lab_dataset_slug.split("/", 1)
        else:
            lab_dir_name = lab_dataset_slug
    bootstrap = _KAGGLE_BOOTSTRAP.format(
        env_json=json.dumps(env),
        lab_dir_name=lab_dir_name,
        lab_owner=lab_owner,
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
