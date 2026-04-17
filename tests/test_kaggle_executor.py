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
            # New Kaggle log path: CLI writes kernel console output into
            # <path>/<kernel_slug>.log, then prints a short download msg.
            slug = argv[2]
            if "-p" in argv:
                out_dir = Path(argv[argv.index("-p") + 1])
                out_dir.mkdir(parents=True, exist_ok=True)
                kernel_slug = slug.split("/", 1)[-1]
                (out_dir / f"{kernel_slug}.log").write_text(
                    "Training started…\n"
                    "Traceback (most recent call last):\n"
                    "RuntimeError: CUDA out of memory. Tried to allocate …\n"
                )
            return _completed(stdout="Kernel log downloaded.\n")
        return _completed()

    monkeypatch.setattr(KaggleExecutor, "_run_kaggle", fake_run)
    result = ex.run("print('hi')", experiment_id="exp_err")
    assert not result.succeeded
    assert result.error is not None
    # classify_error should have bucketed CUDA-OOM into OOM.
    assert result.error.error_type == "OOM"


def test_kernel_slug_sanitises_experiment_id(tmp_path):
    ex = _make(tmp_path)
    # "test" prefix + "exp_AB_12" → avoid exp-exp doubling only when the
    # prefix actually ends in "exp". Here it doesn't, so we keep the id
    # verbatim (sanitised).
    assert ex._kernel_slug("exp_AB_12") == "maxuser/test-exp-ab-12"


def test_kernel_slug_avoids_exp_exp_doubling(tmp_path):
    ex = KaggleExecutor(
        username="maxuser", kernel_prefix="lab-exp",
        sandbox_root=tmp_path, repo_root=tmp_path,
    )
    assert ex._kernel_slug("exp_ca91a6ca0d") == "maxuser/lab-exp-ca91a6ca0d"


def test_bootstrap_wrapper_preserves_future_imports(tmp_path):
    """The kernel must still see ``from __future__ import …`` as the
    first executable statement after the docstring."""
    import ast
    from lab.core.kaggle_executor import _wrap_with_bootstrap

    code = (
        '"""Training skeleton."""\n'
        'from __future__ import annotations\n'
        '\n'
        'import torch\n'
        'print("hello")\n'
    )
    wrapped = _wrap_with_bootstrap(code, {"AGENT_EPOCHS": "3"})
    # Must parse without "__future__ imports must occur at the beginning"
    ast.parse(wrapped)
    # __future__ import must come BEFORE any of the bootstrap's imports.
    future_pos = wrapped.index("from __future__ import")
    bootstrap_pos = wrapped.index("lab Kaggle bootstrap")
    assert future_pos < bootstrap_pos


def test_bootstrap_wrapper_handles_triple_quote_docstring(tmp_path):
    import ast
    from lab.core.kaggle_executor import _wrap_with_bootstrap

    code = (
        '"""Multi-line\n'
        'docstring with blank.\n'
        '"""\n'
        'from __future__ import annotations\n'
        'x = 1\n'
    )
    wrapped = _wrap_with_bootstrap(code, {})
    ast.parse(wrapped)
    assert wrapped.index('"""Multi-line') < wrapped.index("lab Kaggle bootstrap")


def test_bootstrap_wrapper_without_future_import(tmp_path):
    """Plain code (no docstring, no future import) still wraps cleanly."""
    import ast
    from lab.core.kaggle_executor import _wrap_with_bootstrap

    code = 'import os\nprint("hi")\n'
    wrapped = _wrap_with_bootstrap(code, {"X": "y"})
    ast.parse(wrapped)
    # Env var must be set before user code runs
    assert wrapped.index("AGENT") < wrapped.index('print("hi")') or "X" in wrapped


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


def test_classify_kaggle_status_handles_known_phrasings():
    from lab.core.kaggle_executor import _classify_kaggle_status
    # Legacy phrasings
    assert _classify_kaggle_status('Kernel has status "complete"') == "complete"
    assert _classify_kaggle_status('status: complete') == "complete"
    assert _classify_kaggle_status('Kernel has status "error"') == "error"
    assert _classify_kaggle_status('Kernel has status "cancelled"') == "cancelled"
    assert _classify_kaggle_status('Kernel has status "canceled"') == "cancelled"
    assert _classify_kaggle_status('status: incomplete (queued)') == "running"
    # Current Kaggle CLI (seen in production logs, 2026-04)
    assert _classify_kaggle_status(
        'u/foo has status "KernelWorkerStatus.COMPLETE"'
    ) == "complete"
    assert _classify_kaggle_status(
        'u/foo has status "KernelWorkerStatus.ERROR"'
    ) == "error"
    assert _classify_kaggle_status(
        'u/foo has status "KernelWorkerStatus.RUNNING"'
    ) == "running"
    assert _classify_kaggle_status(
        'u/foo has status "KernelWorkerStatus.QUEUED"'
    ) == "running"
    assert _classify_kaggle_status(
        'u/foo has status "KernelWorkerStatus.CANCELLED"'
    ) == "cancelled"
    assert _classify_kaggle_status(
        'u/foo has status "KernelWorkerStatus.CANCEL_ACKNOWLEDGED"'
    ) == "cancelled"


def test_stream_kaggle_log_delta_emits_only_new_lines(caplog, tmp_path):
    from lab.core.kaggle_executor import _stream_kaggle_log_delta
    live = tmp_path / "stdout.log"
    caplog.set_level("INFO", logger="lab.kaggle_executor")

    prev = ""
    prev = _stream_kaggle_log_delta(
        previous=prev,
        current="line1\nline2\n",
        live_stdout_path=live,
    )
    prev = _stream_kaggle_log_delta(
        previous=prev,
        current="line1\nline2\nline3\n",
        live_stdout_path=live,
    )
    # No new lines here -> nothing should be appended/logged
    prev = _stream_kaggle_log_delta(
        previous=prev,
        current="line1\nline2\nline3\n",
        live_stdout_path=live,
    )

    text = live.read_text()
    assert text.splitlines() == ["line1", "line2", "line3"]
    assert any("[kaggle] line1" in rec.message for rec in caplog.records)
    assert any("[kaggle] line3" in rec.message for rec in caplog.records)


def test_fetch_kernel_log_reads_kaggle_log_file(monkeypatch, tmp_path):
    monkeypatch.setattr("lab.core.kaggle_executor.shutil.which", lambda _: "/usr/bin/kaggle")
    ex = _make(tmp_path)

    def fake_run(self, cli, argv, *, timeout):
        if argv[0:2] == ["kernels", "output"]:
            slug = argv[2]
            out_dir = Path(argv[argv.index("-p") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            kernel_slug = slug.split("/", 1)[-1]
            (out_dir / f"{kernel_slug}.log").write_text("[epoch 1] loss=0.1234\n")
            return _completed(stdout="Kernel log downloaded.\n")
        return _completed()

    monkeypatch.setattr(KaggleExecutor, "_run_kaggle", fake_run)
    text = ex._fetch_kernel_log("/usr/bin/kaggle", "maxuser/lab-exp-abc", target_dir=tmp_path)
    assert "[epoch 1] loss=0.1234" in text


def test_normalize_kaggle_json_log_to_plain_lines():
    from lab.core.kaggle_executor import _normalize_kaggle_log_text
    raw = (
        '[{"stream_name":"stdout","time":1.2,"data":"hello\\n"},'
        '{"stream_name":"stderr","time":1.3,"data":"boom\\n"}]'
    )
    out = _normalize_kaggle_log_text(raw)
    assert out.splitlines() == ["hello", "boom"]


def test_run_overrides_agent_device_for_kaggle(monkeypatch, tmp_path):
    """The orchestrator may pass AGENT_DEVICE='cpu' (based on the Mac
    host) but on a Kaggle GPU kernel we need cuda. Executor must
    override the value based on enable_gpu."""
    monkeypatch.setattr("lab.core.kaggle_executor.shutil.which", lambda _: "/usr/bin/kaggle")
    ex = KaggleExecutor(
        username="u", kernel_prefix="test", enable_gpu=True,
        poll_interval_seconds=0, sandbox_root=tmp_path, repo_root=tmp_path,
        training_env={"AGENT_DEVICE": "cpu"},   # host says CPU
    )

    captured: dict[str, str] = {}

    def fake_run(self, cli, argv, *, timeout):
        if argv[0:2] == ["kernels", "push"]:
            # Inspect the written code.py for the env dict
            import re as _re
            kernel_dir = _re.search(r"-p\s+(\S+)", " ".join(argv))
            if kernel_dir:
                code = (tmp_path / "exp_env" / "kernel" / "code.py")
                if code.exists():
                    captured["code"] = code.read_text()
            return _completed(stdout="Kernel pushed.\n")
        if argv[0:2] == ["kernels", "status"]:
            return _completed(stdout='Kernel has status "complete".\n')
        if argv[0:2] == ["kernels", "output"]:
            (tmp_path / "exp_env" / "results.json").write_text(
                '{"primary_metric":"f1_macro","primary_score":0.1,"history":[],"final":{},"best":{}}'
            )
            return _completed(stdout="ok\n")
        if argv[0:2] == ["datasets", "create"] or argv[0:2] == ["datasets", "version"]:
            return _completed(stdout="Dataset.\n")
        return _completed(stdout="")

    monkeypatch.setattr(KaggleExecutor, "_run_kaggle", fake_run)
    ex.run("from lab.tasks.track_b_birdclef import load_audio_dataset\nprint(1)\n",
           experiment_id="exp_env")
    assert "code" in captured, "kernel code.py not captured"
    assert '"AGENT_DEVICE": "cuda"' in captured["code"], (
        "expected AGENT_DEVICE overridden to 'cuda' for GPU kernel, got:\n"
        + "\n".join(l for l in captured["code"].splitlines() if "AGENT_DEVICE" in l)
    )


def test_run_passes_accelerator_as_cli_flag(monkeypatch, tmp_path):
    """executor.kaggle.accelerator must reach `kaggle kernels push` as
    a CLI flag (`--accelerator NvidiaTeslaT4`). The metadata-file field
    of the same name is silently ignored by the current Kaggle API, so
    the flag is the only thing that actually picks the GPU type."""
    monkeypatch.setattr("lab.core.kaggle_executor.shutil.which", lambda _: "/usr/bin/kaggle")
    ex = KaggleExecutor(
        username="u", kernel_prefix="test", enable_gpu=True,
        accelerator="NvidiaTeslaT4",
        poll_interval_seconds=0, sandbox_root=tmp_path, repo_root=tmp_path,
    )
    captured = {"push_argv": None}

    def fake_run(self, cli, argv, *, timeout):
        if argv[0:2] == ["kernels", "push"]:
            captured["push_argv"] = list(argv)
            return _completed()
        if argv[0:2] == ["kernels", "status"]:
            return _completed(stdout='status: complete\n')
        if argv[0:2] == ["kernels", "output"]:
            (tmp_path / "exp_acc" / "results.json").write_text(
                '{"primary_metric":"f1_macro","primary_score":0.1,"history":[],"final":{},"best":{}}'
            )
            return _completed()
        return _completed()

    monkeypatch.setattr(KaggleExecutor, "_run_kaggle", fake_run)
    ex.run("pass\n", experiment_id="exp_acc")
    argv = captured["push_argv"]
    assert argv is not None, "kernels push was never called"
    assert "--accelerator" in argv, f"missing --accelerator flag: {argv}"
    i = argv.index("--accelerator")
    assert argv[i + 1] == "NvidiaTeslaT4", f"wrong value: {argv[i+1]!r}"


def test_run_omits_accelerator_flag_when_empty(monkeypatch, tmp_path):
    monkeypatch.setattr("lab.core.kaggle_executor.shutil.which", lambda _: "/usr/bin/kaggle")
    ex = KaggleExecutor(
        username="u", enable_gpu=True, accelerator="",
        poll_interval_seconds=0, sandbox_root=tmp_path, repo_root=tmp_path,
    )
    captured = {"push_argv": None}

    def fake_run(self, cli, argv, *, timeout):
        if argv[0:2] == ["kernels", "push"]:
            captured["push_argv"] = list(argv)
            return _completed()
        if argv[0:2] == ["kernels", "status"]:
            return _completed(stdout='status: complete\n')
        if argv[0:2] == ["kernels", "output"]:
            (tmp_path / "exp_no_acc" / "results.json").write_text(
                '{"primary_metric":"f1_macro","primary_score":0.1,"history":[],"final":{},"best":{}}'
            )
            return _completed()
        return _completed()

    monkeypatch.setattr(KaggleExecutor, "_run_kaggle", fake_run)
    ex.run("pass\n", experiment_id="exp_no_acc")
    argv = captured["push_argv"]
    assert argv is not None
    assert "--accelerator" not in argv


def test_run_sets_cpu_when_gpu_disabled(monkeypatch, tmp_path):
    monkeypatch.setattr("lab.core.kaggle_executor.shutil.which", lambda _: "/usr/bin/kaggle")
    ex = KaggleExecutor(
        username="u", enable_gpu=False, poll_interval_seconds=0,
        sandbox_root=tmp_path, repo_root=tmp_path,
        training_env={"AGENT_DEVICE": "cuda"},  # caller incorrectly asked for gpu
    )
    captured: dict[str, str] = {}

    def fake_run(self, cli, argv, *, timeout):
        if argv[0:2] == ["kernels", "push"]:
            code = (tmp_path / "exp_nogpu" / "kernel" / "code.py")
            if code.exists():
                captured["code"] = code.read_text()
            return _completed()
        if argv[0:2] == ["kernels", "status"]:
            return _completed(stdout='status: complete\n')
        if argv[0:2] == ["kernels", "output"]:
            (tmp_path / "exp_nogpu" / "results.json").write_text(
                '{"primary_metric":"f1_macro","primary_score":0.1,"history":[],"final":{},"best":{}}'
            )
            return _completed()
        return _completed()

    monkeypatch.setattr(KaggleExecutor, "_run_kaggle", fake_run)
    ex.run("pass\n", experiment_id="exp_nogpu")
    assert "code" in captured
    assert '"AGENT_DEVICE": "cpu"' in captured["code"]


def test_kaggle_safe_env_strips_host_filesystem_paths():
    from lab.core.kaggle_executor import _kaggle_safe_env
    env = {
        "AGENT_EPOCHS": "3",
        "AGENT_BATCH_SIZE": "64",
        "AGENT_PROCESSED_DIR": "/Users/max/Documents/lab/data/processed",
        "AGENT_TRAIN_AUDIO": "/home/user/data/train",
        "AGENT_DEVICE": "cpu",
        "AGENT_KAGGLE_WORKDIR": "/kaggle/working",   # keep — Kaggle-rooted
        "AGENT_WINDOWS_LIKE": "C:\\Users\\Max\\data",
    }
    safe = _kaggle_safe_env(env)
    # Numeric + flag-like values survive
    assert safe["AGENT_EPOCHS"] == "3"
    assert safe["AGENT_BATCH_SIZE"] == "64"
    assert safe["AGENT_DEVICE"] == "cpu"
    # Kaggle-rooted paths survive
    assert safe["AGENT_KAGGLE_WORKDIR"] == "/kaggle/working"
    # Host absolute paths are dropped
    assert "AGENT_PROCESSED_DIR" not in safe
    assert "AGENT_TRAIN_AUDIO" not in safe
    assert "AGENT_WINDOWS_LIKE" not in safe


def test_kill_running_breaks_poll_loop(monkeypatch, tmp_path):
    """kill_running() sets an abort flag the poll loop checks each tick,
    so the orchestrator can stop waiting on a Kaggle kernel without
    blocking for the full poll_timeout_seconds."""
    ex = KaggleExecutor(
        username="u", kernel_prefix="test", poll_interval_seconds=1,
        poll_timeout_seconds=60, sandbox_root=tmp_path, repo_root=tmp_path,
    )
    # Status always reports "running" → normally the loop polls forever.
    def fake_run(self, cli, argv, *, timeout):
        return _completed(stdout='status: running\n')
    monkeypatch.setattr(KaggleExecutor, "_run_kaggle", fake_run)

    # Abort half a second in; the poll should return "aborted" promptly.
    import threading, time
    threading.Timer(0.3, ex.kill_running).start()
    t0 = time.monotonic()
    status, _ = ex._poll_until_done("/usr/bin/kaggle", "u/test-slug")
    duration = time.monotonic() - t0
    assert status == "aborted", status
    assert duration < 3.0, f"poll kept running for {duration:.1f}s after abort"


# ---------------------------------------------------------------------------
# Live log streaming via the Python API (decoupled from status poll)
# ---------------------------------------------------------------------------


def test_poll_loop_streams_live_log_between_status_polls(monkeypatch, tmp_path):
    """The poll loop should emit stdout deltas from the API on every
    `log_poll_interval_seconds` tick, not wait for the 30 s status poll."""
    monkeypatch.setattr("lab.core.kaggle_executor.shutil.which", lambda _: "/usr/bin/kaggle")
    # Deterministic + fast: 1 s between log ticks, 3 s between status polls.
    ex = KaggleExecutor(
        username="maxuser",
        kernel_prefix="test",
        sandbox_root=tmp_path,
        repo_root=tmp_path,
        poll_interval_seconds=3,
        log_poll_interval_seconds=1,
        poll_timeout_seconds=60,
    )

    live = tmp_path / "stdout.log"
    # Sequence of log strings the fake API hands back on successive ticks.
    log_script = iter([
        "[epoch 1] loss=0.5\n",
        "[epoch 1] loss=0.5\n[epoch 1] loss=0.4\n",
        "[epoch 1] loss=0.5\n[epoch 1] loss=0.4\n[epoch 2] loss=0.3\n",
    ])
    ex._fetch_live_log_via_api = lambda slug: next(log_script, "")

    # Status flips to 'complete' on the third status poll.
    status_responses = iter([
        _completed(stdout='has status "running"\n'),
        _completed(stdout='has status "running"\n'),
        _completed(stdout='has status "complete"\n'),
    ])
    monkeypatch.setattr(
        KaggleExecutor,
        "_run_kaggle",
        lambda self, cli, argv, *, timeout: next(
            status_responses, _completed(stdout='has status "complete"\n'),
        ),
    )

    status, log_text = ex._poll_until_done(
        "/usr/bin/kaggle", "maxuser/test-slug", live_stdout_path=live,
    )
    assert status == "complete"
    # All three epochs must have streamed into the UI log file in order.
    contents = live.read_text()
    assert "[epoch 1] loss=0.5" in contents
    assert "[epoch 1] loss=0.4" in contents
    assert "[epoch 2] loss=0.3" in contents


def test_get_kaggle_api_caches_unavailability(monkeypatch, tmp_path):
    """Once auth has failed we should NOT re-attempt every poll tick —
    otherwise the log spam makes the agent log unreadable."""
    ex = KaggleExecutor(
        username="u", kernel_prefix="t", sandbox_root=tmp_path, repo_root=tmp_path,
    )
    attempts: list[int] = []

    def fake_import(name, *args, **kwargs):
        attempts.append(1)
        raise ImportError("no kaggle")

    # First call fails and caches the failure; second call must NOT retry.
    import builtins
    real_import = builtins.__import__

    def stub(name, *a, **kw):
        if name.startswith("kaggle"):
            return fake_import(name)
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", stub)
    assert ex._get_kaggle_api() is None
    assert ex._get_kaggle_api() is None
    assert len(attempts) == 1
