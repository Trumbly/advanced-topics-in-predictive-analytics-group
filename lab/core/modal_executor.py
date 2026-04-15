"""Modal executor.

Runs generated training code inside a Modal function. Optional support
for mounting S3-compatible buckets (e.g. Hetzner Object Storage) via
``modal.CloudBucketMount``.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lab.core.executor import ExecutionResult, classify_error
from lab.core.models import TaskError


logger = logging.getLogger("lab.modal_executor")


@dataclass
class ModalExecutor:
    """Run generated code as a remote Modal function."""

    app_name: str = "lab-agent"
    image_name: str = "debian_slim"
    pip_packages: list[str] = field(default_factory=list)
    gpu: str = ""
    cpu: float | None = None
    memory_mb: int | None = None
    timeout_seconds: int = 1800
    retries: int = 0

    # Optional Volume for persisting artifacts/checkpoints in Modal.
    output_volume_name: str = ""
    output_mount_path: str = "/mnt/output"

    # Optional S3-compatible mount (AWS S3 / Hetzner / R2 / GCS HMAC).
    s3_bucket_name: str = ""
    s3_secret_name: str = ""
    s3_endpoint_url: str = ""
    s3_mount_path: str = "/mnt/data"
    s3_key_prefix: str = ""
    set_processed_dir_from_s3: bool = True
    processed_subpath: str = "processed"

    sandbox_root: Path = Path("sandbox")
    repo_root: Path = field(default_factory=lambda: Path.cwd())
    training_env: dict[str, str] = field(default_factory=dict)
    env_prefix: str = "AGENT"

    backend: str = field(default="modal", init=False)

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
        (workdir / "code.py").write_text(code)

        env = {**self.training_env, **(extra_env or {})}
        processed_key = f"{self.env_prefix}_PROCESSED_DIR"
        payload = {
            "code": code,
            "env": env,
            "workdir": "/tmp/lab-run",
            "processed_key": processed_key,
            "set_processed_dir_from_s3": self.set_processed_dir_from_s3,
            "s3_mount_path": self.s3_mount_path,
            "processed_subpath": self.processed_subpath,
        }

        try:
            remote = self._run_on_modal(payload)
            stdout = str(remote.get("stdout", ""))
            stderr = str(remote.get("stderr", ""))
            exit_code = int(remote.get("exit_code", -1))
            results_raw = remote.get("results_json")
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            err = classify_error(msg, timed_out=False) or TaskError(
                error_type="SpawnError",
                message=f"Modal run failed before execution: {msg[:300]}",
                traceback=msg[-800:],
            )
            return ExecutionResult(
                exit_code=-1,
                stdout="",
                stderr=msg,
                duration_seconds=time.monotonic() - start,
                workdir=workdir,
                results_json_path=None,
                error=err,
                timed_out=False,
            )

        (workdir / "stdout.log").write_text(stdout)
        (workdir / "stderr.log").write_text(stderr)

        results_path: Path | None = None
        if isinstance(results_raw, str) and results_raw.strip():
            results_path = workdir / "results.json"
            results_path.write_text(results_raw)

        duration = time.monotonic() - start
        timed_out = bool(remote.get("timed_out", False))

        if exit_code == 0 and results_path and results_path.exists():
            return ExecutionResult(
                exit_code=0,
                stdout=stdout,
                stderr=stderr,
                duration_seconds=duration,
                workdir=workdir,
                results_json_path=results_path,
                error=None,
                timed_out=False,
            )

        err = classify_error(stderr or stdout, timed_out=timed_out) or TaskError(
            error_type="UnknownError",
            message="Modal run finished without results.json",
            traceback=(stderr or stdout)[-800:],
        )
        return ExecutionResult(
            exit_code=exit_code if exit_code != 0 else -1,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=duration,
            workdir=workdir,
            results_json_path=results_path,
            error=err,
            timed_out=timed_out,
        )

    def _run_on_modal(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            import modal
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Modal SDK not installed. Install with `pip install modal` and run `modal setup`."
            ) from exc

        app = modal.App(self.app_name)
        image = self._build_image(modal)

        fn_kwargs: dict[str, Any] = {
            "image": image,
            "timeout": self.timeout_seconds,
            "retries": self.retries,
        }
        if self.gpu:
            fn_kwargs["gpu"] = self.gpu
        if self.cpu is not None:
            fn_kwargs["cpu"] = self.cpu
        if self.memory_mb is not None:
            fn_kwargs["memory"] = self.memory_mb

        volumes: dict[str, Any] = {}
        if self.output_volume_name:
            out_vol = modal.Volume.from_name(self.output_volume_name, create_if_missing=True)
            volumes[self.output_mount_path] = out_vol
        bucket_mount = self._build_bucket_mount(modal)
        if bucket_mount is not None:
            volumes[self.s3_mount_path] = bucket_mount
        if volumes:
            fn_kwargs["volumes"] = volumes

        @app.function(**fn_kwargs)
        def _execute_remote(run_payload: dict[str, Any]) -> dict[str, Any]:
            import contextlib
            import io
            import os
            import traceback
            from pathlib import Path

            out = io.StringIO()
            err = io.StringIO()

            exit_code = 0
            timed_out = False
            results_text = ""
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                work = Path(run_payload.get("workdir", "/tmp/lab-run"))
                work.mkdir(parents=True, exist_ok=True)
                os.chdir(work)

                env_in = run_payload.get("env", {}) or {}
                for k, v in env_in.items():
                    os.environ[str(k)] = str(v)

                processed_key = str(run_payload.get("processed_key", "AGENT_PROCESSED_DIR"))
                if (
                    run_payload.get("set_processed_dir_from_s3")
                    and not os.environ.get(processed_key)
                ):
                    mount = str(run_payload.get("s3_mount_path", "")).strip()
                    sub = str(run_payload.get("processed_subpath", "")).strip("/")
                    if mount:
                        target = os.path.join(mount, sub) if sub else mount
                        os.environ[processed_key] = target
                        print(f"[modal bootstrap] {processed_key} = {target}", flush=True)

                ns: dict[str, Any] = {"__name__": "__main__"}
                try:
                    src = str(run_payload.get("code", ""))
                    exec(compile(src, "code.py", "exec"), ns, ns)
                except SystemExit as exc:
                    code = exc.code if isinstance(exc.code, int) else 1
                    exit_code = int(code)
                except Exception:  # noqa: BLE001
                    exit_code = 1
                    print(traceback.format_exc(), file=err)

                res = work / "results.json"
                if res.exists():
                    results_text = res.read_text()

            return {
                "exit_code": exit_code,
                "stdout": out.getvalue(),
                "stderr": err.getvalue(),
                "results_json": results_text,
                "timed_out": timed_out,
            }

        with app.run():
            return _execute_remote.remote(payload)

    def _build_image(self, modal_mod):
        if self.image_name == "debian_slim":
            image = modal_mod.Image.debian_slim()
        else:
            image = modal_mod.Image.from_registry(self.image_name)
        if self.pip_packages:
            image = image.pip_install(*self.pip_packages)
        return image

    def _build_bucket_mount(self, modal_mod):
        if not self.s3_bucket_name:
            return None
        if not self.s3_secret_name:
            raise RuntimeError(
                "executor.modal.s3_secret_name is required when s3_bucket_name is set."
            )
        secret = modal_mod.Secret.from_name(self.s3_secret_name)

        kwargs: dict[str, Any] = {"secret": secret}
        if self.s3_endpoint_url:
            kwargs["bucket_endpoint_url"] = self.s3_endpoint_url
        if self.s3_key_prefix:
            kwargs["key_prefix"] = self.s3_key_prefix
        try:
            return modal_mod.CloudBucketMount(self.s3_bucket_name, **kwargs)
        except TypeError:
            kwargs.pop("key_prefix", None)
            return modal_mod.CloudBucketMount(self.s3_bucket_name, **kwargs)

    def infrastructure(self) -> dict[str, Any]:
        return {
            "backend": "modal",
            "app_name": self.app_name,
            "gpu": self.gpu or "none",
            "cpu": self.cpu,
            "memory_mb": self.memory_mb,
            "timeout_seconds": self.timeout_seconds,
            "retries": self.retries,
            "output_volume_name": self.output_volume_name or "",
            "s3_bucket_name": self.s3_bucket_name or "",
            "s3_endpoint_url": self.s3_endpoint_url or "",
            "s3_mount_path": self.s3_mount_path or "",
            "s3_key_prefix": self.s3_key_prefix or "",
            "set_processed_dir_from_s3": self.set_processed_dir_from_s3,
            "processed_subpath": self.processed_subpath or "",
        }

    def kill_running(self) -> bool:
        # We currently run synchronously through `app.run()` + `remote()`.
        # Cooperative cancellation isn't wired yet for Modal backends.
        return False


__all__ = ["ModalExecutor"]
