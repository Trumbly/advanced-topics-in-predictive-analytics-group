"""Typed configuration loader.

The single source of truth is ``config/config.yaml``. Task-specific overrides
come from ``config/tasks/<track>.yaml``. This module merges them and returns
a ``Settings`` object.

Nothing in ``lab.core`` is allowed to read ``os.environ`` directly for
task-specific values — all such access goes through this module so the
precedence (CLI > env > task YAML > global YAML) is enforced in one place.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


REPO_ROOT_DEFAULT = Path(__file__).resolve().parent.parent


class LLMConfig(BaseModel):
    provider: str = "ollama"
    base_url: str = "http://localhost:11434/v1"
    default_model: str = "gemma4:e4b"
    api_key: str = "ollama"
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout_seconds: float = 120.0
    retry_attempts: int = 3
    retry_backoff_seconds: float = 5.0


class ComputeBudget(BaseModel):
    max_experiments: int = 20
    max_wallclock_minutes: int = 240
    max_experiment_seconds: int = 1800
    max_epochs_per_run: int = 3
    max_recovery_attempts: int = 5
    max_codegen_retries: int = 5


class ContextConfig(BaseModel):
    max_prompt_tokens: int = 8000
    memory_top_k: int = 5
    memory_format: str = "markdown"


class TrainingConfig(BaseModel):
    device: str = "auto"
    batch_size: int = 64
    num_workers: int = 4
    persistent_workers: bool = True
    prefetch_factor: int = 2


class PathsConfig(BaseModel):
    data_root: str = "data"
    experiments: str = "experiments/studies"
    sandbox: str = "sandbox"
    tasks_dir: str = "config/tasks"
    prompts_dir: str = "config/prompts"
    skeletons_dir: str = "config/skeletons"
    registry_dir: str = "registry"
    reports_dir: str = "reports"


class LoggingConfig(BaseModel):
    level: str = "INFO"
    console: bool = True
    file: str = "experiments/agent.log"
    structured: bool = True


class UIConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8765
    enable_live_sse: bool = True


class PublishingConfig(BaseModel):
    default_publish: bool = False
    default_tags: list[str] = Field(default_factory=list)


class KaggleConfig(BaseModel):
    username: str = ""
    kernel_prefix: str = "lab-exp"
    enable_gpu: bool = True
    enable_internet: bool = False
    # Accelerator ID passed to `kaggle kernels push --accelerator <X>`.
    # Default is NvidiaTeslaT4 — T4 has sm_75 which matches Kaggle's
    # stock PyTorch image, so the kernel trains on GPU out of the box.
    # Empty string = no flag → Kaggle's fallback, currently P100. The
    # bootstrap installs a sm_60 torch on P100 but that adds ~60 s to
    # every cold start; picking T4 explicitly avoids that.
    # Other valid IDs: NvidiaTeslaT4X2, NvidiaTeslaP100,
    # NvidiaTeslaV100, NvidiaTeslaA100, TpuV3-8, TpuV6E8.
    accelerator: str = "NvidiaTeslaT4"
    poll_interval_seconds: int = 30
    poll_timeout_seconds: int = 36_000
    dataset_sources: list[str] = Field(default_factory=list)
    competition_sources: list[str] = Field(default_factory=list)


class ModalConfig(BaseModel):
    app_name: str = "lab-agent"
    image_name: str = "debian_slim"
    pip_packages: list[str] = Field(default_factory=lambda: [
        "torch",
        "torchvision",
        "torchaudio",
        "scikit-learn>=1.3",
        "numpy>=1.24",
        "pandas>=2.0",
        "pyyaml>=6.0",
    ])
    gpu: str = ""
    cpu: float | None = None
    memory_mb: int | None = None
    timeout_seconds: int = 1800
    retries: int = 0
    output_volume_name: str = ""
    output_mount_path: str = "/mnt/output"
    s3_bucket_name: str = ""
    s3_secret_name: str = ""
    s3_endpoint_url: str = ""
    s3_mount_path: str = "/mnt/data"
    s3_key_prefix: str = ""
    set_processed_dir_from_s3: bool = True
    processed_subpath: str = "processed"


class ExecutorConfig(BaseModel):
    """Where the generated training code physically runs."""
    backend: str = "local"                  # local | kaggle | modal
    kaggle: KaggleConfig = Field(default_factory=KaggleConfig)
    modal: ModalConfig = Field(default_factory=ModalConfig)


class Settings(BaseModel):
    """The full merged configuration."""
    project: str = "lab"
    version: str = "2.0.0"
    repo_root: Path = REPO_ROOT_DEFAULT
    default_task: str = "track_b"
    env_prefix: str = "AGENT"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    compute_budget: ComputeBudget = Field(default_factory=ComputeBudget)
    context: ContextConfig = Field(default_factory=ContextConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    publishing: PublishingConfig = Field(default_factory=PublishingConfig)
    executor: ExecutorConfig = Field(default_factory=ExecutorConfig)

    # Opaque bag of task-config values; the adapter consumes them.
    task_config: dict[str, Any] = Field(default_factory=dict)

    # -- convenience --------------------------------------------------------

    def abspath(self, rel: str | Path) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else (self.repo_root / p)

    def env_var(self, name: str) -> str:
        """Return a namespaced env var name, e.g. ``AGENT_EPOCHS``."""
        return f"{self.env_prefix}_{name.upper()}"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level mapping in {path}")
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base``. Non-dict values replace."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _apply_env_overrides(cfg: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Read ``<PREFIX>__section__key`` env vars and patch them into ``cfg``.

    Example: ``AGENT__COMPUTE_BUDGET__MAX_EXPERIMENTS=5`` overrides
    ``compute_budget.max_experiments``.
    """
    prefix_sep = f"{prefix}__"
    for key, raw in os.environ.items():
        if not key.startswith(prefix_sep):
            continue
        path = key[len(prefix_sep):].lower().split("__")
        if len(path) < 2:
            continue
        node: dict[str, Any] = cfg
        for part in path[:-1]:
            nxt = node.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                node[part] = nxt
            node = nxt
        # Best-effort type coercion
        node[path[-1]] = _coerce(raw)
    return cfg


def _coerce(raw: str) -> Any:
    if raw.lower() in {"true", "false"}:
        return raw.lower() == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def load_settings(
    *,
    repo_root: Path | None = None,
    task: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> Settings:
    """Load and merge config. ``task`` selects a task YAML to merge in."""
    root = Path(repo_root).resolve() if repo_root else REPO_ROOT_DEFAULT
    global_cfg = _read_yaml(root / "config" / "config.yaml")

    task_name = task or global_cfg.get("default_task", "track_b")
    task_cfg_path = root / global_cfg.get("paths", {}).get("tasks_dir", "config/tasks") / f"{task_name}.yaml"
    task_cfg = _read_yaml(task_cfg_path)

    # Task-level compute budget overrides the global one when present.
    merged = dict(global_cfg)
    if "compute_budget" in task_cfg:
        merged["compute_budget"] = _deep_merge(
            merged.get("compute_budget", {}), task_cfg["compute_budget"]
        )
    merged["task_config"] = task_cfg

    merged = _apply_env_overrides(merged, merged.get("env_prefix", "AGENT"))
    if overrides:
        merged = _deep_merge(merged, overrides)

    merged["repo_root"] = root
    return Settings.model_validate(merged)


@lru_cache(maxsize=8)
def cached_settings(task: str | None = None) -> Settings:
    return load_settings(task=task)
