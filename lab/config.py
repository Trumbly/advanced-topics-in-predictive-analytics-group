"""Settings loader. YAML-only, no env-var or runtime overrides (ADR-002)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

_STRICT = ConfigDict(extra="forbid")


class LLMConfig(BaseModel):
    model_config = _STRICT

    provider: Literal["ollama", "openai", "anthropic"]
    base_url: str
    model: str
    temperature: float = Field(ge=0.0, le=2.0)
    max_tokens: int = Field(gt=0)
    retry_attempts: int = Field(ge=0)
    retry_backoff_seconds: float = Field(ge=0.0)
    api_key: str | None = None
    # Per-request HTTP timeout for LLM chat() calls. Larger Ollama models
    # on CPU can need 5+ minutes per response, so the default tolerates
    # that; the settings page lets you tighten it for fast remote APIs.
    timeout_seconds: float = Field(default=300.0, gt=0.0, le=3600.0)


class ComputeBudget(BaseModel):
    model_config = _STRICT

    max_experiments: int = Field(gt=0)
    max_wallclock_minutes: int = Field(gt=0)
    max_experiment_seconds: int = Field(gt=0)
    max_epochs_per_run: int = Field(gt=0)
    max_codegen_retries: int = Field(ge=0)
    max_recovery_attempts: int = Field(ge=0)
    lr_min: float = Field(gt=0.0)
    lr_max: float = Field(gt=0.0)
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    num_workers: int = Field(ge=0, default=0)
    batch_size: int = Field(gt=0, default=8)
    data_subset_percent: int = Field(default=100, ge=10, le=100)

    @field_validator("data_subset_percent")
    @classmethod
    def _step_of_ten(cls, v: int) -> int:
        if v < 10 or v > 100:
            raise ValueError("data_subset_percent must be between 10 and 100 inclusive")
        if v % 10 != 0:
            raise ValueError(f"data_subset_percent must be a multiple of 10, got {v}")
        return v


class AgentConfig(BaseModel):
    model_config = _STRICT

    memory_enabled: bool = False
    personality: Literal["exploratory", "conservative"] = "exploratory"


class ContextConfig(BaseModel):
    model_config = _STRICT

    max_prompt_tokens: int = Field(gt=0)
    memory_top_k: int = Field(ge=0)
    recent_failures: int = Field(ge=0)


class LoggingConfig(BaseModel):
    model_config = _STRICT

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    structured: bool = True


class PathsConfig(BaseModel):
    model_config = _STRICT

    data_root: str
    experiments_dir: str
    sandbox: str
    prompts_dir: str
    offline_weights: str


class UIConfig(BaseModel):
    model_config = _STRICT

    host: str = "127.0.0.1"
    port: int = Field(gt=0, lt=65536, default=8000)


class TaskConfig(BaseModel):
    """Task-specific configuration. Adapter import path drives instantiation."""

    model_config = _STRICT

    name: str
    kind: str
    adapter: str  # "module.path:ClassName"
    primary_metric: str
    skeleton_path: str
    raw_data_dir: str
    processed_data_dir: str
    expected_num_classes: int = Field(gt=0)
    input_tensor_shape: list[int]
    valid_architecture_families: list[str]
    task_description: str
    # Optional embedding-family bookkeeping. Filled in for tasks that
    # support precomputed-embedding architectures (e.g. perch_embedding).
    # The skeleton + input shape + processed dir are looked up by the
    # adapter when a proposal picks an *_embedding family.
    embedding_skeleton_path: str | None = None
    embedding_input_shapes: dict[str, list[int]] = Field(default_factory=dict)
    embedding_processed_dirs: dict[str, str] = Field(default_factory=dict)


class Settings(BaseModel):
    model_config = _STRICT

    project: str
    version: str
    default_task: str
    task_name: str
    llm: LLMConfig
    compute_budget: ComputeBudget
    agent: AgentConfig
    context: ContextConfig
    logging: LoggingConfig
    paths: PathsConfig
    ui: UIConfig
    task: TaskConfig


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(str(path))
    with path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)
    return loaded or {}


def load_settings(task: str = "track_b", *, repo_root: Path | None = None) -> Settings:
    """Load typed settings from `config/config.yaml` + `config/tasks/{task}.yaml`.

    Raises:
        FileNotFoundError: when either YAML file is missing.
        pydantic.ValidationError: when any required key is missing or wrongly typed.
    """
    root = Path(repo_root) if repo_root is not None else _default_root()
    global_path = root / "config" / "config.yaml"
    task_path = root / "config" / "tasks" / f"{task}.yaml"

    raw = _load_yaml(global_path)
    raw["task"] = _load_yaml(task_path)
    raw["task_name"] = task
    return Settings(**raw)


def _default_root() -> Path:
    return Path(__file__).resolve().parent.parent
