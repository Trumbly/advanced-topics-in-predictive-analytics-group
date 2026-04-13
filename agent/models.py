"""Pydantic v2 data models for the BirdCLEF autonomous research agent.

Every persistent object in the system — Study, Experiment, Task, Config, Profile —
is a Pydantic model. All models use `extra="forbid"` to catch typos early, and
expose `to_json_file` / `from_json_file` helpers for disk persistence.

The relationships look like this:

    Study (top level)
      ├── ComputeBudget
      ├── Experiment[]            (ordered)
      │     ├── ModelConfig
      │     ├── TrainingResults
      │     └── Task[]            (ordered)
      │           └── TaskError
      └── DatasetProfile          (loaded from disk, referenced by path)

Supporting models:
    ModelRegistryEntry            — catalog entry for a pretrained model
    PromptTemplate                — loaded from a YAML file, filled with slots
    PipelineDefinition            — loaded from a YAML file, drives the orchestrator
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class AgentBaseModel(BaseModel):
    """Shared config + JSON persistence helpers for all agent data models."""

    model_config = ConfigDict(
        extra="forbid",
        use_enum_values=True,
        validate_assignment=True,
    )

    def to_json_file(self, path: Path) -> None:
        """Serialize this model to a JSON file (creating parents if needed)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def from_json_file(cls, path: Path) -> "AgentBaseModel":
        """Load this model from a JSON file."""
        path = Path(path)
        return cls.model_validate_json(path.read_text())


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class StudyMode(str, Enum):
    AUTONOMOUS = "autonomous"
    INTERACTIVE = "interactive"


class StudyStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    ABORTED = "aborted"


class ExperimentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class TaskType(str, Enum):
    LLM = "llm"
    PREDEFINED = "predefined"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


# ---------------------------------------------------------------------------
# Study
# ---------------------------------------------------------------------------


class ComputeBudget(AgentBaseModel):
    """Upper bounds the orchestrator enforces on a Study."""

    max_experiments: int = 20
    max_wallclock_minutes: int = 240
    max_experiment_seconds: int = 7200      # 2 hours — real BirdCLEF training is slow
    max_epochs_per_run: int = 1             # fast-iteration mode
    max_recovery_attempts: int = 5          # retries for a failed experiment's code (Layer 2, expensive)
    max_codegen_retries: int = 5            # re-generate code on validation failure (Layer 1, cheap)
    recovery_empty_response_retries: int = 5  # inner LLM retries on empty recovery response
    recovery_min_code_chars: int = 200      # responses shorter than this count as empty

    # --- promotion phase (smoke → promote two-stage pipeline) ---
    #
    # The smoke phase (everything above) runs every experiment at
    # `max_epochs_per_run=1` so we can iterate fast. That 1-epoch score is
    # almost pure noise for 234-class multi-label though — bad signal for
    # comparing architectures. The promotion phase re-runs the top-K
    # smoke-phase experiments with a higher epoch budget so we get a
    # realistic score to report.
    #
    # `promoted_epochs` must be > 1 to enable promotion. `promote_top_k`
    # is the number of smoke-phase experiments re-run at the bigger
    # budget. `promote_min_score` (optional) filters: only experiments
    # whose smoke score exceeds this threshold are promoted.
    promoted_epochs: int = 1                # >1 → enable promotion phase
    promote_top_k: int = 3                  # N best smoke-phase candidates
    promote_min_score: float | None = None  # optional filter on smoke score


class Study(AgentBaseModel):
    """A collective of experiments with shared memory, budget, and pipeline.

    Persisted at `experiments/studies/<study_id>/study.json`. The list of
    experiments is stored by ID only — the full Experiment objects live in
    their own subdirectories under the study.
    """

    study_id: str = Field(
        description="Unique identifier, e.g. 'study_20260411_baseline'"
    )
    name: str
    hypothesis: str = Field(
        description="Plain-text description of what this study tests"
    )
    mode: StudyMode
    compute_budget: ComputeBudget

    pipeline_config_path: Path = Field(
        description="Path to the pipeline YAML definition"
    )
    prompt_template_paths: dict[str, Path] = Field(
        default_factory=dict,
        description="Mapping of task_name → prompt YAML path (overrides pipeline defaults)",
    )
    dataset_profile_path: Path
    model_registry_path: Path

    experiment_ids: list[str] = Field(default_factory=list)
    best_experiment_id: str | None = None
    best_score: float | None = None
    submissions: list[Path] = Field(default_factory=list)

    # Study continuation (Feature 3): build on a predecessor study's memory
    predecessor_study_id: str | None = Field(
        default=None,
        description="ID of the study this one builds upon (inherits memory + report context)",
    )

    status: StudyStatus = StudyStatus.ACTIVE
    created_at: datetime
    updated_at: datetime


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


class ModelConfig(AgentBaseModel):
    """Architecture + hyperparameters proposed by the LLM for one experiment."""

    architecture: str = Field(description="Architecture name or free-form description")
    pretrained_model: str | None = Field(
        default=None,
        description="If set, must match an entry in the ModelRegistry",
    )
    hyperparams: dict[str, Any] = Field(
        default_factory=dict,
        description="lr, batch_size, epochs, optimizer, dropout, etc.",
    )
    augmentation: dict[str, Any] = Field(
        default_factory=dict,
        description="time_shift, noise_injection, mixup, specaugment, etc.",
    )


class TrainingResults(AgentBaseModel):
    """Structured output of a single training run."""

    metrics: dict[str, float] = Field(
        default_factory=dict,
        description="e.g. roc_auc_macro, loss, per_class scores",
    )
    training_curves: dict[str, list[float]] = Field(
        default_factory=dict,
        description="Mapping of metric_name → values per epoch",
    )
    duration_seconds: float = 0.0
    peak_ram_mb: float | None = None
    val_predictions_path: Path | None = None


class Experiment(AgentBaseModel):
    """One iteration of the agent loop.

    Persisted at
    `experiments/studies/<study_id>/experiments/<experiment_id>/experiment.json`.
    """

    experiment_id: str = Field(description="Unique within a study, e.g. 'exp_001'")
    study_id: str
    llm_model: str = Field(
        description="e.g. 'gemma4:e4b' — which LLM drove this experiment"
    )
    task_ids: list[str] = Field(
        default_factory=list, description="Ordered list of Task IDs"
    )

    config: ModelConfig | None = None
    results: TrainingResults | None = None
    submission_path: Path | None = None

    status: ExperimentStatus = ExperimentStatus.PENDING
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


class TaskError(AgentBaseModel):
    """Classified error captured when a task fails."""

    error_type: str = Field(
        description="e.g. 'OOM', 'SyntaxError', 'ShapeMismatch', 'Timeout'"
    )
    message: str
    traceback: str | None = None


class Task(AgentBaseModel):
    """A single step in the pipeline for one experiment.

    Either an LLM call (prompt_used, llm_response) or a predefined handler
    (code_used). Always has a structured output dict.
    """

    task_id: str = Field(description="e.g. 'exp_001_task_01_propose_architecture'")
    experiment_id: str
    task_type: TaskType
    task_name: str = Field(
        description="e.g. 'propose_architecture', 'execute_training'"
    )
    status: TaskStatus = TaskStatus.PENDING

    # LLM task fields
    prompt_used: str | None = None
    llm_response: str | None = None

    # Predefined task fields
    code_used: str | None = None

    # Both
    output: dict[str, Any] = Field(default_factory=dict)
    error: TaskError | None = None

    started_at: datetime | None = None
    completed_at: datetime | None = None


# ---------------------------------------------------------------------------
# Dataset Profile
# ---------------------------------------------------------------------------


class ClassStats(AgentBaseModel):
    """Per-class statistics for a multi-label dataset."""

    class_id: str
    sample_count: int
    avg_duration_seconds: float


class DatasetProfile(AgentBaseModel):
    """Precomputed characterization of the dataset.

    Built once by `pipelines/dataset_profile.py` and referenced by every Study.
    Injected into the LLM prompt so architectures are proposed with full
    knowledge of the data shape and class distribution.
    """

    num_classes: int
    num_samples: int
    spectrogram_shape: tuple[int, int, int] = Field(
        description="(channels, n_mels, time_frames)"
    )
    sample_rate: int
    class_stats: list[ClassStats] = Field(default_factory=list)
    imbalance_ratio: float = Field(description="max_class_samples / min_class_samples")
    min_class_samples: int
    max_class_samples: int
    split_strategy: str = Field(description="e.g. 'stratified_kfold', 'fixed_split'")
    split_seed: int
    train_indices: list[int] = Field(default_factory=list)
    val_indices: list[int] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------


class ModelRegistryEntry(AgentBaseModel):
    """One entry in `registry/models.yaml` — a catalog of available models.

    The LLM selects from this catalog by `name`. We use this to prevent
    hallucinated architectures.

    `output_dim` accepts either an explicit integer (when the model
    hardcodes a class count) or the sentinel string ``"num_classes"``,
    which tells the LLM: "use the actual dataset's num_classes variable,
    do not invent a number". The second form is strongly preferred because
    hardcoded counts broke several experiments when the real dataset had
    206 classes but the registry said 234.
    """

    name: str = Field(description="Unique identifier, e.g. 'efficientnet_b0'")
    family: str = Field(description="e.g. 'cnn', 'transformer', 'audio_specific'")
    input_shape: tuple[int, ...]
    output_dim: int | str = Field(
        description="Integer class count, or the sentinel 'num_classes'"
    )
    parameters_millions: float
    pretrained_on: str = Field(description="e.g. 'imagenet', 'audioset', 'none'")
    suitability_notes: str = Field(description="Free-text guidance for the LLM")
    framework: str = Field(description="'torch' or 'tensorflow'")
    import_snippet: str = Field(
        description="Exact Python code to instantiate the model"
    )

    @field_validator("output_dim")
    @classmethod
    def _output_dim_sentinel(cls, v: int | str) -> int | str:
        if isinstance(v, str) and v != "num_classes":
            raise ValueError(
                f"output_dim string must be exactly 'num_classes', got {v!r}"
            )
        return v


class ModelRegistryFile(AgentBaseModel):
    """Root schema for `registry/models.yaml`."""

    models: list[ModelRegistryEntry]


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------


class PromptTemplate(AgentBaseModel):
    """A YAML prompt template loaded from `config/prompts/*.yaml`."""

    name: str
    description: str
    system: str = Field(description="System-role message")
    template: str = Field(description="User-role template with {slot} placeholders")
    slots: list[str] = Field(default_factory=list, description="Required slot names")
    fallback_when_no_memory: str | None = None
    response_schema: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Prompt versioning (A/B testing)
# ---------------------------------------------------------------------------


class PromptVersionMeta(AgentBaseModel):
    """Metadata for one immutable prompt version.

    Stored in `config/prompts/_registry.yaml` under
    `tasks.<task_name>.versions.<version>`.
    """

    version: str = Field(description="e.g. 'v1', 'v2'")
    created_at: datetime
    created_by: str = ""
    description: str = ""


class PromptTaskEntry(AgentBaseModel):
    """Registry entry for one task's prompt versions."""

    default_version: str = Field(description="e.g. 'v1'")
    versions: dict[str, PromptVersionMeta] = Field(default_factory=dict)


class PromptRegistry(AgentBaseModel):
    """Root schema for `config/prompts/_registry.yaml`.

    Tracks which prompt versions exist per task, which is the default,
    and immutability metadata (created_at, created_by).
    """

    tasks: dict[str, PromptTaskEntry] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pipeline definition
# ---------------------------------------------------------------------------


class PipelineStep(AgentBaseModel):
    """One step in a pipeline — either an LLM call or a predefined handler."""

    task_name: str
    task_type: TaskType
    prompt_template: Path | None = Field(
        default=None,
        description="Path to prompt YAML — required when task_type == LLM",
    )
    handler: str | None = Field(
        default=None,
        description="Dotted import path — required when task_type == PREDEFINED",
    )
    config: dict[str, Any] = Field(default_factory=dict)


class PipelineDefinition(AgentBaseModel):
    """A YAML pipeline loaded from `config/pipelines/*.yaml`."""

    name: str
    description: str
    steps: list[PipelineStep]


# ---------------------------------------------------------------------------
# Global runtime config (config/config.yaml)
# ---------------------------------------------------------------------------


class LLMSettings(AgentBaseModel):
    provider: str = "ollama"
    base_url: str = "http://localhost:11434/v1"
    default_model: str = "gemma4:e4b"
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout_seconds: int = 120
    retry_attempts: int = 3
    retry_backoff_seconds: int = 5


class ContextSettings(AgentBaseModel):
    max_prompt_tokens: int = 8000
    memory_top_k: int = 5
    memory_format: str = "markdown"


class PathSettings(AgentBaseModel):
    data_raw: Path = Path("data/raw")
    data_processed: Path = Path("data/processed")
    dataset_profile: Path = Path("data/processed/dataset_profile.json")
    experiments: Path = Path("experiments/studies")
    models: Path = Path("models")
    sandbox: Path = Path("sandbox")
    model_registry: Path = Path("registry/models.yaml")


class LoggingSettings(AgentBaseModel):
    level: str = "INFO"
    console: bool = True
    file: Path = Path("experiments/agent.log")


class TrainingSettings(AgentBaseModel):
    """Training-time knobs the orchestrator forwards to the sandbox.

    These values are exported to the sandbox subprocess as env vars
    (prefixed `BIRDCLEF_`) and read by `pipelines.data_loader.load_
    precomputed_dataset` and the LLM-generated training script. They
    exist so the whole project can be tuned from one YAML file without
    touching any Python code or prompt templates.

    - `device`: torch device the training script will use. `"auto"`
      resolves to `"mps"` on Apple Silicon with MPS available, else
      `"cpu"`. The `SubmissionExporter` always forces cpu regardless
      of this value so the final Kaggle notebook stays CPU-only.
    - `batch_size`: samples per training step. On CPU, <64 causes BLAS
      to run single-threaded because individual matmul matrices are too
      small. 128+ is the sweet spot. On MPS batch size matters less but
      bigger is still faster.
    - `num_workers`: DataLoader worker processes. None means auto-tune
      as `min(6, max(2, cpu_count // 2))`.
    - `persistent_workers`: keep workers alive across epochs. Critical
      on macOS where spawn start method makes worker startup expensive.
    - `prefetch_factor`: batches each worker buffers ahead of the main
      process. 4 is a safe pipeline depth.
    """

    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    batch_size: int = 512
    num_workers: int | None = None
    persistent_workers: bool = True
    prefetch_factor: int = 4


class GlobalConfig(AgentBaseModel):
    """Root schema for `config/config.yaml`."""

    project: str = "birdclef-2026-agent"
    version: str = "0.1.0"
    paths: PathSettings = Field(default_factory=PathSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    compute_budget: ComputeBudget = Field(default_factory=ComputeBudget)
    context: ContextSettings = Field(default_factory=ContextSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)


# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------


__all__ = [
    "AgentBaseModel",
    # Enums
    "StudyMode",
    "StudyStatus",
    "ExperimentStatus",
    "TaskType",
    "TaskStatus",
    # Study
    "ComputeBudget",
    "Study",
    # Experiment
    "ModelConfig",
    "TrainingResults",
    "Experiment",
    # Task
    "TaskError",
    "Task",
    # Dataset
    "ClassStats",
    "DatasetProfile",
    # Registry
    "ModelRegistryEntry",
    "ModelRegistryFile",
    # Prompt
    "PromptTemplate",
    # Pipeline
    "PipelineStep",
    "PipelineDefinition",
    # Config
    "LLMSettings",
    "ContextSettings",
    "PathSettings",
    "LoggingSettings",
    "TrainingSettings",
    "GlobalConfig",
]
