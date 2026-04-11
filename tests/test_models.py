"""Unit tests for the Pydantic data models in `agent.models`.

Covers:
- Successful instantiation with minimal required fields
- Rejection of unknown fields (ConfigDict extra="forbid")
- JSON round-trip serialization (to_json_file / from_json_file)
- Enum coercion from strings
- Default values
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from agent.models import (
    ClassStats,
    ComputeBudget,
    ContextSettings,
    DatasetProfile,
    Experiment,
    ExperimentStatus,
    GlobalConfig,
    LLMSettings,
    LoggingSettings,
    ModelConfig,
    ModelRegistryEntry,
    ModelRegistryFile,
    PathSettings,
    PipelineDefinition,
    PipelineStep,
    PromptTemplate,
    Study,
    StudyMode,
    StudyStatus,
    Task,
    TaskError,
    TaskStatus,
    TaskType,
    TrainingResults,
    TrainingSettings,
)
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 4, 10, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def minimal_study(now: datetime) -> Study:
    return Study(
        study_id="study_test_001",
        name="Baseline Test",
        hypothesis="Small CNNs train fast on CPU.",
        mode=StudyMode.AUTONOMOUS,
        compute_budget=ComputeBudget(),
        pipeline_config_path=Path("config/pipelines/default_pipeline.yaml"),
        dataset_profile_path=Path("data/processed/dataset_profile.json"),
        model_registry_path=Path("registry/models.yaml"),
        created_at=now,
        updated_at=now,
    )


@pytest.fixture
def minimal_experiment(now: datetime) -> Experiment:
    return Experiment(
        experiment_id="exp_001",
        study_id="study_test_001",
        llm_model="gemma4:e4b",
        created_at=now,
    )


@pytest.fixture
def minimal_task() -> Task:
    return Task(
        task_id="exp_001_task_01_propose_architecture",
        experiment_id="exp_001",
        task_type=TaskType.LLM,
        task_name="propose_architecture",
    )


# ---------------------------------------------------------------------------
# Study
# ---------------------------------------------------------------------------


class TestStudy:
    def test_minimal_construction(self, minimal_study: Study) -> None:
        assert minimal_study.study_id == "study_test_001"
        assert minimal_study.status == StudyStatus.ACTIVE.value
        assert minimal_study.experiment_ids == []
        assert minimal_study.best_experiment_id is None

    def test_rejects_unknown_fields(self, now: datetime) -> None:
        with pytest.raises(ValidationError):
            Study(
                study_id="x",
                name="x",
                hypothesis="x",
                mode=StudyMode.AUTONOMOUS,
                compute_budget=ComputeBudget(),
                pipeline_config_path=Path("a.yaml"),
                dataset_profile_path=Path("b.json"),
                model_registry_path=Path("c.yaml"),
                created_at=now,
                updated_at=now,
                nonsense_field="should fail",  # type: ignore[call-arg]
            )

    def test_json_roundtrip(self, minimal_study: Study, tmp_path: Path) -> None:
        path = tmp_path / "study.json"
        minimal_study.to_json_file(path)
        loaded = Study.from_json_file(path)
        assert loaded.study_id == minimal_study.study_id
        assert loaded.mode == minimal_study.mode
        assert loaded.created_at == minimal_study.created_at

    def test_status_accepts_string_enum(self, now: datetime) -> None:
        study = Study(
            study_id="s",
            name="n",
            hypothesis="h",
            mode="interactive",  # type: ignore[arg-type]
            compute_budget=ComputeBudget(),
            pipeline_config_path=Path("a"),
            dataset_profile_path=Path("b"),
            model_registry_path=Path("c"),
            created_at=now,
            updated_at=now,
        )
        assert study.mode == "interactive"


# ---------------------------------------------------------------------------
# ComputeBudget
# ---------------------------------------------------------------------------


class TestComputeBudget:
    def test_defaults(self) -> None:
        budget = ComputeBudget()
        assert budget.max_experiments == 20
        assert budget.max_wallclock_minutes == 240
        assert budget.max_experiment_seconds == 7200
        assert budget.max_epochs_per_run == 1

    def test_override(self) -> None:
        budget = ComputeBudget(max_experiments=5, max_epochs_per_run=3)
        assert budget.max_experiments == 5
        assert budget.max_epochs_per_run == 3


# ---------------------------------------------------------------------------
# Experiment & ModelConfig & TrainingResults
# ---------------------------------------------------------------------------


class TestExperiment:
    def test_minimal_construction(self, minimal_experiment: Experiment) -> None:
        assert minimal_experiment.experiment_id == "exp_001"
        assert minimal_experiment.status == ExperimentStatus.PENDING.value
        assert minimal_experiment.config is None
        assert minimal_experiment.results is None

    def test_with_config_and_results(self, minimal_experiment: Experiment) -> None:
        minimal_experiment.config = ModelConfig(
            architecture="cnn_small_v1",
            hyperparams={"lr": 1e-3, "batch_size": 32, "epochs": 5},
            augmentation={"time_shift": True, "mixup": 0.2},
        )
        minimal_experiment.results = TrainingResults(
            metrics={"roc_auc_macro": 0.72, "loss": 0.35},
            training_curves={"loss": [0.9, 0.6, 0.35]},
            duration_seconds=125.4,
        )
        assert minimal_experiment.config.architecture == "cnn_small_v1"
        assert minimal_experiment.results.metrics["roc_auc_macro"] == 0.72

    def test_json_roundtrip(
        self, minimal_experiment: Experiment, tmp_path: Path
    ) -> None:
        minimal_experiment.config = ModelConfig(
            architecture="cnn_small_v1",
            hyperparams={"lr": 1e-3},
        )
        path = tmp_path / "experiment.json"
        minimal_experiment.to_json_file(path)
        loaded = Experiment.from_json_file(path)
        assert loaded.config is not None
        assert loaded.config.architecture == "cnn_small_v1"


class TestTrainingResults:
    def test_defaults(self) -> None:
        r = TrainingResults()
        assert r.metrics == {}
        assert r.training_curves == {}
        assert r.duration_seconds == 0.0
        assert r.peak_ram_mb is None


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


class TestTask:
    def test_minimal(self, minimal_task: Task) -> None:
        assert minimal_task.task_type == TaskType.LLM.value
        assert minimal_task.status == TaskStatus.PENDING.value
        assert minimal_task.output == {}
        assert minimal_task.error is None

    def test_error_attached(self, minimal_task: Task) -> None:
        minimal_task.error = TaskError(
            error_type="ShapeMismatch",
            message="Expected (32, 128, 256), got (32, 64, 256)",
        )
        minimal_task.status = TaskStatus.FAILED
        assert minimal_task.error.error_type == "ShapeMismatch"

    def test_json_roundtrip(self, minimal_task: Task, tmp_path: Path) -> None:
        minimal_task.prompt_used = "Propose an architecture."
        minimal_task.llm_response = "Use a 3-layer CNN."
        minimal_task.output = {"architecture": "cnn_small_v1"}
        path = tmp_path / "task.json"
        minimal_task.to_json_file(path)
        loaded = Task.from_json_file(path)
        assert loaded.prompt_used == "Propose an architecture."
        assert loaded.output == {"architecture": "cnn_small_v1"}


# ---------------------------------------------------------------------------
# DatasetProfile
# ---------------------------------------------------------------------------


class TestDatasetProfile:
    def test_construction(self) -> None:
        profile = DatasetProfile(
            num_classes=234,
            num_samples=35_000,
            spectrogram_shape=(1, 128, 256),
            sample_rate=32_000,
            class_stats=[
                ClassStats(
                    class_id="sp_001", sample_count=500, avg_duration_seconds=7.2
                ),
                ClassStats(
                    class_id="sp_002", sample_count=50, avg_duration_seconds=6.8
                ),
            ],
            imbalance_ratio=10.0,
            min_class_samples=50,
            max_class_samples=500,
            split_strategy="stratified_kfold",
            split_seed=42,
            train_indices=[0, 1, 2],
            val_indices=[3, 4],
        )
        assert profile.num_classes == 234
        assert profile.spectrogram_shape == (1, 128, 256)
        assert len(profile.class_stats) == 2

    def test_json_roundtrip(self, tmp_path: Path) -> None:
        profile = DatasetProfile(
            num_classes=3,
            num_samples=10,
            spectrogram_shape=(1, 64, 128),
            sample_rate=22_050,
            imbalance_ratio=1.0,
            min_class_samples=3,
            max_class_samples=3,
            split_strategy="fixed_split",
            split_seed=42,
        )
        path = tmp_path / "profile.json"
        profile.to_json_file(path)
        loaded = DatasetProfile.from_json_file(path)
        assert loaded.num_classes == 3
        assert loaded.spectrogram_shape == (1, 64, 128)


# ---------------------------------------------------------------------------
# ModelRegistry
# ---------------------------------------------------------------------------


class TestModelRegistry:
    def test_entry(self) -> None:
        entry = ModelRegistryEntry(
            name="efficientnet_b0",
            family="cnn",
            input_shape=(3, 224, 224),
            output_dim=234,
            parameters_millions=5.3,
            pretrained_on="imagenet",
            suitability_notes="Good transfer learning baseline.",
            framework="torch",
            import_snippet="import torchvision.models as m; m.efficientnet_b0()",
        )
        assert entry.name == "efficientnet_b0"
        assert entry.input_shape == (3, 224, 224)

    def test_file_with_multiple_entries(self) -> None:
        registry = ModelRegistryFile(
            models=[
                ModelRegistryEntry(
                    name="cnn_small_v1",
                    family="cnn",
                    input_shape=(1, 128, 256),
                    output_dim=234,
                    parameters_millions=0.5,
                    pretrained_on="none",
                    suitability_notes="Fast baseline.",
                    framework="torch",
                    import_snippet="from pipelines.models import CnnSmallV1",
                ),
                ModelRegistryEntry(
                    name="efficientnet_b0",
                    family="cnn",
                    input_shape=(3, 224, 224),
                    output_dim=234,
                    parameters_millions=5.3,
                    pretrained_on="imagenet",
                    suitability_notes="Transfer learning.",
                    framework="torch",
                    import_snippet="import torchvision.models as m",
                ),
            ]
        )
        assert len(registry.models) == 2
        assert registry.models[0].name == "cnn_small_v1"


# ---------------------------------------------------------------------------
# PromptTemplate
# ---------------------------------------------------------------------------


class TestPromptTemplate:
    def test_construction(self) -> None:
        template = PromptTemplate(
            name="propose_architecture",
            description="LLM proposes a CNN",
            system="You are an ML agent.",
            template="Profile: {dataset_profile}\nMemory: {experiment_memory}",
            slots=["dataset_profile", "experiment_memory"],
            fallback_when_no_memory="Start with a small CNN.",
        )
        assert template.name == "propose_architecture"
        assert template.slots == ["dataset_profile", "experiment_memory"]
        assert template.response_schema is None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class TestPipelineDefinition:
    def test_construction(self) -> None:
        pipeline = PipelineDefinition(
            name="default_birdclef",
            description="Baseline loop",
            steps=[
                PipelineStep(
                    task_name="propose_architecture",
                    task_type=TaskType.LLM,
                    prompt_template=Path("config/prompts/propose_architecture.yaml"),
                ),
                PipelineStep(
                    task_name="execute_training",
                    task_type=TaskType.PREDEFINED,
                    handler="agent.handlers.execute_training",
                    config={"timeout_seconds": 600},
                ),
            ],
        )
        assert pipeline.name == "default_birdclef"
        assert len(pipeline.steps) == 2
        assert pipeline.steps[0].task_type == TaskType.LLM.value
        assert pipeline.steps[1].config["timeout_seconds"] == 600


# ---------------------------------------------------------------------------
# GlobalConfig
# ---------------------------------------------------------------------------


class TestTrainingSettings:
    def test_defaults(self) -> None:
        t = TrainingSettings()
        assert t.batch_size == 128
        assert t.num_workers is None  # auto-tune
        assert t.persistent_workers is True
        assert t.prefetch_factor == 4

    def test_override(self) -> None:
        t = TrainingSettings(
            batch_size=256, num_workers=10, persistent_workers=False, prefetch_factor=8
        )
        assert t.batch_size == 256
        assert t.num_workers == 10
        assert t.persistent_workers is False
        assert t.prefetch_factor == 8


class TestGlobalConfig:
    def test_defaults(self) -> None:
        config = GlobalConfig()
        assert config.project == "birdclef-2026-agent"
        assert config.llm.default_model == "gemma4:e4b"
        assert config.llm.temperature == 0.7
        assert config.compute_budget.max_experiments == 20
        assert config.context.memory_format == "markdown"
        assert config.paths.data_raw == Path("data/raw")
        assert config.training.batch_size == 128

    def test_training_section_can_be_loaded_from_real_config(
        self, tmp_path: Path
    ) -> None:
        """The checked-in config/config.yaml contains a `training:` block
        and must validate against GlobalConfig."""
        import yaml as _yaml

        data = _yaml.safe_load(Path("config/config.yaml").read_text())
        gc = GlobalConfig.model_validate(data)
        assert gc.training.batch_size >= 1
        assert gc.training.prefetch_factor >= 1

    def test_override_nested(self) -> None:
        config = GlobalConfig(
            llm=LLMSettings(default_model="qwen3:9b", temperature=0.3),
            context=ContextSettings(max_prompt_tokens=4000, memory_top_k=3),
        )
        assert config.llm.default_model == "qwen3:9b"
        assert config.llm.temperature == 0.3
        assert config.context.max_prompt_tokens == 4000

    def test_json_roundtrip(self, tmp_path: Path) -> None:
        config = GlobalConfig(
            logging=LoggingSettings(level="DEBUG", file=Path("x.log")),
            paths=PathSettings(sandbox=Path("/tmp/sandbox")),
        )
        path = tmp_path / "config.json"
        config.to_json_file(path)
        loaded = GlobalConfig.from_json_file(path)
        assert loaded.logging.level == "DEBUG"
        assert loaded.paths.sandbox == Path("/tmp/sandbox")
