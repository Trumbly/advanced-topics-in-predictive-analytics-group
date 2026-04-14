"""Unit tests for `agent.logger.ExperimentLogger`."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from agent.logger import ExperimentLogger
from agent.models import (
    ComputeBudget,
    Experiment,
    ExperimentStatus,
    ModelConfig,
    Study,
    StudyMode,
    StudyStatus,
    Task,
    TaskError,
    TaskStatus,
    TaskType,
    TrainingResults,
)


NOW = datetime(2026, 4, 10, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def logger(tmp_path: Path) -> ExperimentLogger:
    return ExperimentLogger(study_dir=tmp_path / "study_demo")


@pytest.fixture
def study() -> Study:
    return Study(
        study_id="study_demo",
        name="Demo Study",
        hypothesis="Test the logger",
        mode=StudyMode.AUTONOMOUS,
        compute_budget=ComputeBudget(max_experiments=3),
        pipeline_config_path=Path("config/pipelines/default_pipeline.yaml"),
        dataset_profile_path=Path("data/processed/dataset_profile.json"),
        model_registry_path=Path("registry/models.yaml"),
        created_at=NOW,
        updated_at=NOW,
    )


@pytest.fixture
def experiment() -> Experiment:
    return Experiment(
        experiment_id="exp_001",
        study_id="study_demo",
        llm_model="gemma4:e4b",
        status=ExperimentStatus.COMPLETED,
        config=ModelConfig(
            architecture="cnn_small_v1",
            hyperparams={"lr": 1e-3, "epochs": 5},
            augmentation={"time_shift": True},
        ),
        results=TrainingResults(
            metrics={"f1_macro": 0.72, "roc_auc_macro": 0.7, "loss": 0.38},
            training_curves={"loss": [0.9, 0.5, 0.38]},
            duration_seconds=123.4,
        ),
        created_at=NOW,
        task_ids=["exp_001_task_01"],
    )


class TestWriteStudy:
    def test_writes_json_and_markdown(
        self, logger: ExperimentLogger, study: Study
    ) -> None:
        logger.write_study(study)
        assert logger.study_json_path.exists()
        assert logger.study_markdown_path.exists()

        md = logger.study_markdown_path.read_text()
        assert "# Study: Demo Study" in md
        assert "study_demo" in md
        assert "Test the logger" in md


class TestWriteExperiment:
    def test_writes_json_and_markdown(
        self, logger: ExperimentLogger, experiment: Experiment
    ) -> None:
        logger.write_experiment(experiment)
        exp_dir = logger.experiment_dir("exp_001")
        assert (exp_dir / "experiment.json").exists()
        assert (exp_dir / "experiment.md").exists()

        md = (exp_dir / "experiment.md").read_text()
        assert "cnn_small_v1" in md
        assert "f1_macro: 0.7200" in md
        assert "loss (len=3)" in md  # curve summary
        assert "duration: 123.4s" in md


class TestWriteTask:
    def test_llm_task(self, logger: ExperimentLogger) -> None:
        task = Task(
            task_id="exp_001_task_01",
            experiment_id="exp_001",
            task_type=TaskType.LLM,
            task_name="propose_architecture",
            status=TaskStatus.COMPLETED,
            prompt_used="Propose an architecture.",
            llm_response="Use a small CNN.",
            output={"architecture": "cnn_small_v1"},
        )
        logger.write_task(task)
        task_dir = logger.task_dir("exp_001")
        assert (task_dir / "exp_001_task_01.json").exists()
        md = (task_dir / "exp_001_task_01.md").read_text()
        assert "Propose an architecture." in md
        assert "Use a small CNN." in md
        assert "architecture" in md

    def test_failed_task(self, logger: ExperimentLogger) -> None:
        task = Task(
            task_id="exp_001_task_03",
            experiment_id="exp_001",
            task_type=TaskType.PREDEFINED,
            task_name="execute_training",
            status=TaskStatus.FAILED,
            code_used="print('hi')",
            error=TaskError(
                error_type="ShapeMismatch",
                message="expected (32, 128), got (32, 64)",
                traceback="Traceback...",
            ),
        )
        logger.write_task(task)
        md = (logger.task_dir("exp_001") / "exp_001_task_03.md").read_text()
        assert "ShapeMismatch" in md
        assert "expected (32, 128)" in md
        assert "Traceback..." in md
