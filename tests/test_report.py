"""Unit tests for `agent.report.ReportGenerator`.

These tests exercise the full report generation flow against a
synthetic Study directory: we create minimal study.json +
experiments/<id>/experiment.json files on disk, then run the generator
with a mock LLM backend, and assert the output (figures + markdown).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from agent.llm_client import LLMClient
from agent.memory import ExperimentMemory
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
from agent.prompt_engine import PromptEngine
from agent.report import ReportGenerator

NOW = datetime(2026, 4, 10, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Mock LLM backend
# ---------------------------------------------------------------------------


class MockReportBackend:
    """Returns a fixed markdown response regardless of input."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return {"choices": [{"message": {"content": self.response}}]}


class FailingLLMBackend:
    """Backend that always raises — used to test the placeholder path."""

    def create(self, **kwargs: Any) -> Any:
        raise ConnectionError("mock LLM unavailable")


# ---------------------------------------------------------------------------
# Synthetic study builder
# ---------------------------------------------------------------------------


def _make_study_dir(
    tmp_path: Path,
    *,
    n_success: int = 2,
    n_fail: int = 1,
    multi_epoch_best: bool = False,
) -> tuple[Path, Study]:
    """Build a synthetic study on disk with the given number of successes
    and failures. Returns (study_dir, study)."""
    study_dir = tmp_path / "studies" / "study_test_report"
    study_dir.mkdir(parents=True)
    experiments_root = study_dir / "experiments"

    study = Study(
        study_id="study_test_report",
        name="Report Test",
        hypothesis="Verify ReportGenerator builds a markdown report + figures",
        mode=StudyMode.AUTONOMOUS,
        compute_budget=ComputeBudget(
            max_experiments=10,
            max_wallclock_minutes=60,
            max_experiment_seconds=300,
            max_epochs_per_run=1,
            max_recovery_attempts=2,
        ),
        pipeline_config_path=Path("config/pipelines/default_pipeline.yaml"),
        dataset_profile_path=Path("data/processed/dataset_profile.json"),
        model_registry_path=Path("registry/models.yaml"),
        status=StudyStatus.COMPLETED,
        created_at=NOW,
        updated_at=NOW,
    )

    exp_idx = 0
    score_progression = [0.62, 0.78, 0.81]  # increasing successes

    for i in range(n_success):
        exp_idx += 1
        exp_id = f"exp_{exp_idx:03d}"
        score = score_progression[i % len(score_progression)]
        loss = 1.0 - score
        curves: dict[str, list[float]] = {
            "loss": [loss],
            "roc_auc_macro": [score],
        }
        if multi_epoch_best and i == n_success - 1:
            curves = {
                "loss": [0.9, 0.6, loss],
                "roc_auc_macro": [0.55, 0.7, score],
            }
        exp = Experiment(
            experiment_id=exp_id,
            study_id=study.study_id,
            llm_model="gemma4:e4b",
            status=ExperimentStatus.COMPLETED,
            config=ModelConfig(
                architecture="cnn_small_v1" if i == 0 else "custom_cnn_with_se",
                hyperparams={"lr": 1e-3, "batch_size": 32, "epochs": 1},
                augmentation={"time_shift": True},
            ),
            results=TrainingResults(
                metrics={"roc_auc_macro": score, "loss": loss},
                training_curves=curves,
                duration_seconds=120.0 + 10 * i,
            ),
            created_at=NOW,
        )
        (experiments_root / exp_id).mkdir(parents=True)
        exp.to_json_file(experiments_root / exp_id / "experiment.json")
        study.experiment_ids.append(exp_id)
        if study.best_score is None or score > study.best_score:
            study.best_score = score
            study.best_experiment_id = exp_id

    # Failed experiments — each gets a task.json with a different error type
    failure_error_types = ["ShapeMismatch", "AttributeError", "EpochsCapExceeded"]
    for i in range(n_fail):
        exp_idx += 1
        exp_id = f"exp_{exp_idx:03d}"
        err_type = failure_error_types[i % len(failure_error_types)]
        exp = Experiment(
            experiment_id=exp_id,
            study_id=study.study_id,
            llm_model="gemma4:e4b",
            status=ExperimentStatus.FAILED,
            config=ModelConfig(
                architecture=f"broken_arch_{i}",
                hyperparams={"lr": 1e-3},
            ),
            created_at=NOW,
        )
        (experiments_root / exp_id).mkdir(parents=True)
        exp.to_json_file(experiments_root / exp_id / "experiment.json")
        # Drop a task.json with the failure
        tasks_dir = experiments_root / exp_id / "tasks"
        tasks_dir.mkdir()
        task = Task(
            task_id=f"{exp_id}_task_01_execute_training",
            experiment_id=exp_id,
            task_type=TaskType.PREDEFINED,
            task_name="execute_training",
            status=TaskStatus.FAILED,
            error=TaskError(error_type=err_type, message=f"mock {err_type} detail"),
        )
        task.to_json_file(tasks_dir / f"{task.task_id}.json")
        study.experiment_ids.append(exp_id)

    # Persist the study itself
    study.to_json_file(study_dir / "study.json")
    return study_dir, study


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def prompt_engine() -> PromptEngine:
    return PromptEngine()


@pytest.fixture
def llm_client_mock() -> LLMClient:
    backend = MockReportBackend(
        response="# Study: Report Test\n\n"
        "## Executive Summary\nMocked summary.\n\n"
        "## Methodology\nMock.\n\n"
        "## Results\n![Score progression](figures/score_progression.png)\n\n"
        "## Best Experiment\nMock best.\n\n"
        "## Failure Analysis\n![Failures](figures/failure_breakdown.png)\n\n"
        "## Lessons Learned\n- mock lesson\n\n"
        "## Next Steps\n- mock step\n"
    )
    return LLMClient(backend=backend, retry_backoff_seconds=0.0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestReportGenerator:
    def test_generates_report_md_and_figures(
        self,
        tmp_path: Path,
        prompt_engine: PromptEngine,
        llm_client_mock: LLMClient,
    ) -> None:
        study_dir, study = _make_study_dir(tmp_path, n_success=2, n_fail=1)
        memory = ExperimentMemory(study_dir=study_dir)
        gen = ReportGenerator(
            study=study,
            memory=memory,
            llm_client=llm_client_mock,
            prompt_engine=prompt_engine,
            study_dir=study_dir,
        )
        report_path = gen.generate()

        assert report_path.exists()
        assert report_path.name == "report.md"
        body = report_path.read_text()
        assert "# Study: Report Test" in body
        assert "Executive Summary" in body

        figures_dir = report_path.parent / "figures"
        assert (figures_dir / "score_progression.png").exists()
        assert (figures_dir / "failure_breakdown.png").exists()
        # 1-epoch study → no learning curve file
        assert not (figures_dir / "best_learning_curve.png").exists()

    def test_multi_epoch_best_renders_learning_curve(
        self,
        tmp_path: Path,
        prompt_engine: PromptEngine,
        llm_client_mock: LLMClient,
    ) -> None:
        study_dir, study = _make_study_dir(
            tmp_path, n_success=2, n_fail=0, multi_epoch_best=True
        )
        memory = ExperimentMemory(study_dir=study_dir)
        gen = ReportGenerator(
            study=study,
            memory=memory,
            llm_client=llm_client_mock,
            prompt_engine=prompt_engine,
            study_dir=study_dir,
        )
        report_path = gen.generate()
        figures_dir = report_path.parent / "figures"
        assert (figures_dir / "best_learning_curve.png").exists()
        # No failures → no failure breakdown figure
        assert not (figures_dir / "failure_breakdown.png").exists()

    def test_llm_failure_produces_placeholder(
        self, tmp_path: Path, prompt_engine: PromptEngine
    ) -> None:
        study_dir, study = _make_study_dir(tmp_path, n_success=1, n_fail=1)
        memory = ExperimentMemory(study_dir=study_dir)
        failing = LLMClient(
            backend=FailingLLMBackend(),
            retry_attempts=1,
            retry_backoff_seconds=0.0,
        )
        gen = ReportGenerator(
            study=study,
            memory=memory,
            llm_client=failing,
            prompt_engine=prompt_engine,
            study_dir=study_dir,
        )
        report_path = gen.generate()
        body = report_path.read_text()
        assert "placeholder" in body.lower()
        # Figures are still rendered even if LLM fails
        assert (report_path.parent / "figures" / "score_progression.png").exists()
        assert (report_path.parent / "figures" / "failure_breakdown.png").exists()

    def test_strips_wrapping_markdown_fence(
        self, tmp_path: Path, prompt_engine: PromptEngine
    ) -> None:
        """If the LLM wraps its whole response in ```markdown ... ``` the
        outer fence must be stripped so the report renders cleanly."""
        fenced = (
            "```markdown\n"
            "# Study: Report Test\n\n"
            "## Executive Summary\nFenced.\n"
            "```"
        )
        client = LLMClient(
            backend=MockReportBackend(fenced),
            retry_backoff_seconds=0.0,
        )
        study_dir, study = _make_study_dir(tmp_path, n_success=1, n_fail=0)
        memory = ExperimentMemory(study_dir=study_dir)
        gen = ReportGenerator(
            study=study,
            memory=memory,
            llm_client=client,
            prompt_engine=prompt_engine,
            study_dir=study_dir,
        )
        body = gen.generate().read_text()
        assert body.startswith("# Study: Report Test")
        assert "```markdown" not in body

    def test_figures_use_green_for_success_red_for_failure(
        self, tmp_path: Path, prompt_engine: PromptEngine, llm_client_mock: LLMClient
    ) -> None:
        """Sanity check the score_progression figure is readable as a PNG."""
        study_dir, study = _make_study_dir(tmp_path, n_success=2, n_fail=2)
        memory = ExperimentMemory(study_dir=study_dir)
        gen = ReportGenerator(
            study=study,
            memory=memory,
            llm_client=llm_client_mock,
            prompt_engine=prompt_engine,
            study_dir=study_dir,
        )
        gen.generate()
        fig_path = study_dir / "report" / "figures" / "score_progression.png"
        assert fig_path.exists()
        # Valid PNG magic bytes
        assert fig_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    def test_format_experiment_table_contains_every_row(
        self, tmp_path: Path, prompt_engine: PromptEngine, llm_client_mock: LLMClient
    ) -> None:
        study_dir, study = _make_study_dir(tmp_path, n_success=3, n_fail=2)
        memory = ExperimentMemory(study_dir=study_dir)
        gen = ReportGenerator(
            study=study,
            memory=memory,
            llm_client=llm_client_mock,
            prompt_engine=prompt_engine,
            study_dir=study_dir,
        )
        summaries = gen._collect_summaries()
        table = gen._format_experiment_table(summaries)
        for i in range(1, 6):
            assert f"exp_{i:03d}" in table
        # Header is present
        assert "| Exp | Status | Architecture" in table
