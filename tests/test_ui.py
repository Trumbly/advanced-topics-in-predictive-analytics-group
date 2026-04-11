"""Tests for the FastAPI dashboard in `agent.ui`.

These tests build a tiny on-disk study tree in `tmp_path`, point the
app factory at it, and hit the HTML + JSON routes through a Starlette
TestClient. No network, no orchestrator, no live LLM.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from agent.models import (  # noqa: E402
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
from agent.ui.app import create_app  # noqa: E402
from agent.ui.loaders import (  # noqa: E402
    list_studies,
    load_experiment_detail,
    load_study_detail,
)


NOW = datetime(2026, 4, 11, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_study_tree(tmp_path: Path) -> tuple[Path, Path]:
    """Build a small `studies_root` + `sandbox_root` with two studies.

    Study A — 3 experiments: exp_001 succeeds with score 0.73, exp_002
    fails with RuntimeError, exp_003 succeeds with score 0.68.
    Study B — empty, no experiments yet (covers the 'no results' UI path).
    """
    studies_root = tmp_path / "studies"
    sandbox_root = tmp_path / "sandbox"
    studies_root.mkdir()
    sandbox_root.mkdir()

    # --- Study A -----------------------------------------------------------
    study_a = Study(
        study_id="study_a",
        name="Study A",
        hypothesis="Does exp_001 beat the baseline?",
        mode=StudyMode.AUTONOMOUS,
        compute_budget=ComputeBudget(),
        pipeline_config_path=Path("config/pipelines/default_pipeline.yaml"),
        dataset_profile_path=Path("data/processed/dataset_profile.json"),
        model_registry_path=Path("registry/models.yaml"),
        experiment_ids=["exp_001", "exp_002", "exp_003"],
        best_experiment_id="exp_001",
        best_score=0.73,
        status=StudyStatus.COMPLETED,
        created_at=NOW,
        updated_at=NOW,
    )
    (studies_root / "study_a").mkdir()
    (studies_root / "study_a" / "study.json").write_text(
        study_a.model_dump_json()
    )

    experiments_dir_a = studies_root / "study_a" / "experiments"
    experiments_dir_a.mkdir()

    # exp_001 — best, succeeds
    exp1 = Experiment(
        experiment_id="exp_001",
        study_id="study_a",
        llm_model="nemotron-3-nano:4b",
        task_ids=[
            "exp_001_task_01_propose_architecture",
            "exp_001_task_02_generate_code",
            "exp_001_task_03_execute_training",
            "exp_001_task_04_capture_metrics",
        ],
        config=ModelConfig(architecture="[custom_cnn] cnn_small_v1 baseline"),
        results=TrainingResults(
            metrics={"roc_auc_macro": 0.73, "loss": 0.42},
            training_curves={"loss": [0.9, 0.6, 0.42], "roc_auc_macro": [0.5, 0.68, 0.73]},
            duration_seconds=120.5,
        ),
        status=ExperimentStatus.COMPLETED,
        created_at=NOW,
        started_at=NOW,
        completed_at=NOW,
    )
    _write_experiment(experiments_dir_a, exp1)
    _write_task(
        experiments_dir_a / "exp_001" / "tasks",
        Task(
            task_id="exp_001_task_01_propose_architecture",
            experiment_id="exp_001",
            task_type=TaskType.LLM,
            task_name="propose_architecture",
            status=TaskStatus.COMPLETED,
            started_at=NOW,
            completed_at=NOW,
            prompt_used="[system] be smart\n[user] propose an arch",
            llm_response='{"architecture": "[custom_cnn] cnn_small_v1"}',
            output={"architecture_proposal": {"architecture": "cnn_small_v1"}},
        ),
    )

    # exp_002 — fails in execute_training
    exp2 = Experiment(
        experiment_id="exp_002",
        study_id="study_a",
        llm_model="nemotron-3-nano:4b",
        task_ids=["exp_002_task_01_execute_training"],
        config=ModelConfig(architecture="[deep_cnn] 6-block deep CNN"),
        status=ExperimentStatus.FAILED,
        created_at=NOW,
        started_at=NOW,
        completed_at=NOW,
    )
    _write_experiment(experiments_dir_a, exp2)
    _write_task(
        experiments_dir_a / "exp_002" / "tasks",
        Task(
            task_id="exp_002_task_01_execute_training",
            experiment_id="exp_002",
            task_type=TaskType.PREDEFINED,
            task_name="execute_training",
            status=TaskStatus.FAILED,
            started_at=NOW,
            completed_at=NOW,
            error=TaskError(
                error_type="RuntimeError",
                message="shape mismatch at conv2",
                traceback="...",
            ),
        ),
    )

    # exp_003 — succeeds with a lower score than the best
    exp3 = Experiment(
        experiment_id="exp_003",
        study_id="study_a",
        llm_model="nemotron-3-nano:4b",
        task_ids=["exp_003_task_01_execute_training"],
        config=ModelConfig(
            architecture="[efficientnet_b0] TorchvisionAdapter(efficientnet_b0)"
        ),
        results=TrainingResults(
            metrics={"roc_auc_macro": 0.68, "loss": 0.5},
            training_curves={"loss": [0.8, 0.5]},
            duration_seconds=210.0,
        ),
        status=ExperimentStatus.COMPLETED,
        created_at=NOW,
        started_at=NOW,
        completed_at=NOW,
    )
    _write_experiment(experiments_dir_a, exp3)

    # Sandbox artifacts for exp_001 — code.py + stdout + stderr
    sandbox_a = sandbox_root / "study_a" / "exp_001"
    sandbox_a.mkdir(parents=True)
    (sandbox_a / "code.py").write_text("print('hello, birdclef')\n")
    (sandbox_a / "stdout.log").write_text(
        "loading data...\nmodel built: 42 parameters\nall done\n"
    )
    (sandbox_a / "stderr.log").write_text("")

    # --- Study B — empty ---------------------------------------------------
    study_b = Study(
        study_id="study_b",
        name="Study B",
        hypothesis="Empty study — no experiments yet.",
        mode=StudyMode.AUTONOMOUS,
        compute_budget=ComputeBudget(),
        pipeline_config_path=Path("config/pipelines/default_pipeline.yaml"),
        dataset_profile_path=Path("data/processed/dataset_profile.json"),
        model_registry_path=Path("registry/models.yaml"),
        status=StudyStatus.ACTIVE,
        created_at=NOW,
        updated_at=NOW,
    )
    (studies_root / "study_b").mkdir()
    (studies_root / "study_b" / "study.json").write_text(
        study_b.model_dump_json()
    )

    return studies_root, sandbox_root


def _write_experiment(experiments_dir: Path, exp: Experiment) -> None:
    exp_dir = experiments_dir / exp.experiment_id
    exp_dir.mkdir()
    (exp_dir / "experiment.json").write_text(exp.model_dump_json())
    (exp_dir / "tasks").mkdir()


def _write_task(tasks_dir: Path, task: Task) -> None:
    (tasks_dir / f"{task.task_id}.json").write_text(task.model_dump_json())


@pytest.fixture
def on_disk_studies(tmp_path: Path) -> tuple[Path, Path]:
    return _write_study_tree(tmp_path)


@pytest.fixture
def client(on_disk_studies: tuple[Path, Path]) -> TestClient:
    studies_root, sandbox_root = on_disk_studies
    app = create_app(studies_root=studies_root, sandbox_root=sandbox_root)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Loader unit tests (no FastAPI)
# ---------------------------------------------------------------------------


class TestLoaders:
    def test_list_studies_returns_both(
        self, on_disk_studies: tuple[Path, Path]
    ) -> None:
        studies_root, _ = on_disk_studies
        rows = list_studies(studies_root)
        ids = {r.study_id for r in rows}
        assert ids == {"study_a", "study_b"}

    def test_list_studies_counts_completed_and_failed(
        self, on_disk_studies: tuple[Path, Path]
    ) -> None:
        studies_root, _ = on_disk_studies
        a = next(r for r in list_studies(studies_root) if r.study_id == "study_a")
        assert a.experiment_count == 3
        assert a.completed_count == 2
        assert a.failed_count == 1
        assert a.best_experiment_id == "exp_001"
        assert a.best_score == 0.73

    def test_empty_studies_root(self, tmp_path: Path) -> None:
        assert list_studies(tmp_path / "no-such-dir") == []

    def test_study_detail_score_progression(
        self, on_disk_studies: tuple[Path, Path]
    ) -> None:
        studies_root, _ = on_disk_studies
        detail = load_study_detail(studies_root, "study_a")
        assert detail is not None
        assert len(detail.experiments) == 3
        # scores in insertion order: 0.73, None (failed), 0.68
        scores = [p["score"] for p in detail.score_progression]
        assert scores == [0.73, None, 0.68]
        # best_so_far tracks the running max
        best = [p["best_so_far"] for p in detail.score_progression]
        assert best == [0.73, 0.73, 0.73]

    def test_study_detail_failure_breakdown(
        self, on_disk_studies: tuple[Path, Path]
    ) -> None:
        studies_root, _ = on_disk_studies
        detail = load_study_detail(studies_root, "study_a")
        assert detail is not None
        assert detail.failure_breakdown.total_failed == 1
        assert detail.failure_breakdown.by_error_type == {"RuntimeError": 1}

    def test_unknown_study_returns_none(
        self, on_disk_studies: tuple[Path, Path]
    ) -> None:
        studies_root, _ = on_disk_studies
        assert load_study_detail(studies_root, "does_not_exist") is None

    def test_experiment_detail_includes_code_and_logs(
        self, on_disk_studies: tuple[Path, Path]
    ) -> None:
        studies_root, sandbox_root = on_disk_studies
        detail = load_experiment_detail(
            studies_root, sandbox_root, "study_a", "exp_001"
        )
        assert detail is not None
        assert detail.code is not None
        assert "hello, birdclef" in detail.code
        assert detail.stdout_tail is not None
        assert "all done" in detail.stdout_tail
        assert detail.training_curves["loss"] == [0.9, 0.6, 0.42]

    def test_experiment_detail_with_failed_task(
        self, on_disk_studies: tuple[Path, Path]
    ) -> None:
        studies_root, sandbox_root = on_disk_studies
        detail = load_experiment_detail(
            studies_root, sandbox_root, "study_a", "exp_002"
        )
        assert detail is not None
        assert len(detail.tasks) == 1
        assert detail.tasks[0].status == "failed"
        assert detail.tasks[0].error_type == "RuntimeError"
        assert "shape mismatch" in (detail.tasks[0].error_message or "")


# ---------------------------------------------------------------------------
# HTTP smoke tests
# ---------------------------------------------------------------------------


class TestRoutes:
    def test_index_lists_both_studies(self, client: TestClient) -> None:
        r = client.get("/")
        assert r.status_code == 200
        assert "Study A" in r.text
        assert "Study B" in r.text

    def test_index_shows_best_score(self, client: TestClient) -> None:
        r = client.get("/")
        assert "0.7300" in r.text  # formatted best score for Study A

    def test_study_detail_shows_experiments(self, client: TestClient) -> None:
        r = client.get("/studies/study_a")
        assert r.status_code == 200
        assert "exp_001" in r.text
        assert "exp_002" in r.text
        assert "exp_003" in r.text
        assert "Does exp_001 beat the baseline?" in r.text

    def test_study_detail_highlights_best(self, client: TestClient) -> None:
        r = client.get("/studies/study_a")
        # best row has the star marker
        assert "★" in r.text

    def test_study_detail_404_for_unknown(
        self, client: TestClient
    ) -> None:
        assert client.get("/studies/nope").status_code == 404

    def test_experiment_detail_renders_code_and_tasks(
        self, client: TestClient
    ) -> None:
        r = client.get("/studies/study_a/experiments/exp_001")
        assert r.status_code == 200
        assert "hello, birdclef" in r.text
        assert "propose_architecture" in r.text
        assert "0.7300" in r.text

    def test_experiment_detail_404_for_unknown(
        self, client: TestClient
    ) -> None:
        assert (
            client.get("/studies/study_a/experiments/nope").status_code == 404
        )

    def test_score_progression_api(self, client: TestClient) -> None:
        r = client.get("/api/studies/study_a/score_progression")
        assert r.status_code == 200
        body = r.json()
        assert body["metric"] == "roc_auc_macro"
        assert len(body["points"]) == 3
        assert body["points"][0]["score"] == 0.73
        assert body["points"][1]["score"] is None  # failed exp
        assert body["points"][2]["score"] == 0.68

    def test_failure_breakdown_api(self, client: TestClient) -> None:
        r = client.get("/api/studies/study_a/failure_breakdown")
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 1
        assert body["by_error_type"] == {"RuntimeError": 1}

    def test_training_curves_api(self, client: TestClient) -> None:
        r = client.get("/api/studies/study_a/experiments/exp_001/curves")
        assert r.status_code == 200
        body = r.json()
        assert body["loss"] == [0.9, 0.6, 0.42]
        assert body["roc_auc_macro"] == [0.5, 0.68, 0.73]

    def test_healthz(self, client: TestClient) -> None:
        r = client.get("/api/healthz")
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_empty_study_still_renders(self, client: TestClient) -> None:
        """Study B has no experiments — must not 500."""
        r = client.get("/studies/study_b")
        assert r.status_code == 200
        assert "Study B" in r.text
