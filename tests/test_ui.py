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
    import os
    import time as _time

    sandbox_a = sandbox_root / "study_a" / "exp_001"
    sandbox_a.mkdir(parents=True)
    (sandbox_a / "code.py").write_text("print('hello, birdclef')\n")
    (sandbox_a / "stdout.log").write_text(
        "loading data...\nmodel built: 42 parameters\nall done\n"
    )
    (sandbox_a / "stderr.log").write_text("")
    # Set old mtime so the live-monitoring loader does NOT treat these
    # completed experiments as "currently running".
    old = _time.time() - 3600  # 1 hour ago
    for f in sandbox_a.iterdir():
        os.utime(f, (old, old))

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

    def test_study_detail_experiment_counts(
        self, on_disk_studies: tuple[Path, Path]
    ) -> None:
        """`completed_count` / `failed_count` are pre-computed on the
        loader side because the Jinja `selectattr('status.value', ...)`
        filter doesn't behave consistently across Enum / str instances."""
        studies_root, _ = on_disk_studies
        detail = load_study_detail(studies_root, "study_a")
        assert detail is not None
        assert detail.completed_count == 2
        assert detail.failed_count == 1

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


# ---------------------------------------------------------------------------
# Kaggle submission
# ---------------------------------------------------------------------------
#
# The Kaggle button on the study-detail page POSTs to
#   /api/studies/{id}/submit
# which wraps SubmissionExporter.export_best. These tests verify the
# whole flow against the fake study tree: happy path (new notebook
# file created), 404 for unknown studies, 409 when there's no
# best_experiment_id.


class TestKaggleSubmission:
    def test_submit_happy_path_creates_notebook(
        self, on_disk_studies: tuple[Path, Path], client: TestClient
    ) -> None:
        """Study A has best_experiment_id=exp_001 and a sandbox code.py
        that passes SubmissionExporter validation. POSTing to
        /api/studies/study_a/submit must write a .ipynb file under
        study_a/submissions/ and return its path."""
        studies_root, _ = on_disk_studies
        r = client.post("/api/studies/study_a/submit")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["ok"] is True
        assert body["best_experiment_id"] == "exp_001"
        assert body["submission_filename"].endswith(".ipynb")
        assert "exp_001" in body["submission_filename"]

        # Notebook file exists on disk.
        submission_path = Path(body["submission_path"])
        assert submission_path.exists()
        assert submission_path.parent == studies_root / "study_a" / "submissions"

        # study.json was updated with the new submission entry.
        study_json = json.loads(
            (studies_root / "study_a" / "study.json").read_text()
        )
        assert len(study_json["submissions"]) == 1

    def test_submit_404_for_unknown_study(
        self, client: TestClient
    ) -> None:
        r = client.post("/api/studies/does_not_exist/submit")
        assert r.status_code == 404

    def test_submit_409_when_no_best(
        self, client: TestClient
    ) -> None:
        """Study B has no best_experiment_id yet — submit must refuse
        with HTTP 409 Conflict."""
        r = client.post("/api/studies/study_b/submit")
        assert r.status_code == 409

    def test_submissions_appear_in_loader_after_export(
        self,
        on_disk_studies: tuple[Path, Path],
        client: TestClient,
    ) -> None:
        """After a successful POST, `load_study_detail().submissions`
        must list the new notebook — that's what the study detail
        page reads to render the Kaggle submissions card."""
        studies_root, _ = on_disk_studies

        # Before export
        detail = load_study_detail(studies_root, "study_a")
        assert detail is not None
        assert len(detail.submissions) == 0

        # Export
        r = client.post("/api/studies/study_a/submit")
        assert r.status_code == 200

        # After export
        detail2 = load_study_detail(studies_root, "study_a")
        assert detail2 is not None
        assert len(detail2.submissions) == 1
        sub = detail2.submissions[0]
        assert sub.filename.endswith(".ipynb")
        assert sub.size_bytes > 0

    def test_study_detail_page_renders_kaggle_button_when_best_exists(
        self, client: TestClient
    ) -> None:
        """The Kaggle button is only rendered when `best_experiment_id`
        is set — study_a has it, study_b doesn't."""
        r = client.get("/studies/study_a")
        assert "Export Kaggle submission" in r.text

        r = client.get("/studies/study_b")
        assert "Export Kaggle submission" not in r.text


# ---------------------------------------------------------------------------
# Live monitoring
# ---------------------------------------------------------------------------
#
# HTMX partials + JSON endpoints that auto-refresh while a study is
# running. These test both the "no running experiment" case (our fake
# study tree is static) and the "fresh sandbox log" case.


class TestLiveMonitoring:
    def test_running_json_returns_false_for_completed_study(
        self, client: TestClient
    ) -> None:
        """Study A is completed — no experiment is running."""
        r = client.get("/api/studies/study_a/running")
        assert r.status_code == 200
        assert r.json()["running"] is False

    def test_running_partial_renders_empty_when_nothing_running(
        self, client: TestClient
    ) -> None:
        """The partial is an empty div with the polling trigger — it
        will auto-fill when an experiment starts."""
        r = client.get("/partials/studies/study_a/running")
        assert r.status_code == 200
        assert "running-card" in r.text
        # no "Currently running" label since nothing is running
        assert "Currently running" not in r.text

    def test_stats_partial_renders_correctly(
        self, client: TestClient
    ) -> None:
        r = client.get("/partials/studies/study_a/stats")
        assert r.status_code == 200
        assert "study-stats" in r.text

    def test_stdout_partial_returns_content(
        self, client: TestClient
    ) -> None:
        """exp_001 in our fake tree has a stdout.log with 'all done'."""
        r = client.get(
            "/partials/studies/study_a/experiments/exp_001/stdout"
        )
        assert r.status_code == 200
        assert "all done" in r.text

    def test_stdout_partial_for_missing_log(
        self, client: TestClient
    ) -> None:
        """exp_002 has no sandbox directory — must return a placeholder,
        not 500."""
        r = client.get(
            "/partials/studies/study_a/experiments/exp_002/stdout"
        )
        assert r.status_code == 200
        assert "no stdout output" in r.text

    def test_study_detail_includes_htmx_polling(
        self, client: TestClient
    ) -> None:
        """The study detail page must contain HTMX `hx-get` polling
        attributes so the stat cards and running card auto-refresh."""
        r = client.get("/studies/study_a")
        assert r.status_code == 200
        assert "hx-trigger" in r.text
        assert "hx-get" in r.text
        assert "running-card" in r.text

    def test_experiment_detail_has_live_stdout(
        self, client: TestClient
    ) -> None:
        """The experiment detail page must poll the stdout partial."""
        r = client.get("/studies/study_a/experiments/exp_001")
        assert r.status_code == 200
        assert "stdout-tail" in r.text
        assert "hx-trigger" in r.text

    def test_running_json_with_fresh_log(
        self, on_disk_studies: tuple[Path, Path]
    ) -> None:
        """Simulate a running experiment: create a fresh stdout.log
        with a recent mtime. The loader should detect it as running."""
        import time

        from agent.ui.loaders import find_running_experiment

        studies_root, sandbox_root = on_disk_studies

        # Create a "currently writing" sandbox log for exp_003
        sandbox_003 = sandbox_root / "study_a" / "exp_003"
        sandbox_003.mkdir(parents=True, exist_ok=True)
        log = sandbox_003 / "stdout.log"
        log.write_text("epoch 1/5 starting...\n")
        # Touch to ensure mtime is NOW
        import os

        os.utime(log, None)

        result = find_running_experiment(studies_root, sandbox_root, "study_a")
        assert result is not None
        assert result.experiment_id == "exp_003"
        assert "epoch 1/5" in result.stdout_tail

    def test_running_returns_none_for_stale_log(
        self, on_disk_studies: tuple[Path, Path]
    ) -> None:
        """A log that hasn't been written to in >60s is NOT running."""
        import os
        import time

        from agent.ui.loaders import find_running_experiment

        studies_root, sandbox_root = on_disk_studies

        # Create a sandbox log but set mtime 120s ago
        sandbox_004 = sandbox_root / "study_a" / "exp_stale"
        sandbox_004.mkdir(parents=True, exist_ok=True)
        log = sandbox_004 / "stdout.log"
        log.write_text("stale\n")
        stale_time = time.time() - 120
        os.utime(log, (stale_time, stale_time))

        result = find_running_experiment(studies_root, sandbox_root, "study_a")
        # Should be None (exp_001 also has a log but its mtime is old too)
        assert result is None


# ---------------------------------------------------------------------------
# Start / Stop controls
# ---------------------------------------------------------------------------


class TestParseAgentStatus:
    """Test the orchestrator.log parser that feeds the running card."""

    def test_no_log_returns_starting(self, tmp_path: Path) -> None:
        from agent.ui.loaders import parse_agent_status

        assert parse_agent_status(tmp_path / "no_such_dir") == "(starting...)"

    def test_empty_log_returns_starting(self, tmp_path: Path) -> None:
        from agent.ui.loaders import parse_agent_status

        (tmp_path / "orchestrator.log").write_text("")
        assert parse_agent_status(tmp_path) == "(starting...)"

    def test_detects_experiment_header(self, tmp_path: Path) -> None:
        from agent.ui.loaders import parse_agent_status

        (tmp_path / "orchestrator.log").write_text(
            "==============\n"
            "Starting study: test\n\n"
            "──── Experiment 3/25 ────\n"
        )
        assert "3/25" in parse_agent_status(tmp_path)

    def test_detects_task_start(self, tmp_path: Path) -> None:
        from agent.ui.loaders import parse_agent_status

        (tmp_path / "orchestrator.log").write_text(
            "──── Experiment 2/10 ────\n"
            "  [exp_002] generate_code (llm) ...\n"
        )
        result = parse_agent_status(tmp_path)
        assert "exp_002" in result
        assert "generate_code" in result
        assert "llm" in result

    def test_detects_error_recovery(self, tmp_path: Path) -> None:
        from agent.ui.loaders import parse_agent_status

        (tmp_path / "orchestrator.log").write_text(
            "  [exp_005] attempting error recovery 2/3\n"
        )
        result = parse_agent_status(tmp_path)
        assert "exp_005" in result
        assert "error recovery" in result
        assert "2/3" in result

    def test_detects_study_finished(self, tmp_path: Path) -> None:
        from agent.ui.loaders import parse_agent_status

        (tmp_path / "orchestrator.log").write_text(
            "Study finished: MAX_EXPERIMENTS\n"
        )
        assert parse_agent_status(tmp_path) == "Study finished"

    def test_detects_promotion(self, tmp_path: Path) -> None:
        from agent.ui.loaders import parse_agent_status

        (tmp_path / "orchestrator.log").write_text(
            "──── Promoting exp_007 (smoke roc_auc_macro=0.56) ────\n"
        )
        result = parse_agent_status(tmp_path)
        assert "Promoting" in result
        assert "exp_007" in result

    def test_most_recent_status_wins(self, tmp_path: Path) -> None:
        """When multiple status lines exist, the LAST one wins."""
        from agent.ui.loaders import parse_agent_status

        (tmp_path / "orchestrator.log").write_text(
            "──── Experiment 1/5 ────\n"
            "  [exp_001] propose_architecture (llm) ...\n"
            "    ✓ propose_architecture (5.6s)\n"
            "  [exp_001] generate_code (llm) ...\n"
        )
        result = parse_agent_status(tmp_path)
        assert "generate_code" in result


class TestStartStopControls:
    def test_process_status_returns_false_when_no_pidfile(
        self, client: TestClient
    ) -> None:
        r = client.get("/api/studies/study_a/process")
        assert r.status_code == 200
        assert r.json()["has_process"] is False

    def test_stop_404_for_unknown_study(
        self, client: TestClient
    ) -> None:
        r = client.post("/api/studies/does_not_exist/stop")
        assert r.status_code == 404

    def test_stop_404_when_no_running_process(
        self, client: TestClient
    ) -> None:
        """Study A exists but has no pidfile → no process to stop."""
        r = client.post("/api/studies/study_a/stop")
        assert r.status_code == 404

    def test_start_rejects_empty_name(
        self, client: TestClient
    ) -> None:
        r = client.post(
            "/api/studies/start",
            json={"name": "", "hypothesis": "test"},
        )
        assert r.status_code == 422

    def test_index_page_has_new_study_button(
        self, client: TestClient
    ) -> None:
        r = client.get("/")
        assert r.status_code == 200
        assert "New study" in r.text
        assert "new-study-form" in r.text

    def test_study_detail_shows_stop_for_active_study(
        self, client: TestClient
    ) -> None:
        """study_b has status='active' — it should show the Stop button."""
        r = client.get("/studies/study_b")
        assert r.status_code == 200
        assert "Stop" in r.text or "stop-btn" in r.text

    def test_study_detail_hides_stop_for_completed_study(
        self, client: TestClient
    ) -> None:
        """study_a has status='completed' — no Stop button (the JS
        handler text may still be in the page, but the actual
        `<button id="stop-btn">` must not be rendered)."""
        r = client.get("/studies/study_a")
        # The actual button element is gated by status==active/running.
        # For completed studies, the template skips the button.
        assert 'id="stop-btn"' not in r.text
