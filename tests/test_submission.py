"""Unit tests for `agent.submission.SubmissionExporter`."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agent.models import (
    ComputeBudget,
    Experiment,
    ExperimentStatus,
    ModelConfig,
    Study,
    StudyMode,
    StudyStatus,
    TrainingResults,
)
from agent.submission import (
    NoBestExperimentError,
    SubmissionExporter,
    SubmissionValidationError,
)


NOW = datetime(2026, 4, 10, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def study(tmp_path: Path) -> Study:
    return Study(
        study_id="study_demo",
        name="Demo",
        hypothesis="",
        mode=StudyMode.AUTONOMOUS,
        compute_budget=ComputeBudget(max_experiments=5),
        pipeline_config_path=Path("config/pipelines/default_pipeline.yaml"),
        dataset_profile_path=Path("data/processed/dataset_profile.json"),
        model_registry_path=Path("registry/models.yaml"),
        status=StudyStatus.COMPLETED,
        best_experiment_id="exp_001",
        best_score=0.72,
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
        ),
        results=TrainingResults(
            metrics={"roc_auc_macro": 0.72, "loss": 0.38},
            duration_seconds=120.0,
        ),
        created_at=NOW,
    )


@pytest.fixture
def clean_code() -> str:
    return (
        "import torch\n"
        "model = torch.nn.Linear(10, 1)\n"
        "import pandas as pd\n"
        "pd.DataFrame({'id': [1], 'target': [0]}).to_csv('submission.csv', index=False)\n"
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_clean_code_passes(
        self,
        study: Study,
        experiment: Experiment,
        clean_code: str,
        tmp_path: Path,
    ) -> None:
        exporter = SubmissionExporter()
        out = exporter.export(
            study=study,
            experiment=experiment,
            code=clean_code,
            output_path=tmp_path / "submission.ipynb",
        )
        assert out.exists()

    def test_cuda_reference_rejected(
        self,
        study: Study,
        experiment: Experiment,
        tmp_path: Path,
    ) -> None:
        bad_code = "import torch\nmodel = model.cuda()\n"
        exporter = SubmissionExporter(cpu_only=True)
        with pytest.raises(SubmissionValidationError, match="cuda"):
            exporter.export(
                study=study,
                experiment=experiment,
                code=bad_code,
                output_path=tmp_path / "submission.ipynb",
            )

    def test_jax_import_rejected(
        self,
        study: Study,
        experiment: Experiment,
        tmp_path: Path,
    ) -> None:
        bad_code = "import jax\n"
        exporter = SubmissionExporter(cpu_only=True)
        with pytest.raises(SubmissionValidationError, match="jax"):
            exporter.export(
                study=study,
                experiment=experiment,
                code=bad_code,
                output_path=tmp_path / "submission.ipynb",
            )

    def test_empty_code_rejected(
        self,
        study: Study,
        experiment: Experiment,
        tmp_path: Path,
    ) -> None:
        exporter = SubmissionExporter()
        with pytest.raises(SubmissionValidationError, match="empty"):
            exporter.export(
                study=study,
                experiment=experiment,
                code="   ",
                output_path=tmp_path / "submission.ipynb",
            )

    def test_runtime_too_long_rejected(
        self,
        study: Study,
        experiment: Experiment,
        clean_code: str,
        tmp_path: Path,
    ) -> None:
        # Training took 6000s, target runtime is 5400s → should fail
        experiment.results.duration_seconds = 6000  # type: ignore[union-attr]
        exporter = SubmissionExporter(target_runtime_seconds=5400)
        with pytest.raises(SubmissionValidationError, match="exceeds"):
            exporter.export(
                study=study,
                experiment=experiment,
                code=clean_code,
                output_path=tmp_path / "submission.ipynb",
            )


# ---------------------------------------------------------------------------
# Notebook structure
# ---------------------------------------------------------------------------


class TestNotebookStructure:
    def test_notebook_has_expected_cells(
        self,
        study: Study,
        experiment: Experiment,
        clean_code: str,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "submission.ipynb"
        exporter = SubmissionExporter()
        exporter.export(
            study=study,
            experiment=experiment,
            code=clean_code,
            output_path=path,
        )
        nb = json.loads(path.read_text())
        assert nb["nbformat"] == 4
        assert len(nb["cells"]) >= 4
        # Header + config + code header + code + footer
        cell_types = [c["cell_type"] for c in nb["cells"]]
        assert "markdown" in cell_types
        assert "code" in cell_types

    def test_config_cell_contains_study_id(
        self,
        study: Study,
        experiment: Experiment,
        clean_code: str,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "submission.ipynb"
        SubmissionExporter().export(
            study=study,
            experiment=experiment,
            code=clean_code,
            output_path=path,
        )
        text = path.read_text()
        assert "study_demo" in text
        assert "exp_001" in text
        assert "CUDA_VISIBLE_DEVICES" in text

    def test_config_cell_forces_birdclef_device_cpu(
        self,
        study: Study,
        experiment: Experiment,
        clean_code: str,
        tmp_path: Path,
    ) -> None:
        """The submission notebook must pin BIRDCLEF_DEVICE=cpu in its
        config cell — otherwise experiments trained locally on MPS would
        try to run on MPS inside Kaggle (where it does not exist).
        """
        path = tmp_path / "submission.ipynb"
        SubmissionExporter().export(
            study=study,
            experiment=experiment,
            code=clean_code,
            output_path=path,
        )
        nb = json.loads(path.read_text())
        # Find the first code cell — that is the auto-generated config cell.
        config_cell = next(c for c in nb["cells"] if c["cell_type"] == "code")
        source = "".join(config_cell["source"])
        assert 'os.environ["BIRDCLEF_DEVICE"] = "cpu"' in source
        assert 'os.environ["CUDA_VISIBLE_DEVICES"] = ""' in source


# ---------------------------------------------------------------------------
# export_best
# ---------------------------------------------------------------------------


class TestExportBest:
    def test_export_best_happy_path(
        self,
        study: Study,
        experiment: Experiment,
        clean_code: str,
        tmp_path: Path,
    ) -> None:
        study_dir = tmp_path / "study"
        sandbox_dir = tmp_path / "sandbox"

        # Lay out the files export_best expects
        exp_dir = study_dir / "experiments" / "exp_001"
        exp_dir.mkdir(parents=True)
        experiment.to_json_file(exp_dir / "experiment.json")

        sandbox_exp = sandbox_dir / "exp_001"
        sandbox_exp.mkdir(parents=True)
        (sandbox_exp / "code.py").write_text(clean_code)

        exporter = SubmissionExporter()
        out_path = exporter.export_best(
            study=study,
            study_dir=study_dir,
            sandbox_dir=sandbox_dir,
        )
        assert out_path.exists()
        assert out_path.name == "exp_001_submission.ipynb"

    def test_export_best_without_best_raises(
        self, tmp_path: Path
    ) -> None:
        study = Study(
            study_id="empty",
            name="x",
            hypothesis="",
            mode=StudyMode.AUTONOMOUS,
            compute_budget=ComputeBudget(),
            pipeline_config_path=Path("a"),
            dataset_profile_path=Path("b"),
            model_registry_path=Path("c"),
            status=StudyStatus.ACTIVE,
            best_experiment_id=None,
            created_at=NOW,
            updated_at=NOW,
        )
        exporter = SubmissionExporter()
        with pytest.raises(NoBestExperimentError):
            exporter.export_best(
                study=study,
                study_dir=tmp_path,
                sandbox_dir=tmp_path,
            )

    def test_export_best_missing_code_file_raises(
        self, study: Study, experiment: Experiment, tmp_path: Path
    ) -> None:
        study_dir = tmp_path / "study"
        exp_dir = study_dir / "experiments" / "exp_001"
        exp_dir.mkdir(parents=True)
        experiment.to_json_file(exp_dir / "experiment.json")
        # sandbox dir exists but code.py does not
        sandbox_dir = tmp_path / "sandbox"
        sandbox_dir.mkdir()

        exporter = SubmissionExporter()
        with pytest.raises(NoBestExperimentError, match="Code file"):
            exporter.export_best(
                study=study,
                study_dir=study_dir,
                sandbox_dir=sandbox_dir,
            )
