"""End-to-end tests for the Orchestrator with a mocked LLM.

These tests wire up a real ContextHandler + PromptEngine + Memory + Logger
+ Executor against a mock LLM backend that returns canned responses. This
exercises the full agent loop without touching Ollama or BirdCLEF data.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
import yaml

from agent.context_handler import ContextHandler
from agent.executor import CodeExecutor
from agent.llm_client import LLMClient
from agent.logger import ExperimentLogger
from agent.memory import ExperimentMemory
from agent.models import (
    ClassStats,
    ComputeBudget,
    DatasetProfile,
    ExperimentStatus,
    StudyMode,
    StudyStatus,
    Study,
)
from agent.orchestrator import Orchestrator, StopReason
from agent.prompt_engine import PromptEngine
from registry import ModelRegistry


NOW = datetime(2026, 4, 10, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Mock LLM backend that returns different responses per call
# ---------------------------------------------------------------------------


class ScriptedBackend:
    """Returns a predetermined sequence of responses, one per LLM call."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if not self.responses:
            raise RuntimeError("ScriptedBackend ran out of responses")
        text = self.responses.pop(0)
        return {"choices": [{"message": {"content": text, "role": "assistant"}}]}


def _round_trip_responses(score: float = 0.5) -> list[str]:
    """One experiment needs 3 LLM calls: propose / generate / analyze."""
    propose = json.dumps(
        {
            "architecture": "cnn_small_v1",
            "pretrained_model": None,
            "hyperparams": {"lr": 0.001, "batch_size": 16, "epochs": 2, "optimizer": "adam"},
            "augmentation": {"time_shift": True},
        }
    )
    # Code that simply writes a results.json with the requested score
    code = f"""
import json, pathlib
pathlib.Path('results.json').write_text(json.dumps({{
    'metrics': {{'roc_auc_macro': {score}, 'loss': {1.0 - score}}},
    'training_curves': {{'loss': [1.0, 0.6, {1.0 - score}]}},
    'duration_seconds': 1.2
}}))
print('ok')
"""
    analyze = f"ROC-AUC is {score:.3f}. The small CNN baseline is a sensible starting point. Next, try EfficientNet transfer learning."
    return [propose, code, analyze]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dataset_profile(tmp_path: Path) -> Path:
    profile = DatasetProfile(
        num_classes=3,
        num_samples=30,
        spectrogram_shape=(1, 64, 128),
        sample_rate=32_000,
        class_stats=[
            ClassStats(class_id="sp_a", sample_count=10, avg_duration_seconds=5.0),
            ClassStats(class_id="sp_b", sample_count=10, avg_duration_seconds=5.0),
            ClassStats(class_id="sp_c", sample_count=10, avg_duration_seconds=5.0),
        ],
        imbalance_ratio=1.0,
        min_class_samples=10,
        max_class_samples=10,
        split_strategy="fixed_split",
        split_seed=42,
        train_indices=list(range(24)),
        val_indices=list(range(24, 30)),
    )
    path = tmp_path / "profile.json"
    profile.to_json_file(path)
    return path


@pytest.fixture
def pipeline_yaml(tmp_path: Path) -> Path:
    """A minimal pipeline: propose → generate → execute → capture → analyze.

    Note: we deliberately skip the validate_code step in this fixture so
    the scripted code can be as simple as possible. The real default
    pipeline keeps validate_code enabled.
    """
    data = {
        "name": "orchestrator_test",
        "description": "Test pipeline",
        "steps": [
            {
                "task_name": "propose_architecture",
                "task_type": "llm",
                "prompt_template": "config/prompts/propose_architecture.yaml",
            },
            {
                "task_name": "generate_code",
                "task_type": "llm",
                "prompt_template": "config/prompts/generate_code.yaml",
            },
            {
                "task_name": "execute_training",
                "task_type": "predefined",
                "handler": "agent.handlers.execute_training",
                "config": {"timeout_seconds": 30},
            },
            {
                "task_name": "capture_metrics",
                "task_type": "predefined",
                "handler": "agent.handlers.capture_metrics",
            },
            {
                "task_name": "analyze_results",
                "task_type": "llm",
                "prompt_template": "config/prompts/analyze_results.yaml",
            },
        ],
    }
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


@pytest.fixture
def study(pipeline_yaml: Path, dataset_profile: Path) -> Study:
    return Study(
        study_id="study_test_orch",
        name="Orchestrator Test",
        hypothesis="Test the loop",
        mode=StudyMode.AUTONOMOUS,
        compute_budget=ComputeBudget(
            max_experiments=2,
            max_wallclock_minutes=10,
            max_experiment_seconds=30,
            max_epochs_per_run=2,
        ),
        pipeline_config_path=pipeline_yaml,
        dataset_profile_path=dataset_profile,
        model_registry_path=Path("registry/models.yaml"),
        status=StudyStatus.ACTIVE,
        created_at=NOW,
        updated_at=NOW,
    )


def _make_orchestrator(
    study: Study,
    tmp_path: Path,
    responses: list[str],
) -> tuple[Orchestrator, ScriptedBackend]:
    backend = ScriptedBackend(responses)
    llm_client = LLMClient(
        backend=backend,
        model="gemma4:e4b",
        retry_backoff_seconds=0.0,
    )
    prompt_engine = PromptEngine()
    registry = ModelRegistry("registry/models.yaml")
    dataset_profile = DatasetProfile.from_json_file(study.dataset_profile_path)
    study_dir = tmp_path / "study_dir"
    memory = ExperimentMemory(study_dir=study_dir)
    experiment_logger = ExperimentLogger(study_dir=study_dir)
    executor = CodeExecutor(
        sandbox_root=tmp_path / "sandbox",
        timeout_seconds=30,
        python_executable=sys.executable,
    )
    context_handler = ContextHandler(
        prompt_engine=prompt_engine,
        llm_client=llm_client,
        dataset_profile=dataset_profile,
        registry=registry,
        memory=memory,
        max_prompt_tokens=100_000,  # give tests plenty of budget
    )
    orch = Orchestrator(
        study=study,
        llm_client=llm_client,
        prompt_engine=prompt_engine,
        context_handler=context_handler,
        memory=memory,
        experiment_logger=experiment_logger,
        executor=executor,
    )
    return orch, backend


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOrchestratorSmoke:
    def test_two_experiments_complete(
        self, study: Study, tmp_path: Path
    ) -> None:
        responses = _round_trip_responses(score=0.5) + _round_trip_responses(score=0.7)
        orch, backend = _make_orchestrator(study, tmp_path, responses)
        final_study, stop_reason = orch.run()

        assert stop_reason == StopReason.MAX_EXPERIMENTS
        assert len(final_study.experiment_ids) == 2
        assert final_study.best_experiment_id == "exp_002"
        assert final_study.best_score == pytest.approx(0.7)
        assert final_study.status == StudyStatus.COMPLETED.value

        # Every experiment was persisted
        study_dir = tmp_path / "study_dir"
        assert (study_dir / "study.json").exists()
        assert (study_dir / "memory.json").exists()
        for exp_id in final_study.experiment_ids:
            assert (study_dir / "experiments" / exp_id / "experiment.json").exists()

        # Memory reflects both successes
        assert len(orch.memory) == 2
        assert orch.memory.best_score() == pytest.approx(0.7)

    def test_failed_experiment_does_not_crash_loop(
        self, study: Study, tmp_path: Path
    ) -> None:
        # First experiment returns code that crashes; second one succeeds
        bad_propose = json.dumps(
            {
                "architecture": "cnn_small_v1",
                "pretrained_model": None,
                "hyperparams": {"lr": 1e-3, "batch_size": 8, "epochs": 1, "optimizer": "adam"},
                "augmentation": {},
            }
        )
        bad_code = "raise ValueError('deliberate failure')"
        # analyze_results is skipped when the pipeline short-circuits on failure
        responses = [
            bad_propose,
            bad_code,
            # Now the second experiment's full round trip
            *_round_trip_responses(score=0.6),
        ]
        orch, _ = _make_orchestrator(study, tmp_path, responses)
        final_study, _ = orch.run()

        assert len(final_study.experiment_ids) == 2
        assert final_study.best_experiment_id == "exp_002"
        assert final_study.best_score == pytest.approx(0.6)

        # First experiment is marked FAILED
        exp1 = orch.memory.get("exp_001")
        assert exp1 is not None
        assert exp1.status == ExperimentStatus.FAILED.value

    def test_invalid_llm_json_marks_experiment_failed(
        self, study: Study, tmp_path: Path
    ) -> None:
        study.compute_budget.max_experiments = 1
        responses = ["this is not json at all"]
        orch, _ = _make_orchestrator(study, tmp_path, responses)
        final_study, _ = orch.run()
        assert len(final_study.experiment_ids) == 1
        assert final_study.best_experiment_id is None

    def test_budget_zero_runs_nothing(
        self, study: Study, tmp_path: Path
    ) -> None:
        study.compute_budget.max_experiments = 0
        orch, backend = _make_orchestrator(study, tmp_path, [])
        final_study, stop_reason = orch.run()
        assert stop_reason == StopReason.MAX_EXPERIMENTS
        assert len(final_study.experiment_ids) == 0
        assert backend.calls == []


# ---------------------------------------------------------------------------
# Regression: validate_code between generate_code and execute_training
# must not shadow the generated code. Production bug seen in
# study_20260410_131833_first_run where execute_training ran a file
# containing only "cnn_small_v1" because the orchestrator pulled code
# from the *previous* task's output (validate_code), not generate_code.
# ---------------------------------------------------------------------------


@pytest.fixture
def pipeline_yaml_with_validate(tmp_path: Path) -> Path:
    """Pipeline that includes validate_code between generate_code and
    execute_training — the real default layout."""
    data = {
        "name": "orchestrator_test_with_validate",
        "description": "Test pipeline with validate_code",
        "steps": [
            {
                "task_name": "propose_architecture",
                "task_type": "llm",
                "prompt_template": "config/prompts/propose_architecture.yaml",
            },
            {
                "task_name": "generate_code",
                "task_type": "llm",
                "prompt_template": "config/prompts/generate_code.yaml",
            },
            {
                "task_name": "validate_code",
                "task_type": "predefined",
                "handler": "agent.handlers.validate_code",
                "config": {"check_imports": ["json"]},
            },
            {
                "task_name": "execute_training",
                "task_type": "predefined",
                "handler": "agent.handlers.execute_training",
                "config": {"timeout_seconds": 30},
            },
            {
                "task_name": "capture_metrics",
                "task_type": "predefined",
                "handler": "agent.handlers.capture_metrics",
            },
            {
                "task_name": "analyze_results",
                "task_type": "llm",
                "prompt_template": "config/prompts/analyze_results.yaml",
            },
        ],
    }
    path = tmp_path / "pipeline_with_validate.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


class TestOrchestratorWithValidateCode:
    def test_execute_training_sees_generate_code_output_not_validate(
        self,
        pipeline_yaml_with_validate: Path,
        dataset_profile: Path,
        tmp_path: Path,
    ) -> None:
        """Regression for the production bug: execute_training was running
        `experiment.config.architecture` as code because it pulled from the
        preceding task output (validate_code), which does not contain 'code'.
        """
        study = Study(
            study_id="study_regression",
            name="Regression",
            hypothesis="validate_code must not shadow generate_code output",
            mode=StudyMode.AUTONOMOUS,
            compute_budget=ComputeBudget(
                max_experiments=1,
                max_wallclock_minutes=5,
                max_experiment_seconds=30,
                max_epochs_per_run=1,
            ),
            pipeline_config_path=pipeline_yaml_with_validate,
            dataset_profile_path=dataset_profile,
            model_registry_path=Path("registry/models.yaml"),
            status=StudyStatus.ACTIVE,
            created_at=NOW,
            updated_at=NOW,
        )
        responses = _round_trip_responses(score=0.42)
        orch, _ = _make_orchestrator(study, tmp_path, responses)
        final_study, _ = orch.run()

        assert len(final_study.experiment_ids) == 1
        exp = orch.memory.get("exp_001")
        assert exp is not None
        assert exp.status == ExperimentStatus.COMPLETED.value, (
            f"Experiment failed: {exp.status}. "
            f"Most likely execute_training was handed the wrong code."
        )
        assert exp.results is not None
        assert exp.results.metrics["roc_auc_macro"] == pytest.approx(0.42)

        # Sanity: the execute_training task must have run the actual generated
        # code, not a short fallback like "cnn_small_v1".
        study_dir = tmp_path / "study_dir"
        exec_task_file = next(
            (study_dir / "experiments" / "exp_001" / "tasks").glob(
                "*_execute_training.json"
            )
        )
        import json as _json
        task_data = _json.loads(exec_task_file.read_text())
        code_used = task_data.get("code_used", "")
        assert "results.json" in code_used, (
            "execute_training.code_used does not look like real training code; "
            f"got: {code_used[:200]!r}"
        )
        assert len(code_used) > 50, (
            f"code_used is suspiciously short ({len(code_used)} chars): "
            f"{code_used!r}"
        )
