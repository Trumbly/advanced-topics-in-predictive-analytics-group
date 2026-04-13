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
from agent.orchestrator import Orchestrator, StopReason, _parse_model_config
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
                "prompt_template": "config/prompts/propose_architecture/v1.yaml",
            },
            {
                "task_name": "generate_code",
                "task_type": "llm",
                "prompt_template": "config/prompts/generate_code/v1.yaml",
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
                "prompt_template": "config/prompts/analyze_results/v1.yaml",
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
    memory = ExperimentMemory(study_dir=study_dir, score_metric="roc_auc_macro")
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
        generate_report_at_end=False,  # avoid consuming extra scripted responses
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
        # Disable recovery so this test keeps its original intent: "when
        # an experiment fails, the loop must continue to the next one".
        # Recovery behavior has its own dedicated test below.
        study.compute_budget.max_recovery_attempts = 0

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

    def test_empty_proposal_dict_marks_experiment_failed(
        self, study: Study, tmp_path: Path
    ) -> None:
        """An empty dict `{}` is valid JSON but not a valid proposal.
        This is the exp_023 scenario — previously the task was marked
        COMPLETED and the experiment crashed later. Now it must fail
        the propose_architecture task right away so recovery can run."""
        study.compute_budget.max_experiments = 1
        study.compute_budget.max_recovery_attempts = 0  # no recovery
        responses = ["{}"]
        orch, _ = _make_orchestrator(study, tmp_path, responses)
        orch.run()
        exp = orch.memory.get("exp_001")
        assert exp is not None
        assert exp.status == ExperimentStatus.FAILED.value

    def test_proposal_with_null_architecture_marks_experiment_failed(
        self, study: Study, tmp_path: Path
    ) -> None:
        """`{"architecture": null}` — another exp_023 flavor."""
        study.compute_budget.max_experiments = 1
        study.compute_budget.max_recovery_attempts = 0
        responses = ['{"architecture": null, "hyperparams": {}}']
        orch, _ = _make_orchestrator(study, tmp_path, responses)
        orch.run()
        exp = orch.memory.get("exp_001")
        assert exp is not None
        assert exp.status == ExperimentStatus.FAILED.value

    def test_proposal_with_placeholder_architecture_rejected(
        self, study: Study, tmp_path: Path
    ) -> None:
        """`"architecture": "unknown"` is a placeholder that historically
        slipped through — now it must be rejected."""
        study.compute_budget.max_experiments = 1
        study.compute_budget.max_recovery_attempts = 0
        responses = ['{"architecture": "unknown", "hyperparams": {}}']
        orch, _ = _make_orchestrator(study, tmp_path, responses)
        orch.run()
        exp = orch.memory.get("exp_001")
        assert exp is not None
        assert exp.status == ExperimentStatus.FAILED.value


class TestParseModelConfig:
    """Unit tests for the strict `_parse_model_config` helper."""

    def test_rejects_non_dict(self) -> None:
        with pytest.raises(ValueError, match="Expected dict"):
            _parse_model_config("not a dict")
        with pytest.raises(ValueError, match="Expected dict"):
            _parse_model_config(None)
        with pytest.raises(ValueError, match="Expected dict"):
            _parse_model_config(["list", "not", "dict"])

    def test_rejects_empty_dict(self) -> None:
        with pytest.raises(ValueError, match="architecture"):
            _parse_model_config({})

    def test_rejects_null_architecture(self) -> None:
        with pytest.raises(ValueError, match="architecture"):
            _parse_model_config({"architecture": None})

    def test_rejects_empty_string_architecture(self) -> None:
        with pytest.raises(ValueError, match="architecture"):
            _parse_model_config({"architecture": ""})

    def test_rejects_placeholder_architecture(self) -> None:
        for placeholder in ("unknown", "null", "none", "?", "UNKNOWN"):
            with pytest.raises(ValueError, match="placeholder"):
                _parse_model_config({"architecture": placeholder})

    def test_rejects_non_dict_hyperparams(self) -> None:
        with pytest.raises(ValueError, match="hyperparams"):
            _parse_model_config(
                {"architecture": "cnn_small_v1", "hyperparams": "not a dict"}
            )

    def test_rejects_non_dict_augmentation(self) -> None:
        with pytest.raises(ValueError, match="augmentation"):
            _parse_model_config(
                {"architecture": "cnn_small_v1", "augmentation": [1, 2]}
            )

    def test_accepts_minimal_valid_proposal(self) -> None:
        cfg = _parse_model_config({"architecture": "cnn_small_v1"})
        assert cfg.architecture == "cnn_small_v1"
        assert cfg.hyperparams == {}
        assert cfg.augmentation == {}
        assert cfg.pretrained_model is None

    def test_strips_architecture_whitespace(self) -> None:
        cfg = _parse_model_config({"architecture": "  cnn_small_v1  "})
        assert cfg.architecture == "cnn_small_v1"

    def test_accepts_full_proposal(self) -> None:
        cfg = _parse_model_config(
            {
                "architecture": "efficientnet_b0",
                "pretrained_model": "efficientnet_b0",
                "hyperparams": {"lr": 0.001, "batch_size": 128},
                "augmentation": {"mixup": 0.2},
            }
        )
        assert cfg.architecture == "efficientnet_b0"
        assert cfg.pretrained_model == "efficientnet_b0"
        assert cfg.hyperparams["lr"] == 0.001
        assert cfg.augmentation["mixup"] == 0.2


# ---------------------------------------------------------------------------
# Promotion phase
# ---------------------------------------------------------------------------
#
# The smoke phase runs every experiment for 1 epoch (fast iteration). At
# end-of-study, `_run_promotion_phase` re-runs the top-K smoke candidates
# with BIRDCLEF_EPOCHS bumped up so we get a realistic final score. The
# re-run reuses the SAME sandbox code — no new LLM calls.


def _smoke_code_that_reports_epochs_as_score(base_score: float) -> str:
    """Return Python code that writes a results.json where the metric
    value scales with BIRDCLEF_EPOCHS. This lets us verify that the
    promoted run actually sees the bumped env var."""
    return f"""
import json, os, pathlib
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "1"))
score = {base_score} + 0.1 * (EPOCHS - 1)
pathlib.Path('results.json').write_text(json.dumps({{
    'metrics': {{'roc_auc_macro': score, 'loss': 0.5}},
    'training_curves': {{'loss': [0.5]*EPOCHS, 'roc_auc_macro': [score]*EPOCHS}},
    'duration_seconds': 0.1 * EPOCHS,
}}))
print(f'epochs={{EPOCHS}} score={{score}}')
"""


class TestOrchestratorPromotion:
    def test_promotion_phase_disabled_by_default(
        self, study: Study, tmp_path: Path
    ) -> None:
        """When promoted_epochs <= 1, no promotion runs."""
        study.compute_budget.max_experiments = 1
        study.compute_budget.promoted_epochs = 1  # disabled
        responses = _round_trip_responses(score=0.5)
        orch, _ = _make_orchestrator(study, tmp_path, responses)
        orch.run()
        # Only the smoke experiment — no `promoted_` IDs.
        assert all(
            not eid.startswith("promoted_") for eid in study.experiment_ids
        )

    def test_promotion_phase_runs_top_k(
        self, study: Study, tmp_path: Path
    ) -> None:
        """With promoted_epochs > 1 and top_k=1, the best smoke
        experiment gets re-run under BIRDCLEF_EPOCHS=5 and a
        `promoted_exp_001` experiment is added to the study."""
        study.compute_budget.max_experiments = 2
        study.compute_budget.promoted_epochs = 5
        study.compute_budget.promote_top_k = 1

        # Two experiments with different smoke scores. The better one
        # should be promoted.
        propose1 = json.dumps(
            {
                "architecture": "cnn_small_v1",
                "pretrained_model": None,
                "hyperparams": {"lr": 0.001, "batch_size": 16, "epochs": 1, "optimizer": "adam"},
                "augmentation": {},
            }
        )
        code1 = _smoke_code_that_reports_epochs_as_score(0.5)
        analyze1 = "baseline"
        propose2 = json.dumps(
            {
                "architecture": "cnn_small_v1_v2",
                "pretrained_model": None,
                "hyperparams": {"lr": 0.001, "batch_size": 16, "epochs": 1, "optimizer": "adam"},
                "augmentation": {},
            }
        )
        code2 = _smoke_code_that_reports_epochs_as_score(0.7)
        analyze2 = "better"
        responses = [propose1, code1, analyze1, propose2, code2, analyze2]

        orch, _ = _make_orchestrator(study, tmp_path, responses)
        orch.run()

        # Smoke run of exp_002 got 0.7 (base), exp_001 got 0.5.
        # Promotion should have picked exp_002 and re-run it at EPOCHS=5,
        # producing 0.7 + 0.1 * (5 - 1) = 1.1.
        promoted_ids = [
            eid for eid in study.experiment_ids if eid.startswith("promoted_")
        ]
        assert len(promoted_ids) == 1
        assert promoted_ids[0] == "promoted_exp_002"

        promoted = orch.memory.get("promoted_exp_002")
        assert promoted is not None
        assert promoted.status == ExperimentStatus.COMPLETED.value
        assert promoted.results is not None
        promoted_score = promoted.results.metrics["roc_auc_macro"]
        assert promoted_score == pytest.approx(1.1, abs=1e-6), (
            f"expected 0.7 + 0.4 = 1.1, got {promoted_score}"
        )

        # best_score should reflect the promoted score, not the smoke one.
        assert study.best_experiment_id == "promoted_exp_002"
        assert study.best_score == pytest.approx(1.1, abs=1e-6)

    def test_promotion_phase_respects_min_score_filter(
        self, study: Study, tmp_path: Path
    ) -> None:
        """When promote_min_score is set, only experiments with smoke
        score >= threshold are promoted."""
        study.compute_budget.max_experiments = 2
        study.compute_budget.promoted_epochs = 3
        study.compute_budget.promote_top_k = 5
        study.compute_budget.promote_min_score = 0.6

        propose1 = json.dumps(
            {
                "architecture": "lowscore",
                "pretrained_model": None,
                "hyperparams": {"lr": 0.001, "batch_size": 16, "epochs": 1, "optimizer": "adam"},
                "augmentation": {},
            }
        )
        code1 = _smoke_code_that_reports_epochs_as_score(0.4)  # below 0.6
        analyze1 = "ok"
        propose2 = json.dumps(
            {
                "architecture": "highscore",
                "pretrained_model": None,
                "hyperparams": {"lr": 0.001, "batch_size": 16, "epochs": 1, "optimizer": "adam"},
                "augmentation": {},
            }
        )
        code2 = _smoke_code_that_reports_epochs_as_score(0.8)  # above 0.6
        analyze2 = "good"
        responses = [propose1, code1, analyze1, propose2, code2, analyze2]

        orch, _ = _make_orchestrator(study, tmp_path, responses)
        orch.run()

        promoted_ids = [
            eid for eid in study.experiment_ids if eid.startswith("promoted_")
        ]
        assert promoted_ids == ["promoted_exp_002"], (
            f"only exp_002 should be promoted, got {promoted_ids}"
        )


class TestOrchestratorCornerCases:
    """Edge cases that don't cleanly belong to any other test class."""

    def test_budget_zero_runs_nothing(
        self, study: Study, tmp_path: Path
    ) -> None:
        study.compute_budget.max_experiments = 0
        orch, backend = _make_orchestrator(study, tmp_path, [])
        final_study, stop_reason = orch.run()
        assert stop_reason == StopReason.MAX_EXPERIMENTS
        assert len(final_study.experiment_ids) == 0
        assert backend.calls == []

    def test_cold_start_prompt_requests_json(
        self, study: Study, tmp_path: Path
    ) -> None:
        """Regression: when memory is empty, the orchestrator routes
        propose_architecture through the `use_fallback=True` path. That
        fallback text must contain explicit JSON instructions so the LLM
        doesn't reply with English prose (which was causing exp_001 to
        ALWAYS fail before this fix)."""
        study.compute_budget.max_experiments = 1
        responses = _round_trip_responses(score=0.5)
        orch, backend = _make_orchestrator(study, tmp_path, responses)
        orch.run()

        # The first LLM call is propose_architecture
        first_call_messages = backend.calls[0]["messages"]
        user_msg = next(
            m["content"] for m in first_call_messages if m["role"] == "user"
        )
        assert "JSON" in user_msg
        assert '"architecture"' in user_msg
        assert '"hyperparams"' in user_msg


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
                "prompt_template": "config/prompts/propose_architecture/v1.yaml",
            },
            {
                "task_name": "generate_code",
                "task_type": "llm",
                "prompt_template": "config/prompts/generate_code/v1.yaml",
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
                "prompt_template": "config/prompts/analyze_results/v1.yaml",
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


# ---------------------------------------------------------------------------
# Recovery loop: when execute_training fails with a code-level error,
# the orchestrator asks the LLM to rewrite the code and re-runs. The
# experiment then succeeds through the recovery path.
# ---------------------------------------------------------------------------


def _recovery_round_trip_responses(final_score: float) -> list[str]:
    """One full recovery success path.

    Sequence:
      1. propose_architecture → valid JSON proposal
      2. generate_code → code that deliberately crashes
      3. error_recovery → fixed code that writes a valid results.json
      4. analyze_results → free-form analysis text
    """
    propose = json.dumps(
        {
            "architecture": "cnn_small_v1",
            "pretrained_model": None,
            "hyperparams": {"lr": 0.001, "batch_size": 16, "epochs": 1, "optimizer": "adam"},
            "augmentation": {},
        }
    )
    broken = "raise RuntimeError('shape mismatch at conv1')\n"
    fixed = f"""
import json, pathlib
pathlib.Path('results.json').write_text(json.dumps({{
    'metrics': {{'roc_auc_macro': {final_score}, 'loss': {1.0 - final_score}}},
    'training_curves': {{'loss': [0.9, {1.0 - final_score}]}},
    'duration_seconds': 1.1
}}))
print('recovered')
"""
    analyze = "Recovered successfully. Next time, reduce batch size."
    return [propose, broken, fixed, analyze]


class TestOrchestratorRecovery:
    def test_recovery_turns_failure_into_success(
        self,
        pipeline_yaml_with_validate: Path,
        dataset_profile: Path,
        tmp_path: Path,
    ) -> None:
        """The first experiment's generate_code returns broken code that
        crashes in execute_training. The orchestrator invokes error_recovery,
        gets fixed code, re-runs validate_code + execute_training +
        capture_metrics, and the experiment completes with a real score.
        """
        study = Study(
            study_id="study_recovery",
            name="Recovery",
            hypothesis="Error recovery brings failed runs back to success",
            mode=StudyMode.AUTONOMOUS,
            compute_budget=ComputeBudget(
                max_experiments=1,
                max_wallclock_minutes=5,
                max_experiment_seconds=30,
                max_epochs_per_run=1,
                max_recovery_attempts=2,
                max_codegen_retries=0,  # this test covers Layer 2 only
            ),
            pipeline_config_path=pipeline_yaml_with_validate,
            dataset_profile_path=dataset_profile,
            model_registry_path=Path("registry/models.yaml"),
            status=StudyStatus.ACTIVE,
            created_at=NOW,
            updated_at=NOW,
        )

        responses = _recovery_round_trip_responses(final_score=0.71)
        orch, backend = _make_orchestrator(study, tmp_path, responses)
        final_study, _ = orch.run()

        assert len(final_study.experiment_ids) == 1
        exp = orch.memory.get("exp_001")
        assert exp is not None, "experiment not in memory"
        assert exp.status == ExperimentStatus.COMPLETED.value, (
            f"expected COMPLETED via recovery, got {exp.status}"
        )
        assert exp.results is not None
        assert exp.results.metrics["roc_auc_macro"] == pytest.approx(0.71)

        # The experiment must now have a recovery task in its task_ids
        task_ids = exp.task_ids
        assert any("recovery" in tid for tid in task_ids), (
            f"no recovery task found in task_ids: {task_ids}"
        )

        # Backend must have been called 4 times:
        # propose, generate_code, error_recovery, analyze
        assert len(backend.calls) == 4, (
            f"expected 4 LLM calls (propose/generate/recovery/analyze), "
            f"got {len(backend.calls)}"
        )

    def test_recovery_disabled_with_zero_attempts(
        self,
        pipeline_yaml_with_validate: Path,
        dataset_profile: Path,
        tmp_path: Path,
    ) -> None:
        """When max_recovery_attempts=0, a code-level failure is not
        retried — the experiment is marked FAILED immediately."""
        study = Study(
            study_id="study_no_recovery",
            name="No recovery",
            hypothesis="Zero budget → no recovery",
            mode=StudyMode.AUTONOMOUS,
            compute_budget=ComputeBudget(
                max_experiments=1,
                max_wallclock_minutes=5,
                max_experiment_seconds=30,
                max_epochs_per_run=1,
                max_recovery_attempts=0,
                max_codegen_retries=0,
            ),
            pipeline_config_path=pipeline_yaml_with_validate,
            dataset_profile_path=dataset_profile,
            model_registry_path=Path("registry/models.yaml"),
            status=StudyStatus.ACTIVE,
            created_at=NOW,
            updated_at=NOW,
        )
        responses = _recovery_round_trip_responses(final_score=0.71)
        orch, backend = _make_orchestrator(study, tmp_path, responses)
        orch.run()

        exp = orch.memory.get("exp_001")
        assert exp is not None
        assert exp.status == ExperimentStatus.FAILED.value
        # Only propose + generate + broken-exec, no recovery
        assert len(backend.calls) == 2, (
            f"expected 2 LLM calls (propose/generate), got {len(backend.calls)}"
        )

    def test_recovery_retries_on_empty_response(
        self,
        pipeline_yaml_with_validate: Path,
        dataset_profile: Path,
        tmp_path: Path,
    ) -> None:
        """When the LLM returns an empty string, the recovery handler
        MUST retry internally before consuming the recovery attempt.
        This models real nemotron-3-nano behavior: occasional empty
        draws interleaved with good ones."""
        study = Study(
            study_id="study_recovery_retry",
            name="Recovery with empty retries",
            hypothesis="Inner retries absorb LLM sampling variance",
            mode=StudyMode.AUTONOMOUS,
            compute_budget=ComputeBudget(
                max_experiments=1,
                max_wallclock_minutes=5,
                max_experiment_seconds=30,
                max_epochs_per_run=1,
                max_recovery_attempts=1,  # only ONE outer attempt
                max_codegen_retries=0,  # Layer 2 test, skip Layer 1
                recovery_empty_response_retries=3,  # three inner retries
                recovery_min_code_chars=50,
            ),
            pipeline_config_path=pipeline_yaml_with_validate,
            dataset_profile_path=dataset_profile,
            model_registry_path=Path("registry/models.yaml"),
            status=StudyStatus.ACTIVE,
            created_at=NOW,
            updated_at=NOW,
        )

        propose, broken, fixed, analyze = _recovery_round_trip_responses(0.62)
        # Two empty drawings before the successful one — the handler
        # should retry internally and still consume only ONE outer attempt.
        responses = [propose, broken, "", "   ", fixed, analyze]

        orch, backend = _make_orchestrator(study, tmp_path, responses)
        orch.run()

        exp = orch.memory.get("exp_001")
        assert exp is not None
        assert exp.status == ExperimentStatus.COMPLETED.value, (
            f"inner retries should have rescued this experiment, got {exp.status}"
        )
        # 1 propose + 1 generate + 3 recovery LLM calls + 1 analyze = 6
        assert len(backend.calls) == 6, (
            f"expected 6 LLM calls (propose, generate, 3x recovery, analyze), "
            f"got {len(backend.calls)}"
        )
        # Recovery task should have recorded that it needed 2 empty retries
        recovery_ids = [tid for tid in exp.task_ids if "recovery" in tid]
        assert len(recovery_ids) == 1
        rt_json = (
            orch.experiment_logger.task_dir(exp.experiment_id)
            / f"{recovery_ids[0]}.json"
        )
        assert rt_json.exists()
        rt_data = json.loads(rt_json.read_text())
        assert rt_data["status"] == "completed"
        assert rt_data.get("output", {}).get("inner_retries") == 2

    def test_recovery_fails_after_all_inner_retries_empty(
        self,
        pipeline_yaml_with_validate: Path,
        dataset_profile: Path,
        tmp_path: Path,
    ) -> None:
        """If every inner retry is empty, the outer attempt fails
        with EmptyRecoveryResponse and the inner retry count is
        reflected in the error message."""
        study = Study(
            study_id="study_recovery_all_empty",
            name="All empty",
            hypothesis="Exhausted retries fail cleanly",
            mode=StudyMode.AUTONOMOUS,
            compute_budget=ComputeBudget(
                max_experiments=1,
                max_wallclock_minutes=5,
                max_experiment_seconds=30,
                max_epochs_per_run=1,
                max_recovery_attempts=1,
                max_codegen_retries=0,
                recovery_empty_response_retries=3,
                recovery_min_code_chars=50,
            ),
            pipeline_config_path=pipeline_yaml_with_validate,
            dataset_profile_path=dataset_profile,
            model_registry_path=Path("registry/models.yaml"),
            status=StudyStatus.ACTIVE,
            created_at=NOW,
            updated_at=NOW,
        )

        propose, broken, _fixed, _analyze = _recovery_round_trip_responses(0.62)
        # All three inner retries empty.
        responses = [propose, broken, "", "", ""]

        orch, backend = _make_orchestrator(study, tmp_path, responses)
        orch.run()

        exp = orch.memory.get("exp_001")
        assert exp is not None
        assert exp.status == ExperimentStatus.FAILED.value
        recovery_ids = [tid for tid in exp.task_ids if "recovery" in tid]
        assert len(recovery_ids) == 1
        rt_json = (
            orch.experiment_logger.task_dir(exp.experiment_id)
            / f"{recovery_ids[0]}.json"
        )
        assert rt_json.exists()
        rt_data = json.loads(rt_json.read_text())
        assert rt_data["status"] == "failed"
        assert rt_data["error"]["error_type"] == "EmptyRecoveryResponse"
        assert "3" in rt_data["error"]["message"]  # mentions the retry count
