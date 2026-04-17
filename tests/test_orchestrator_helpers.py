"""Pure-helper tests for the orchestrator (no LLM, no subprocess).

Targets the two hardenings that prevent prose-as-code crashes:
  * ``_strip_fences`` extracts the FIRST fenced ```python``` block
    when the LLM wraps its prose around code.
  * Bare prose (no fence) goes through unchanged so the validator
    rejects it cleanly with SyntaxError on the next round-trip.
"""
from __future__ import annotations

from lab.core.models import Experiment, ExperimentStatus, StudyStatus
from lab.core.models import TaskError
from lab.core.executor import ExecutionResult
from lab.core.validator import ValidationResult
from lab.core.orchestrator import (
    Orchestrator,
    _apply_validator_autofix,
    _compose_code_from_model_response,
    _extract_build_model_block,
    _insert_local_import_into_build_model,
    _inject_model_block,
    _proposal_hyperparam_env,
    _rewrite_torch_hub_load_to_torchvision,
    _strip_fences,
)


def test_strip_fences_extracts_python_block_among_prose():
    raw = (
        "Looking at the error, the issue is …\n"
        "Here's the fix:\n"
        "```python\n"
        "import torch\n"
        "print('hi')\n"
        "```\n"
        "Hope this helps!"
    )
    out = _strip_fences(raw)
    assert out == "import torch\nprint('hi')"


def test_strip_fences_extracts_first_block_when_multiple():
    raw = "```python\nx = 1\n```\n\nand\n\n```python\ny = 2\n```"
    assert _strip_fences(raw) == "x = 1"


def test_strip_fences_handles_bare_outer_fence():
    raw = "```\nimport torch\n```"
    assert _strip_fences(raw) == "import torch"


def test_strip_fences_passthrough_when_no_fences():
    raw = "import torch\nprint('hi')\n"
    assert _strip_fences(raw).startswith("import torch")


def test_strip_fences_handles_pythonless_fence_marker():
    raw = "```py\nx = 42\n```"
    assert _strip_fences(raw) == "x = 42"


def test_strip_fences_strips_pure_prose_to_itself():
    """Bare prose has no fence — return as-is so the validator rejects
    it cleanly with a SyntaxError (don't try to be clever and "fix" it
    here; the recovery loop should re-prompt the LLM)."""
    raw = "Looking at the error, I think we should just give up."
    out = _strip_fences(raw)
    assert "Looking at the error" in out


def test_extract_build_model_block_from_full_script():
    raw = (
        "import torch\n"
        "import torch.nn as nn\n"
        "def helper():\n"
        "    return 1\n"
        "def build_model(num_classes: int) -> nn.Module:\n"
        "    return nn.Linear(10, num_classes)\n"
        "def main():\n"
        "    pass\n"
    )
    block = _extract_build_model_block(raw)
    assert block is not None
    assert block.startswith("def build_model(")
    assert "main" not in block


def test_inject_model_block_replaces_marker_region_only():
    skeleton = (
        "import torch.nn as nn\n"
        "### MODEL ###\n"
        "# old\n"
        "### END MODEL ###\n"
        "def main():\n"
        "    return 1\n"
    )
    model_block = (
        "def build_model(num_classes: int) -> nn.Module:\n"
        "    return nn.Linear(10, num_classes)\n"
    )
    out = _inject_model_block(skeleton, model_block)
    assert out is not None
    assert "def build_model(" in out
    assert "def main():" in out
    assert "# old" not in out


def test_compose_code_from_model_response_uses_template_injection():
    skeleton = (
        "import torch.nn as nn\n"
        "### MODEL ###\n"
        "# to replace\n"
        "### END MODEL ###\n"
        "def main():\n"
        "    return 1\n"
    )
    raw = (
        "```python\n"
        "def build_model(num_classes: int) -> nn.Module:\n"
        "    return nn.Linear(10, num_classes)\n"
        "```"
    )
    out = _compose_code_from_model_response(raw, template_code=skeleton)
    assert "def build_model(" in out
    assert "def main():" in out
    assert "# to replace" not in out


def test_compose_code_from_model_response_falls_back_when_no_model_found():
    skeleton = (
        "### MODEL ###\n"
        "# to replace\n"
        "### END MODEL ###\n"
    )
    raw = "this is not code"
    out = _compose_code_from_model_response(raw, template_code=skeleton)
    assert out == "this is not code"


def test_insert_local_import_into_build_model_inserts_after_docstring():
    code = (
        "import torch\n"
        "import torch.nn as nn\n"
        "def build_model(num_classes: int) -> nn.Module:\n"
        "    \"\"\"builder\"\"\"\n"
        "    return nn.Linear(10, num_classes)\n"
    )
    out = _insert_local_import_into_build_model(code, "import torchvision")
    assert out is not None
    assert "    \"\"\"builder\"\"\"\n    import torchvision\n" in out


def test_apply_validator_autofix_adds_missing_torchvision_import():
    code = (
        "import torch\n"
        "import torch.nn as nn\n"
        "def build_model(num_classes: int) -> nn.Module:\n"
        "    model = torchvision.models.resnet18(pretrained=False)\n"
        "    model.fc = nn.Linear(model.fc.in_features, num_classes)\n"
        "    return model\n"
    )
    val = ValidationResult(
        ok=False,
        error_type="DryRunFailed",
        message="Dry-run build_model failed: NameError: name 'torchvision' is not defined",
    )
    fixed = _apply_validator_autofix(code, val)
    assert fixed is not None
    assert "import torchvision" in fixed


def test_rewrite_torch_hub_load_to_torchvision_for_known_model():
    code = (
        "def build_model(num_classes: int):\n"
        "    m = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v3_small', pretrained=True)\n"
        "    return m\n"
    )
    out = _rewrite_torch_hub_load_to_torchvision(code)
    assert out is not None
    assert "torchvision.models.mobilenet_v3_small(pretrained=True)" in out


def test_apply_validator_autofix_rewrites_forbidden_torch_hub_source():
    code = (
        "import torch\n"
        "import torch.nn as nn\n"
        "def build_model(num_classes: int):\n"
        "    model = torch.hub.load('NVIDIA/DeepLearningExamples', 'resnet50', pretrained=True)\n"
        "    model.fc = nn.Linear(model.fc.in_features, num_classes)\n"
        "    return model\n"
    )
    val = ValidationResult(
        ok=False,
        error_type="ForbiddenModelSource",
        message="`torch.hub.load(...)` is disallowed",
    )
    fixed = _apply_validator_autofix(code, val)
    assert fixed is not None
    assert "torchvision.models.resnet50(pretrained=True)" in fixed
    assert "import torchvision" in fixed


# ---------------------------------------------------------------------------
# Final study status derivation
# ---------------------------------------------------------------------------


def _exp(status: ExperimentStatus) -> Experiment:
    return Experiment(study_id="s", primary_metric="x", status=status)


def test_final_status_aborted_wins_even_with_a_success():
    derive = Orchestrator._derive_final_status
    exps = [_exp(ExperimentStatus.COMPLETED), _exp(ExperimentStatus.FAILED)]
    assert derive(exps, aborted=True) == StudyStatus.ABORTED


def test_final_status_completed_when_at_least_one_succeeded():
    derive = Orchestrator._derive_final_status
    exps = [_exp(ExperimentStatus.FAILED), _exp(ExperimentStatus.COMPLETED), _exp(ExperimentStatus.FAILED)]
    assert derive(exps, aborted=False) == StudyStatus.COMPLETED


def test_final_status_failed_when_all_experiments_failed():
    derive = Orchestrator._derive_final_status
    exps = [_exp(ExperimentStatus.FAILED), _exp(ExperimentStatus.FAILED)]
    assert derive(exps, aborted=False) == StudyStatus.FAILED


def test_final_status_failed_when_no_experiments_ran():
    derive = Orchestrator._derive_final_status
    assert derive([], aborted=False) == StudyStatus.FAILED


# ---------------------------------------------------------------------------
# Hard-failure short-circuit
# ---------------------------------------------------------------------------


def test_hard_failure_set_includes_environmental_errors():
    """Recovery should NOT be attempted on errors the LLM can't fix
    from inside the script: missing data, OOM, timeouts."""
    from lab.core.orchestrator import _HARD_FAILURE_ERROR_TYPES
    assert "Timeout" in _HARD_FAILURE_ERROR_TYPES
    assert "OOM" in _HARD_FAILURE_ERROR_TYPES
    assert "FileNotFound" in _HARD_FAILURE_ERROR_TYPES
    # Code-level errors stay recoverable.
    assert "SyntaxError" not in _HARD_FAILURE_ERROR_TYPES
    assert "ShapeMismatch" not in _HARD_FAILURE_ERROR_TYPES
    assert "RuntimeError" not in _HARD_FAILURE_ERROR_TYPES


# ---------------------------------------------------------------------------
# In-band error field in results.json must be honoured
# ---------------------------------------------------------------------------


def test_capture_metrics_returns_error_field_in_raw(tmp_path):
    """_capture_metrics surfaces the raw results dict so the caller can
    inspect an in-band error field even when the script exited 0."""
    import json
    from lab.core.executor import ExecutionResult
    from lab.core.orchestrator import Orchestrator

    results = tmp_path / "results.json"
    results.write_text(json.dumps({
        "primary_metric": "f1_macro",
        "primary_score": 0.0,
        "history": [],
        "final": {"f1_macro": 0.0},
        "best": {"f1_macro": 0.0},
        "error": "FileNotFoundError: missing labels.csv",
    }))
    exec_result = ExecutionResult(
        exit_code=0,
        stdout="",
        stderr="",
        duration_seconds=1.0,
        workdir=tmp_path,
        results_json_path=results,
        error=None,
        timed_out=False,
    )
    # Build a stub orchestrator just to call _capture_metrics — we don't
    # need the full LLM/executor wiring for this, only the adapter shim.
    class _StubAdapter:
        primary_metric = "f1_macro"

    orch = Orchestrator.__new__(Orchestrator)
    orch.adapter = _StubAdapter()
    out = orch._capture_metrics(exec_result)
    assert out["raw"]["error"].startswith("FileNotFoundError")


def test_capture_metrics_uses_top_level_metrics_when_final_missing(tmp_path):
    """Some training scripts emit `results.metrics` without `final`.
    `_capture_metrics` should still surface ROC/F1 into experiment.metrics."""
    import json

    results = tmp_path / "results.json"
    results.write_text(json.dumps({
        "primary_metric": "f1_macro",
        "primary_score": 0.42,
        "metrics": {
            "f1_macro": 0.42,
            "roc_auc_macro": 0.73,
            "loss": 1.11,
        },
    }))
    exec_result = ExecutionResult(
        exit_code=0,
        stdout="",
        stderr="",
        duration_seconds=1.0,
        workdir=tmp_path,
        results_json_path=results,
        error=None,
        timed_out=False,
    )
    class _StubAdapter:
        primary_metric = "f1_macro"

    orch = Orchestrator.__new__(Orchestrator)
    orch.adapter = _StubAdapter()
    out = orch._capture_metrics(exec_result)
    assert out["metrics"]["f1_macro"] == 0.42
    assert out["metrics"]["roc_auc_macro"] == 0.73
    assert out["metrics"]["loss"] == 1.11
    assert out["primary_score"] == 0.42


def test_capture_metrics_uses_expected_primary_metric_when_overridden(tmp_path):
    import json

    results = tmp_path / "results.json"
    results.write_text(json.dumps({
        "primary_metric": "f1_macro",
        "primary_score": 0.25,
        "history": [
            {"f1_macro": 0.20, "roc_auc_macro": 0.61},
            {"f1_macro": 0.25, "roc_auc_macro": 0.73},
        ],
        "final": {"f1_macro": 0.25, "roc_auc_macro": 0.73},
        "best": {"f1_macro": 0.25, "roc_auc_macro": 0.73},
    }))
    exec_result = ExecutionResult(
        exit_code=0,
        stdout="",
        stderr="",
        duration_seconds=1.0,
        workdir=tmp_path,
        results_json_path=results,
        error=None,
        timed_out=False,
    )
    class _StubAdapter:
        primary_metric = "f1_macro"

    orch = Orchestrator.__new__(Orchestrator)
    orch.adapter = _StubAdapter()
    out = orch._capture_metrics(exec_result, expected_primary_metric="roc_auc_macro")
    assert out["primary_metric"] == "roc_auc_macro"
    assert out["primary_score"] == 0.73


def test_metric_for_experiment_uses_round_robin_schedule():
    orch = Orchestrator.__new__(Orchestrator)
    orch._primary_metric_schedule = ["f1_macro", "roc_auc_macro"]
    orch.adapter = type("_A", (), {"primary_metric": "f1_macro"})()
    assert orch._metric_for_experiment(0) == "f1_macro"
    assert orch._metric_for_experiment(1) == "roc_auc_macro"
    assert orch._metric_for_experiment(2) == "f1_macro"


def test_execute_with_recovery_skips_reruns_for_kaggle_backend(tmp_path):
    """Kaggle failures should not trigger layer-2 reruns of the same
    experiment — one remote attempt is enough."""
    class _FailingKaggleExecutor:
        backend = "kaggle"

        def __init__(self):
            self.calls = 0

        def run(self, code: str, *, experiment_id: str, extra_env=None):
            self.calls += 1
            return ExecutionResult(
                exit_code=-1,
                stdout="",
                stderr="Kernel log downloaded to -/lab-exp-abc.log",
                duration_seconds=1.0,
                workdir=tmp_path,
                results_json_path=None,
                error=TaskError(error_type="UnknownError", message="remote failed"),
                timed_out=False,
            )

    orch = Orchestrator.__new__(Orchestrator)
    orch.executor = _FailingKaggleExecutor()
    orch._abort_requested = False
    orch.settings = type(
        "_S", (), {"compute_budget": type("_B", (), {"max_recovery_attempts": 5})()}
    )()
    orch._recover_code = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not recover"))
    orch._revalidate_with_retries = lambda code, exp: code

    exp = Experiment(study_id="s", primary_metric="f1_macro")
    result = orch._execute_with_recovery("print('x')", exp)
    assert result.error is not None
    assert orch.executor.calls == 1


# ---------------------------------------------------------------------------
# Proposal hyperparameter env translation (Feature 7)
# ---------------------------------------------------------------------------


def test_proposal_hyperparam_env_passes_valid_lr_and_schedule():
    env = _proposal_hyperparam_env(
        {"lr": 5e-4, "lr_schedule": "cosine"}, env_prefix="AGENT",
    )
    assert env == {
        "AGENT_LR": repr(5e-4),
        "AGENT_LR_SCHEDULE": "cosine",
    }


def test_proposal_hyperparam_env_drops_out_of_range_lr():
    # 10 is way outside [1e-5, 1e-1] — drop silently.
    env = _proposal_hyperparam_env({"lr": 10.0}, env_prefix="AGENT")
    assert "AGENT_LR" not in env


def test_proposal_hyperparam_env_drops_unknown_schedule():
    env = _proposal_hyperparam_env(
        {"lr_schedule": "exponential_warmup"}, env_prefix="AGENT",
    )
    assert "AGENT_LR_SCHEDULE" not in env


def test_proposal_hyperparam_env_passes_init_from_experiment_id():
    env = _proposal_hyperparam_env(
        {"init_from_experiment_id": "exp_abc123"}, env_prefix="AGENT",
    )
    assert env["AGENT_INIT_FROM_EXPERIMENT_ID"] == "exp_abc123"


def test_proposal_hyperparam_env_handles_none():
    assert _proposal_hyperparam_env(None, env_prefix="AGENT") == {}
    assert _proposal_hyperparam_env("not a dict", env_prefix="AGENT") == {}


def test_proposal_hyperparam_env_passes_valid_epochs():
    env = _proposal_hyperparam_env({"epochs": 5}, env_prefix="AGENT")
    assert env["AGENT_EPOCHS"] == "5"


def test_proposal_hyperparam_env_clamps_epochs_at_max():
    env = _proposal_hyperparam_env({"epochs": 99}, env_prefix="AGENT")
    assert "AGENT_EPOCHS" not in env


def test_proposal_hyperparam_env_rejects_non_int_epochs():
    env = _proposal_hyperparam_env({"epochs": 3.5}, env_prefix="AGENT")
    assert "AGENT_EPOCHS" not in env
