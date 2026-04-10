"""Unit tests for the predefined task handlers under `agent.handlers`."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from agent.executor import CodeExecutor
from agent.handlers import capture_metrics, execute_training, validate_code
from agent.metrics import MetricsCollector
from agent.models import Task, TaskStatus, TaskType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _task(task_type: TaskType = TaskType.PREDEFINED) -> Task:
    return Task(
        task_id="exp_test_task_01",
        experiment_id="exp_test",
        task_type=task_type,
        task_name="test_handler",
    )


# ---------------------------------------------------------------------------
# validate_code
# ---------------------------------------------------------------------------


class TestValidateCodePure:
    def test_valid_code(self) -> None:
        result = validate_code.validate("import torch\nprint('hi')\n")
        assert result.ok

    def test_syntax_error(self) -> None:
        result = validate_code.validate("def broken(:\n  pass\n")
        assert not result.ok
        assert result.error_type == "SyntaxError"

    def test_forbidden_import_subprocess(self) -> None:
        result = validate_code.validate("import subprocess\n")
        assert not result.ok
        assert result.error_type == "ForbiddenImport"

    def test_forbidden_import_from_urllib(self) -> None:
        result = validate_code.validate("from urllib.request import urlopen\n")
        assert not result.ok
        assert result.error_type == "ForbiddenImport"

    def test_forbidden_pattern_os_system(self) -> None:
        result = validate_code.validate("import os\nos.system('ls')\n")
        assert not result.ok
        assert result.error_type == "ForbiddenPattern"

    def test_bare_eval_call_rejected(self) -> None:
        result = validate_code.validate("x = eval('1+1')\n")
        assert not result.ok
        assert result.error_type == "ForbiddenBareCall"
        assert "eval" in result.message

    def test_bare_exec_call_rejected(self) -> None:
        result = validate_code.validate("exec('x = 1')\n")
        assert not result.ok
        assert result.error_type == "ForbiddenBareCall"

    def test_model_eval_method_allowed(self) -> None:
        """Regression: `model.eval()` is PyTorch's legitimate inference switch
        and must not be rejected by the eval() check."""
        code = (
            "import torch\n"
            "import torch.nn as nn\n"
            "model = nn.Linear(10, 1)\n"
            "model.eval()\n"
            "with torch.no_grad():\n"
            "    out = model(torch.randn(1, 10))\n"
        )
        result = validate_code.validate(code, check_imports=("torch",))
        assert result.ok, f"model.eval() should pass, got {result.error_type}: {result.message}"

    def test_attribute_exec_method_allowed(self) -> None:
        """`something.exec(...)` (e.g. database cursors) must not trip the
        bare-exec check."""
        result = validate_code.validate(
            "import torch\ncursor = None\ncursor and cursor.exec('SELECT 1')\n",
            check_imports=("torch",),
        )
        assert result.ok

    def test_extra_reject_pattern(self) -> None:
        result = validate_code.validate(
            "torch.cuda.is_available()\n",
            extra_reject_patterns=("torch.cuda",),
        )
        assert not result.ok
        assert result.error_type == "ForbiddenPattern"

    def test_missing_expected_import(self) -> None:
        result = validate_code.validate(
            "print('hi')\n",
            check_imports=("torch", "tensorflow"),
        )
        assert not result.ok
        assert result.error_type == "MissingExpectedImport"

    def test_expected_import_satisfied(self) -> None:
        result = validate_code.validate(
            "import torch\nprint('hi')\n",
            check_imports=("torch", "tensorflow"),
        )
        assert result.ok


class TestValidateCodeHandler:
    def test_updates_task_on_success(self) -> None:
        task = _task()
        task.code_used = "import torch\nprint('hi')\n"
        validate_code.run(task, config={"check_imports": ["torch"]})
        assert task.status == TaskStatus.COMPLETED.value
        assert task.error is None
        assert task.output.get("validation") == "passed"

    def test_updates_task_on_failure(self) -> None:
        task = _task()
        task.code_used = "import subprocess\n"
        validate_code.run(task)
        assert task.status == TaskStatus.FAILED.value
        assert task.error is not None
        assert task.error.error_type == "ForbiddenImport"

    def test_empty_code_fails(self) -> None:
        task = _task()
        task.code_used = "   "
        validate_code.run(task)
        assert task.status == TaskStatus.FAILED.value
        assert task.error is not None
        assert task.error.error_type == "NoCode"


# ---------------------------------------------------------------------------
# execute_training
# ---------------------------------------------------------------------------


class TestExecuteTrainingHandler:
    @pytest.fixture
    def executor(self, tmp_path: Path) -> CodeExecutor:
        return CodeExecutor(
            sandbox_root=tmp_path / "sandbox",
            timeout_seconds=10,
            python_executable=sys.executable,
        )

    def test_successful_run(self, executor: CodeExecutor) -> None:
        task = _task()
        task.code_used = (
            "import json, pathlib\n"
            "pathlib.Path('results.json').write_text("
            "json.dumps({'metrics': {'roc_auc_macro': 0.5, 'loss': 1.0}}))\n"
        )
        execute_training.run(task, executor=executor)
        assert task.status == TaskStatus.COMPLETED.value
        assert task.error is None
        assert task.output["exit_code"] == 0
        assert task.output["results_json_path"] is not None

    def test_failed_run(self, executor: CodeExecutor) -> None:
        task = _task()
        task.code_used = "raise ValueError('boom')\n"
        execute_training.run(task, executor=executor)
        assert task.status == TaskStatus.FAILED.value
        assert task.error is not None
        assert task.error.error_type == "ValueError"

    def test_missing_code_fails(self, executor: CodeExecutor) -> None:
        task = _task()
        execute_training.run(task, executor=executor)
        assert task.status == TaskStatus.FAILED.value
        assert task.error is not None
        assert task.error.error_type == "NoCode"


# ---------------------------------------------------------------------------
# capture_metrics
# ---------------------------------------------------------------------------


class TestCaptureMetricsHandler:
    def test_parses_previous_results(self, tmp_path: Path) -> None:
        results_path = tmp_path / "results.json"
        results_path.write_text(
            json.dumps(
                {
                    "metrics": {"roc_auc_macro": 0.8, "loss": 0.2},
                    "duration_seconds": 120.0,
                }
            )
        )
        task = _task()
        previous_output = {"results_json_path": str(results_path)}
        capture_metrics.run(
            task,
            collector=MetricsCollector(),
            previous_task_output=previous_output,
        )
        assert task.status == TaskStatus.COMPLETED.value
        assert task.output["training_results"]["metrics"]["roc_auc_macro"] == 0.8

    def test_missing_results_path_fails(self) -> None:
        task = _task()
        capture_metrics.run(task, previous_task_output={})
        assert task.status == TaskStatus.FAILED.value
        assert task.error is not None
        assert task.error.error_type == "NoResults"

    def test_invalid_results_json_fails(self, tmp_path: Path) -> None:
        bad_path = tmp_path / "bad.json"
        bad_path.write_text("{broken json}")
        task = _task()
        capture_metrics.run(
            task,
            previous_task_output={"results_json_path": str(bad_path)},
        )
        assert task.status == TaskStatus.FAILED.value
        assert task.error is not None
        assert task.error.error_type == "MetricsParseError"
