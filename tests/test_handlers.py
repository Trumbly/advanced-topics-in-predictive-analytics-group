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

    def test_custom_nn_module_architecture_allowed(self) -> None:
        """Free-form architectures: the validator must accept a fully custom
        nn.Module defined inline, without any reference to the model
        registry. This is what the agent should produce when the LLM
        designs a novel architecture from primitives.
        """
        custom = (
            "import torch\n"
            "import torch.nn as nn\n"
            "import torch.nn.functional as F\n"
            "import torchvision.models as tv_models\n"
            "from pipelines.data_loader import load_precomputed_dataset\n"
            "from sklearn.metrics import roc_auc_score\n"
            "import numpy as np\n"
            "import json\n"
            "\n"
            "class SEBlock(nn.Module):\n"
            "    def __init__(self, channels, reduction=8):\n"
            "        super().__init__()\n"
            "        self.avg = nn.AdaptiveAvgPool2d(1)\n"
            "        self.fc = nn.Sequential(\n"
            "            nn.Linear(channels, channels // reduction),\n"
            "            nn.ReLU(inplace=True),\n"
            "            nn.Linear(channels // reduction, channels),\n"
            "            nn.Sigmoid(),\n"
            "        )\n"
            "    def forward(self, x):\n"
            "        b, c, _, _ = x.size()\n"
            "        y = self.avg(x).view(b, c)\n"
            "        y = self.fc(y).view(b, c, 1, 1)\n"
            "        return x * y\n"
            "\n"
            "class CustomCnnWithSE(nn.Module):\n"
            "    def __init__(self, num_classes=234):\n"
            "        super().__init__()\n"
            "        self.conv = nn.Conv2d(1, 32, 3, padding=1)\n"
            "        self.se = SEBlock(32)\n"
            "        self.pool = nn.AdaptiveAvgPool2d(1)\n"
            "        self.head = nn.Linear(32, num_classes)\n"
            "    def forward(self, x):\n"
            "        x = F.relu(self.conv(x))\n"
            "        x = self.se(x)\n"
            "        x = self.pool(x).flatten(1)\n"
            "        return self.head(x)\n"
            "\n"
            "model = CustomCnnWithSE(num_classes=234)\n"
            "model.eval()\n"
        )
        result = validate_code.validate(
            custom,
            check_imports=("torch", "tensorflow", "keras", "pipelines"),
        )
        assert result.ok, (
            f"Custom architecture should validate, got "
            f"{result.error_type}: {result.message}"
        )

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

    def test_epochs_cap_rejects_higher_value(self) -> None:
        """Fast-iteration mode: EPOCHS > max_epochs must be rejected."""
        code = (
            "import torch\n"
            "EPOCHS = 5\n"
            "print(EPOCHS)\n"
        )
        result = validate_code.validate(code, max_epochs=1)
        assert not result.ok
        assert result.error_type == "EpochsCapExceeded"
        assert "EPOCHS = 5" in result.message
        assert "hard cap of 1" in result.message

    def test_epochs_cap_allows_exact_value(self) -> None:
        code = "import torch\nEPOCHS = 1\n"
        result = validate_code.validate(code, max_epochs=1)
        assert result.ok

    def test_epochs_cap_ignored_when_not_set(self) -> None:
        """If the pipeline doesn't enable max_epochs, any value is fine."""
        code = "import torch\nEPOCHS = 50\n"
        result = validate_code.validate(code)  # no max_epochs argument
        assert result.ok

    def test_epochs_cap_only_checks_module_scope(self) -> None:
        """`EPOCHS = 5` inside a function body is allowed — we only
        enforce the cap on the top-level constant."""
        code = (
            "import torch\n"
            "def make_epochs():\n"
            "    EPOCHS = 5  # local, not the training budget\n"
            "    return EPOCHS\n"
            "EPOCHS = 1\n"
        )
        result = validate_code.validate(code, max_epochs=1)
        assert result.ok

    def test_main_guard_required_for_load_precomputed_dataset(self) -> None:
        """`load_precomputed_dataset(...)` at module scope is rejected
        because DataLoader workers (spawn) would re-run it."""
        code = (
            "from pipelines.data_loader import load_precomputed_dataset\n"
            "train, val, nc = load_precomputed_dataset()\n"
        )
        result = validate_code.validate(code)
        assert not result.ok
        assert result.error_type == "MissingMainGuard"
        assert "load_precomputed_dataset" in result.message
        assert "__main__" in result.message

    def test_main_guard_required_for_dataloader(self) -> None:
        """Bare `DataLoader(...)` at module scope is also rejected."""
        code = (
            "from torch.utils.data import DataLoader\n"
            "loader = DataLoader(ds, batch_size=4, num_workers=2)\n"
        )
        result = validate_code.validate(code)
        assert not result.ok
        assert result.error_type == "MissingMainGuard"
        assert "DataLoader" in result.message

    def test_main_guard_accepts_call_inside_if_main(self) -> None:
        """Inside the guard block the call is allowed."""
        code = (
            "from pipelines.data_loader import load_precomputed_dataset\n"
            "if __name__ == '__main__':\n"
            "    train, val, nc = load_precomputed_dataset()\n"
        )
        result = validate_code.validate(code)
        assert result.ok

    def test_main_guard_accepts_call_inside_function(self) -> None:
        """Inside a function def the call is allowed too — the function
        body is not re-executed on module import."""
        code = (
            "from pipelines.data_loader import load_precomputed_dataset\n"
            "def setup():\n"
            "    return load_precomputed_dataset()\n"
            "if __name__ == '__main__':\n"
            "    setup()\n"
        )
        result = validate_code.validate(code)
        assert result.ok

    def test_main_guard_accepts_call_when_no_multiprocessing_triggers(
        self,
    ) -> None:
        """Code without DataLoader/load_precomputed_dataset at all
        passes (nothing to guard)."""
        code = "import torch\nmodel = torch.nn.Linear(10, 2)\n"
        result = validate_code.validate(code)
        assert result.ok

    def test_main_guard_detects_attribute_call(self) -> None:
        """`pipelines.data_loader.load_precomputed_dataset(...)` at module
        scope is also caught."""
        code = (
            "import pipelines.data_loader\n"
            "train, val, nc = pipelines.data_loader.load_precomputed_dataset()\n"
        )
        result = validate_code.validate(code)
        assert not result.ok
        assert result.error_type == "MissingMainGuard"


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
        # Raw content (even as truncated text blob) must be in the output
        assert "raw_results" in task.output

    def test_script_reported_error_surfaces_cleanly(self, tmp_path: Path) -> None:
        """Regression: the LLM script caught its own error and wrote
        `{"error": "..."}`. capture_metrics must (a) mark the task failed
        with ScriptReportedError as the error_type, (b) carry the LLM's
        actual error message (not 'missing required metrics'), and
        (c) stash the raw results.json content in task.output for debugging."""
        results_path = tmp_path / "results.json"
        results_path.write_text(
            json.dumps(
                {
                    "metrics": {},
                    "error": "RuntimeError: shape mismatch on conv1",
                }
            )
        )
        task = _task()
        capture_metrics.run(
            task,
            previous_task_output={"results_json_path": str(results_path)},
        )
        assert task.status == TaskStatus.FAILED.value
        assert task.error is not None
        assert task.error.error_type == "ScriptReportedError"
        assert "shape mismatch" in task.error.message
        assert "missing required metrics" not in task.error.message
        # The raw JSON content must be preserved in the task log
        raw = task.output.get("raw_results")
        assert isinstance(raw, dict)
        assert raw["error"] == "RuntimeError: shape mismatch on conv1"

    def test_raw_results_embedded_on_success(self, tmp_path: Path) -> None:
        """Even on success, the raw JSON content is embedded in task.output
        so the experiment log captures everything the script wrote."""
        results_path = tmp_path / "results.json"
        payload = {
            "metrics": {"roc_auc_macro": 0.7, "loss": 0.3},
            "training_curves": {"loss": [1.0, 0.5, 0.3]},
            "duration_seconds": 42.0,
        }
        results_path.write_text(json.dumps(payload))
        task = _task()
        capture_metrics.run(
            task,
            previous_task_output={"results_json_path": str(results_path)},
        )
        assert task.status == TaskStatus.COMPLETED.value
        assert task.output["raw_results"] == payload
        # Parsed training_results is also there
        assert (
            task.output["training_results"]["metrics"]["roc_auc_macro"] == 0.7
        )
