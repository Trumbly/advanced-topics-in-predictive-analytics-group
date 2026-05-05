"""Validator captures full Python traceback for smoke failures."""

from __future__ import annotations

from pathlib import Path

import pytest

from lab.config import load_settings
from lab.core.validator import Validator

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def validator():
    return Validator(load_settings("track_b", repo_root=REPO_ROOT))


def test_smoke_runtime_failure_captures_traceback(validator):
    bad = (
        "import torch.nn as nn\n"
        "\n"
        "def build_model(num_classes: int) -> nn.Module:\n"
        "    raise AttributeError('missing kernel_size')\n"
    )
    result = validator.validate(
        bad,
        signature=("build_model", "num_classes"),
        smoke_input_shape=(1, 8, 8),
        smoke_num_classes=4,
    )
    assert not result.ok
    assert result.error_type == "SmokeFailed"
    assert result.findings, "expected traceback in findings"
    trace = "\n".join(result.findings)
    assert "AttributeError" in trace
    # The validator's sandbox file shows up in the trace; that's the line ref.
    assert "validator-sandbox" in trace or "build_model" in trace


def test_smoke_module_level_failure_captures_traceback(validator):
    bad = (
        "import torch.nn as nn\n"
        "\n"
        "x = 1 / 0  # module-level bomb\n"
        "\n"
        "def build_model(num_classes: int) -> nn.Module:\n"
        "    return nn.Linear(1, num_classes)\n"
    )
    result = validator.validate(
        bad,
        signature=("build_model", "num_classes"),
        smoke_input_shape=(1,),
        smoke_num_classes=4,
    )
    assert not result.ok
    trace = "\n".join(result.findings)
    assert "ZeroDivisionError" in trace


def test_smoke_forward_failure_captures_traceback(validator):
    bad = (
        "import torch.nn as nn\n"
        "import torch\n"
        "\n"
        "class _M(nn.Module):\n"
        "    def forward(self, x):\n"
        "        raise RuntimeError('forward bomb')\n"
        "\n"
        "def build_model(num_classes: int) -> nn.Module:\n"
        "    return _M()\n"
    )
    result = validator.validate(
        bad,
        signature=("build_model", "num_classes"),
        smoke_input_shape=(1, 8, 8),
        smoke_num_classes=4,
    )
    assert not result.ok
    trace = "\n".join(result.findings)
    assert "RuntimeError" in trace
    assert "forward bomb" in trace
