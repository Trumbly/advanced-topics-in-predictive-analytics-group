"""I-09 acceptance: deterministic autofix + LLM re-prompt."""

from __future__ import annotations

from pathlib import Path

import pytest

from lab.config import LLMConfig, load_settings
from lab.core.llm import LLMClient
from lab.core.models import TaskError, ValidationResult
from lab.core.recovery import Recovery, _HARD_FAILURE_ERROR_TYPES
from lab.prompts.engine import PromptEngine
from lab.prompts.registry import PromptRegistry

REPO_ROOT = Path(__file__).resolve().parent.parent
SHIPPED_PROMPTS = REPO_ROOT / "config" / "prompts"


class _FakePoster:
    def __init__(self, payload: dict):
        self.calls = []
        self.payload = payload

    def post(self, url, body, headers):
        import json as _json

        self.calls.append((url, body, headers))
        return 200, _json.dumps(self.payload).encode("utf-8")


@pytest.fixture
def settings():
    return load_settings("track_b", repo_root=REPO_ROOT)


@pytest.fixture
def recovery(settings):
    cfg = LLMConfig(
        provider="ollama",
        base_url="http://localhost:11434",
        model="m",
        temperature=0.0,
        max_tokens=128,
        retry_attempts=0,
        retry_backoff_seconds=0.0,
    )
    poster = _FakePoster(
        {"choices": [{"message": {"content": "FIXED"}}]}
    )
    client = LLMClient(cfg, http=poster, sleep=lambda _: None)
    engine = PromptEngine(PromptRegistry(SHIPPED_PROMPTS))
    return Recovery(client, engine, settings), poster


# -------- autofix --------

def test_unknown_nn_rename_via_autofix(recovery):
    rec, _ = recovery
    code = "import torch.nn as nn\n\ndef build_model(num_classes: int):\n    return nn.Conv2x2d(1, 8, 3)\n"
    finding = ValidationResult(
        ok=False,
        error_type="UnknownTorchNN",
        message="unknown torch.nn attribute: nn.Conv2x2d",
        autofix_hint="rename nn.Conv2x2d -> nn.Conv2d",
    )
    out = rec.try_autofix(code, finding)
    assert out is not None
    assert "nn.Conv2d(" in out
    assert "nn.Conv2x2d" not in out


def test_unknown_nn_arrow_unicode_also_works(recovery):
    rec, _ = recovery
    code = "import torch.nn as nn\n\nx = nn.Foo()\n"
    finding = ValidationResult(
        ok=False,
        error_type="UnknownTorchNN",
        message="x",
        autofix_hint="rename nn.Foo → nn.Bar",
    )
    out = rec.try_autofix(code, finding)
    assert out is not None and "nn.Bar()" in out


def test_bad_signature_injects_num_classes(recovery):
    rec, _ = recovery
    code = "import torch.nn as nn\n\ndef build_model():\n    return nn.Linear(1, 2)\n"
    finding = ValidationResult(
        ok=False,
        error_type="BadSignature",
        message="missing num_classes",
        autofix_hint="add num_classes parameter",
    )
    out = rec.try_autofix(code, finding)
    assert out is not None
    assert "def build_model(num_classes: int):" in out


def test_no_autofix_returns_none_on_unrelated_error(recovery):
    rec, _ = recovery
    finding = ValidationResult(ok=False, error_type="Syntax", message="oops")
    assert rec.try_autofix("x = 1", finding) is None


def test_autofix_passes_through_on_ok_finding(recovery):
    rec, _ = recovery
    assert rec.try_autofix("x = 1", ValidationResult(ok=True)) is None


# -------- LLM re-prompt --------

def test_ask_llm_invokes_recover_prompt(recovery):
    rec, poster = recovery
    err = TaskError(error_type="ShapeMismatch", message="x", traceback="trace")
    out = rec.ask_llm(
        "broken code",
        err,
        slots={
            "task_description": "x",
            "num_classes": "234",
            "input_tensor_shape": "(1,128,313)",
            "model_block_signature": "build_model(num_classes)",
            "experiment_memory": "(none)",
        },
    )
    assert out == "FIXED"
    assert len(poster.calls) == 1
    body = poster.calls[0][1]
    # verify the broken code + error type made it into the user prompt
    decoded = body.decode("utf-8")
    assert "broken code" in decoded
    assert "ShapeMismatch" in decoded


def test_ask_llm_accepts_validation_result(recovery):
    rec, poster = recovery
    finding = ValidationResult(
        ok=False,
        error_type="UnknownTorchNN",
        message="weird",
        findings=["nn.Foo"],
    )
    rec.ask_llm(
        "x",
        finding,
        slots={
            "task_description": "x",
            "num_classes": "234",
            "input_tensor_shape": "(1,128,313)",
            "model_block_signature": "build_model(num_classes)",
            "experiment_memory": "(none)",
        },
    )
    decoded = poster.calls[0][1].decode("utf-8")
    assert "UnknownTorchNN" in decoded
    assert "nn.Foo" in decoded


# -------- hard failure exposure --------

def test_hard_failure_set_exposed():
    assert "Timeout" in _HARD_FAILURE_ERROR_TYPES
    assert "OOM" in _HARD_FAILURE_ERROR_TYPES
    assert "FileNotFound" in _HARD_FAILURE_ERROR_TYPES
    assert "ValueError" not in _HARD_FAILURE_ERROR_TYPES
