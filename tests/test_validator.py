"""I-07 acceptance: static + smoke validator."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from lab.config import load_settings
from lab.core.validator import Validator

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).parent / "fixtures" / "code"

SIGNATURE = ("build_model", "num_classes")
SMOKE_INPUT_SHAPE = (1, 32, 32)  # tiny enough to keep smoke <5s
SMOKE_CLASSES = 4


@pytest.fixture(scope="module")
def validator():
    settings = load_settings("track_b", repo_root=REPO_ROOT)
    return Validator(settings)


def _read(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


# ----- happy path -----

def test_ok_fixture_passes(validator):
    result = validator.validate(
        _read("ok.py"),
        signature=SIGNATURE,
        smoke_input_shape=SMOKE_INPUT_SHAPE,
        smoke_num_classes=SMOKE_CLASSES,
    )
    assert result.ok, result.message


# ----- forbidden imports -----

def test_subprocess_rejected_loop_mode(validator):
    result = validator.validate(
        _read("forbidden_import.py"),
        signature=SIGNATURE,
        smoke_input_shape=SMOKE_INPUT_SHAPE,
        smoke_num_classes=SMOKE_CLASSES,
    )
    assert not result.ok
    assert result.error_type == "ForbiddenImport"
    assert "subprocess" in result.message


def test_urllib_allowed_in_loop_but_rejected_in_submission(validator):
    code = _read("urllib_only.py")

    loop = validator.validate(
        code,
        signature=SIGNATURE,
        smoke_input_shape=(10,),
        smoke_num_classes=SMOKE_CLASSES,
    )
    assert loop.ok, loop.message  # urllib OK in loop mode

    sub = validator.validate(
        code,
        signature=SIGNATURE,
        smoke_input_shape=(10,),
        smoke_num_classes=SMOKE_CLASSES,
        submission_mode=True,
    )
    assert not sub.ok
    assert sub.error_type == "ForbiddenImport"
    assert "urllib" in sub.message


def test_shell_pip_install_rejected_in_submission(validator):
    code = "# !pip install foo\n" + _read("ok.py")
    result = validator.validate(
        code,
        signature=SIGNATURE,
        smoke_input_shape=SMOKE_INPUT_SHAPE,
        smoke_num_classes=SMOKE_CLASSES,
        submission_mode=True,
    )
    assert not result.ok
    assert result.error_type == "ForbiddenImport"


# ----- signature -----

def test_bad_signature_carries_autofix_hint(validator):
    result = validator.validate(
        _read("bad_signature.py"),
        signature=SIGNATURE,
        smoke_input_shape=SMOKE_INPUT_SHAPE,
        smoke_num_classes=SMOKE_CLASSES,
    )
    assert not result.ok
    assert result.error_type == "BadSignature"
    assert result.autofix_hint
    assert "num_classes" in result.autofix_hint


def test_missing_function_is_bad_signature(validator):
    code = "import torch.nn as nn\n\nx = 1\n"
    result = validator.validate(
        code,
        signature=SIGNATURE,
        smoke_input_shape=SMOKE_INPUT_SHAPE,
        smoke_num_classes=SMOKE_CLASSES,
    )
    assert not result.ok
    assert result.error_type == "BadSignature"


# ----- nn attribute reflection -----

def test_hallucinated_nn_carries_autofix_hint(validator):
    result = validator.validate(
        _read("hallucinated_nn.py"),
        signature=SIGNATURE,
        smoke_input_shape=SMOKE_INPUT_SHAPE,
        smoke_num_classes=SMOKE_CLASSES,
    )
    assert not result.ok
    assert result.error_type == "UnknownTorchNN"
    assert result.autofix_hint and "Conv2d" in result.autofix_hint


# ----- smoke forward -----

def test_shape_mismatch_caught_by_smoke(validator):
    result = validator.validate(
        _read("shape_smoke_fail.py"),
        signature=SIGNATURE,
        smoke_input_shape=(1, 128, 313),
        smoke_num_classes=SMOKE_CLASSES,
    )
    assert not result.ok
    assert result.error_type == "SmokeFailed"
    assert "shape" in result.message.lower() or "expected" in result.message.lower()


def test_smoke_finishes_under_5_seconds(validator):
    t0 = time.monotonic()
    result = validator.validate(
        _read("ok.py"),
        signature=SIGNATURE,
        smoke_input_shape=SMOKE_INPUT_SHAPE,
        smoke_num_classes=SMOKE_CLASSES,
    )
    dt = time.monotonic() - t0
    assert result.ok
    assert dt < 5.0, f"smoke took {dt:.2f}s, must be <5s"


# ----- syntax -----

def test_syntax_error_classified():
    settings = load_settings("track_b", repo_root=REPO_ROOT)
    v = Validator(settings)
    result = v.validate(
        "def broken(:\n  pass\n",
        signature=SIGNATURE,
        smoke_input_shape=SMOKE_INPUT_SHAPE,
        smoke_num_classes=SMOKE_CLASSES,
    )
    assert not result.ok
    assert result.error_type == "Syntax"
