"""I-DELETE acceptance: ensure the dropped components stay dropped.

The rewrite never reintroduced kaggle_executor / modal_executor /
config_editor; this test fails loudly if any of those names creep back in.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

LAB_ROOT = Path(__file__).resolve().parent.parent / "lab"
FORBIDDEN_PATTERN = re.compile(
    r"\b(kaggle_executor|modal_executor|config_editor)\b"
)


@pytest.mark.parametrize("path", sorted(LAB_ROOT.rglob("*.py")))
def test_no_dropped_components(path: Path):
    assert not FORBIDDEN_PATTERN.search(path.read_text(encoding="utf-8")), (
        f"{path} mentions a dropped component (ADR-010)"
    )


def test_lab_help_imports_cleanly(capsys):
    """`python -m lab --help` must not ImportError because of dead deps."""
    import importlib

    cli = importlib.import_module("lab.cli")
    parser = cli._build_parser()
    parser.parse_args(["--help"]) if False else parser.format_help()
