"""SEARCH/REPLACE patch parsing + application + recovery wiring."""

from __future__ import annotations

import pytest

from lab.core.experiment import _apply_recovery_output, _current_block
from lab.core.patcher import apply_patches, looks_like_patches, parse_patches


_REPLY_OK = """\
<<<<<<< SEARCH
nn.Conv2x2d(1, 8, kernel_size=3)
=======
nn.Conv2d(1, 8, kernel_size=3)
>>>>>>> REPLACE
"""

_REPLY_MULTI = """\
<<<<<<< SEARCH
nn.Foo()
=======
nn.Linear(10, 1)
>>>>>>> REPLACE

<<<<<<< SEARCH
return out
=======
return out.sigmoid()
>>>>>>> REPLACE
"""

_REPLY_BAD_SEARCH = """\
<<<<<<< SEARCH
this text is not in the source
=======
something else
>>>>>>> REPLACE
"""

_REPLY_FENCED_FALLBACK = """\
def build_model(num_classes: int) -> nn.Module:
    return nn.Linear(10, num_classes)
"""


# ---- parser ----

def test_parse_extracts_one_patch():
    p = parse_patches(_REPLY_OK)
    assert p == [
        ("nn.Conv2x2d(1, 8, kernel_size=3)", "nn.Conv2d(1, 8, kernel_size=3)"),
    ]


def test_parse_extracts_multiple_patches():
    assert len(parse_patches(_REPLY_MULTI)) == 2


def test_parse_returns_empty_on_plain_code():
    assert parse_patches(_REPLY_FENCED_FALLBACK) == []


def test_looks_like_patches_detects_marker():
    assert looks_like_patches(_REPLY_OK)
    assert not looks_like_patches(_REPLY_FENCED_FALLBACK)


# ---- applier ----

def test_apply_patches_replaces_first_match():
    original = "x = nn.Conv2x2d(1, 8, kernel_size=3)\nout = x"
    patched = apply_patches(original, parse_patches(_REPLY_OK))
    assert patched == "x = nn.Conv2d(1, 8, kernel_size=3)\nout = x"


def test_apply_patches_returns_none_on_missing_search():
    original = "totally different code"
    patched = apply_patches(original, parse_patches(_REPLY_BAD_SEARCH))
    assert patched is None


def test_apply_patches_empty_list_returns_none():
    assert apply_patches("anything", []) is None


# ---- _apply_recovery_output (experiment integration) ----

def test_recovery_output_uses_patch_when_patch_applies():
    block = "x = nn.Conv2x2d(1, 8, kernel_size=3)\nreturn x"
    out, kind = _apply_recovery_output(block, _REPLY_OK)
    assert kind == "llm_patch"
    assert "nn.Conv2d" in out


def test_recovery_output_falls_back_to_full_block_when_search_misses():
    block = "totally different code"
    full_reply = (
        _REPLY_BAD_SEARCH
        + "\n# --- AGENT_BUILD_MODEL_START ---\n"
        "def build_model(num_classes: int):\n    return None\n"
        "# --- AGENT_BUILD_MODEL_END ---\n"
    )
    out, kind = _apply_recovery_output(block, full_reply)
    assert kind == "llm_patch_failed_fallback"
    assert "def build_model" in out


def test_recovery_output_fenced_fallback():
    out, kind = _apply_recovery_output("orig", _REPLY_FENCED_FALLBACK)
    assert kind == "llm_reprompt"
    assert "def build_model" in out


# ---- _current_block ----

def test_current_block_strips_skeleton():
    spliced = (
        "import torch\n"
        "# --- AGENT_BUILD_MODEL_START ---\n"
        "def build_model(num_classes: int):\n    return None\n"
        "# --- AGENT_BUILD_MODEL_END ---\n"
        "...\n"
    )
    block = _current_block(spliced)
    assert "import torch" not in block
    assert "def build_model" in block


def test_current_block_returns_input_when_no_markers():
    raw = "def build_model(num_classes):\n    return None\n"
    assert _current_block(raw).strip() == raw.strip()
