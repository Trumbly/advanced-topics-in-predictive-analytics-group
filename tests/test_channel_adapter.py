"""Skeleton's _ensure_channel_compat must wrap any model whose first conv
expects a different channel count than what the dataset actually provides.

Renders the audio-multilabel skeleton, splices a tiny LLM-style block that
asks for a 3-channel Conv2d (the typical ImageNet-pretrained shape), then
imports the rendered file and confirms _ensure_channel_compat injects a
1->3 adapter and the wrapped model accepts a 1-channel forward pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from lab.config import load_settings
from lab.tasks.skeleton import render_skeleton, splice_build_model

REPO_ROOT = Path(__file__).resolve().parent.parent


_BUILD_MODEL_3CH = """
def build_model(num_classes: int):
    import torch.nn as nn

    return nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, num_classes),
    )
""".strip()


_BUILD_MODEL_1CH = """
def build_model(num_classes: int):
    import torch.nn as nn

    return nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, num_classes),
    )
""".strip()


def _render(tmp_path: Path, build_model_block: str) -> object:
    """Render the skeleton with ``build_model_block`` and import it."""
    s = load_settings("track_b", repo_root=REPO_ROOT)
    s = s.model_copy(
        update={
            "task": s.task.model_copy(update={"input_tensor_shape": [1, 16, 32]})
        }
    )
    code = splice_build_model(render_skeleton(s), build_model_block)
    target = tmp_path / "skel_under_test.py"
    target.write_text(code, encoding="utf-8")

    # Import the just-written file under a unique module name so each test
    # gets a fresh module without colliding with previous renders.
    import importlib.util

    name = f"_skel_under_test_{id(target)}"
    spec = importlib.util.spec_from_file_location(name, target)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_3channel_first_conv_gets_wrapped_with_1to3_adapter(tmp_path):
    mod = _render(tmp_path, _BUILD_MODEL_3CH)
    raw = mod.build_model(num_classes=4)
    wrapped = mod._ensure_channel_compat(raw, in_chan=1)

    # adapter inserted: not the same instance
    assert wrapped is not raw

    x = torch.zeros(2, 1, 16, 32)
    out = wrapped(x)
    assert out.shape == (2, 4)


def test_matching_channels_skip_adapter(tmp_path):
    mod = _render(tmp_path, _BUILD_MODEL_1CH)
    raw = mod.build_model(num_classes=4)
    wrapped = mod._ensure_channel_compat(raw, in_chan=1)
    # No wrapping when channels already match -- avoids burning params on
    # a no-op adapter for cnn_scratch builds.
    assert wrapped is raw


def test_first_conv_in_channels_handles_models_without_conv(tmp_path):
    mod = _render(tmp_path, _BUILD_MODEL_1CH)
    pure_linear = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(10, 4))
    assert mod._first_conv_in_channels(pure_linear) is None
    # No conv -> no wrap (we cannot guess channel expectations).
    assert mod._ensure_channel_compat(pure_linear, in_chan=1) is pure_linear
