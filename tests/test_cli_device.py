"""Tests for the device-resolution helpers in `agent.cli`.

The CLI is responsible for turning `training.device` from config.yaml
(a `Literal["auto", "cpu", "mps", "cuda"]`) into a concrete backend
name that gets exported as the `BIRDCLEF_DEVICE` env var. This module
verifies that:

  1. `_resolve_device("cpu")` is always "cpu".
  2. `_resolve_device("mps")` / `"cuda"` returns the requested backend
     when it is available, and gracefully falls back to "cpu" (with a
     warning on stderr) when it is not.
  3. `_resolve_device("auto")` picks MPS > CUDA > CPU in that order,
     using whatever backends are available.
  4. `_training_env_from_config` builds a `BIRDCLEF_*` dict that matches
     what the executor expects, including `BIRDCLEF_DEVICE` set to the
     already-resolved concrete value (not "auto").

MPS / CUDA availability is mocked so these tests run on any machine.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.cli import _resolve_device, _training_env_from_config
from agent.models import GlobalConfig, TrainingSettings


class TestResolveDevice:
    def test_cpu_always_returns_cpu(self) -> None:
        assert _resolve_device("cpu") == "cpu"

    def test_mps_returns_mps_when_available(self) -> None:
        with patch("agent.cli._mps_available", return_value=True):
            assert _resolve_device("mps") == "mps"

    def test_mps_falls_back_to_cpu_when_unavailable(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with patch("agent.cli._mps_available", return_value=False):
            assert _resolve_device("mps") == "cpu"
        captured = capsys.readouterr()
        assert "mps" in captured.err.lower()

    def test_cuda_returns_cuda_when_available(self) -> None:
        with patch("agent.cli._cuda_available", return_value=True):
            assert _resolve_device("cuda") == "cuda"

    def test_cuda_falls_back_to_cpu_when_unavailable(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with patch("agent.cli._cuda_available", return_value=False):
            assert _resolve_device("cuda") == "cpu"
        captured = capsys.readouterr()
        assert "cuda" in captured.err.lower()

    def test_auto_prefers_mps_over_cuda_over_cpu(self) -> None:
        with patch("agent.cli._mps_available", return_value=True), patch(
            "agent.cli._cuda_available", return_value=True
        ):
            assert _resolve_device("auto") == "mps"

    def test_auto_uses_cuda_when_mps_unavailable(self) -> None:
        with patch("agent.cli._mps_available", return_value=False), patch(
            "agent.cli._cuda_available", return_value=True
        ):
            assert _resolve_device("auto") == "cuda"

    def test_auto_falls_back_to_cpu_when_neither_available(self) -> None:
        with patch("agent.cli._mps_available", return_value=False), patch(
            "agent.cli._cuda_available", return_value=False
        ):
            assert _resolve_device("auto") == "cpu"

    def test_none_or_unknown_defaults_to_cpu_via_auto(self) -> None:
        """Empty / unexpected values behave like `auto` then fall through."""
        with patch("agent.cli._mps_available", return_value=False), patch(
            "agent.cli._cuda_available", return_value=False
        ):
            assert _resolve_device("") == "cpu"


class TestTrainingEnvFromConfig:
    def test_env_contains_resolved_device_not_auto(self) -> None:
        """BIRDCLEF_DEVICE must be the concrete device, not 'auto'."""
        gc = GlobalConfig(training=TrainingSettings(device="auto"))
        with patch("agent.cli._mps_available", return_value=False), patch(
            "agent.cli._cuda_available", return_value=False
        ):
            env = _training_env_from_config(gc)
        assert env["BIRDCLEF_DEVICE"] == "cpu"  # not "auto"

    def test_env_passes_through_explicit_device(self) -> None:
        gc = GlobalConfig(training=TrainingSettings(device="mps"))
        with patch("agent.cli._mps_available", return_value=True):
            env = _training_env_from_config(gc)
        assert env["BIRDCLEF_DEVICE"] == "mps"

    def test_env_includes_batch_and_workers(self) -> None:
        gc = GlobalConfig(
            training=TrainingSettings(
                device="cpu",
                batch_size=256,
                num_workers=8,
                persistent_workers=False,
                prefetch_factor=6,
            )
        )
        env = _training_env_from_config(gc)
        assert env["BIRDCLEF_DEVICE"] == "cpu"
        assert env["BIRDCLEF_BATCH_SIZE"] == "256"
        assert env["BIRDCLEF_NUM_WORKERS"] == "8"
        assert env["BIRDCLEF_PERSISTENT_WORKERS"] == "false"
        assert env["BIRDCLEF_PREFETCH_FACTOR"] == "6"

    def test_env_omits_num_workers_when_none(self) -> None:
        """`num_workers=None` means 'auto-tune' — we do NOT forward it so
        the sandbox falls back to `_default_num_workers()`."""
        gc = GlobalConfig(
            training=TrainingSettings(device="cpu", num_workers=None)
        )
        env = _training_env_from_config(gc)
        assert "BIRDCLEF_NUM_WORKERS" not in env
