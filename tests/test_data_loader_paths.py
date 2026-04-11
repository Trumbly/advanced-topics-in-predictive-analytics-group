"""Tests for the env-var fallback path resolution in `pipelines.data_loader`.

We don't spin up the whole dataset here — just verify that the helper
functions read BIRDCLEF_* env vars and fall back to the repo-relative
defaults when not set.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from pipelines.data_loader import (
    _default_labels_csv,
    _default_profile_path,
    _default_spectrograms_dir,
)


class TestDefaultPathResolution:
    """Env-var presence controls which path the data loader picks up."""

    def test_profile_env_var_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(
            "BIRDCLEF_DATASET_PROFILE", "/absolute/path/to/profile.json"
        )
        assert _default_profile_path() == Path("/absolute/path/to/profile.json")

    def test_spectrograms_env_var_wins(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("BIRDCLEF_SPECTROGRAMS_DIR", "/abs/specs")
        assert _default_spectrograms_dir() == Path("/abs/specs")

    def test_labels_env_var_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BIRDCLEF_LABELS_CSV", "/abs/labels.csv")
        assert _default_labels_csv() == Path("/abs/labels.csv")

    def test_fallback_without_env_vars(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without env vars, the defaults are the repo-relative paths."""
        monkeypatch.delenv("BIRDCLEF_DATASET_PROFILE", raising=False)
        monkeypatch.delenv("BIRDCLEF_SPECTROGRAMS_DIR", raising=False)
        monkeypatch.delenv("BIRDCLEF_LABELS_CSV", raising=False)
        assert _default_profile_path() == Path(
            "data/processed/dataset_profile.json"
        )
        assert _default_spectrograms_dir() == Path("data/processed/spectrograms")
        assert _default_labels_csv() == Path("data/processed/labels.csv")


class TestLoaderSignature:
    """The public `load_precomputed_dataset` signature exposes the CPU-
    saturation knobs as optional kwargs that default to None. None means
    "read from BIRDCLEF_* env vars, then fall back to hardcoded defaults"."""

    def test_signature_has_cpu_saturation_params(self) -> None:
        import inspect

        from pipelines.data_loader import load_precomputed_dataset

        params = inspect.signature(load_precomputed_dataset).parameters
        assert "num_workers" in params
        assert "persistent_workers" in params
        assert "prefetch_factor" in params
        assert "batch_size" in params

        # All four defaults must be None — the actual values come from
        # env vars (set by the CLI from config.yaml) or hardcoded fallbacks.
        assert params["num_workers"].default is None
        assert params["persistent_workers"].default is None
        assert params["prefetch_factor"].default is None
        assert params["batch_size"].default is None


class TestTrainingConfigFallbacks:
    """The _default_* helpers read BIRDCLEF_* env vars and fall back to
    safe hardcoded values. These are the single source of truth for how
    config.yaml flows into a sandbox subprocess."""

    def test_batch_size_from_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pipelines.data_loader import _default_batch_size

        monkeypatch.setenv("BIRDCLEF_BATCH_SIZE", "256")
        assert _default_batch_size() == 256

    def test_batch_size_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pipelines.data_loader import _default_batch_size

        monkeypatch.delenv("BIRDCLEF_BATCH_SIZE", raising=False)
        assert _default_batch_size() == 128

    def test_batch_size_ignores_garbage_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pipelines.data_loader import _default_batch_size

        monkeypatch.setenv("BIRDCLEF_BATCH_SIZE", "not-a-number")
        assert _default_batch_size() == 128

    def test_num_workers_from_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pipelines.data_loader import _default_num_workers

        monkeypatch.setenv("BIRDCLEF_NUM_WORKERS", "10")
        assert _default_num_workers() == 10

    def test_num_workers_auto_tune(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without env var: auto-tune → clamped to [2, 6]."""
        from pipelines.data_loader import _default_num_workers

        monkeypatch.delenv("BIRDCLEF_NUM_WORKERS", raising=False)
        n = _default_num_workers()
        assert 2 <= n <= 6

    def test_persistent_workers_from_env_truthy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pipelines.data_loader import _default_persistent_workers

        for truthy in ("true", "True", "1", "yes", "on"):
            monkeypatch.setenv("BIRDCLEF_PERSISTENT_WORKERS", truthy)
            assert _default_persistent_workers() is True

    def test_persistent_workers_from_env_falsy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pipelines.data_loader import _default_persistent_workers

        for falsy in ("false", "0", "no", "off"):
            monkeypatch.setenv("BIRDCLEF_PERSISTENT_WORKERS", falsy)
            assert _default_persistent_workers() is False

    def test_persistent_workers_default_true(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pipelines.data_loader import _default_persistent_workers

        monkeypatch.delenv("BIRDCLEF_PERSISTENT_WORKERS", raising=False)
        assert _default_persistent_workers() is True

    def test_prefetch_factor_from_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pipelines.data_loader import _default_prefetch_factor

        monkeypatch.setenv("BIRDCLEF_PREFETCH_FACTOR", "8")
        assert _default_prefetch_factor() == 8

    def test_prefetch_factor_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pipelines.data_loader import _default_prefetch_factor

        monkeypatch.delenv("BIRDCLEF_PREFETCH_FACTOR", raising=False)
        assert _default_prefetch_factor() == 4
