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
    """The public `load_precomputed_dataset` signature must expose the CPU-
    saturation knobs (persistent_workers, prefetch_factor, auto-tuned
    num_workers). LLM-generated code relies on these defaults."""

    def test_signature_has_cpu_saturation_params(self) -> None:
        import inspect

        from pipelines.data_loader import load_precomputed_dataset

        params = inspect.signature(load_precomputed_dataset).parameters
        assert "num_workers" in params
        assert "persistent_workers" in params
        assert "prefetch_factor" in params
        assert "batch_size" in params

        # num_workers default must be None so the loader auto-tunes it
        assert params["num_workers"].default is None
        # persistent_workers default must be True so workers survive epochs
        assert params["persistent_workers"].default is True
        # prefetch_factor default must be > 2 so batches pipeline
        assert params["prefetch_factor"].default >= 4
        # batch_size default must be >= 128 (BLAS-friendly for CPU)
        assert params["batch_size"].default >= 128
