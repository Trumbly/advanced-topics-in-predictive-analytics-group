"""Unit tests for `pipelines.audio_pipeline`."""

from __future__ import annotations

import numpy as np
import pytest

from pipelines.audio_pipeline import (
    AudioPipeline,
    AudioPipelineConfig,
    AugmentationConfig,
)


# ---------------------------------------------------------------------------
# AudioPipelineConfig
# ---------------------------------------------------------------------------


class TestAudioPipelineConfig:
    def test_defaults(self) -> None:
        cfg = AudioPipelineConfig()
        assert cfg.sample_rate == 32_000
        assert cfg.n_mels == 128
        assert cfg.window_seconds == 5.0

    def test_window_samples_computed(self) -> None:
        cfg = AudioPipelineConfig(sample_rate=16_000, window_seconds=2.0)
        assert cfg.window_samples == 32_000

    def test_hop_samples_no_overlap(self) -> None:
        cfg = AudioPipelineConfig(sample_rate=32_000, window_seconds=5.0, window_overlap=0.0)
        assert cfg.hop_samples == cfg.window_samples

    def test_hop_samples_half_overlap(self) -> None:
        cfg = AudioPipelineConfig(sample_rate=32_000, window_seconds=5.0, window_overlap=0.5)
        assert cfg.hop_samples == cfg.window_samples // 2


# ---------------------------------------------------------------------------
# AugmentationConfig
# ---------------------------------------------------------------------------


class TestAugmentationConfig:
    def test_from_dict_ignores_unknown_keys(self) -> None:
        cfg = AugmentationConfig.from_dict(
            {
                "time_shift": True,
                "mixup": 0.3,
                "garbage_from_llm": "ignored",
                "another_bogus_field": 42,
            }
        )
        assert cfg.time_shift is True
        assert cfg.mixup == 0.3

    def test_from_dict_empty(self) -> None:
        cfg = AugmentationConfig.from_dict({})
        # Default values must still apply
        assert cfg.time_shift is True
        assert cfg.noise_injection is True


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------


class TestWindowing:
    @pytest.fixture
    def pipeline(self) -> AudioPipeline:
        return AudioPipeline(
            AudioPipelineConfig(sample_rate=1000, window_seconds=1.0, window_overlap=0.0)
        )

    def test_short_audio_is_padded(self, pipeline: AudioPipeline) -> None:
        audio = np.ones(500, dtype=np.float32)
        windows = pipeline.window(audio)
        assert len(windows) == 1
        assert len(windows[0]) == 1000
        assert (windows[0][:500] == 1.0).all()
        assert (windows[0][500:] == 0.0).all()

    def test_exact_length_produces_one_window(self, pipeline: AudioPipeline) -> None:
        audio = np.arange(1000, dtype=np.float32)
        windows = pipeline.window(audio)
        assert len(windows) == 1
        np.testing.assert_array_equal(windows[0], audio)

    def test_long_audio_multiple_windows(self, pipeline: AudioPipeline) -> None:
        audio = np.arange(2500, dtype=np.float32)
        windows = pipeline.window(audio)
        assert len(windows) == 2  # incomplete tail is dropped
        assert all(len(w) == 1000 for w in windows)

    def test_windows_with_overlap(self) -> None:
        pipeline = AudioPipeline(
            AudioPipelineConfig(sample_rate=1000, window_seconds=1.0, window_overlap=0.5)
        )
        audio = np.arange(2500, dtype=np.float32)
        windows = pipeline.window(audio)
        # hop = 500, so windows start at 0, 500, 1000, 1500
        assert len(windows) == 4


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------


class TestAugmentation:
    @pytest.fixture
    def pipeline(self) -> AudioPipeline:
        return AudioPipeline()

    @pytest.fixture
    def spec(self) -> np.ndarray:
        rng = np.random.default_rng(0)
        return rng.normal(0, 1, size=(128, 256)).astype(np.float32)

    def test_none_config_returns_input(
        self, pipeline: AudioPipeline, spec: np.ndarray
    ) -> None:
        out = pipeline.augment(spec, None)
        np.testing.assert_array_equal(out, spec)

    def test_augment_does_not_mutate_input(
        self, pipeline: AudioPipeline, spec: np.ndarray
    ) -> None:
        spec_copy = spec.copy()
        pipeline.augment(spec, AugmentationConfig(time_shift=True, noise_injection=True))
        np.testing.assert_array_equal(spec, spec_copy)

    def test_augment_changes_output(
        self, pipeline: AudioPipeline, spec: np.ndarray
    ) -> None:
        out = pipeline.augment(
            spec,
            AugmentationConfig(time_shift=True, noise_injection=True, noise_level=0.1),
            rng=np.random.default_rng(42),
        )
        assert not np.array_equal(out, spec)
        assert out.shape == spec.shape

    def test_specaugment_masks_bands(
        self, pipeline: AudioPipeline, spec: np.ndarray
    ) -> None:
        out = pipeline.augment(
            spec,
            AugmentationConfig(
                time_shift=False,
                noise_injection=False,
                specaugment=True,
                spec_freq_mask=10,
                spec_time_mask=20,
            ),
            rng=np.random.default_rng(0),
        )
        assert out.shape == spec.shape

    def test_accepts_dict_from_llm(
        self, pipeline: AudioPipeline, spec: np.ndarray
    ) -> None:
        # The LLM returns a dict — verify the pipeline handles it
        out = pipeline.augment(
            spec,
            {"time_shift": True, "noise_injection": True, "unknown_key": "ignored"},
            rng=np.random.default_rng(0),
        )
        assert out.shape == spec.shape
