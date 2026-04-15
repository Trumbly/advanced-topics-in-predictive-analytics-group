"""Audio preprocessing pipeline — fixed, deterministic, agent-configurable.

Responsibility
--------------
Convert raw BirdCLEF audio recordings into mel-spectrogram tensors that a
CNN can consume. Applies optional augmentation during training.

The agent may tune hyperparameters via `AudioPipelineConfig` (n_mels,
window size, augmentation flags, etc.) but must NOT edit this file. This
guarantees every experiment in a Study is comparable because the
preprocessing function is the same.

Pipeline stages
---------------
1. `load_audio(path)`       — load and resample to target sample rate
2. `to_mel_spectrogram(y)`  — compute log-mel-spectrogram (librosa)
3. `window(spec)`           — split into fixed-length windows with overlap
4. `augment(spec, cfg)`     — optional: time-shift, noise, mixup, specaugment

The preprocessing script (`scripts/build_profile.py`) uses stages 1-3 offline
to produce `.npy` files. Stage 4 (augmentation) is applied only at training
time by the data loader.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class AudioPipelineConfig:
    """Hyperparameters for mel-spectrogram generation and augmentation.

    Defaults are sensible for BirdCLEF (32 kHz field recordings, 5s windows).
    """

    sample_rate: int = 32_000
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 320                  # 320 -> 500 time frames per 5s (finer than 512 -> 313)
    fmin: float = 20.0
    fmax: float | None = 14_000.0           # 14 kHz covers virtually all bird vocalizations
    normalize: bool = True                  # normalize spectrogram to [0, 1]
    window_seconds: float = 5.0
    window_overlap: float = 0.0            # 0.0 = no overlap, 0.5 = 50% overlap
    power: float = 2.0                     # 2.0 = power spectrogram, 1.0 = amplitude
    to_db: bool = True                     # convert to log (dB) scale
    top_db: float = 80.0

    @property
    def window_samples(self) -> int:
        return int(self.sample_rate * self.window_seconds)

    @property
    def hop_samples(self) -> int:
        return int(self.window_samples * (1.0 - self.window_overlap))


@dataclass
class AugmentationConfig:
    """Augmentation settings applied at training time only."""

    time_shift: bool = True
    time_shift_max: float = 0.2            # fraction of window length
    noise_injection: bool = True
    noise_level: float = 0.005             # stddev of added Gaussian noise
    mixup: float = 0.2                     # 0.2 = typical mixup alpha, 0.0 = disabled
    specaugment: bool = True
    spec_freq_mask: int = 15               # max freq bins to mask
    spec_time_mask: int = 30               # max time frames to mask
    background_noise: bool = True          # mix pink noise to simulate ambient soundscapes
    background_noise_snr_db: float = 15.0  # signal-to-noise ratio in dB (lower = more noise)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AugmentationConfig":
        """Permissive constructor: ignores unknown keys.

        The LLM may return arbitrary augmentation dicts. We accept only the
        known fields and silently drop the rest so the agent loop never
        crashes on a typo from the model.
        """
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class AudioPipeline:
    """Load audio, compute mel-spectrograms, apply augmentation."""

    def __init__(self, config: AudioPipelineConfig | None = None) -> None:
        self.config = config or AudioPipelineConfig()

    # -- loading ------------------------------------------------------------

    def load_audio(self, path: Path) -> np.ndarray:
        """Load an audio file and resample to `config.sample_rate`.

        Returns a 1D float32 numpy array.
        """
        # Lazy import so the models package can be imported without librosa
        # installed (for CI / tests that only touch Pydantic models).
        import librosa  # type: ignore[import-not-found]

        path = Path(path)
        y, _ = librosa.load(str(path), sr=self.config.sample_rate, mono=True)
        return y.astype(np.float32)

    # -- spectrogram --------------------------------------------------------

    def to_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Compute a log-mel-spectrogram.

        Returns a 2D array of shape (n_mels, time_frames). If `to_db` is
        True the values are in dB, otherwise raw power/amplitude.
        """
        import librosa  # type: ignore[import-not-found]

        cfg = self.config
        spec = librosa.feature.melspectrogram(
            y=audio,
            sr=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            fmin=cfg.fmin,
            fmax=cfg.fmax if cfg.fmax is not None else cfg.sample_rate / 2,
            power=cfg.power,
        )
        if cfg.to_db:
            spec = librosa.power_to_db(spec, top_db=cfg.top_db)
        if cfg.normalize:
            # Normalize to [0, 1] for stable training across recordings
            s_min = spec.min()
            s_max = spec.max()
            if s_max - s_min > 1e-6:
                spec = (spec - s_min) / (s_max - s_min)
            else:
                spec = np.zeros_like(spec)
        return spec.astype(np.float32)

    # -- windowing ----------------------------------------------------------

    def window(self, audio: np.ndarray) -> list[np.ndarray]:
        """Split a 1D audio signal into fixed-length windows.

        The final window is padded with zeros if the audio is shorter than
        `window_samples`. If the audio is longer than one window, we slide
        with `hop_samples` and drop any incomplete tail window.
        """
        cfg = self.config
        n = len(audio)
        window_samples = cfg.window_samples
        hop_samples = cfg.hop_samples

        if n <= window_samples:
            padded = np.zeros(window_samples, dtype=audio.dtype)
            padded[:n] = audio
            return [padded]

        windows: list[np.ndarray] = []
        start = 0
        while start + window_samples <= n:
            windows.append(audio[start : start + window_samples])
            start += hop_samples
        return windows

    # -- augmentation (train-time only) -------------------------------------

    def augment(
        self,
        spec: np.ndarray,
        config: AugmentationConfig | dict[str, Any] | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Apply augmentation to a single mel-spectrogram.

        Accepts either an `AugmentationConfig` or a dict from the LLM.
        Returns a new array — never mutates the input.
        """
        if config is None:
            return spec
        if isinstance(config, dict):
            config = AugmentationConfig.from_dict(config)

        rng = rng or np.random.default_rng()
        out = spec.copy()

        if config.time_shift and out.shape[1] > 1:
            max_shift = int(out.shape[1] * config.time_shift_max)
            if max_shift > 0:
                shift = int(rng.integers(-max_shift, max_shift + 1))
                out = np.roll(out, shift, axis=1)

        if config.noise_injection:
            noise = rng.normal(0.0, config.noise_level, size=out.shape).astype(out.dtype)
            out = out + noise

        if config.background_noise:
            out = self._apply_background_noise(out, config, rng)

        if config.specaugment:
            out = self._apply_specaugment(out, config, rng)

        # NOTE: mixup is applied at the batch level by the data loader, not here.
        return out

    def _apply_background_noise(
        self,
        spec: np.ndarray,
        config: AugmentationConfig,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Mix pink-ish noise into the spectrogram to simulate ambient soundscapes.

        Pink noise has more energy at low frequencies, similar to real
        environmental recordings (wind, rain, distant traffic). This helps
        the model generalize from clean focal recordings to noisy test
        soundscapes.

        The SNR is randomized around the configured value (±5 dB) so the
        model sees a range of noise conditions.
        """
        out = spec.copy()
        # Randomize SNR around the target (±5 dB)
        snr_db = config.background_noise_snr_db + rng.uniform(-5.0, 5.0)

        # Generate pink-ish noise: scale white noise by 1/sqrt(freq_bin+1)
        freq_bins, time_frames = out.shape
        white = rng.normal(0.0, 1.0, size=out.shape).astype(out.dtype)
        # Pink noise filter: 1/sqrt(f) scaling per frequency bin
        freq_scale = 1.0 / np.sqrt(np.arange(1, freq_bins + 1, dtype=out.dtype))
        pink = white * freq_scale[:, np.newaxis]

        # Scale noise to achieve target SNR relative to signal power
        signal_power = np.mean(out ** 2) + 1e-10
        noise_power = np.mean(pink ** 2) + 1e-10
        snr_linear = 10.0 ** (snr_db / 10.0)
        scale = np.sqrt(signal_power / (noise_power * snr_linear))
        out = out + scale * pink
        return out

    def _apply_specaugment(
        self,
        spec: np.ndarray,
        config: AugmentationConfig,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Mask random frequency and time bands with the mean value of the spec."""
        out = spec.copy()
        fill = float(out.mean())

        freq_bins, time_frames = out.shape

        if config.spec_freq_mask > 0 and freq_bins > 0:
            f = int(rng.integers(0, min(config.spec_freq_mask, freq_bins) + 1))
            if f > 0:
                f0 = int(rng.integers(0, freq_bins - f + 1))
                out[f0 : f0 + f, :] = fill

        if config.spec_time_mask > 0 and time_frames > 0:
            t = int(rng.integers(0, min(config.spec_time_mask, time_frames) + 1))
            if t > 0:
                t0 = int(rng.integers(0, time_frames - t + 1))
                out[:, t0 : t0 + t] = fill

        return out

    # -- full pipeline for one audio file ----------------------------------

    def process_file(self, path: Path) -> list[np.ndarray]:
        """Load, window, and convert one audio file to mel-spectrograms.

        Returns a list of (n_mels, time_frames) arrays — one per window.
        This is the main entry point used by `scripts/build_profile.py`
        during offline preprocessing.
        """
        audio = self.load_audio(path)
        windows = self.window(audio)
        return [self.to_mel_spectrogram(w) for w in windows]
