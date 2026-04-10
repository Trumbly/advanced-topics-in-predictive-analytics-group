"""Fixed audio preprocessing pipeline for BirdCLEF 2026.

This package contains the NON-LLM building blocks that the agent uses but
never modifies. The agent configures these modules via dicts (e.g. augmentation
settings) but the code itself is stable, tested, and deterministic.

Modules:
    audio_pipeline    — load audio, compute mel-spectrograms, apply augmentation
    dataset_profile   — build a DatasetProfile from a preprocessed dataset
    data_loader       — load precomputed spectrograms into torch datasets
"""

from pipelines.audio_pipeline import AudioPipeline, AudioPipelineConfig
from pipelines.dataset_profile import build_dataset_profile
from pipelines.data_loader import PrecomputedSpectrogramDataset, load_precomputed_dataset

__all__ = [
    "AudioPipeline",
    "AudioPipelineConfig",
    "build_dataset_profile",
    "PrecomputedSpectrogramDataset",
    "load_precomputed_dataset",
]
