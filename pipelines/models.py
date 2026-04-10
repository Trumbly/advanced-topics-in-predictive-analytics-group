"""Fixed model architectures the agent can import by name.

The model registry (`registry/models.yaml`) references these classes via
`import_snippet` strings. LLM-generated training code imports them with:

    from pipelines.models import CnnSmallV1
    model = CnnSmallV1(num_classes=234, in_channels=1)

Everything in this module is deliberately minimal — the building blocks,
not the experiment logic. The agent varies instantiation, hyperparameters,
and augmentation, but the classes themselves are stable.

Note: transfer-learning models like EfficientNet / ResNet are NOT exported
from here. Those are imported directly from torchvision in the
`import_snippet` field of their registry entry, because torchvision
already provides them. This file is for our own from-scratch baselines.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CnnSmallV1(nn.Module):
    """Small from-scratch CNN baseline for mel-spectrograms.

    Architecture:
        3 × (Conv2d → BatchNorm2d → ReLU → MaxPool2d)
        → AdaptiveAvgPool2d(1)
        → Linear(64 → num_classes)

    ~90k parameters — trains in under two minutes on CPU for the sample
    subset, making it an ideal pipeline smoke test and the first entry
    in the exploration phase. Produces logits (NOT probabilities); the
    caller is expected to apply `torch.sigmoid` or use `BCEWithLogitsLoss`.
    """

    def __init__(self, num_classes: int = 234, in_channels: int = 1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        flat = features.view(features.size(0), -1)
        return self.classifier(flat)


__all__ = ["CnnSmallV1"]
