"""Fixed model architectures the agent can import by name.

The model registry (`registry/models.yaml`) references these classes via
`import_snippet` strings. LLM-generated training code imports them with:

    from pipelines.models import CnnSmallV1
    model = CnnSmallV1(num_classes=num_classes, in_channels=1)

Everything in this module is deliberately minimal — the building blocks,
not the experiment logic. The agent varies instantiation, hyperparameters,
and augmentation, but the classes themselves are stable.

Transfer-learning wrappers
--------------------------
`TorchvisionAdapter` wraps a torchvision classifier (EfficientNet-B0,
ResNet18, etc.) so that our single-channel 128-mel spectrograms can feed
into it without the LLM having to reinvent the 1→3-channel conversion
and the 224×224 resize every time. It also uses `weights=None` by
default because the sandbox has no internet access, so pretrained
checkpoints would fail to download.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class TorchvisionAdapter(nn.Module):
    """Wrap a torchvision image classifier for single-channel spectrograms.

    Input tensors coming from our data loader have shape
    ``(batch, 1, n_mels, time_frames)`` — typically ``(B, 1, 128, 313)``.
    Torchvision models expect ``(batch, 3, 224, 224)``. This adapter:

      1. Resizes the spatial dims to ``(224, 224)`` using bilinear
         interpolation, and
      2. Repeats the single channel three times to produce an RGB-shaped
         tensor the backbone understands,

    before forwarding through the backbone. The backbone's classifier head
    is replaced with ``nn.LazyLinear(num_classes)`` so you never have to
    compute ``in_features`` manually and the wrapper works across every
    torchvision architecture (EfficientNet uses ``backbone.classifier``,
    ResNet uses ``backbone.fc``, etc.).

    Example::

        from pipelines.models import TorchvisionAdapter
        model = TorchvisionAdapter("efficientnet_b0", num_classes=num_classes)

    Parameters
    ----------
    backbone_name:
        Name of the torchvision constructor, e.g. ``"efficientnet_b0"``,
        ``"resnet18"``, ``"mobilenet_v3_small"``.
    num_classes:
        Number of output logits (must match the actual dataset — do NOT
        hardcode 234 when the profile says 206).
    pretrained:
        If ``True``, attempts to load the default pretrained weights
        (requires a cached download; the sandbox does not have internet,
        so this defaults to ``False``).
    input_size:
        Spatial size the backbone expects, default ``(224, 224)``.

    Notes
    -----
    ``nn.LazyLinear`` materializes on the first forward pass. The caller
    MUST move the model to the training device BEFORE the first forward
    so the lazy layer ends up on the right device — this is what the
    standard training skeleton does, in that order.
    """

    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        *,
        pretrained: bool = False,
        input_size: tuple[int, int] = (224, 224),
    ) -> None:
        super().__init__()

        import torchvision.models as tvm

        ctor = getattr(tvm, backbone_name, None)
        if ctor is None:
            available = [
                n for n in dir(tvm) if not n.startswith("_") and n.islower()
            ]
            raise ValueError(
                f"Unknown torchvision backbone {backbone_name!r}. "
                f"Try one of: {sorted(available)[:10]}..."
            )

        weights: Any = None
        if pretrained:
            # Torchvision exposes `<Name>_Weights.DEFAULT` per model.
            weights_cls_name = "".join(
                p.capitalize() for p in backbone_name.split("_")
            ) + "_Weights"
            weights_cls = getattr(tvm, weights_cls_name, None)
            if weights_cls is not None:
                weights = getattr(weights_cls, "DEFAULT", None)

        try:
            backbone = ctor(weights=weights)
        except Exception:
            # No internet / no cached weights — fall back to random init.
            backbone = ctor(weights=None)

        # Replace the classifier head with a LazyLinear → num_classes.
        # Torchvision exposes the final FC either as `.fc` (ResNet-family)
        # or `.classifier` (EfficientNet, MobileNet, VGG). We swap whichever
        # one exists with `nn.Identity()` and put our own head after the
        # forward pass.
        self._head_attr: str | None = None
        if hasattr(backbone, "fc") and isinstance(backbone.fc, nn.Module):
            self._head_attr = "fc"
            backbone.fc = nn.Identity()
        elif hasattr(backbone, "classifier") and isinstance(
            backbone.classifier, nn.Module
        ):
            self._head_attr = "classifier"
            backbone.classifier = nn.Identity()

        self.backbone = backbone
        self.head = nn.LazyLinear(num_classes)
        self.input_size = input_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, 1, n_mels, T) → (B, 3, 224, 224)
        if x.dim() != 4:
            raise ValueError(
                f"TorchvisionAdapter expects 4D (B, C, H, W) input, got {tuple(x.shape)}"
            )
        if x.size(1) == 1:
            x = x.expand(-1, 3, -1, -1)
        if x.shape[-2:] != self.input_size:
            x = F.interpolate(
                x, size=self.input_size, mode="bilinear", align_corners=False
            )
        features = self.backbone(x)
        if features.dim() > 2:
            features = features.flatten(1)
        return self.head(features)


__all__ = ["CnnSmallV1", "TorchvisionAdapter"]
