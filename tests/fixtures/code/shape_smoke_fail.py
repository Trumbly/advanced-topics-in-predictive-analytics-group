import torch.nn as nn


def build_model(num_classes: int) -> nn.Module:
    # output shape will be wrong: (B, 99) instead of (B, num_classes)
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(1 * 128 * 313, 99),
    )
