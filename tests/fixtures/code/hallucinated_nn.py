import torch.nn as nn


def build_model(num_classes: int) -> nn.Module:
    return nn.Conv2x2d(1, 8, kernel_size=3)  # not a real layer
