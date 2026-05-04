import subprocess  # forbidden in any mode
import torch.nn as nn


def build_model(num_classes: int) -> nn.Module:
    return nn.Linear(10, num_classes)
