import torch.nn as nn


def build_model() -> nn.Module:  # missing num_classes
    return nn.Linear(10, 2)
