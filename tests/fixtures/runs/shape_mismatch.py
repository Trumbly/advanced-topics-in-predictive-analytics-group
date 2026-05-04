import torch
import torch.nn as nn

m = nn.Linear(10, 2)
x = torch.randn(4, 11)  # wrong feature dim
m(x)
