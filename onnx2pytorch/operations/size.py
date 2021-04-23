import torch
from torch import nn


class Size(nn.Module):
    def forward(self, input: torch.Tensor):
        return torch.tensor(input.numel())
