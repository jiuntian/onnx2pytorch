import torch
from torch import nn


class Expand(nn.Module):
    def forward(self, input: torch.Tensor, shape: torch.Tensor):
        return torch.tensor(input.expand(shape))
