from __future__ import annotations

from torch import nn
import torch


class ProjectionMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], projection_dim: int, dropout: float):
        super().__init__()
        layers = []
        prev = input_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(prev, hidden), nn.ReLU(), nn.Dropout(dropout)])
            prev = hidden
        layers.append(nn.Linear(prev, projection_dim))
        self.projector = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.projector(x)
        return torch.nn.functional.normalize(z, p=2, dim=-1)

