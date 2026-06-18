from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        hidden_layers: int = 1,
        dropout: float = 0.1,
        residual: bool = True,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = output_dim

        self.num_hidden_layers = hidden_layers
        self.residual = residual

        if hidden_layers == 0:
            self.linear = nn.Linear(input_dim, output_dim)
            return

        self.linear_layers = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim)]
            + [nn.Linear(hidden_dim, hidden_dim) for _ in range(hidden_layers - 1)]
            + [nn.Linear(hidden_dim, output_dim)]
        )
        self.dropout = nn.Dropout(dropout)
        self.norm_layers = nn.ModuleList(
            [nn.LayerNorm(hidden_dim, bias=False) for _ in range(hidden_layers - 1)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_hidden_layers == 0:
            return self.linear(x)

        x = self.linear_layers[0](x)
        for linear, norm in zip(self.linear_layers[1:-1], self.norm_layers):
            h = linear(self.dropout(F.gelu(x)))
            x = norm(h + x) if self.residual else norm(h)
        return self.linear_layers[-1](self.dropout(F.gelu(x)))
