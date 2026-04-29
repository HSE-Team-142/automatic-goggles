import torch
from torch import nn


class MLPBlock(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            dropout: float,
            residual: bool,
    ):
        super().__init__()
        self.residual = residual and input_dim == output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.net(x)
        if self.residual:
            output = output + x
        return output


class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dim: int = 256,
            hidden_layers: int = 3,
            dropout: float = 0.1,
            residual: bool = True,
    ):
        super().__init__()
        
        modules: list[nn.Module] = []

        dim = input_dim

        for _ in range(hidden_layers):
            modules.append(
                MLPBlock(
                    input_dim=dim,
                    output_dim=hidden_dim,
                    dropout=dropout,
                    residual=residual,
                )
            )
            dim = hidden_dim
        
        modules.append(nn.Linear(dim, output_dim))

        self.net = nn.Sequential(*modules)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
