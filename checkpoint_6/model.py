import torch
from torch import nn

from dataclasses import dataclass

from frozen_pretrained_model import FronzenPretrainedModel
from mlp import MLP

from transformers import AutoConfig

@dataclass
class PAWNConfig:
    max_length: int = 512
    metric_features: int = 256
    gates: int = 256
    mlp_hidden_features: int = 256
    mlp_hidden_layers: int = 3
    mlp_dropout: float = 0.0
    token_dropout: float = 0.15
    model_name: str = "openai-community/gpt2"

    def __post_init__(self) -> None:
        if self.metric_features % self.gates != 0:
            raise ValueError("metric_features must be divisible by gates")

class PAWN(nn.Module):
    def __init__(self, config: PAWNConfig):
        super().__init__()
        self.config = config

        pretrained_model_hidden_dim = AutoConfig.from_pretrained(config.model_name).hidden_size
        gate_nn_input_dim = pretrained_model_hidden_dim * 2 + 1

        self.frozen_pretrained_model = FronzenPretrainedModel(config.model_name, self.config.max_length)

        self.metrics_nn = MLP(
            input_dim=5,
            output_dim=config.metric_features,
            hidden_dim=config.mlp_hidden_features,
            hidden_layers=config.mlp_hidden_layers,
            dropout=config.mlp_dropout,
        )
        self.gate_nn = MLP(
            input_dim=gate_nn_input_dim,
            output_dim=config.gates,
            hidden_dim=config.mlp_hidden_features,
            hidden_layers=config.mlp_hidden_layers,
            dropout=config.mlp_dropout,
        )
        self.aggregate_nn = MLP(
            input_dim=config.metric_features,
            output_dim=1,
            hidden_dim=config.mlp_hidden_features,
            hidden_layers=config.mlp_hidden_layers,
            dropout=config.mlp_dropout,
        )
    
    def forward(self, texts: list[str]) -> torch.Tensor:
        metrics, hidden_states, attention_mask = self.frozen_pretrained_model(texts)
        B, L, _ = hidden_states.size()

        current_hs = hidden_states[:, :-1, :]
        next_hs =  hidden_states[:, 1:, :]
        pos_embeddings = self._pos_embeddings(L-1, B, hidden_states.device, self.config.max_length)
        gate_inputs = torch.cat([current_hs, next_hs, pos_embeddings], dim=-1)
        gate_mask = self._gate_mask(attention_mask[:, 1:] == 0)
        gate_logits = self.gate_nn(gate_inputs)
        gate_logits = gate_logits.masked_fill(gate_mask.unsqueeze(-1), float("-inf"))

        weights = torch.softmax(gate_logits, dim=-2)
        weights = weights.repeat_interleave(self.config.metric_features // self.config.gates, dim=-1)

        metrics_features = self.metrics_nn(metrics)

        aggregate = (weights * metrics_features).sum(dim=1)
        output = self.aggregate_nn(aggregate)

        return output.squeeze(-1)

    def _pos_embeddings(self, length: int, batch_size: int, device: torch.device, max_length: int) -> torch.Tensor:
        pos_embeddings = torch.arange(length, device=device, dtype=torch.float32)
        pos_embeddings = pos_embeddings.unsqueeze(0).expand(batch_size, -1)
        return (pos_embeddings / max_length).unsqueeze(-1)
    
    def _gate_mask(self, mask: torch.Tensor) -> torch.Tensor:
        B, L = mask.size()
        device = mask.device

        dropout_mask = (torch.rand(B, L, device=device) < self.config.token_dropout)
        final_mask = dropout_mask | mask
        while final_mask.all(dim=-1).any().item() is True:
            dropout_mask = (torch.rand(B, L, device=device) < self.config.token_dropout)
            final_mask = dropout_mask | mask
        
        return final_mask
