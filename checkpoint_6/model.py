import torch
from torch import nn
import os

from frozen_pretrained_model import FronzenPretrainedModel
from mlp import MLP
from configs import ModelConfig

from dotenv import dotenv_values
from transformers import AutoConfig


try:
    ENV_CONFIG = dotenv_values(".env")
    HF_TOKEN = ENV_CONFIG.get("HF_TOKEN")
except:
    HF_TOKEN = None


class PAWN(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.pawn_config = config

        pretrained_model_hidden_dim = AutoConfig.from_pretrained(config.model_name, token=HF_TOKEN).hidden_size
        gate_nn_input_dim = pretrained_model_hidden_dim * 2 + 1

        self.frozen_pretrained_model = FronzenPretrainedModel(config.model_name, config.max_length, hf_token=HF_TOKEN)

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
        pos_embeddings = self._pos_embeddings(L-1, B, hidden_states.device, self.pawn_config.max_length)
        gate_inputs = torch.cat([current_hs, next_hs, pos_embeddings], dim=-1)
        attention_mask = attention_mask[:, :-1]
        gate_mask = self._gate_mask(attention_mask == 0)
        gate_logits = self.gate_nn(gate_inputs)
        gate_logits = gate_logits.masked_fill(gate_mask.unsqueeze(-1), float("-inf"))

        metrics_features = self.metrics_nn(metrics)
        G, M = gate_logits.size(-1), metrics_features.size(-1)
        if 1 < G < M:
            gate_logits = gate_logits.repeat(1, 1, M // G)

        aggregated_input = (gate_logits.softmax(dim=-2) * metrics_features).sum(dim=-2)
        aggregated_output = self.aggregate_nn(aggregated_input)

        return aggregated_output.squeeze(-1)

    def _pos_embeddings(self, length: int, batch_size: int, device: torch.device, max_length: int) -> torch.Tensor:
        pos_embeddings = torch.arange(length, device=device, dtype=torch.float32)
        pos_embeddings = pos_embeddings.unsqueeze(0).expand(batch_size, -1)
        return (pos_embeddings / max_length).unsqueeze(-1)
    
    def _gate_mask(self, mask: torch.Tensor) -> torch.Tensor:
        if not self.training or self.pawn_config.token_dropout == 0:
            return mask

        B, L = mask.size()
        device = mask.device

        dropout_mask = (torch.rand(B, L, device=device) < self.pawn_config.token_dropout)
        final_mask = dropout_mask | mask
        while final_mask.all(dim=-1).any().item() is True:
            dropout_mask = (torch.rand(B, L, device=device) < self.pawn_config.token_dropout)
            final_mask = dropout_mask | mask
        
        return final_mask
