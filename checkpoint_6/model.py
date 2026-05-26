import torch
from torch import nn
import os

from extract_features import FeatureExtractor
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

        pretrained_model_hidden_dim = AutoConfig.from_pretrained(config.primary_model_name, token=HF_TOKEN).hidden_size

        if config.return_second_model_hs:
            gate_nn_input_dim = pretrained_model_hidden_dim * 4 + 1
        else:
            gate_nn_input_dim = pretrained_model_hidden_dim * 2 + 1

        metrics_nn_input_dim = len(config.primary_model_metrics)
        if config.second_model_metrics is not None:
            metrics_nn_input_dim += len(config.second_model_metrics)
        if config.return_xppl:
            metrics_nn_input_dim += 1
        
        agg_metrics_dim = len(config.primary_model_agg_metrics or [])
        if config.second_model_agg_metrics is not None:
            agg_metrics_dim += len(config.second_model_agg_metrics)
        if config.cross_model_agg_features is not None:
            agg_metrics_dim += len(config.cross_model_agg_features)

        self.feature_extractor = FeatureExtractor(
            primary_model_name=config.primary_model_name, 
            primary_model_metrics=config.primary_model_metrics,
            primary_model_agg_metrics=config.primary_model_agg_metrics,
            max_length=config.max_length, 
            second_model_name=config.second_model_name,
            second_model_metrics=config.second_model_metrics,
            second_model_agg_metrics=config.second_model_agg_metrics,
            cross_model_agg_features=config.cross_model_agg_features,
            return_xppl=config.return_xppl,
            return_second_model_hs=config.return_second_model_hs,
            hidden_state_fusion=config.hidden_state_fusion,
            hf_token=HF_TOKEN,
        )

        self.metrics_nn = MLP(
            input_dim=metrics_nn_input_dim,
            output_dim=config.metric_features,
            hidden_dim=config.mlp_hidden_features,
            hidden_layers=config.mlp_hidden_layers,
            dropout=config.mlp_dropout,
            residual=config.residual,
        )
        self.gate_nn = MLP(
            input_dim=gate_nn_input_dim,
            output_dim=config.gates,
            hidden_dim=config.mlp_hidden_features,
            hidden_layers=config.mlp_hidden_layers,
            dropout=config.mlp_dropout,
            residual=config.residual,
        )
        self.aggregate_nn = MLP(
            input_dim=config.metric_features,
            output_dim=1,
            hidden_dim=config.mlp_hidden_features,
            hidden_layers=config.mlp_hidden_layers,
            dropout=config.mlp_dropout,
            residual=config.residual,
        )
        if agg_metrics_dim > 0:
            self.agg_metrics_norm = nn.LayerNorm(agg_metrics_dim)
            self.agg_film = nn.Linear(agg_metrics_dim, 2 * config.metric_features)
            nn.init.zeros_(self.agg_film.weight)
            nn.init.zeros_(self.agg_film.bias)
        else:
            self.agg_metrics_norm = None
            self.agg_film = None
    
    def forward(self, texts: list[str], labels: torch.Tensor | None = None) -> torch.Tensor:
        features = self.feature_extractor(texts)
        metrics = features["metrics"]
        agg_metrics = features["agg_metrics"]
        primary_hidden_states = features["primary_hidden_states"]
        second_hidden_states = features["second_hidden_states"]
        attention_mask = features["attention_mask"]
        B, L, _ = primary_hidden_states.size()

        primary_current_hs = primary_hidden_states[:, :-1, :] # [B, L-1, H]
        primary_next_hs =  primary_hidden_states[:, 1:, :] # [B, L-1, H]
        pos_embeddings = self._pos_embeddings(L-1, B, primary_current_hs.device, self.pawn_config.max_length) # [B, L-1, 1]

        if second_hidden_states is not None:
            second_current_hs = second_hidden_states[:, :-1, :] # [B, L-1, H]
            second_next_hs =  second_hidden_states[:, 1:, :] # [B, L-1, H]
            gate_inputs = torch.cat([primary_current_hs, primary_next_hs, second_current_hs, second_next_hs, pos_embeddings], dim=-1) # [B, L-1, 4H + 1]
        else:
            gate_inputs = torch.cat([primary_current_hs, primary_next_hs, pos_embeddings], dim=-1) # [B, L-1, 2H + 1]
        
        attention_mask = attention_mask[:, :-1]
        gate_mask = self._gate_mask(attention_mask == 0)
        gate_logits = self.gate_nn(gate_inputs)
        gate_logits = gate_logits.masked_fill(gate_mask.unsqueeze(-1), float("-inf"))

        metrics_features = self.metrics_nn(metrics)
        G, M = gate_logits.size(-1), metrics_features.size(-1)
        if 1 < G < M:
            gate_logits = gate_logits.repeat(1, 1, M // G)

        aggregated_input = (gate_logits.softmax(dim=-2) * metrics_features).sum(dim=-2)

        if self.agg_film is not None:
            agg_metrics = self.agg_metrics_norm(agg_metrics)
            gamma, beta = self.agg_film(agg_metrics).chunk(2, dim=-1)
            gamma = 1.0 + gamma
            aggregated_input = gamma * aggregated_input + beta
        
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
