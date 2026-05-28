from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, model_validator


class ModelConfig(BaseModel):
    primary_model_name: str = "openai-community/gpt2"
    max_length: int = 512
    metric_features: int = 256
    gates: int = 256
    mlp_hidden_features: int = 256
    mlp_hidden_layers: int = 3
    mlp_dropout: float = 0.0
    token_dropout: float = 0.15
    residual: bool = True
    primary_model_metrics: Optional[list[str]] = ["entropy", "max_log_probs", "next_token_log_probs", "rank", "top_p"]
    primary_model_agg_metrics: Optional[list[str]] = None
    second_model_name: Optional[str] = None
    second_model_metrics: Optional[list[str]] = None
    second_model_agg_metrics: Optional[list[str]] = None
    cross_model_agg_features: Optional[list[str]] = None
    return_xppl: Optional[bool] = False
    return_second_model_hs: Optional[bool] = False
    hidden_state_fusion: Optional[str] = None

    @model_validator(mode="after")
    def validate_dimensions(self) -> ModelConfig:
        if self.metric_features % self.gates != 0:
            raise ValueError("metric_features must be divisible by gates")
        if self.hidden_state_fusion not in (None, "last", "uniform"):
            raise ValueError('hidden_state_fusion must be one of: None, "last", "uniform"')
        if self.second_model_metrics and self.second_model_name is None:
            raise ValueError("second_model_name must be set when second_model_metrics is used")
        if self.second_model_agg_metrics and self.second_model_name is None:
            raise ValueError("second_model_name must be set when second_model_agg_metrics is used")
        if self.cross_model_agg_features and self.second_model_name is None:
            raise ValueError("second_model_name must be set when cross_model_agg_features is used")
        if self.return_xppl and self.second_model_name is None:
            raise ValueError("second_model_name must be set when return_xppl is used")
        if self.return_second_model_hs and self.second_model_name is None:
            raise ValueError("second_model_name must be set when return_second_model_hs is used")
        return self


class OptimizerConfig(BaseModel):
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: Optional[float] = 1.0
    gradient_accumulation_steps: Optional[int] = 1
    label_smoothing: Optional[float]= 0.0
    pos_weight: Optional[float] = 1.0


class TrainerConfig(BaseModel):
    device: Optional[str] = None
    seed: int = 42
    epochs: int = 5


class DataConfig(BaseModel):
    batch_size: int = 4
    eval_batch_size: int = 4


class ExperimentConfig(BaseModel):
    model: ModelConfig
    optimizer: OptimizerConfig
    trainer: TrainerConfig
    data: DataConfig


def build_experiment_config(raw_config: dict[str, Any]) -> ExperimentConfig:
    return ExperimentConfig.model_validate(raw_config)
