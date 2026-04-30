from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(StrictConfig):
    model_name: str = "openai-community/gpt2"
    max_length: int = 512
    metric_features: int = 256
    gates: int = 256
    mlp_hidden_features: int = 256
    mlp_hidden_layers: int = 3
    mlp_dropout: float = 0.0
    token_dropout: float = 0.15
    residual: bool = True

    @model_validator(mode="after")
    def validate_dimensions(self) -> ModelConfig:
        if self.metric_features % self.gates != 0:
            raise ValueError("metric_features must be divisible by gates")
        return self


class OptimizerConfig(StrictConfig):
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float | None = 1.0


class TrainerConfig(StrictConfig):
    device: str | None = None
    seed: int = 42
    epochs: int = 5


class DataConfig(StrictConfig):
    batch_size: int = 4
    eval_batch_size: int = 4
    num_workers: int = 0


class ExperimentConfig(StrictConfig):
    model: ModelConfig
    optimizer: OptimizerConfig
    trainer: TrainerConfig
    data: DataConfig


def build_experiment_config(raw_config: dict[str, Any]) -> ExperimentConfig:
    return ExperimentConfig.model_validate(raw_config)
