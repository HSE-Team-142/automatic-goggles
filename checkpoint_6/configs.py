from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any


@dataclass
class ModelConfig:
    model_name: str = "openai-community/gpt2"
    max_length: int = 512
    metric_features: int = 256
    gates: int = 256
    mlp_hidden_features: int = 256
    mlp_hidden_layers: int = 3
    mlp_dropout: float = 0.0
    token_dropout: float = 0.15
    residual: bool = True

    def __post_init__(self) -> None:
        if self.metric_features % self.gates != 0:
            raise ValueError("metric_features must be divisible by gates")


@dataclass
class OptimizerConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float | None = 1.0


@dataclass
class TrainerConfig:
    output_dir: str = "checkpoint_6/outputs"
    device: str | None = None
    seed: int = 42
    epochs: int = 5


@dataclass
class DataConfig:
    batch_size: int = 4
    eval_batch_size: int = 4
    num_workers: int = 0


@dataclass
class ExperimentConfig:
    model: ModelConfig
    optimizer: OptimizerConfig
    trainer: TrainerConfig
    data: DataConfig


def build_experiment_config(raw_config: dict[str, Any]) -> ExperimentConfig:
    return ExperimentConfig(
        model=build_dataclass(ModelConfig, raw_config, "model"),
        optimizer=build_dataclass(OptimizerConfig, raw_config, "optimizer"),
        trainer=build_dataclass(TrainerConfig, raw_config, "trainer"),
        data=build_dataclass(DataConfig, raw_config, "data"),
    )


def build_dataclass(cls: type, raw_config: dict[str, Any], section: str):
    section_config = raw_config.get(section)
    if not isinstance(section_config, dict):
        raise ValueError(f"config section `{section}` must be a mapping")

    known_fields = {field.name for field in fields(cls)}
    unknown_fields = sorted(set(section_config) - known_fields)
    if unknown_fields:
        raise ValueError(f"unknown keys in config section `{section}`: {unknown_fields}")

    return cls(**section_config)
