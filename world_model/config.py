"""Validated YAML configuration for deterministic world-model training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DataConfig:
    cache_dir: str
    history: int
    batch_size: int
    validation_batch_size: int
    workers: int
    prefetch_factor: int


@dataclass(frozen=True)
class ModelConfig:
    n_actions: int
    base_channels: int
    channel_multipliers: tuple[int, ...]
    blocks_per_level: int
    action_embed_dim: int
    condition_dim: int


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float
    warmup_steps: int
    gradient_clip: float


@dataclass(frozen=True)
class RuntimeConfig:
    output_dir: str
    seed: int
    epochs: int
    device: str
    compile: bool
    use_bfloat16: bool
    use_tf32: bool
    channels_last: bool
    ema_decay: float
    validation_every_epochs: int
    rollout_horizon: int


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    runtime: RuntimeConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        required = {"data", "model", "optimizer", "runtime"}
        if not isinstance(raw, dict) or set(raw) != required:
            raise ValueError(
                f"configuration sections must be exactly {sorted(required)}"
            )

        model_values = dict(raw["model"])
        model_values["channel_multipliers"] = tuple(
            model_values["channel_multipliers"]
        )
        try:
            config = cls(
                data=DataConfig(**raw["data"]),
                model=ModelConfig(**model_values),
                optimizer=OptimizerConfig(**raw["optimizer"]),
                runtime=RuntimeConfig(**raw["runtime"]),
            )
        except TypeError as error:
            raise ValueError(f"invalid configuration field: {error}") from error
        config.validate()
        return config

    def validate(self) -> None:
        if self.data.history < 1 or self.data.batch_size < 1:
            raise ValueError("history and batch size must be positive")
        if self.data.validation_batch_size < 1 or self.data.workers < 0:
            raise ValueError(
                "validation batch size must be positive and workers non-negative"
            )
        if self.data.prefetch_factor < 1:
            raise ValueError("prefetch factor must be positive")
        if self.runtime.epochs < 1 or self.runtime.validation_every_epochs < 1:
            raise ValueError("epoch counts must be positive")
        if self.runtime.rollout_horizon < 1:
            raise ValueError("rollout horizon must be positive")
        if not 0 < self.runtime.ema_decay < 1:
            raise ValueError("EMA decay must be between zero and one")
        if self.optimizer.learning_rate <= 0 or self.optimizer.warmup_steps < 0:
            raise ValueError(
                "learning rate must be positive and warmup non-negative"
            )
        if self.optimizer.gradient_clip <= 0:
            raise ValueError("gradient clip must be positive")
        if not 0 <= self.optimizer.weight_decay:
            raise ValueError("weight decay must be non-negative")
        if not 0 <= self.optimizer.beta1 < 1 or not 0 <= self.optimizer.beta2 < 1:
            raise ValueError("optimizer betas must be in [0, 1)")
        if len(self.model.channel_multipliers) < 2:
            raise ValueError("at least two channel multipliers are required")
        if any(value < 1 for value in self.model.channel_multipliers):
            raise ValueError("channel multipliers must be positive")
        if (
            self.model.n_actions < 1
            or self.model.base_channels < 1
            or self.model.blocks_per_level < 1
            or self.model.action_embed_dim < 1
            or self.model.condition_dim < 1
        ):
            raise ValueError("model dimensions must be positive")

    def to_dict(self) -> dict:
        return asdict(self)
