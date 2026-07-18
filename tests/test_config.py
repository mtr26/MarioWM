from pathlib import Path

import pytest

from world_model.config import ExperimentConfig


VALID_CONFIG = """
data:
  cache_dir: dataset/cache
  history: 4
  batch_size: 8
  validation_batch_size: 4
  workers: 0
  prefetch_factor: 2
model:
  n_actions: 7
  base_channels: 16
  channel_multipliers: [1, 2]
  blocks_per_level: 1
  action_embed_dim: 8
  condition_dim: 16
optimizer:
  learning_rate: 0.001
  weight_decay: 0.0001
  beta1: 0.9
  beta2: 0.999
  warmup_steps: 2
  gradient_clip: 1.0
runtime:
  output_dir: runs/test
  seed: 42
  epochs: 2
  device: cpu
  compile: false
  use_bfloat16: false
  use_tf32: false
  channels_last: false
  ema_decay: 0.99
  validation_every_epochs: 1
  rollout_horizon: 2
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_config_loads_and_normalizes_channel_multipliers(tmp_path):
    config = ExperimentConfig.from_yaml(_write(tmp_path, VALID_CONFIG))

    assert config.runtime.device == "cpu"
    assert config.model.channel_multipliers == (1, 2)
    assert config.to_dict()["data"]["history"] == 4


def test_config_rejects_unknown_top_level_section(tmp_path):
    invalid = VALID_CONFIG + "unknown:\n  value: true\n"

    with pytest.raises(ValueError, match="sections must be exactly"):
        ExperimentConfig.from_yaml(_write(tmp_path, invalid))


def test_config_rejects_invalid_ema_decay(tmp_path):
    invalid = VALID_CONFIG.replace("ema_decay: 0.99", "ema_decay: 1.0")

    with pytest.raises(ValueError, match="EMA decay"):
        ExperimentConfig.from_yaml(_write(tmp_path, invalid))
