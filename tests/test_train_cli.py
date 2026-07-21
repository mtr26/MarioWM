import subprocess
import sys
from pathlib import Path

import torch

from train_world_model import build_loader, build_model
from world_model.config import ExperimentConfig
from world_model.conversion import ConversionConfig, convert_dataset
from world_model.data import MarioWindowDataset


def _config(cache: Path, output: Path, *, epochs: int = 1) -> str:
    return f"""
data:
  cache_dir: {cache}
  history: 2
  batch_size: 2
  validation_batch_size: 2
  workers: 0
  prefetch_factor: 2
model:
  n_actions: 7
  base_channels: 8
  channel_multipliers: [1, 2]
  blocks_per_level: 1
  action_embed_dim: 8
  condition_dim: 16
optimizer:
  learning_rate: 0.001
  weight_decay: 0.0001
  beta1: 0.9
  beta2: 0.999
  warmup_steps: 1
  gradient_clip: 1.0
runtime:
  output_dir: {output}
  seed: 42
  epochs: {epochs}
  device: cpu
  compile: false
  use_bfloat16: false
  use_tf32: false
  channels_last: false
  ema_decay: 0.9
  validation_every_epochs: 1
  rollout_horizon: 1
"""


def _prepared(synthetic_h5, tmp_path):
    return convert_dataset(
        ConversionConfig(
            input_path=synthetic_h5,
            output_dir=tmp_path / "cache",
            height=4,
            width=6,
            history=2,
            break_indices=(6,),
            workers=1,
        )
    )


def test_build_model_and_zero_worker_loader(synthetic_h5, tmp_path):
    cache = _prepared(synthetic_h5, tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(_config(cache, tmp_path / "run"), encoding="utf-8")
    config = ExperimentConfig.from_yaml(config_path)
    dataset = MarioWindowDataset(cache, "train", history=2)

    model = build_model(config)
    loader = build_loader(
        dataset,
        config.data,
        training=True,
        device=torch.device("cpu"),
    )
    batch = next(iter(loader))

    assert model.history == 2
    assert batch["context"].shape[1:] == (6, 4, 6)


def test_training_cli_writes_best_and_latest_checkpoints(synthetic_h5, tmp_path):
    cache = _prepared(synthetic_h5, tmp_path)
    output = tmp_path / "run"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(_config(cache, output), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "train_world_model.py",
            "--config",
            str(config_path),
            "--overfit-batches",
            "1",
            "--device",
            "cpu",
            "--no-compile",
        ],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert (output / "latest.pt").is_file()
    assert (output / "best.pt").is_file()
    assert (output / "previews" / "epoch-0000.png").is_file()
    assert "Training epoch 1/1" in result.stderr
    assert "Validating epoch 1/1" in result.stderr


def test_resume_preserves_best_checkpoint_without_improvement(
    synthetic_h5, tmp_path
):
    cache = _prepared(synthetic_h5, tmp_path)
    output = tmp_path / "run"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(_config(cache, output), encoding="utf-8")
    command = [
        sys.executable,
        "train_world_model.py",
        "--config",
        str(config_path),
        "--overfit-batches",
        "1",
        "--device",
        "cpu",
        "--no-compile",
    ]
    first = subprocess.run(
        command,
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert first.returncode == 0, first.stderr

    latest_path = output / "latest.pt"
    latest = torch.load(latest_path, map_location="cpu", weights_only=False)
    latest["best_validation_l1"] = -1.0
    torch.save(latest, latest_path)
    best_before = (output / "best.pt").read_bytes()
    config_path.write_text(_config(cache, output, epochs=2), encoding="utf-8")

    resumed = subprocess.run(
        [*command, "--resume", str(latest_path)],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        timeout=60,
    )

    assert resumed.returncode == 0, resumed.stderr
    assert (output / "best.pt").read_bytes() == best_before
