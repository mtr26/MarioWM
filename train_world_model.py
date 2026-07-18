"""Train the deterministic action-conditioned Mario world model."""

from __future__ import annotations

import argparse
import inspect
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from world_model.checkpointing import load_checkpoint, save_checkpoint
from world_model.config import DataConfig, ExperimentConfig, OptimizerConfig, RuntimeConfig
from world_model.data import MarioWindowDataset
from world_model.model import ActionConditionedUNet
from world_model.training import (
    EMA,
    autoregressive_rollout,
    evaluate,
    save_prediction_grid,
    save_rollout_grid,
    train_one_epoch,
    warmup_cosine_factor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a deterministic action-conditioned Mario U-Net"
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/deterministic_unet.yaml")
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--overfit-batches",
        type=int,
        help="Restrict training to this many fixed batches for pipeline validation",
    )
    parser.add_argument("--device", type=str, help="Override configured device")
    parser.add_argument("--no-compile", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_torch(runtime: RuntimeConfig) -> None:
    torch.backends.cuda.matmul.allow_tf32 = runtime.use_tf32
    torch.backends.cudnn.allow_tf32 = runtime.use_tf32
    if runtime.use_tf32:
        torch.set_float32_matmul_precision("high")


def build_loader(
    dataset,
    config: DataConfig,
    *,
    training: bool,
    device: torch.device,
) -> DataLoader:
    workers = config.workers
    kwargs = {
        "dataset": dataset,
        "batch_size": (
            config.batch_size if training else config.validation_batch_size
        ),
        "shuffle": training,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "drop_last": training and len(dataset) >= config.batch_size,
    }
    if workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = config.prefetch_factor
    return DataLoader(**kwargs)


def build_model(config: ExperimentConfig) -> ActionConditionedUNet:
    model = config.model
    return ActionConditionedUNet(
        history=config.data.history,
        n_actions=model.n_actions,
        base_channels=model.base_channels,
        channel_multipliers=model.channel_multipliers,
        blocks_per_level=model.blocks_per_level,
        action_embed_dim=model.action_embed_dim,
        cond_dim=model.condition_dim,
    )


def build_optimizer(
    model: torch.nn.Module,
    config: OptimizerConfig,
    device: torch.device,
) -> torch.optim.AdamW:
    kwargs = {
        "params": model.parameters(),
        "lr": config.learning_rate,
        "weight_decay": config.weight_decay,
        "betas": (config.beta1, config.beta2),
    }
    if "fused" in inspect.signature(torch.optim.AdamW).parameters:
        kwargs["fused"] = device.type == "cuda"
    return torch.optim.AdamW(**kwargs)


def _validation_preview(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_path: Path,
    use_bfloat16: bool,
) -> None:
    batch = next(iter(loader))
    context = batch["context"].to(device)
    actions = batch["actions"].to(device)
    target = batch["target"].to(device)
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if use_bfloat16 and device.type == "cuda"
        else torch.no_grad()
    )
    with torch.inference_mode(), autocast:
        prediction = model(context, actions)
    save_prediction_grid(output_path, context, target, prediction)


def _rollout_preview(
    model: torch.nn.Module,
    dataset: MarioWindowDataset | None,
    horizon: int,
    device: torch.device,
    output_path: Path,
) -> bool:
    if dataset is None:
        return False
    rollout = None
    for index in range(min(len(dataset), 10_000)):
        try:
            rollout = dataset.get_rollout(index, horizon)
            break
        except ValueError:
            continue
    if rollout is None:
        return False

    initial = rollout["initial_context"].unsqueeze(0).to(device)
    actions = rollout["action_sequence"].unsqueeze(0).to(device)
    targets = rollout["targets"].unsqueeze(0)
    predictions = autoregressive_rollout(
        model, initial, actions, horizon=horizon
    ).cpu()
    save_rollout_grid(output_path, targets, predictions)
    return True


def run_training_loop(
    *,
    raw_model: torch.nn.Module,
    train_model: torch.nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    rollout_dataset: MarioWindowDataset | None,
    optimizer: torch.optim.Optimizer,
    scheduler,
    ema: EMA,
    config: ExperimentConfig,
    device: torch.device,
    start_epoch: int,
    global_step: int,
    best_validation_l1: float,
) -> None:
    output_dir = Path(config.runtime.output_dir)
    preview_dir = output_dir / "previews"
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
    best_l1 = best_validation_l1

    try:
        for epoch in range(start_epoch, config.runtime.epochs):
            train_metrics, global_step = train_one_epoch(
                model=train_model,
                loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                ema=ema,
                device=device,
                use_bfloat16=config.runtime.use_bfloat16,
                channels_last=config.runtime.channels_last,
                gradient_clip=config.optimizer.gradient_clip,
                global_step=global_step,
            )
            writer.add_scalar("train/l1", train_metrics["l1"], global_step)
            writer.add_scalar(
                "train/learning_rate", optimizer.param_groups[0]["lr"], global_step
            )

            validation_metrics = None
            should_validate = (
                (epoch + 1) % config.runtime.validation_every_epochs == 0
                or epoch + 1 == config.runtime.epochs
            )
            if should_validate:
                with ema.average_parameters(raw_model):
                    validation_metrics = evaluate(
                        model=raw_model,
                        loader=validation_loader,
                        device=device,
                        use_bfloat16=config.runtime.use_bfloat16,
                        channels_last=config.runtime.channels_last,
                        n_actions=config.model.n_actions,
                    )
                    _validation_preview(
                        raw_model,
                        validation_loader,
                        device,
                        preview_dir / f"epoch-{epoch:04d}.png",
                        config.runtime.use_bfloat16,
                    )
                    _rollout_preview(
                        raw_model,
                        rollout_dataset,
                        config.runtime.rollout_horizon,
                        device,
                        preview_dir / f"rollout-{epoch:04d}.png",
                    )

                writer.add_scalar(
                    "validation/l1", validation_metrics["l1"], global_step
                )
                writer.add_scalar(
                    "validation/mse", validation_metrics["mse"], global_step
                )
                for action, value in validation_metrics["per_action_l1"].items():
                    if math.isfinite(value):
                        writer.add_scalar(
                            f"validation/action_{action}_l1", value, global_step
                        )

            improved = bool(
                validation_metrics and validation_metrics["l1"] < best_l1
            )
            if improved:
                best_l1 = validation_metrics["l1"]

            save_checkpoint(
                output_dir / "latest.pt",
                model=raw_model,
                ema_state=ema.state_dict(),
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                config=config.to_dict(),
                best_validation_l1=best_l1,
            )
            if improved:
                save_checkpoint(
                    output_dir / "best.pt",
                    model=raw_model,
                    ema_state=ema.state_dict(),
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    global_step=global_step,
                    config=config.to_dict(),
                    best_validation_l1=best_l1,
                )
            message = (
                f"epoch={epoch} step={global_step} train_l1={train_metrics['l1']:.6f}"
            )
            if validation_metrics:
                message += f" validation_l1={validation_metrics['l1']:.6f}"
            print(message, flush=True)
    finally:
        writer.close()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    set_seed(config.runtime.seed)
    configure_torch(config.runtime)

    device = torch.device(args.device or config.runtime.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training was requested but CUDA is unavailable")
    if args.overfit_batches is not None and args.overfit_batches < 1:
        raise ValueError("overfit batches must be positive")

    base_train_dataset = MarioWindowDataset(
        config.data.cache_dir, "train", config.data.history
    )
    base_validation_dataset = MarioWindowDataset(
        config.data.cache_dir, "validation", config.data.history
    )
    train_dataset = base_train_dataset
    validation_dataset = base_validation_dataset
    rollout_dataset: MarioWindowDataset | None = base_validation_dataset
    if args.overfit_batches:
        limit = min(
            len(base_train_dataset),
            args.overfit_batches * config.data.batch_size,
        )
        fixed_indices = list(range(limit))
        train_dataset = Subset(base_train_dataset, fixed_indices)
        validation_dataset = train_dataset
        rollout_dataset = None

    train_loader = build_loader(
        train_dataset, config.data, training=True, device=device
    )
    validation_loader = build_loader(
        validation_dataset, config.data, training=False, device=device
    )
    raw_model = build_model(config).to(device)
    if config.runtime.channels_last:
        raw_model.to(memory_format=torch.channels_last)
    optimizer = build_optimizer(raw_model, config.optimizer, device)
    total_steps = config.runtime.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: warmup_cosine_factor(
            step, config.optimizer.warmup_steps, total_steps
        ),
    )
    ema = EMA(raw_model, config.runtime.ema_decay)
    start_epoch = 0
    global_step = 0
    best_validation_l1 = math.inf
    if args.resume:
        state = load_checkpoint(
            args.resume,
            model=raw_model,
            optimizer=optimizer,
            scheduler=scheduler,
            restore_rng=True,
        )
        ema.load_state_dict(state["ema"])
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state["global_step"])
        best_validation_l1 = float(state.get("best_validation_l1", math.inf))

    should_compile = config.runtime.compile and not args.no_compile
    train_model = torch.compile(raw_model) if should_compile else raw_model
    run_training_loop(
        raw_model=raw_model,
        train_model=train_model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        rollout_dataset=rollout_dataset,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        config=config,
        device=device,
        start_epoch=start_epoch,
        global_step=global_step,
        best_validation_l1=best_validation_l1,
    )


if __name__ == "__main__":
    main()
