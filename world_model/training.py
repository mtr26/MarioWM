"""Training and evaluation primitives for the deterministic world model."""

from __future__ import annotations

import math
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Iterator

import torch
from torch.nn import functional as F


class EMA:
    """Exponential moving average of a model state dictionary."""

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        if not 0 < decay < 1:
            raise ValueError("EMA decay must be between zero and one")
        self.decay = decay
        self.shadow = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        state = model.state_dict()
        if state.keys() != self.shadow.keys():
            raise ValueError("EMA model structure changed during training")
        for name, value in state.items():
            if value.is_floating_point():
                self.shadow[name].lerp_(value.detach(), 1 - self.decay)
            else:
                self.shadow[name].copy_(value)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: value.clone() for name, value in self.shadow.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        if state.keys() != self.shadow.keys():
            raise ValueError("EMA checkpoint does not match model structure")
        self.shadow = {
            name: value.detach().to(
                device=self.shadow[name].device,
                dtype=self.shadow[name].dtype,
            ).clone()
            for name, value in state.items()
        }

    @contextmanager
    def average_parameters(
        self, model: torch.nn.Module
    ) -> Iterator[torch.nn.Module]:
        original = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }
        model.load_state_dict(self.shadow)
        try:
            yield model
        finally:
            model.load_state_dict(original)


def warmup_cosine_factor(step: int, warmup_steps: int, total_steps: int) -> float:
    """Return a linear-warmup then cosine-decay learning-rate multiplier."""
    if warmup_steps and step < warmup_steps:
        return max(1, step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1 + math.cos(math.pi * progress))


def _autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def _move_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
    channels_last: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    context = batch["context"].to(device, non_blocking=True)
    actions = batch["actions"].to(device, non_blocking=True)
    target = batch["target"].to(device, non_blocking=True)
    if channels_last:
        context = context.contiguous(memory_format=torch.channels_last)
        target = target.contiguous(memory_format=torch.channels_last)
    return context, actions, target


def train_one_epoch(
    *,
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    ema: EMA,
    device: torch.device,
    use_bfloat16: bool,
    channels_last: bool,
    gradient_clip: float,
    global_step: int,
) -> tuple[dict[str, float], int]:
    model.train()
    total_loss = 0.0
    total_items = 0
    ema_source = getattr(model, "_orig_mod", model)

    for batch in loader:
        context, actions, target = _move_batch(batch, device, channels_last)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, use_bfloat16):
            prediction = model(context, actions)
            loss = F.l1_loss(prediction, target)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss at step {global_step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        ema.update(ema_source)

        batch_size = int(context.shape[0])
        total_loss += float(loss.detach()) * batch_size
        total_items += batch_size
        global_step += 1

    if total_items == 0:
        raise ValueError("training loader produced no samples")
    return {"l1": total_loss / total_items}, global_step


@torch.inference_mode()
def evaluate(
    *,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    use_bfloat16: bool,
    channels_last: bool,
    n_actions: int,
) -> dict:
    model.eval()
    l1_sum = 0.0
    mse_sum = 0.0
    pixel_count = 0
    action_sums = torch.zeros(n_actions, dtype=torch.float64)
    action_counts = torch.zeros(n_actions, dtype=torch.int64)

    for batch in loader:
        context, actions, target = _move_batch(batch, device, channels_last)
        with _autocast(device, use_bfloat16):
            prediction = model(context, actions)
        absolute = (prediction.float() - target.float()).abs()
        squared = (prediction.float() - target.float()).square()
        l1_sum += float(absolute.sum())
        mse_sum += float(squared.sum())
        pixel_count += absolute.numel()

        sample_l1 = absolute.flatten(1).mean(1).cpu()
        current_actions = actions[:, -1].cpu()
        for action in range(n_actions):
            mask = current_actions == action
            action_sums[action] += sample_l1[mask].sum().double()
            action_counts[action] += mask.sum()

    if pixel_count == 0:
        raise ValueError("validation loader produced no samples")
    per_action = {
        action: (
            float(action_sums[action] / action_counts[action])
            if int(action_counts[action])
            else float("nan")
        )
        for action in range(n_actions)
    }
    return {
        "l1": l1_sum / pixel_count,
        "mse": mse_sum / pixel_count,
        "per_action_l1": per_action,
    }


@torch.inference_mode()
def autoregressive_rollout(
    model: torch.nn.Module,
    initial_context: torch.Tensor,
    action_sequence: torch.Tensor,
    horizon: int,
) -> torch.Tensor:
    if initial_context.ndim != 5:
        raise ValueError("initial context must have shape (B, history, 3, H, W)")
    if horizon < 1:
        raise ValueError("rollout horizon must be positive")
    context_frames = list(initial_context.unbind(dim=1))
    history = len(context_frames)
    if action_sequence.ndim != 2 or action_sequence.shape[1] < history + horizon - 1:
        raise ValueError("action sequence is too short for the requested rollout")

    predictions = []
    for step in range(horizon):
        action_history = action_sequence[:, step : step + history]
        stacked = torch.cat(context_frames[-history:], dim=1)
        prediction = model(stacked, action_history)
        predictions.append(prediction)
        context_frames.append(prediction)
    return torch.stack(predictions, dim=1)


@torch.inference_mode()
def save_prediction_grid(
    path: str | Path,
    context: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
    max_items: int = 8,
) -> None:
    from torchvision.utils import save_image

    count = min(max_items, context.shape[0])
    last_context = context[:count, -3:].float().cpu()
    targets = target[:count].float().cpu()
    predictions = prediction[:count].float().cpu()
    errors = (predictions - targets).abs()
    images = torch.stack(
        (last_context, targets, predictions, errors), dim=1
    ).flatten(0, 1)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_image(images.clamp(0, 1), output, nrow=4)


@torch.inference_mode()
def save_rollout_grid(
    path: str | Path,
    targets: torch.Tensor,
    predictions: torch.Tensor,
) -> None:
    from torchvision.utils import save_image

    real = targets[0].float().cpu()
    predicted = predictions[0].float().cpu()
    errors = (predicted - real).abs()
    images = torch.stack((real, predicted, errors), dim=1).flatten(0, 1)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_image(images.clamp(0, 1), output, nrow=3)
