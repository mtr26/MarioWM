"""Deterministic action-conditioned U-Net for Mario frame prediction."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def _group_count(channels: int) -> int:
    for value in (32, 16, 8, 4, 2, 1):
        if channels % value == 0:
            return value
    return 1


class ActionConditioner(nn.Module):
    """Embed an ordered action history into one FiLM condition vector."""

    def __init__(
        self,
        n_actions: int,
        history: int,
        embed_dim: int,
        cond_dim: int,
    ) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.history = history
        self.action_embedding = nn.Embedding(n_actions, embed_dim)
        self.position_embedding = nn.Embedding(history, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(history * embed_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim != 2 or actions.shape[1] != self.history:
            raise ValueError(f"actions must have shape (B, {self.history})")
        if actions.shape[0] == 0:
            raise ValueError("action batch cannot be empty")
        if int(actions.min()) < 0 or int(actions.max()) >= self.n_actions:
            raise ValueError(
                f"action indices must be in [0, {self.n_actions - 1}]"
            )

        positions = torch.arange(self.history, device=actions.device)
        embedded = self.action_embedding(actions) + self.position_embedding(
            positions
        ).unsqueeze(0)
        return self.mlp(embedded.flatten(1))


class FiLMResBlock(nn.Module):
    """Residual convolution block with action-dependent scale and shift."""

    def __init__(self, in_channels: int, out_channels: int, cond_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.film = nn.Linear(cond_dim, 2 * out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(
        self, x: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        residual = self.skip(x)
        hidden = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.film(condition).chunk(2, dim=1)
        hidden = self.norm2(hidden)
        hidden = hidden * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        hidden = self.conv2(F.silu(hidden))
        return residual + hidden


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 3, stride=2, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class ActionConditionedUNet(nn.Module):
    """Predict one RGB frame from recent RGB frames and discrete actions."""

    def __init__(
        self,
        history: int = 4,
        n_actions: int = 7,
        base_channels: int = 64,
        channel_multipliers: Sequence[int] = (1, 2, 3, 4),
        blocks_per_level: int = 2,
        action_embed_dim: int = 64,
        cond_dim: int = 256,
    ) -> None:
        super().__init__()
        if history < 1 or n_actions < 1 or base_channels < 1:
            raise ValueError("history, n_actions, and base_channels must be positive")
        if len(channel_multipliers) < 2 or any(
            value < 1 for value in channel_multipliers
        ):
            raise ValueError("channel multipliers must contain at least two positives")
        if blocks_per_level < 1:
            raise ValueError("blocks_per_level must be positive")

        self.history = history
        self.n_actions = n_actions
        channels = [base_channels * int(value) for value in channel_multipliers]
        self.conditioner = ActionConditioner(
            n_actions=n_actions,
            history=history,
            embed_dim=action_embed_dim,
            cond_dim=cond_dim,
        )
        self.input_conv = nn.Conv2d(history * 3, channels[0], 3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        current_channels = channels[0]
        for level, level_channels in enumerate(channels):
            blocks = nn.ModuleList()
            for block_index in range(blocks_per_level):
                block_input = (
                    current_channels if block_index == 0 else level_channels
                )
                blocks.append(
                    FiLMResBlock(block_input, level_channels, cond_dim)
                )
                current_channels = level_channels
            self.down_blocks.append(blocks)
            if level < len(channels) - 1:
                self.downsamples.append(
                    Downsample(current_channels, channels[level + 1])
                )
                current_channels = channels[level + 1]

        self.mid_blocks = nn.ModuleList(
            (
                FiLMResBlock(current_channels, current_channels, cond_dim),
                FiLMResBlock(current_channels, current_channels, cond_dim),
            )
        )

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for level in reversed(range(len(channels))):
            level_channels = channels[level]
            blocks = nn.ModuleList(
                (
                    FiLMResBlock(
                        current_channels + level_channels,
                        level_channels,
                        cond_dim,
                    ),
                )
            )
            for _ in range(blocks_per_level - 1):
                blocks.append(
                    FiLMResBlock(level_channels, level_channels, cond_dim)
                )
            self.up_blocks.append(blocks)
            current_channels = level_channels
            if level > 0:
                self.upsamples.append(
                    Upsample(current_channels, channels[level - 1])
                )
                current_channels = channels[level - 1]

        self.output_norm = nn.GroupNorm(
            _group_count(current_channels), current_channels
        )
        self.output_conv = nn.Conv2d(current_channels, 3, 3, padding=1)

    def forward(
        self, context: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        expected_channels = self.history * 3
        if context.ndim != 4 or context.shape[1] != expected_channels:
            raise ValueError(f"expected {expected_channels} context channels")
        if actions.shape[0] != context.shape[0]:
            raise ValueError("context and action batch sizes must match")
        factor = 2 ** (len(self.down_blocks) - 1)
        if context.shape[-2] % factor or context.shape[-1] % factor:
            raise ValueError(f"height and width must be divisible by {factor}")

        condition = self.conditioner(actions)
        hidden = self.input_conv(context)
        skips: list[torch.Tensor] = []
        for level, blocks in enumerate(self.down_blocks):
            for block in blocks:
                hidden = block(hidden, condition)
            skips.append(hidden)
            if level < len(self.downsamples):
                hidden = self.downsamples[level](hidden)

        for block in self.mid_blocks:
            hidden = block(hidden, condition)

        for decoder_index, blocks in enumerate(self.up_blocks):
            hidden = torch.cat((hidden, skips.pop()), dim=1)
            for block in blocks:
                hidden = block(hidden, condition)
            if decoder_index < len(self.upsamples):
                hidden = self.upsamples[decoder_index](hidden)

        return torch.sigmoid(
            self.output_conv(F.silu(self.output_norm(hidden)))
        )
