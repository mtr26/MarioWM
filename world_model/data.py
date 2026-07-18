"""PyTorch access to boundary-safe Mario trajectory windows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset


SplitName = Literal["train", "validation", "test", "all"]
SPLIT_IDS = {"train": 0, "validation": 1, "test": 2}


class MarioWindowDataset(Dataset):
    """Return aligned frame/action histories without crossing trajectories."""

    def __init__(
        self,
        cache_dir: str | Path,
        split: SplitName,
        history: int = 4,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.metadata = json.loads(
            (self.cache_dir / "metadata.json").read_text(encoding="utf-8")
        )
        self.history = int(history)
        if self.history < 1:
            raise ValueError("history must be positive")
        if split != "all" and split not in SPLIT_IDS:
            raise ValueError(f"unknown split: {split}")

        self.episode_offsets = np.load(self.cache_dir / "episode_offsets.npy")
        episode_splits = np.load(self.cache_dir / "episode_splits.npy")
        if split == "all":
            selected = np.arange(len(episode_splits), dtype=np.int64)
        else:
            selected = np.flatnonzero(
                episode_splits == SPLIT_IDS[split]
            ).astype(np.int64)

        transitions: list[np.ndarray] = []
        episodes: list[np.ndarray] = []
        for episode in selected:
            start = int(self.episode_offsets[episode])
            end = int(self.episode_offsets[episode + 1])
            first = start + self.history - 1
            if first < end:
                values = np.arange(first, end, dtype=np.int64)
                transitions.append(values)
                episodes.append(
                    np.full(len(values), int(episode), dtype=np.int64)
                )

        self.sample_transitions = (
            np.concatenate(transitions)
            if transitions
            else np.empty(0, dtype=np.int64)
        )
        self.sample_episodes = (
            np.concatenate(episodes)
            if episodes
            else np.empty(0, dtype=np.int64)
        )
        self._frames: np.ndarray | None = None
        self._actions: np.ndarray | None = None

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_frames"] = None
        state["_actions"] = None
        return state

    def _open(self) -> None:
        if self._frames is None:
            self._frames = np.load(self.cache_dir / "frames.npy", mmap_mode="r")
            self._actions = np.load(
                self.cache_dir / "actions.npy", mmap_mode="r"
            )

    def __len__(self) -> int:
        return len(self.sample_transitions)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        self._open()
        assert self._frames is not None
        assert self._actions is not None

        transition = int(self.sample_transitions[index])
        episode = int(self.sample_episodes[index])
        transition_start = int(self.episode_offsets[episode])
        local_t = transition - transition_start
        frame_start = transition_start + episode

        context_frames = np.array(
            self._frames[
                frame_start
                + local_t
                - self.history
                + 1 : frame_start
                + local_t
                + 1
            ],
            copy=True,
        )
        target_frame = np.array(
            self._frames[frame_start + local_t + 1], copy=True
        )
        action_history = np.array(
            self._actions[
                transition - self.history + 1 : transition + 1
            ],
            copy=True,
        )

        context = (
            torch.from_numpy(context_frames)
            .permute(0, 3, 1, 2)
            .float()
            .div_(255)
        )
        return {
            "context": context.reshape(
                -1, context.shape[-2], context.shape[-1]
            ),
            "actions": torch.from_numpy(
                action_history.astype(np.int64, copy=False)
            ),
            "target": (
                torch.from_numpy(target_frame)
                .permute(2, 0, 1)
                .float()
                .div_(255)
            ),
            "transition_index": torch.tensor(transition, dtype=torch.int64),
            "episode_index": torch.tensor(episode, dtype=torch.int64),
        }

    def get_rollout(self, index: int, horizon: int) -> dict[str, torch.Tensor]:
        """Return an initial context, recorded actions, and real future frames."""
        if horizon < 1:
            raise ValueError("rollout horizon must be positive")
        self._open()
        assert self._frames is not None
        assert self._actions is not None

        transition = int(self.sample_transitions[index])
        episode = int(self.sample_episodes[index])
        transition_start = int(self.episode_offsets[episode])
        transition_end = int(self.episode_offsets[episode + 1])
        if transition + horizon > transition_end:
            raise ValueError("rollout horizon crosses trajectory boundary")

        local_t = transition - transition_start
        frame_start = transition_start + episode
        initial_context = np.array(
            self._frames[
                frame_start
                + local_t
                - self.history
                + 1 : frame_start
                + local_t
                + 1
            ],
            copy=True,
        )
        action_sequence = np.array(
            self._actions[
                transition - self.history + 1 : transition + horizon
            ],
            copy=True,
        )
        targets = np.array(
            self._frames[
                frame_start
                + local_t
                + 1 : frame_start
                + local_t
                + horizon
                + 1
            ],
            copy=True,
        )
        return {
            "initial_context": (
                torch.from_numpy(initial_context)
                .permute(0, 3, 1, 2)
                .float()
                .div_(255)
            ),
            "action_sequence": torch.from_numpy(
                action_sequence.astype(np.int64, copy=False)
            ),
            "targets": (
                torch.from_numpy(targets)
                .permute(0, 3, 1, 2)
                .float()
                .div_(255)
            ),
        }
