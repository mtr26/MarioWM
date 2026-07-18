"""Convert collected Mario transitions into a training-friendly cache."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np


REQUIRED_DATASETS = ("observations", "next_obs", "actions", "rewards", "dones")


class SourceValidationError(ValueError):
    """Raised when the input HDF5 file cannot be converted safely."""


@dataclass(frozen=True)
class SourceSchema:
    n_transitions: int
    frame_shape: tuple[int, int, int]
    n_actions: int


@dataclass(frozen=True)
class ConversionConfig:
    input_path: Path
    output_dir: Path
    height: int = 120
    width: int = 128
    history: int = 4
    break_indices: tuple[int, ...] = ()
    split_seed: int = 42
    split_fractions: tuple[float, float, float] = (0.9, 0.05, 0.05)
    workers: int = 16


def validate_source(handle: h5py.File) -> SourceSchema:
    """Validate the collector schema and return its model-relevant dimensions."""
    missing = [name for name in REQUIRED_DATASETS if name not in handle]
    if missing:
        raise SourceValidationError(
            f"missing required datasets: {', '.join(sorted(missing))}"
        )

    observations = handle["observations"]
    next_obs = handle["next_obs"]
    actions = handle["actions"]
    rewards = handle["rewards"]
    dones = handle["dones"]

    if observations.ndim != 4 or observations.shape[-1] != 3:
        raise SourceValidationError("observations must have shape (N, H, W, 3)")
    n_transitions = int(observations.shape[0])
    if n_transitions == 0:
        raise SourceValidationError("source dataset must contain transitions")
    if next_obs.shape != observations.shape:
        raise SourceValidationError("next_obs shape must match observations")
    if observations.dtype != np.uint8 or next_obs.dtype != np.uint8:
        raise SourceValidationError("frame datasets must use uint8")

    for name, dataset in (("actions", actions), ("rewards", rewards), ("dones", dones)):
        if dataset.shape != (n_transitions,):
            raise SourceValidationError(
                f"{name} must have shape ({n_transitions},), got {dataset.shape}"
            )
    if not np.issubdtype(actions.dtype, np.integer):
        raise SourceValidationError("actions must use an integer dtype")
    if rewards.dtype != np.float32:
        raise SourceValidationError("rewards must use float32")
    if dones.dtype != np.bool_:
        raise SourceValidationError("dones must use bool")

    action_values = actions[:]
    if int(action_values.min()) < 0:
        raise SourceValidationError("actions must be non-negative")

    return SourceSchema(
        n_transitions=n_transitions,
        frame_shape=tuple(int(value) for value in observations.shape[1:]),
        n_actions=int(action_values.max()) + 1,
    )


def build_episode_offsets(
    dones: np.ndarray, break_indices: Sequence[int]
) -> np.ndarray:
    """Return transition offsets for done-delimited and explicit trajectories."""
    done_values = np.asarray(dones, dtype=bool)
    if done_values.ndim != 1 or done_values.size == 0:
        raise ValueError("dones must be a non-empty one-dimensional array")

    n_transitions = int(done_values.size)
    explicit_breaks = {int(index) for index in break_indices}
    invalid = sorted(
        index for index in explicit_breaks if index <= 0 or index >= n_transitions
    )
    if invalid:
        raise ValueError(
            f"break index must satisfy 0 < index < {n_transitions}: {invalid}"
        )

    starts = {0, *explicit_breaks}
    starts.update(
        int(index) + 1
        for index in np.flatnonzero(done_values)
        if int(index) + 1 < n_transitions
    )
    return np.asarray([*sorted(starts), n_transitions], dtype=np.int64)


def assign_episode_splits(
    n_episodes: int,
    seed: int,
    fractions: tuple[float, float, float],
) -> np.ndarray:
    """Assign exact-count, deterministic train/validation/test labels."""
    if n_episodes < 3:
        raise ValueError(
            "at least three trajectories are required for train/validation/test"
        )
    values = np.asarray(fractions, dtype=np.float64)
    if (
        values.shape != (3,)
        or np.any(values <= 0)
        or not np.isclose(values.sum(), 1.0)
    ):
        raise ValueError("split fractions must be three positive values summing to one")

    counts = np.floor(values * n_episodes).astype(np.int64)
    counts[counts == 0] = 1
    while int(counts.sum()) > n_episodes:
        counts[int(np.argmax(counts))] -= 1
    while int(counts.sum()) < n_episodes:
        deficits = values * n_episodes - counts
        counts[int(np.argmax(deficits))] += 1

    labels = np.repeat(np.arange(3, dtype=np.uint8), counts)
    return np.random.default_rng(seed).permutation(labels)
