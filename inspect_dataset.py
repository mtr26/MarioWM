"""Inspect Mario HDF5 structure, alignment, boundaries, and visual samples."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np

from world_model.conversion import ACTION_NAMES, build_episode_offsets


def _dataset_info(dataset: h5py.Dataset) -> dict:
    return {
        "shape": list(dataset.shape),
        "dtype": str(dataset.dtype),
        "chunks": None if dataset.chunks is None else list(dataset.chunks),
        "compression": dataset.compression,
        "storage_bytes": int(dataset.id.get_storage_size()),
    }


def _length_summary(lengths: np.ndarray) -> dict[str, float | int]:
    if lengths.size == 0:
        return {"min": 0, "median": 0.0, "mean": 0.0, "p95": 0.0, "max": 0}
    return {
        "min": int(lengths.min()),
        "median": float(np.median(lengths)),
        "mean": float(lengths.mean()),
        "p95": float(np.quantile(lengths, 0.95)),
        "max": int(lengths.max()),
    }


def _print_stats(stats: dict) -> None:
    print("\n── Metadata ──────────────────────────────────────────────")
    for key, value in stats["attributes"].items():
        print(f"  {key:24s}: {value}")

    print("\n── Shape / storage ───────────────────────────────────────")
    for name, values in stats["datasets"].items():
        print(
            f"  {name:18s}: {tuple(values['shape'])} {values['dtype']} "
            f"chunks={values['chunks']} compression={values['compression']} "
            f"stored={values['storage_bytes'] / 2**20:.1f} MiB"
        )

    lengths = stats["trajectory_length_summary"]
    print("\n── Trajectories ──────────────────────────────────────────")
    print(f"  Transitions             : {stats['n_transitions']:,}")
    print(f"  Complete episodes       : {stats['complete_episodes']:,}")
    print(f"  Trajectory segments     : {stats['n_trajectories']:,}")
    print(f"  Trailing partial length : {stats['trailing_partial_length']:,}")
    print(
        "  Length min/median/mean/p95/max: "
        f"{lengths['min']}/{lengths['median']:.1f}/{lengths['mean']:.1f}/"
        f"{lengths['p95']:.1f}/{lengths['max']}"
    )

    print("\n── Actions ───────────────────────────────────────────────")
    for item in stats["actions"]:
        print(
            f"  [{item['index']}] {item['name']:18s} "
            f"{item['count']:>8,} ({item['percent']:5.1f}%)"
        )

    print("\n── I/O / continuity ──────────────────────────────────────")
    print(f"  Sequential read         : {stats['sequential_read_mib_s']:.1f} MiB/s")
    for item in stats["explicit_breaks"]:
        print(
            f"  New trajectory at {item['index']:,}: "
            f"previous next == current obs: {item['continuous']}"
        )


def inspect(
    h5_path: str,
    n_preview: int = 16,
    traj_len: int = 64,
    *,
    stats_only: bool = False,
    break_indices: Sequence[int] = (),
) -> dict:
    """Return dataset statistics and optionally create boundary-safe visuals."""
    source_path = Path(h5_path)
    print(f"\nInspecting: {source_path}")

    with h5py.File(source_path, "r") as handle:
        required = ("observations", "next_obs", "actions", "rewards", "dones")
        missing = [name for name in required if name not in handle]
        if missing:
            raise ValueError(f"missing required datasets: {', '.join(missing)}")

        observations = handle["observations"]
        next_observations = handle["next_obs"]
        actions = handle["actions"][:]
        rewards = handle["rewards"][:]
        dones = handle["dones"][:].astype(bool)
        n_transitions = int(observations.shape[0])
        offsets = build_episode_offsets(dones, break_indices)
        trajectory_lengths = np.diff(offsets)
        done_indices = np.flatnonzero(dones)
        trailing_partial = (
            0
            if dones[-1]
            else n_transitions - (int(done_indices[-1]) + 1 if done_indices.size else 0)
        )

        unique_actions, action_counts = np.unique(actions, return_counts=True)
        action_items = []
        for index, count in zip(unique_actions, action_counts):
            action_index = int(index)
            action_items.append(
                {
                    "index": action_index,
                    "name": (
                        ACTION_NAMES[action_index]
                        if action_index < len(ACTION_NAMES)
                        else f"action-{action_index}"
                    ),
                    "count": int(count),
                    "percent": float(count * 100 / n_transitions),
                }
            )

        explicit_breaks = []
        for index in sorted(set(int(value) for value in break_indices)):
            if index <= 0 or index >= n_transitions:
                raise ValueError(f"break index out of range: {index}")
            explicit_breaks.append(
                {
                    "index": index,
                    "continuous": bool(
                        np.array_equal(
                            next_observations[index - 1], observations[index]
                        )
                    ),
                }
            )

        chunk_frames = (
            int(observations.chunks[0]) if observations.chunks else min(1024, n_transitions)
        )
        read_count = min(chunk_frames, n_transitions)
        started = time.perf_counter()
        benchmark = observations[:read_count]
        elapsed = max(time.perf_counter() - started, 1e-9)
        sequential_read_mib_s = float(benchmark.nbytes / 2**20 / elapsed)
        del benchmark

        stats = {
            "attributes": {
                key: value.item() if hasattr(value, "item") else value
                for key, value in handle.attrs.items()
            },
            "datasets": {name: _dataset_info(handle[name]) for name in required},
            "n_transitions": n_transitions,
            "complete_episodes": int(dones.sum()),
            "n_trajectories": len(offsets) - 1,
            "trailing_partial_length": trailing_partial,
            "trajectory_length_summary": _length_summary(trajectory_lengths),
            "reward_min": float(rewards.min()),
            "reward_mean": float(rewards.mean()),
            "reward_max": float(rewards.max()),
            "actions": action_items,
            "explicit_breaks": explicit_breaks,
            "sequential_read_mib_s": sequential_read_mib_s,
        }
        _print_stats(stats)

        if stats_only:
            return stats

        import imageio.v3 as iio
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if n_preview < 1 or traj_len < 1:
            raise ValueError("preview count and trajectory length must be positive")

        preview_count = min(n_preview, n_transitions)
        preview_indices = np.random.default_rng(0).choice(
            n_transitions, size=preview_count, replace=False
        )
        columns = min(8, preview_count)
        rows = math.ceil(preview_count / columns)
        figure, axes = plt.subplots(
            rows, columns, figsize=(columns * 1.7, rows * 1.7), squeeze=False
        )
        flat_axes = axes.flatten()
        for axis, index in zip(flat_axes, preview_indices):
            axis.imshow(observations[index])
            axis.axis("off")
            axis.set_title(f"t={index}, a={int(actions[index])}", fontsize=6)
        for axis in flat_axes[preview_count:]:
            axis.axis("off")
        figure.suptitle("Random aligned observation/action samples", fontsize=10)
        figure.tight_layout()
        preview_path = source_path.parent / "preview.png"
        figure.savefig(preview_path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        print(f"\nSaved frame mosaic: {preview_path}")

        candidates = np.flatnonzero(trajectory_lengths >= traj_len)
        if candidates.size:
            episode = int(np.random.default_rng(1).choice(candidates))
            start_min = int(offsets[episode])
            start_max = int(offsets[episode + 1]) - traj_len
            start = int(np.random.default_rng(2).integers(start_min, start_max + 1))
            trajectory_frames = observations[start : start + traj_len]
            gif_path = source_path.parent / "sample_trajectory.gif"
            iio.imwrite(gif_path, trajectory_frames, fps=10, loop=0)
            print(f"Saved trajectory GIF: {gif_path}")
        else:
            print(f"No trajectory is long enough for a {traj_len}-frame GIF")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a collected Mario dataset")
    parser.add_argument("h5_path", type=str)
    parser.add_argument("--n-preview", type=int, default=16)
    parser.add_argument("--traj-len", type=int, default=64)
    parser.add_argument("--stats-only", action="store_true")
    parser.add_argument("--break-index", action="append", type=int, default=[])
    arguments = parser.parse_args()
    inspect(
        arguments.h5_path,
        n_preview=arguments.n_preview,
        traj_len=arguments.traj_len,
        stats_only=arguments.stats_only,
        break_indices=tuple(arguments.break_index),
    )
