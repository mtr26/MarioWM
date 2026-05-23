"""
Dataset visualization & integrity checks.

Usage:
    python inspect_dataset.py dataset/mario_1-1_100k_eps20.h5

Outputs:
  - Dataset stats (action distribution, reward histogram, done count)
  - A mosaic of random frames saved to dataset/preview.png
  - A short GIF of a sampled trajectory saved to dataset/sample_trajectory.gif
"""

import os
import sys
import argparse

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import imageio.v3 as iio


def inspect(h5_path: str, n_preview: int = 16, traj_len: int = 64):
    print(f"\nInspecting: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        # ── Metadata ──────────────────────────────────────────────────────────
        print("\n── Metadata ──────────────────────────────────────────────")
        for k, v in f.attrs.items():
            print(f"  {k:20s}: {v}")

        obs      = f["observations"]    # (N, H, W, 3)
        next_obs = f["next_obs"]
        actions  = f["actions"][:]      # load to RAM for stats
        rewards  = f["rewards"][:]
        dones    = f["dones"][:]

        N = obs.shape[0]
        print(f"\n── Shape / dtype ─────────────────────────────────────────")
        print(f"  observations : {obs.shape}  {obs.dtype}")
        print(f"  next_obs     : {next_obs.shape}  {next_obs.dtype}")
        print(f"  actions      : {actions.shape}  {actions.dtype}")
        print(f"  rewards      : {rewards.shape}  {rewards.dtype}")
        print(f"  dones        : {dones.shape}  {dones.dtype}")

        print(f"\n── Statistics ────────────────────────────────────────────")
        print(f"  Total steps  : {N:,}")
        print(f"  Episodes     : {dones.sum():,}")
        print(f"  Avg ep len   : {N / max(dones.sum(), 1):.1f} steps")
        print(f"  Reward range : [{rewards.min():.1f}, {rewards.max():.1f}]")
        print(f"  Mean reward  : {rewards.mean():.2f}")

        print(f"\n── Action distribution ───────────────────────────────────")
        from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
        for a_idx, count in enumerate(np.bincount(actions, minlength=len(SIMPLE_MOVEMENT))):
            bar = "█" * (count * 40 // N)
            label = str(SIMPLE_MOVEMENT[a_idx])
            print(f"  [{a_idx}] {label:25s} {count:>7,} ({count/N*100:5.1f}%)  {bar}")

        # ── Preview mosaic ─────────────────────────────────────────────────
        print(f"\n── Generating frame mosaic ({n_preview} frames) …")
        idxs = np.random.default_rng(0).choice(N, size=n_preview, replace=False)
        cols = 8
        rows = max(1, n_preview // cols)

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
        axes = np.array(axes).flatten()

        for i, idx in enumerate(idxs):
            frame = obs[idx]   # uint8 RGB
            axes[i].imshow(frame)
            axes[i].axis("off")
            axes[i].set_title(f"a={actions[idx]}", fontsize=6)

        plt.suptitle("Random frames from dataset", fontsize=10, y=1.01)
        plt.tight_layout()

        preview_path = os.path.join(os.path.dirname(h5_path), "preview.png")
        plt.savefig(preview_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {preview_path}")

        # ── Sample trajectory GIF ──────────────────────────────────────────
        print(f"\n── Generating trajectory GIF ({traj_len} steps) …")
        start = np.random.default_rng(1).integers(0, N - traj_len)
        frames_gif = [obs[i] for i in range(start, start + traj_len)]

        gif_path = os.path.join(os.path.dirname(h5_path), "sample_trajectory.gif")
        iio.imwrite(gif_path, frames_gif, fps=10, loop=0)
        print(f"  Saved → {gif_path}")

    print("\n✓ Inspection complete.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a collected Mario dataset")
    parser.add_argument("h5_path", type=str, help="Path to the HDF5 dataset file")
    parser.add_argument("--n-preview", type=int, default=16,
                        help="Number of random frames in mosaic (default: 16)")
    parser.add_argument("--traj-len", type=int, default=64,
                        help="Length of the sampled trajectory GIF (default: 64)")
    args = parser.parse_args()
    inspect(args.h5_path, n_preview=args.n_preview, traj_len=args.traj_len)
