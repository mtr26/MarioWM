"""
Offline dataset collection for the Mario World Model.

After training the PPO agent, run this script to collect a rich,
diverse dataset of (obs_t, action_t, obs_{t+1}, reward_t, done_t) tuples
stored in an HDF5 file for fast random-access during world model training.

The key insight for diversity: we mix the trained PPO policy with random
actions using an ε-greedy strategy. We default ε=0.20 so that 20% of steps
are random — this ensures the dataset covers:
  - Mario dying (hits Goomba, falls in pit)
  - Mario standing still
  - Edge-of-level transitions
  - Varied enemy/pipe encounters

Dataset format (HDF5):
    /observations   uint8  (N, H, W, 3)   — RGB frames at time t
    /next_obs       uint8  (N, H, W, 3)   — RGB frames at time t+1
    /actions        int32  (N,)           — discrete action index
    /rewards        float32 (N,)
    /dones          bool    (N,)          — True if episode ended

Usage:
    python collect_data.py --model checkpoints/ppo_mario_final.zip
    python collect_data.py --model checkpoints/best_model.zip --n-steps 200000 --epsilon 0.3
"""

import os
import argparse
import numpy as np
import h5py
from tqdm import tqdm

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from env.wrappers import (
    make_collect_env, make_ppo_env,
    N_STACK, N_ACTIONS,
    FRAME_HEIGHT, FRAME_WIDTH,
    RGB_HEIGHT, RGB_WIDTH,
)

# ─── Paths ────────────────────────────────────────────────────────────────────
DATASET_DIR = "dataset"
os.makedirs(DATASET_DIR, exist_ok=True)


# ─── Helper: build stacked grayscale obs from last k raw RGB frames ──────────
class FrameBuffer:
    """
    Maintains a rolling buffer of grayscale frames so we can feed the
    PPO agent the same frame-stacked observation it was trained on,
    even though we are storing RGB frames in the dataset.
    """

    def __init__(self, n_stack: int = N_STACK, h: int = FRAME_HEIGHT, w: int = FRAME_WIDTH):
        self.n_stack = n_stack
        # buffer shape: (k, h, w, 1) — grayscale
        self.buf = np.zeros((n_stack, h, w, 1), dtype=np.uint8)

    def reset(self, first_gray_frame: np.ndarray):
        """Fill all slots with the first frame (mimics VecFrameStack reset)."""
        for i in range(self.n_stack):
            self.buf[i] = first_gray_frame
        return self._obs()

    def push(self, gray_frame: np.ndarray):
        self.buf = np.roll(self.buf, shift=-1, axis=0)
        self.buf[-1] = gray_frame
        return self._obs()

    def _obs(self) -> np.ndarray:
        # SB3 VecFrameStack with channels_order="last" produces (h, w, n_stack)
        return np.concatenate(self.buf, axis=-1)  # (h, w, k)


def rgb_to_gray_norm(rgb: np.ndarray) -> np.ndarray:
    """Convert uint8 RGB (H,W,3) → uint8 grayscale (H,W,1)."""
    import cv2
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return gray[:, :, np.newaxis]  # uint8, no normalization


# ─── Main collection loop ─────────────────────────────────────────────────────

def collect(
    model_path: str,
    n_steps: int = 100_000,
    epsilon: float = 0.20,
    output_path: str | None = None,
    seed: int = 42,
):
    if output_path is None:
        output_path = os.path.join(DATASET_DIR, f"mario_1-1_{n_steps//1000}k_eps{int(epsilon*100)}.h5")

    print(f"\n{'='*60}")
    print(f"  Mario 1-1 Data Collection")
    print(f"  Model   : {model_path}")
    print(f"  Steps   : {n_steps:,}")
    print(f"  Epsilon : {epsilon:.0%} random actions")
    print(f"  Output  : {output_path}")
    print(f"{'='*60}\n")

    rng = np.random.default_rng(seed)

    # ── Load agent ────────────────────────────────────────────────────────────
    # The agent expects (h, w, n_stack) grayscale obs — we build that ourselves.
    model = PPO.load(model_path)
    print(f"[ok] Loaded policy from {model_path}")

    # ── Environment (RGB, no grayscale, no normalization) ─────────────────────
    collect_env = make_collect_env()
    frame_buf = FrameBuffer(n_stack=N_STACK)

    # ── Pre-allocate numpy arrays (faster than growing lists) ─────────────────
    obs_buf      = np.empty((n_steps, RGB_HEIGHT, RGB_WIDTH, 3), dtype=np.uint8)
    next_obs_buf = np.empty((n_steps, RGB_HEIGHT, RGB_WIDTH, 3), dtype=np.uint8)
    act_buf      = np.empty((n_steps,), dtype=np.int32)
    rew_buf      = np.empty((n_steps,), dtype=np.float32)
    done_buf     = np.empty((n_steps,), dtype=bool)

    # ── Collection loop ───────────────────────────────────────────────────────
    raw_obs, _ = collect_env.reset(seed=seed)             # uint8 RGB (H, W, 3)
    gray = rgb_to_gray_norm(raw_obs)
    stacked_obs = frame_buf.reset(gray)                   # (H, W, n_stack)

    episodes_done = 0
    step = 0

    with tqdm(total=n_steps, unit="step", desc="Collecting") as pbar:
        while step < n_steps:
            # ε-greedy action selection
            if rng.random() < epsilon:
                action = rng.integers(0, N_ACTIONS)
            else:
                # SB3 predict expects (1, h, w, n_stack)
                action, _ = model.predict(
                    stacked_obs[np.newaxis], deterministic=True
                )
                action = int(action[0])

            raw_next_obs, reward, terminated, truncated, info = collect_env.step(action)
            done = terminated or truncated

            # Store raw RGB transition
            obs_buf[step]      = raw_obs
            next_obs_buf[step] = raw_next_obs
            act_buf[step]      = action
            rew_buf[step]      = float(reward)
            done_buf[step]     = done

            # Advance frame buffer
            gray = rgb_to_gray_norm(raw_next_obs)
            stacked_obs = frame_buf.push(gray)

            raw_obs = raw_next_obs
            step += 1
            pbar.update(1)

            if done:
                raw_obs, _ = collect_env.reset()
                gray = rgb_to_gray_norm(raw_obs)
                stacked_obs = frame_buf.reset(gray)
                episodes_done += 1

    collect_env.close()

    # ── Write HDF5 ────────────────────────────────────────────────────────────
    print(f"\n[saving] Writing {n_steps:,} transitions to {output_path} …")
    with h5py.File(output_path, "w") as f:
        f.create_dataset("observations",  data=obs_buf,      compression="lzf")
        f.create_dataset("next_obs",      data=next_obs_buf, compression="lzf")
        f.create_dataset("actions",       data=act_buf)
        f.create_dataset("rewards",       data=rew_buf)
        f.create_dataset("dones",         data=done_buf)

        # ── Metadata ──────────────────────────────────────────────────────────
        f.attrs["world"]         = "1-1"
        f.attrs["n_steps"]       = n_steps
        f.attrs["epsilon"]       = epsilon
        f.attrs["n_actions"]     = N_ACTIONS
        f.attrs["frame_h"]       = RGB_HEIGHT
        f.attrs["frame_w"]       = RGB_WIDTH
        f.attrs["n_stack_agent"] = N_STACK
        f.attrs["episodes"]      = episodes_done
        f.attrs["model_path"]    = model_path

    size_mb = os.path.getsize(output_path) / (1024 ** 2)
    print(f"✓ Dataset saved  |  {n_steps:,} steps  |  {episodes_done} episodes  |  {size_mb:.1f} MB")
    return output_path


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect offline dataset from trained PPO agent")
    parser.add_argument("--model",    type=str, default="checkpoints/ppo_mario_final.zip",
                        help="Path to the trained PPO model checkpoint")
    parser.add_argument("--n-steps",  type=int, default=100_000,
                        help="Number of env steps to collect (default: 100k)")
    parser.add_argument("--epsilon",  type=float, default=0.20,
                        help="Fraction of random actions for diversity (default: 0.20)")
    parser.add_argument("--output",   type=str, default=None,
                        help="Output HDF5 path (auto-named if not set)")
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    collect(
        model_path=args.model,
        n_steps=args.n_steps,
        epsilon=args.epsilon,
        output_path=args.output,
        seed=args.seed,
    )
