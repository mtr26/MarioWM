"""
Quick sanity check — renders the trained PPO agent playing Mario 1-1
and optionally saves a video. Run this after training to visually verify
the policy before spending time on data collection.

Usage:
    python evaluate_agent.py                              # print rewards only
    python evaluate_agent.py --save-video eval.mp4       # record to file
    python evaluate_agent.py --n-episodes 3 --deterministic
"""

import argparse
import numpy as np
import imageio.v3 as iio

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from env.wrappers import make_ppo_env, make_video_env, N_STACK, N_SKIP


def evaluate(
    model_path: str = "checkpoints/ppo_mario_final.zip",
    n_episodes: int = 1,
    deterministic: bool = True,
    save_video: str | None = None,
    fps: int = 60,
):
    print(f"\nEvaluating: {model_path}")

    model = PPO.load(model_path)

    # ── Agent env (preprocessed: grayscale, stacked, uint8) ───────────────────
    vec = DummyVecEnv([make_ppo_env])
    vec = VecFrameStack(vec, n_stack=N_STACK, channels_order="last")

    # ── Raw RGB env for video capture (native 256×240, full colour) ──────────
    # Skips the ResizeFrame wrapper so video is recorded at NES native resolution.
    raw_env = make_video_env() if save_video else None

    video_frames = []
    all_rewards  = []

    for ep in range(n_episodes):
        obs = vec.reset()
        if raw_env is not None:
            raw_obs, _ = raw_env.reset()
            video_frames.append(raw_obs)   # first frame

        ep_reward = 0.0
        done = [False]

        while not done[0]:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = vec.step(action)
            ep_reward += float(reward[0])

            # Mirror the same action in the raw env and collect N_SKIP frames
            if raw_env is not None:
                raw_obs, _, term, trunc, _ = raw_env.step(int(action[0]))
                video_frames.append(raw_obs)
                if term or trunc:
                    raw_obs, _ = raw_env.reset()

        all_rewards.append(ep_reward)
        print(f"  Episode {ep+1}: reward = {ep_reward:.1f}")

    vec.close()
    if raw_env is not None:
        raw_env.close()

    mean_r = np.mean(all_rewards)
    print(f"\n  Mean reward over {n_episodes} episode(s): {mean_r:.1f}")

    if save_video and video_frames:
        # imageio expects (T, H, W, C) uint8
        frames_arr = np.stack(video_frames).astype(np.uint8)
        iio.imwrite(save_video, frames_arr, fps=fps)
        print(f"  Video saved → {save_video}  ({len(video_frames)} frames @ {fps}fps)")

    return mean_r


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",         type=str,   default="checkpoints/best_model.zip")
    parser.add_argument("--n-episodes",    type=int,   default=1)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--save-video",    type=str,   default=None,
                        help="Save an mp4 or gif (e.g. eval.mp4)")
    parser.add_argument("--fps",           type=int,   default=60,
                        help="Video FPS (default: 60 to match NES)")
    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        n_episodes=args.n_episodes,
        deterministic=args.deterministic,
        save_video=args.save_video,
        fps=args.fps,
    )
