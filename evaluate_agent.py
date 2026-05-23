"""
Quick sanity check — renders the trained PPO agent playing Mario 1-1
and optionally saves a video. Run this after training to visually verify
the policy before spending time on data collection.

Usage:
    python evaluate_agent.py                              # render to screen
    python evaluate_agent.py --save-video eval.mp4       # record to file
    python evaluate_agent.py --n-episodes 3 --deterministic
"""

import os
import argparse
import numpy as np
import imageio.v3 as iio

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from env.wrappers import make_ppo_env, N_STACK


def evaluate(
    model_path: str = "checkpoints/ppo_mario_final.zip",
    n_episodes: int = 1,
    deterministic: bool = True,
    save_video: str | None = None,
    fps: int = 30,
):
    print(f"\nEvaluating: {model_path}")

    model = PPO.load(model_path)

    # Wrap in VecEnv so it matches training setup
    def _make():
        return make_ppo_env(render_mode="rgb_array")

    vec = DummyVecEnv([_make])
    vec = VecFrameStack(vec, n_stack=N_STACK, channels_order="last")

    video_frames = []
    all_rewards = []

    for ep in range(n_episodes):
        obs = vec.reset()
        ep_reward = 0.0
        done = [False]

        while not done[0]:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = vec.step(action)
            ep_reward += float(reward[0])

            # Grab raw render for video
            raw = vec.env_method("render")[0]
            if raw is not None:
                video_frames.append(raw)

        all_rewards.append(ep_reward)
        print(f"  Episode {ep+1}: reward = {ep_reward:.1f}")

    vec.close()

    mean_r = np.mean(all_rewards)
    print(f"\n  Mean reward over {n_episodes} episode(s): {mean_r:.1f}")

    if save_video and video_frames:
        iio.imwrite(save_video, video_frames, fps=fps)
        print(f"  Video saved → {save_video}")

    return mean_r


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",         type=str, default="checkpoints/ppo_mario_final.zip")
    parser.add_argument("--n-episodes",    type=int, default=1)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--save-video",    type=str, default=None,
                        help="Save an mp4 or gif of the episode (e.g. eval.mp4)")
    parser.add_argument("--fps",           type=int, default=30)
    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        n_episodes=args.n_episodes,
        deterministic=args.deterministic,
        save_video=args.save_video,
        fps=args.fps,
    )
