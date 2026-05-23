"""
Environment wrappers for Super Mario Bros.
Stacks preprocessing steps needed for both the PPO agent and the
offline dataset collection:
  - Convert to grayscale (saves memory; color is not needed for gameplay)
  - Resize to 84×84 (standard DQN convention)
  - Skip n frames (repeat action for n steps, take max over last 2)
  - Stack k consecutive frames (gives the agent a sense of velocity)
  - CustomReward: replaces the broken default reward with x_pos delta,
    a strong death penalty (-50), and a level-completion bonus (+500).

For the world model's decoder we also expose a function that wraps the
env WITHOUT grayscaling so we can save RGB frames for reconstruction.

Note on gym vs gymnasium:
  gym-super-mario-bros is registered with old OpenAI gym, not gymnasium.
  We create it with the old-gym API, wrap it with JoypadSpace, then bridge
  to gymnasium with a thin OldGymBridge wrapper that converts the 4-tuple
  step output to the 5-tuple gymnasium API that SB3 2.x expects.
"""

import numpy as np
import cv2
import gym as old_gym            # old OpenAI gym (pulled in by nes-py / gym-super-mario-bros)
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
import gym_super_mario_bros      # noqa: F401  (registers env IDs with old gym)
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


# ─── Constants ────────────────────────────────────────────────────────────────
WORLD = "1-1"
ENV_ID = f"SuperMarioBros-{WORLD}-v0"

# SIMPLE_MOVEMENT gives 7 discrete actions:
# 0: NOOP, 1: right, 2: right+A, 3: right+B, 4: right+A+B, 5: A, 6: left
N_ACTIONS = len(SIMPLE_MOVEMENT)

FRAME_HEIGHT = 84
FRAME_WIDTH = 84
N_STACK = 4          # frames to stack for the PPO agent
N_SKIP = 4           # frames to skip (action repeat)

# RGB output shape used by the *world model* dataset (no grayscale)
RGB_HEIGHT = 84
RGB_WIDTH = 84


# ─── Old-gym → Gymnasium Bridge ───────────────────────────────────────────────

class OldGymBridge(gym.Env):
    """
    Wraps an old-style OpenAI gym env (4-tuple step) into the modern
    gymnasium API (5-tuple step with terminated / truncated split).

    gymnasium.wrappers.StepAPICompatibility was removed in gymnasium 1.x,
    so we maintain our own minimal shim.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, old_env: old_gym.Env, render_mode: str = "rgb_array"):
        self._env = old_env
        self.render_mode = render_mode

        # Mirror old env's spaces into gymnasium types
        obs_sp = old_env.observation_space
        self.observation_space = spaces.Box(
            low=obs_sp.low, high=obs_sp.high,
            shape=obs_sp.shape, dtype=obs_sp.dtype,
        )
        act_sp = old_env.action_space
        self.action_space = spaces.Discrete(act_sp.n)

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        obs = self._env.reset()
        if isinstance(obs, tuple):          # some old gym envs return (obs, info)
            obs = obs[0]
        return obs, {}

    def step(self, action):
        # self._env is a JoypadSpace wrapping the raw SuperMarioBrosEnv.
        # JoypadSpace.step calls env.step(mapped_action) on the raw NES env
        # which returns a clean 4-tuple.  We extract terminated/truncated from
        # the 'done' flag and the info dict.
        result = self._env.step(action)
        obs, reward, done, info = result
        truncated = bool(info.get("TimeLimit.truncated", False))
        terminated = bool(done) and not truncated
        return obs, float(reward), terminated, truncated, info

    def render(self):
        return self._env.render(mode="rgb_array")

    def close(self):
        self._env.close()

    @property
    def unwrapped(self):
        return self._env.unwrapped


# ─── Individual Wrappers ──────────────────────────────────────────────────────

class SkipFrame(gym.Wrapper):
    """Repeat action for `skip` steps, sum rewards, max-pool last 2 frames."""

    def __init__(self, env: gym.Env, skip: int = N_SKIP):
        super().__init__(env)
        self._skip = skip
        self._obs_buffer = np.zeros(
            (2, *env.observation_space.shape), dtype=np.uint8
        )

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        info = {}
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if terminated or truncated:
                break
        # Max-pool to reduce motion blur artifacts
        obs = self._obs_buffer.max(axis=0)
        return obs, total_reward, terminated, truncated, info


class GrayScaleFrame(gym.ObservationWrapper):
    """Convert RGB (H, W, 3) → grayscale (H, W, 1)."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        h, w, _ = self.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(h, w, 1), dtype=np.uint8
        )

    def observation(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        return gray[:, :, np.newaxis]


class ResizeFrame(gym.ObservationWrapper):
    """Resize (H, W, C) → (target_h, target_w, C)."""

    def __init__(self, env: gym.Env, height: int = FRAME_HEIGHT, width: int = FRAME_WIDTH):
        super().__init__(env)
        self._h = height
        self._w = width
        _, _, c = self.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(height, width, c), dtype=np.uint8
        )

    def observation(self, obs):
        resized = cv2.resize(obs, (self._w, self._h), interpolation=cv2.INTER_AREA)
        if resized.ndim == 2:
            resized = resized[:, :, np.newaxis]
        return resized


class NormalizeFrame(gym.ObservationWrapper):
    """Scale pixel values from uint8 [0,255] to float32 [0.0,1.0]."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        h, w, c = self.observation_space.shape
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(h, w, c), dtype=np.float32
        )

    def observation(self, obs):
        return obs.astype(np.float32) / 255.0


class CustomReward(gym.Wrapper):
    """
    Replaces the default gym-super-mario-bros reward with a shaped signal
    that actually incentivises completing the level:

      +Δx_pos   : reward for moving right (dense progress signal)
      -Δclock   : penalty for time passing (encourages speed)
      -50        : heavy penalty for dying (discourages sprint-and-die)
      +500       : bonus for reaching the flagpole (completes the level)

    The default reward uses a -15 death penalty that is too small to deter
    the sprint-and-die local optimum. We increase it to -50.
    """

    # Tunable constants — easy to adjust without touching the rest of the code
    DEATH_PENALTY    = -50
    FLAG_BONUS       = 500
    PROGRESS_SCALE   = 1.0   # multiply x_pos delta (1.0 = raw pixels)
    CLOCK_PENALTY    = 0.1   # penalty per clock tick lost

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._last_x_pos = 0
        self._last_life  = 2   # Mario starts with 3 lives (0-indexed)
        self._last_time  = 400  # game clock starts at 400

    # ------------------------------------------------------------------
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # info may be empty on first reset for bridged envs — default safely
        self._last_x_pos = info.get("x_pos", 0)
        self._last_life  = info.get("life",  2)
        self._last_time  = info.get("time",  400)
        return obs, info

    def step(self, action):
        obs, _default_reward, terminated, truncated, info = self.env.step(action)

        x_pos = info.get("x_pos", self._last_x_pos)
        life  = info.get("life",  self._last_life)
        time  = info.get("time",  self._last_time)

        # ── Progress reward ───────────────────────────────────────────
        reward = self.PROGRESS_SCALE * (x_pos - self._last_x_pos)

        # ── Clock penalty (time ticks down → positive delta = penalty) ─
        time_lost = self._last_time - time
        if time_lost > 0:
            reward -= self.CLOCK_PENALTY * time_lost

        # ── Death penalty ─────────────────────────────────────────────
        if life < self._last_life:
            reward += self.DEATH_PENALTY

        # ── Level-completion bonus ────────────────────────────────────
        if info.get("flag_get", False):
            reward += self.FLAG_BONUS

        self._last_x_pos = x_pos
        self._last_life  = life
        self._last_time  = time

        return obs, float(reward), terminated, truncated, info


# ─── Wrapper Factories ────────────────────────────────────────────────────────

def _base_env() -> gym.Env:
    """
    Create the Mario World 1-1 env bridged to the gymnasium API.

    gym-super-mario-bros uses the old 'gym' namespace. The gym.make() call
    wraps the NES env in several old-gym shims (TimeLimit, OrderEnforcing,
    PassiveEnvChecker) that have inconsistent step() signatures across gym
    versions.  We bypass all of them with .unwrapped and apply our own clean
    wrappers on top.
    """
    raw_gym_env = old_gym.make(ENV_ID)         # old_gym.TimeLimit → ... → NesEnv
    raw_nes_env = raw_gym_env.unwrapped        # SuperMarioBrosEnv (pure 4-tuple step)
    mario_env = JoypadSpace(raw_nes_env, SIMPLE_MOVEMENT)
    env = OldGymBridge(mario_env)
    return env


def make_ppo_env() -> gym.Env:
    """
    Build the fully preprocessed env for PPO training:
      raw RGB → skip frames → custom reward → grayscale → resize
    Observations are uint8 [0,255] — SB3 handles normalization internally.
    Returns a single (non-vectorized) gymnasium env.
    """
    env = _base_env()
    env = SkipFrame(env, skip=N_SKIP)
    env = CustomReward(env)      # reshape reward before obs transforms
    env = GrayScaleFrame(env)
    env = ResizeFrame(env)
    return env


def make_vec_ppo_env(n_envs: int = 8, seed: int = 42) -> VecFrameStack:
    """
    Build a vectorized, frame-stacked env suitable for SB3's PPO.
    Uses SubprocVecEnv so each NES emulator runs in its own process.
    Returns VecFrameStack wrapping n_envs parallel environments.
    """
    def _thunk():
        return make_ppo_env()

    vec_env = SubprocVecEnv([_thunk] * n_envs)
    vec_env = VecFrameStack(vec_env, n_stack=N_STACK, channels_order="last")
    return vec_env


def make_collect_env() -> gym.Env:
    """
    Build an env used during data *collection* only:
    - Applies frame-skip so action timing matches the trained policy.
    - Does NOT grayscale: we store full RGB for the world model decoder.
    - Resizes to RGB_HEIGHT × RGB_WIDTH.
    - Does NOT normalize: we store uint8 to save disk space.

    The collector script queries the trained agent with a grayscale+stacked
    observation (rebuilt on the fly from a FrameBuffer) while storing the
    raw RGB frame into the HDF5 dataset.
    """
    env = _base_env()
    env = SkipFrame(env, skip=N_SKIP)
    env = CustomReward(env)      # consistent reward shaping during collection
    env = ResizeFrame(env, height=RGB_HEIGHT, width=RGB_WIDTH)
    return env
