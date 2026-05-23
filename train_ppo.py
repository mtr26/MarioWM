"""
PPO training script for Super Mario Bros World 1-1.

Data collection is integrated directly into training:
  1. A random rollout phase runs BEFORE training to seed the dataset with
     diverse, chaotic behaviors (Mario dying, standing still, etc.)
  2. A DataCollectionCallback runs DURING training, collecting RGB frames
     using the current (evolving) policy + ε-greedy exploration.
     Early checkpoints → exploratory. Late checkpoints → skilled play.
     Together they give the world model the full behavioral spectrum.

The HDF5 dataset grows incrementally during training — you can kill the
run at any time and still have a valid dataset.

Usage:
    python train_ppo.py                          # train + collect with defaults
    python train_ppo.py --timesteps 2_000_000    # longer run
    python train_ppo.py --resume                 # continue from checkpoint
    python train_ppo.py --no-collect             # skip data collection
"""

import os
import argparse
import numpy as np
import h5py

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from env.wrappers import (
    make_vec_ppo_env, make_ppo_env, make_collect_env,
    N_STACK, N_ACTIONS, RGB_HEIGHT, RGB_WIDTH,
)

# ─── Paths ────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = "checkpoints"
LOG_DIR        = "logs/ppo_mario"
DATASET_DIR    = "dataset"
FINAL_MODEL    = os.path.join(CHECKPOINT_DIR, "ppo_mario_final")
DATASET_PATH   = os.path.join(DATASET_DIR,    "mario_1-1_live.h5")

for d in (CHECKPOINT_DIR, LOG_DIR, DATASET_DIR):
    os.makedirs(d, exist_ok=True)


# ─── PPO Hyperparameters ──────────────────────────────────────────────────────
PPO_KWARGS = dict(
    policy="CnnPolicy",
    learning_rate=2.5e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=4,
    gamma=0.9,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    tensorboard_log=LOG_DIR,
    verbose=1,
)


# ─── Frame buffer (grayscale stack for agent inference) ───────────────────────

class FrameBuffer:
    """Rolling grayscale frame buffer matching the agent's training obs format."""

    def __init__(self):
        import cv2
        self._cv2 = cv2
        self.buf = np.zeros((N_STACK, RGB_HEIGHT, RGB_WIDTH, 1), dtype=np.uint8)

    def reset(self, rgb_frame: np.ndarray) -> np.ndarray:
        gray = self._to_gray(rgb_frame)
        for i in range(N_STACK):
            self.buf[i] = gray
        return self._stacked()

    def push(self, rgb_frame: np.ndarray) -> np.ndarray:
        self.buf = np.roll(self.buf, shift=-1, axis=0)
        self.buf[-1] = self._to_gray(rgb_frame)
        return self._stacked()

    def _to_gray(self, rgb: np.ndarray) -> np.ndarray:
        gray = self._cv2.cvtColor(rgb, self._cv2.COLOR_RGB2GRAY)
        return gray[:, :, np.newaxis]  # uint8, no normalization

    def _stacked(self) -> np.ndarray:
        # (H, W, N_STACK) — matches VecFrameStack channels_order="last"
        return np.concatenate(self.buf, axis=-1)


# ─── HDF5 dataset writer ──────────────────────────────────────────────────────

class DatasetWriter:
    """
    Incrementally appends transitions to an HDF5 file using resizable datasets.
    Safe to instantiate multiple times — will append to an existing file.
    """

    CHUNK = 1024   # HDF5 chunk size along axis 0

    def __init__(self, path: str, max_steps: int = 1_000_000):
        self.path = path
        existed = os.path.exists(path)
        self._f = h5py.File(path, "a")
        self._cursor = 0

        if not existed or "observations" not in self._f:
            # Create resizable datasets
            kw = dict(maxshape=(None,), compression="lzf", chunks=(self.CHUNK,))
            kw_frame = dict(
                maxshape=(None, RGB_HEIGHT, RGB_WIDTH, 3),
                compression="lzf",
                chunks=(self.CHUNK, RGB_HEIGHT, RGB_WIDTH, 3),
            )
            self._f.create_dataset("observations",  shape=(0, RGB_HEIGHT, RGB_WIDTH, 3),
                                   dtype=np.uint8,   **kw_frame)
            self._f.create_dataset("next_obs",       shape=(0, RGB_HEIGHT, RGB_WIDTH, 3),
                                   dtype=np.uint8,   **kw_frame)
            self._f.create_dataset("actions",        shape=(0,), dtype=np.int32,   **kw)
            self._f.create_dataset("rewards",        shape=(0,), dtype=np.float32, **kw)
            self._f.create_dataset("dones",          shape=(0,), dtype=bool,       **kw)
        else:
            self._cursor = len(self._f["actions"])
            print(f"[dataset] Resuming — {self._cursor:,} transitions already stored.")

    def write_batch(
        self,
        obs_batch:      np.ndarray,   # (B, H, W, 3) uint8
        next_obs_batch: np.ndarray,
        act_batch:      np.ndarray,   # (B,) int32
        rew_batch:      np.ndarray,   # (B,) float32
        done_batch:     np.ndarray,   # (B,) bool
    ):
        B = len(act_batch)
        new_len = self._cursor + B
        for key in ("observations", "next_obs", "actions", "rewards", "dones"):
            self._f[key].resize(new_len, axis=0)

        sl = slice(self._cursor, new_len)
        self._f["observations"][sl] = obs_batch
        self._f["next_obs"][sl]     = next_obs_batch
        self._f["actions"][sl]      = act_batch
        self._f["rewards"][sl]      = rew_batch
        self._f["dones"][sl]        = done_batch
        self._f.flush()
        self._cursor = new_len

    @property
    def n_stored(self) -> int:
        return self._cursor

    def close(self):
        self._f.attrs["n_steps"]   = self._cursor
        self._f.attrs["world"]     = "1-1"
        self._f.attrs["n_actions"] = N_ACTIONS
        self._f.attrs["frame_h"]   = RGB_HEIGHT
        self._f.attrs["frame_w"]   = RGB_WIDTH
        self._f.close()


# ─── Random collection (pre-training) ─────────────────────────────────────────

def collect_random(writer: DatasetWriter, n_steps: int, seed: int = 42):
    """
    Run a fully random policy on the collect env and write transitions.
    This seeds the dataset with diverse, chaotic behaviors before training.
    """
    from tqdm import tqdm
    rng = np.random.default_rng(seed)

    env = make_collect_env()
    obs, _ = env.reset(seed=seed)
    episodes = 0

    print(f"\n[pre-training] Collecting {n_steps:,} random steps …")
    # Pre-allocate batch buffers
    obs_buf      = np.empty((n_steps, RGB_HEIGHT, RGB_WIDTH, 3), dtype=np.uint8)
    next_obs_buf = np.empty((n_steps, RGB_HEIGHT, RGB_WIDTH, 3), dtype=np.uint8)
    act_buf      = np.empty(n_steps, dtype=np.int32)
    rew_buf      = np.empty(n_steps, dtype=np.float32)
    done_buf     = np.empty(n_steps, dtype=bool)

    for i in tqdm(range(n_steps), unit="step", desc="random rollout"):
        action = rng.integers(0, N_ACTIONS)
        next_obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated

        obs_buf[i]      = obs
        next_obs_buf[i] = next_obs
        act_buf[i]      = action
        rew_buf[i]      = float(reward)
        done_buf[i]     = done

        obs = next_obs
        if done:
            obs, _ = env.reset()
            episodes += 1

    env.close()
    writer.write_batch(obs_buf, next_obs_buf, act_buf, rew_buf, done_buf)
    print(f"[pre-training] Done — {n_steps:,} steps, {episodes} episodes stored.")


# ─── Data collection callback ─────────────────────────────────────────────────

class DataCollectionCallback(BaseCallback):
    """
    Runs alongside PPO training. After every `collect_every` training steps,
    it steps a side RGB env with the current policy + ε-greedy for
    `collect_n_steps` steps and appends the transitions to the HDF5 dataset.

    Because this runs throughout training, the dataset naturally captures:
      - Early training: noisy, exploratory (high entropy policy)
      - Mid training:   partially skilled
      - Late training:  competent play
    """

    def __init__(
        self,
        writer:          DatasetWriter,
        collect_every:   int   = 10_000,   # collect after every N training steps
        collect_n_steps: int   = 512,      # steps to collect per callback call
        epsilon:         float = 0.15,     # random action fraction
        seed:            int   = 0,
    ):
        super().__init__(verbose=0)
        self.writer          = writer
        self.collect_every   = collect_every
        self.collect_n_steps = collect_n_steps
        self.epsilon         = epsilon
        self._rng            = np.random.default_rng(seed)
        self._last_collect   = 0

        # Side env for RGB collection — created once in _on_training_start
        self._collect_env  = None
        self._frame_buf    = None
        self._current_obs  = None   # last RGB obs (uint8)
        self._stacked_obs  = None   # grayscale stacked obs for agent

    def _on_training_start(self):
        self._collect_env = make_collect_env()
        self._frame_buf   = FrameBuffer()
        rgb_obs, _        = self._collect_env.reset()
        self._current_obs = rgb_obs
        self._stacked_obs = self._frame_buf.reset(rgb_obs)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_collect >= self.collect_every:
            self._collect_batch()
            self._last_collect = self.num_timesteps
        return True

    def _collect_batch(self):
        obs_buf      = np.empty((self.collect_n_steps, RGB_HEIGHT, RGB_WIDTH, 3), dtype=np.uint8)
        next_obs_buf = np.empty((self.collect_n_steps, RGB_HEIGHT, RGB_WIDTH, 3), dtype=np.uint8)
        act_buf      = np.empty(self.collect_n_steps, dtype=np.int32)
        rew_buf      = np.empty(self.collect_n_steps, dtype=np.float32)
        done_buf     = np.empty(self.collect_n_steps, dtype=bool)

        for i in range(self.collect_n_steps):
            # ε-greedy: random or current policy
            if self._rng.random() < self.epsilon:
                action = int(self._rng.integers(0, N_ACTIONS))
            else:
                stacked = self._stacked_obs[np.newaxis]   # (1, H, W, N_STACK)
                action, _ = self.model.predict(stacked, deterministic=True)
                action = int(action[0])

            next_rgb, reward, terminated, truncated, info = self._collect_env.step(action)
            done = terminated or truncated

            obs_buf[i]      = self._current_obs
            next_obs_buf[i] = next_rgb
            act_buf[i]      = action
            rew_buf[i]      = float(reward)
            done_buf[i]     = done

            # Advance state
            self._stacked_obs = self._frame_buf.push(next_rgb)
            self._current_obs = next_rgb

            if done:
                rgb_obs, _        = self._collect_env.reset()
                self._current_obs = rgb_obs
                self._stacked_obs = self._frame_buf.reset(rgb_obs)

        self.writer.write_batch(obs_buf, next_obs_buf, act_buf, rew_buf, done_buf)

    def _on_training_end(self):
        if self._collect_env is not None:
            self._collect_env.close()
        total = self.writer.n_stored
        print(f"\n[dataset] Training complete — {total:,} transitions collected.")


# ─── Eval env factory ─────────────────────────────────────────────────────────

def make_eval_env():
    env = make_ppo_env()
    env = Monitor(env)
    vec = DummyVecEnv([lambda: env])
    vec = VecFrameStack(vec, n_stack=N_STACK, channels_order="last")
    return vec


# ─── Main training function ───────────────────────────────────────────────────

def train(
    timesteps:        int   = 1_000_000,
    n_envs:           int   = 8,
    resume:           bool  = False,
    seed:             int   = 42,
    # Data collection
    collect:          bool  = True,
    n_random_steps:   int   = 10_000,    # random steps before training starts
    collect_every:    int   = 10_000,    # collect every N training steps
    collect_n_steps:  int   = 512,       # steps per collection call
    epsilon:          float = 0.15,      # ε-greedy fraction during collection
    dataset_path:     str   = DATASET_PATH,
):
    print(f"\n{'='*60}")
    print(f"  Mario 1-1 PPO Training")
    print(f"  Total timesteps : {timesteps:,}")
    print(f"  Parallel envs   : {n_envs}")
    print(f"  Resume          : {resume}")
    print(f"  Data collection : {collect}")
    if collect:
        print(f"    Random steps  : {n_random_steps:,}")
        print(f"    Collect every : {collect_every:,} steps")
        print(f"    Steps/call    : {collect_n_steps}")
        print(f"    Epsilon       : {epsilon:.0%}")
        print(f"    Dataset       : {dataset_path}")
    print(f"{'='*60}\n")

    # ── Dataset writer ────────────────────────────────────────────────────────
    writer = DatasetWriter(dataset_path) if collect else None

    # ── Random pre-training collection ───────────────────────────────────────
    if collect and n_random_steps > 0 and not resume:
        collect_random(writer, n_steps=n_random_steps, seed=seed)

    # ── Training environments ─────────────────────────────────────────────────
    train_env = make_vec_ppo_env(n_envs=n_envs, seed=seed)
    eval_env  = make_eval_env()

    # ── Callbacks ─────────────────────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq=max(50_000 // n_envs, 1),
        save_path=CHECKPOINT_DIR,
        name_prefix="ppo_mario",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=CHECKPOINT_DIR,
        log_path=LOG_DIR,
        eval_freq=max(100_000 // n_envs, 1),
        n_eval_episodes=2,
        deterministic=True,
        render=False,
    )
    callbacks = [checkpoint_cb, eval_cb]

    if collect:
        data_cb = DataCollectionCallback(
            writer=writer,
            collect_every=collect_every,
            collect_n_steps=collect_n_steps,
            epsilon=epsilon,
            seed=seed + 1,
        )
        callbacks.append(data_cb)

    # ── Model ─────────────────────────────────────────────────────────────────
    if resume and os.path.exists(FINAL_MODEL + ".zip"):
        print(f"[resume] Loading model from {FINAL_MODEL}.zip")
        model = PPO.load(FINAL_MODEL, env=train_env, tensorboard_log=LOG_DIR)
        model.set_env(train_env)
    else:
        model = PPO(env=train_env, **PPO_KWARGS)

    # ── Train ─────────────────────────────────────────────────────────────────
    model.learn(
        total_timesteps=timesteps,
        callback=CallbackList(callbacks),
        reset_num_timesteps=not resume,
        tb_log_name="ppo_run",
        progress_bar=True,
    )

    model.save(FINAL_MODEL)
    print(f"\n✓ Model saved → {FINAL_MODEL}.zip")

    train_env.close()
    eval_env.close()

    if writer:
        writer.close()
        size_mb = os.path.getsize(dataset_path) / (1024 ** 2)
        print(f"✓ Dataset saved → {dataset_path}  ({size_mb:.1f} MB)")

    return model


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on Mario 1-1 + collect dataset")
    parser.add_argument("--timesteps",       type=int,   default=1_000_000)
    parser.add_argument("--n-envs",          type=int,   default=8)
    parser.add_argument("--resume",          action="store_true")
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--no-collect",      action="store_true",
                        help="Disable data collection (train-only mode)")
    parser.add_argument("--n-random-steps",  type=int,   default=10_000,
                        help="Random steps before training (default: 10k)")
    parser.add_argument("--collect-every",   type=int,   default=10_000,
                        help="Collect every N training steps (default: 10k)")
    parser.add_argument("--collect-n-steps", type=int,   default=512,
                        help="Steps collected per callback call (default: 512)")
    parser.add_argument("--epsilon",         type=float, default=0.15,
                        help="Fraction of random actions during collection (default: 0.15)")
    parser.add_argument("--dataset-path",    type=str,   default=DATASET_PATH)
    args = parser.parse_args()

    train(
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        resume=args.resume,
        seed=args.seed,
        collect=not args.no_collect,
        n_random_steps=args.n_random_steps,
        collect_every=args.collect_every,
        collect_n_steps=args.collect_n_steps,
        epsilon=args.epsilon,
        dataset_path=args.dataset_path,
    )
