"""Convert collected Mario transitions into a training-friendly cache."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2
import h5py
import numpy as np
from tqdm.auto import tqdm


REQUIRED_DATASETS = ("observations", "next_obs", "actions", "rewards", "dones")
ACTION_NAMES = ("NOOP", "right", "right+A", "right+B", "right+A+B", "A", "left")
ARRAY_FILES = (
    "frames.npy",
    "actions.npy",
    "rewards.npy",
    "episode_offsets.npy",
    "episode_splits.npy",
    "source_transition_indices.npy",
)


class SourceValidationError(ValueError):
    """Raised when the input HDF5 file cannot be converted safely."""


class CacheValidationError(ValueError):
    """Raised when generated cache artifacts disagree with their metadata."""


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
    limit_transitions: int | None = None


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


def _validate_conversion_config(config: ConversionConfig) -> None:
    if config.height < 1 or config.width < 1:
        raise ValueError("output height and width must be positive")
    if config.history < 1:
        raise ValueError("history must be positive")
    if config.workers < 1:
        raise ValueError("workers must be positive")
    if (
        config.limit_transitions is not None
        and config.limit_transitions < config.history
    ):
        raise ValueError("transition limit must be at least the history length")


def _resize_frame(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def _resize_batch(
    frames: np.ndarray, height: int, width: int, workers: int
) -> np.ndarray:
    if len(frames) == 0:
        return np.empty((0, height, width, 3), dtype=np.uint8)
    if workers == 1:
        resized = [_resize_frame(frame, height, width) for frame in frames]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            resized = list(
                pool.map(
                    lambda frame: _resize_frame(frame, height, width),
                    frames,
                )
            )
    return np.stack(resized).astype(np.uint8, copy=False)


def _sha256(
    path: Path,
    block_size: int = 8 * 1024 * 1024,
    progress: Any = None,
) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
            if progress is not None:
                progress.update(len(block))
    return digest.hexdigest()


def _dataset_card(metadata: dict) -> str:
    return (
        "---\n"
        "license: other\n"
        "task_categories:\n"
        "- video-prediction\n"
        "---\n\n"
        "# Mario 1-1 action-conditioned transitions\n\n"
        "This repository contains a NumPy memory-mapped cache derived from "
        "recorded Super Mario Bros. World 1-1 trajectories. The action at "
        "transition t produces frame t+1. Download the repository to local "
        "SSD before training.\n\n"
        f"- Transitions: {metadata['n_transitions']}\n"
        f"- Trajectories: {metadata['n_trajectories']}\n"
        f"- Frame shape: {metadata['frame_shape']}\n"
        f"- History: {metadata['history']}\n"
        f"- Explicit breaks: {metadata['break_indices']}\n\n"
        "The cache is not an official Nintendo dataset and contains gameplay "
        "imagery whose use remains subject to applicable rights and local law.\n"
    )


def _write_cache(config: ConversionConfig, temporary_dir: Path) -> dict:
    cv2.setNumThreads(1)
    input_path = Path(config.input_path)
    with ExitStack() as stack:
        handle = stack.enter_context(h5py.File(input_path, "r"))
        source_schema = validate_source(handle)
        if source_schema.n_actions > 256:
            raise SourceValidationError("at most 256 discrete actions are supported")
        n_transitions = (
            source_schema.n_transitions
            if config.limit_transitions is None
            else int(config.limit_transitions)
        )
        if n_transitions > source_schema.n_transitions:
            raise ValueError(
                "transition limit cannot exceed the source transition count"
            )
        active_breaks = tuple(
            sorted(index for index in config.break_indices if index < n_transitions)
        )
        ignored_breaks = tuple(
            sorted(index for index in config.break_indices if index >= n_transitions)
        )
        if ignored_breaks:
            print(
                "Ignoring break indices outside converted prefix: "
                f"{list(ignored_breaks)}"
            )

        actions = handle["actions"][:n_transitions].astype(np.uint8)
        rewards = handle["rewards"][:n_transitions].astype(np.float32)
        dones = handle["dones"][:n_transitions].astype(bool)
        offsets = build_episode_offsets(dones, active_breaks)
        n_trajectories = len(offsets) - 1
        splits = assign_episode_splits(
            n_trajectories, config.split_seed, config.split_fractions
        )
        frame_count = n_transitions + n_trajectories

        frames_out = np.lib.format.open_memmap(
            temporary_dir / "frames.npy",
            mode="w+",
            dtype=np.uint8,
            shape=(frame_count, config.height, config.width, 3),
        )
        np.save(temporary_dir / "actions.npy", actions)
        np.save(temporary_dir / "rewards.npy", rewards)
        np.save(temporary_dir / "episode_offsets.npy", offsets)
        np.save(temporary_dir / "episode_splits.npy", splits)
        np.save(
            temporary_dir / "source_transition_indices.npy",
            np.arange(n_transitions, dtype=np.int64),
        )

        observations = handle["observations"]
        next_observations = handle["next_obs"]
        source_chunk = int(observations.chunks[0]) if observations.chunks else 1024
        trajectory_ends = set((offsets[1:] - 1).tolist())
        pending_next_frame: np.ndarray | None = None
        pending_transition: int | None = None
        conversion_progress = stack.enter_context(
            tqdm(
                total=n_transitions,
                desc="Converting transitions",
                unit="transition",
                unit_scale=True,
                dynamic_ncols=True,
            )
        )

        for block_start in range(0, n_transitions, source_chunk):
            block_end = min(block_start + source_chunk, n_transitions)
            observation_block = observations[block_start:block_end]
            next_observation_block = next_observations[block_start:block_end]
            had_pending_transition = pending_next_frame is not None

            if pending_next_frame is not None and not np.array_equal(
                pending_next_frame, observation_block[0]
            ):
                assert pending_transition is not None
                raise SourceValidationError(
                    f"next_obs[{pending_transition}] does not match "
                    f"observations[{pending_transition + 1}]; if the collector "
                    "restarted there, pass "
                    f"--break-index {pending_transition + 1}"
                )
            for relative_index in range(len(observation_block) - 1):
                transition_index = block_start + relative_index
                if transition_index in trajectory_ends:
                    continue
                if not np.array_equal(
                    next_observation_block[relative_index],
                    observation_block[relative_index + 1],
                ):
                    raise SourceValidationError(
                        f"next_obs[{transition_index}] does not match "
                        f"observations[{transition_index + 1}]; if the collector "
                        "restarted there, pass "
                        f"--break-index {transition_index + 1}"
                    )

            last_transition = block_end - 1
            if (
                last_transition < n_transitions - 1
                and last_transition not in trajectory_ends
            ):
                pending_next_frame = next_observation_block[-1].copy()
                pending_transition = last_transition
            else:
                pending_next_frame = None
                pending_transition = None

            resized_observations = _resize_batch(
                observation_block,
                config.height,
                config.width,
                config.workers,
            )

            first_episode = int(
                np.searchsorted(offsets[1:], block_start, side="right")
            )
            last_episode = int(
                np.searchsorted(offsets[1:], block_end - 1, side="right")
            )
            for episode in range(first_episode, last_episode + 1):
                start = max(block_start, int(offsets[episode]))
                end = min(block_end, int(offsets[episode + 1]))
                source_slice = slice(start - block_start, end - block_start)
                output_start = start + episode
                frames_out[output_start : output_start + (end - start)] = (
                    resized_observations[source_slice]
                )

            terminal_episodes = np.flatnonzero(
                (offsets[1:] - 1 >= block_start)
                & (offsets[1:] - 1 < block_end)
            )
            if terminal_episodes.size:
                terminal_indices = offsets[terminal_episodes + 1] - 1
                terminal_frames = _resize_batch(
                    next_observation_block[terminal_indices - block_start],
                    config.height,
                    config.width,
                    config.workers,
                )
                for frame, episode, transition_end in zip(
                    terminal_frames,
                    terminal_episodes,
                    offsets[terminal_episodes + 1],
                ):
                    frames_out[int(transition_end) + int(episode)] = frame
            completed_transitions = (
                block_end
                - block_start
                + int(had_pending_transition)
                - int(pending_next_frame is not None)
            )
            conversion_progress.update(completed_transitions)

        frames_out.flush()
        del frames_out

    metadata = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_name": input_path.name,
        "source_size_bytes": input_path.stat().st_size,
        "source_frame_shape": list(source_schema.frame_shape),
        "source_n_transitions": source_schema.n_transitions,
        "n_transitions": n_transitions,
        "n_trajectories": n_trajectories,
        "n_frames": frame_count,
        "frame_shape": [config.height, config.width, 3],
        "history": config.history,
        "frame_skip": 4,
        "n_actions": source_schema.n_actions,
        "action_names": list(ACTION_NAMES[: source_schema.n_actions]),
        "break_indices": list(active_breaks),
        "ignored_break_indices": list(ignored_breaks),
        "limit_transitions": config.limit_transitions,
        "split_seed": config.split_seed,
        "split_fractions": list(config.split_fractions),
        "resize": "opencv.INTER_AREA",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "h5py_version": h5py.__version__,
        "opencv_version": cv2.__version__,
    }
    hash_total = sum((temporary_dir / name).stat().st_size for name in ARRAY_FILES)
    with tqdm(
        total=hash_total,
        desc="Hashing cache",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        dynamic_ncols=True,
    ) as hash_progress:
        metadata["sha256"] = {
            name: _sha256(temporary_dir / name, progress=hash_progress)
            for name in ARRAY_FILES
        }
    (temporary_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (temporary_dir / "README.md").write_text(
        _dataset_card(metadata), encoding="utf-8"
    )
    return metadata


def validate_cache(path: str | Path) -> dict:
    """Verify cache structure, metadata, and artifact hashes."""
    cache_path = Path(path)
    try:
        metadata = json.loads(
            (cache_path / "metadata.json").read_text(encoding="utf-8")
        )
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise CacheValidationError("cache metadata is missing or invalid") from error

    try:
        frames = np.load(cache_path / "frames.npy", mmap_mode="r")
        actions = np.load(cache_path / "actions.npy", mmap_mode="r")
        rewards = np.load(cache_path / "rewards.npy", mmap_mode="r")
        offsets = np.load(cache_path / "episode_offsets.npy")
        splits = np.load(cache_path / "episode_splits.npy")
        source_indices = np.load(
            cache_path / "source_transition_indices.npy", mmap_mode="r"
        )
    except (FileNotFoundError, OSError, ValueError) as error:
        raise CacheValidationError("cache arrays are missing or unreadable") from error

    expected_frame_shape = (
        int(metadata["n_frames"]),
        *tuple(int(value) for value in metadata["frame_shape"]),
    )
    if frames.shape != expected_frame_shape or frames.dtype != np.uint8:
        raise CacheValidationError("frames.npy shape or dtype does not match metadata")
    if not (
        actions.shape
        == rewards.shape
        == source_indices.shape
        == (int(metadata["n_transitions"]),)
    ):
        raise CacheValidationError("transition arrays have inconsistent shapes")
    if offsets.shape != (int(metadata["n_trajectories"]) + 1,):
        raise CacheValidationError("episode offsets do not match trajectory count")
    if int(offsets[0]) != 0 or int(offsets[-1]) != int(metadata["n_transitions"]):
        raise CacheValidationError("episode offsets do not cover all transitions")
    if np.any(np.diff(offsets) <= 0):
        raise CacheValidationError("episode offsets must be strictly increasing")
    if splits.shape != (int(metadata["n_trajectories"]),):
        raise CacheValidationError("episode splits do not match trajectory count")
    if not set(np.unique(splits).tolist()).issubset({0, 1, 2}):
        raise CacheValidationError("episode splits contain unknown labels")

    hashes = metadata.get("sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(ARRAY_FILES):
        raise CacheValidationError("cache metadata has an invalid SHA-256 manifest")
    for name, expected in hashes.items():
        if _sha256(cache_path / name) != expected:
            raise CacheValidationError(f"SHA-256 mismatch for {name}")

    return metadata


def convert_dataset(config: ConversionConfig) -> Path:
    """Create, validate, and atomically publish a local memory-mapped cache."""
    _validate_conversion_config(config)
    input_path = Path(config.input_path).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"input HDF5 file does not exist: {input_path}")

    output = Path(config.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"refusing to replace non-empty output directory: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent)
    )

    resolved_config = ConversionConfig(
        input_path=input_path,
        output_dir=output,
        height=config.height,
        width=config.width,
        history=config.history,
        break_indices=config.break_indices,
        split_seed=config.split_seed,
        split_fractions=config.split_fractions,
        workers=config.workers,
        limit_transitions=config.limit_transitions,
    )
    _write_cache(resolved_config, temporary)
    validate_cache(temporary)
    if output.exists():
        output.rmdir()
    os.replace(temporary, output)
    return output


def publish_cache(
    output_dir: str | Path,
    repo_id: str,
    private: bool,
    api: Any = None,
) -> str:
    """Validate and upload a cache to a Hugging Face dataset repository."""
    parts = repo_id.split("/")
    if len(parts) != 2 or any(not part.strip() for part in parts):
        raise ValueError("Hugging Face repo id must use namespace/name")

    cache_path = Path(output_dir).resolve()
    validate_cache(cache_path)
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()

    repo_url = api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )
    result = api.upload_folder(
        folder_path=str(cache_path),
        repo_id=repo_id,
        repo_type="dataset",
    )
    return str(getattr(result, "repo_url", repo_url))
