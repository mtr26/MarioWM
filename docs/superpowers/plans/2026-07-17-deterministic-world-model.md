# Deterministic Mario World Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the Mario HDF5 transitions into a fast memory-mapped trajectory cache, optionally publish it to Hugging Face Hub, and train a deterministic action-conditioned U-Net efficiently on one H100.

**Architecture:** A conversion CLI streams the immutable HDF5 source into unique, resized trajectory frames stored as `.npy` memory maps with episode offsets and deterministic splits. A PyTorch dataset builds boundary-safe four-frame windows, and a FiLM-conditioned U-Net predicts the next RGB frame from the frame and action histories. A single-GPU trainer uses BF16, TF32, EMA, warmup/cosine scheduling, resumable checkpoints, TensorBoard, and recorded-action validation rollouts.

**Tech Stack:** Python 3.11, NumPy, h5py, OpenCV, PyTorch, torchvision, PyYAML, TensorBoard, pytest, Hugging Face Hub/Xet.

## Global Constraints

- Keep `mario_1-1_live.h5` immutable and untracked.
- Treat transition index 10,000 as a new trajectory start even though `dones[9999]` is false.
- Preserve the native 240×256 aspect ratio by resizing to 120×128 with OpenCV `INTER_AREA`.
- Use four context frames and four aligned actions; `actions[t]` produces `next_obs[t]`.
- Split complete trajectories or valid partial trajectory segments, never individual frames.
- Use the local memory-mapped cache for training; Hugging Face Hub is transport and storage, not the live training filesystem.
- Never accept, print, or persist a Hugging Face token; use the credential store or `HF_TOKEN`.
- Do not add attention, diffusion, transformers, multi-GPU training, or remote-checkpoint collection in this implementation.

---

## File Structure

- Create `world_model/__init__.py`: public package exports.
- Create `world_model/conversion.py`: HDF5 validation, trajectory layout, resizing, cache writing, validation, hashing, dataset card, and Hub publication.
- Create `prepare_world_model_data.py`: conversion and optional Hub upload CLI.
- Create `world_model/data.py`: memory-mapped trajectory-window dataset and DataLoader helpers.
- Create `world_model/model.py`: action conditioner, FiLM residual blocks, and deterministic U-Net.
- Create `world_model/config.py`: validated YAML configuration dataclasses.
- Create `world_model/checkpointing.py`: atomic checkpoint and RNG capture/restore.
- Create `world_model/training.py`: EMA, schedule, train/evaluate loops, prediction grids, and autoregressive rollout.
- Create `train_world_model.py`: single-H100 training CLI.
- Create `configs/deterministic_unet.yaml`: baseline H100 configuration.
- Create `tests/conftest.py`: compact synthetic HDF5 fixture.
- Create `tests/test_conversion.py`: schema, boundary, cache, and metadata tests.
- Create `tests/test_huggingface.py`: mocked Hub publication tests.
- Create `tests/test_data.py`: sample alignment and split-isolation tests.
- Create `tests/test_model.py`: shape, range, determinism, and action-path tests.
- Create `tests/test_checkpointing.py`: atomic checkpoint and exact-resume tests.
- Create `tests/test_training.py`: CPU train/evaluate smoke tests.
- Modify `requirements.txt`: add Hub and test dependencies.
- Modify `README.md`: document conversion, upload, download, overfit, training, and resume commands.

---

### Task 1: Establish Conversion Contracts and Trajectory Boundaries

**Files:**
- Create: `world_model/__init__.py`
- Create: `world_model/conversion.py`
- Create: `tests/conftest.py`
- Create: `tests/test_conversion.py`

**Interfaces:**
- Consumes: source datasets `observations`, `next_obs`, `actions`, `rewards`, and `dones`.
- Produces: `SourceSchema`, `ConversionConfig`, `validate_source()`, `build_episode_offsets()`, and `assign_episode_splits()`.

- [ ] **Step 1: Write the synthetic source fixture and failing boundary tests**

```python
# tests/conftest.py
from pathlib import Path

import h5py
import numpy as np
import pytest


@pytest.fixture
def synthetic_h5(tmp_path: Path) -> Path:
    path = tmp_path / "source.h5"
    n, height, width = 12, 8, 10
    observations = np.empty((n, height, width, 3), dtype=np.uint8)
    next_obs = np.empty_like(observations)
    for index in range(n):
        observations[index].fill(index)
        next_obs[index].fill(index + 1)
    observations[6].fill(90)
    next_obs[5].fill(89)
    actions = np.arange(n, dtype=np.int32) % 7
    rewards = np.arange(n, dtype=np.float32)
    dones = np.zeros(n, dtype=bool)
    dones[[3, 9]] = True
    with h5py.File(path, "w") as handle:
        handle.create_dataset("observations", data=observations, chunks=(4, height, width, 3))
        handle.create_dataset("next_obs", data=next_obs, chunks=(4, height, width, 3))
        handle.create_dataset("actions", data=actions)
        handle.create_dataset("rewards", data=rewards)
        handle.create_dataset("dones", data=dones)
    return path
```

```python
# tests/test_conversion.py
import h5py
import numpy as np
import pytest

from world_model.conversion import (
    SourceValidationError,
    assign_episode_splits,
    build_episode_offsets,
    validate_source,
)


def test_validate_source_reports_expected_schema(synthetic_h5):
    with h5py.File(synthetic_h5, "r") as handle:
        schema = validate_source(handle)
    assert schema.n_transitions == 12
    assert schema.frame_shape == (8, 10, 3)
    assert schema.n_actions == 7


def test_build_episode_offsets_combines_dones_breaks_and_file_end():
    dones = np.zeros(12, dtype=bool)
    dones[[3, 9]] = True
    offsets = build_episode_offsets(dones, break_indices=(6,))
    np.testing.assert_array_equal(offsets, np.array([0, 4, 6, 10, 12], dtype=np.int64))


def test_build_episode_offsets_rejects_out_of_range_break():
    with pytest.raises(ValueError, match="break index"):
        build_episode_offsets(np.zeros(8, dtype=bool), break_indices=(8,))


def test_assign_episode_splits_is_stable_and_contains_all_splits():
    first = assign_episode_splits(20, seed=42, fractions=(0.9, 0.05, 0.05))
    second = assign_episode_splits(20, seed=42, fractions=(0.9, 0.05, 0.05))
    np.testing.assert_array_equal(first, second)
    assert set(first.tolist()) == {0, 1, 2}


def test_validate_source_rejects_missing_dataset(synthetic_h5):
    with h5py.File(synthetic_h5, "a") as handle:
        del handle["rewards"]
    with h5py.File(synthetic_h5, "r") as handle:
        with pytest.raises(SourceValidationError, match="rewards"):
            validate_source(handle)
```

- [ ] **Step 2: Run the tests and verify they fail because the package does not exist**

Run: `pytest tests/test_conversion.py -v`

Expected: collection fails with `ModuleNotFoundError: No module named 'world_model'`.

- [ ] **Step 3: Implement schema validation, boundary construction, and deterministic splits**

```python
# world_model/__init__.py
"""Deterministic action-conditioned Mario world model."""

from .model import ActionConditionedUNet

__all__ = ["ActionConditionedUNet"]
```

Create `world_model/conversion.py` with these exact public contracts:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np


REQUIRED_DATASETS = ("observations", "next_obs", "actions", "rewards", "dones")


class SourceValidationError(ValueError):
    """The input HDF5 file cannot be converted safely."""


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
    missing = [name for name in REQUIRED_DATASETS if name not in handle]
    if missing:
        raise SourceValidationError(f"missing required datasets: {', '.join(missing)}")
    observations = handle["observations"]
    next_obs = handle["next_obs"]
    actions = handle["actions"]
    rewards = handle["rewards"]
    dones = handle["dones"]
    n = int(observations.shape[0])
    if observations.ndim != 4 or observations.shape[-1] != 3:
        raise SourceValidationError("observations must have shape (N, H, W, 3)")
    if next_obs.shape != observations.shape:
        raise SourceValidationError("next_obs shape must match observations")
    if observations.dtype != np.uint8 or next_obs.dtype != np.uint8:
        raise SourceValidationError("frame datasets must use uint8")
    for name, dataset in (("actions", actions), ("rewards", rewards), ("dones", dones)):
        if dataset.shape != (n,):
            raise SourceValidationError(f"{name} must have shape ({n},)")
    if not np.issubdtype(actions.dtype, np.integer):
        raise SourceValidationError("actions must use an integer dtype")
    if rewards.dtype != np.float32:
        raise SourceValidationError("rewards must use float32")
    if dones.dtype != np.bool_:
        raise SourceValidationError("dones must use bool")
    action_values = actions[:]
    if action_values.size == 0 or int(action_values.min()) < 0:
        raise SourceValidationError("actions must be non-empty non-negative integers")
    return SourceSchema(
        n_transitions=n,
        frame_shape=tuple(int(value) for value in observations.shape[1:]),
        n_actions=int(action_values.max()) + 1,
    )


def build_episode_offsets(dones: np.ndarray, break_indices: Sequence[int]) -> np.ndarray:
    dones = np.asarray(dones, dtype=bool)
    if dones.ndim != 1 or dones.size == 0:
        raise ValueError("dones must be a non-empty one-dimensional array")
    n = int(dones.size)
    breaks = {int(index) for index in break_indices}
    invalid = sorted(index for index in breaks if index <= 0 or index >= n)
    if invalid:
        raise ValueError(f"break index must satisfy 0 < index < {n}: {invalid}")
    starts = {0, *breaks}
    starts.update(int(index) + 1 for index in np.flatnonzero(dones) if int(index) + 1 < n)
    return np.asarray([*sorted(starts), n], dtype=np.int64)


def assign_episode_splits(
    n_episodes: int,
    seed: int,
    fractions: tuple[float, float, float],
) -> np.ndarray:
    if n_episodes < 3:
        raise ValueError("at least three trajectories are required for train/validation/test")
    values = np.asarray(fractions, dtype=np.float64)
    if values.shape != (3,) or np.any(values <= 0) or not np.isclose(values.sum(), 1.0):
        raise ValueError("split fractions must be three positive values summing to one")
    counts = np.floor(values * n_episodes).astype(np.int64)
    counts[counts == 0] = 1
    while counts.sum() > n_episodes:
        counts[int(np.argmax(counts))] -= 1
    while counts.sum() < n_episodes:
        deficits = values * n_episodes - counts
        counts[int(np.argmax(deficits))] += 1
    labels = np.repeat(np.arange(3, dtype=np.uint8), counts)
    return np.random.default_rng(seed).permutation(labels)
```

During this step, keep `world_model/__init__.py` from importing `.model` until Task 5 creates it; initially set `__all__: list[str] = []` so test collection succeeds.

- [ ] **Step 4: Run conversion contract tests**

Run: `pytest tests/test_conversion.py -v`

Expected: 5 tests pass.

- [ ] **Step 5: Commit the boundary contracts**

```bash
git add world_model/__init__.py world_model/conversion.py tests/conftest.py tests/test_conversion.py
git commit -m "feat: define world model conversion contracts"
```

---

### Task 2: Write and Validate the Memory-Mapped Cache

**Files:**
- Modify: `world_model/conversion.py`
- Modify: `tests/test_conversion.py`

**Interfaces:**
- Consumes: `ConversionConfig`, `validate_source()`, `build_episode_offsets()`, and `assign_episode_splits()`.
- Produces: `convert_dataset(config: ConversionConfig) -> Path`, `validate_cache(path: Path) -> dict`, and the eight cache artifacts defined in the design.

- [ ] **Step 1: Add failing end-to-end cache tests**

Append these tests to `tests/test_conversion.py`:

```python
import json

from world_model.conversion import ConversionConfig, convert_dataset, validate_cache


def test_convert_dataset_writes_unique_trajectory_frames(synthetic_h5, tmp_path):
    output = tmp_path / "cache"
    result = convert_dataset(
        ConversionConfig(
            input_path=synthetic_h5,
            output_dir=output,
            height=4,
            width=6,
            history=2,
            break_indices=(6,),
            workers=2,
        )
    )
    assert result == output
    frames = np.load(output / "frames.npy", mmap_mode="r")
    actions = np.load(output / "actions.npy", mmap_mode="r")
    offsets = np.load(output / "episode_offsets.npy")
    splits = np.load(output / "episode_splits.npy")
    np.testing.assert_array_equal(offsets, np.array([0, 4, 6, 10, 12], dtype=np.int64))
    assert frames.shape == (16, 4, 6, 3)
    assert actions.shape == (12,)
    assert splits.shape == (4,)
    assert int(frames[0, 0, 0, 0]) == 0
    assert int(frames[4, 0, 0, 0]) == 4
    assert int(frames[7, 0, 0, 0]) == 89
    assert int(frames[8, 0, 0, 0]) == 90


def test_convert_dataset_refuses_nonempty_output(synthetic_h5, tmp_path):
    output = tmp_path / "cache"
    output.mkdir()
    (output / "keep.txt").write_text("owned by user", encoding="utf-8")
    config = ConversionConfig(input_path=synthetic_h5, output_dir=output, height=4, width=6)
    with pytest.raises(FileExistsError, match="non-empty"):
        convert_dataset(config)
    assert (output / "keep.txt").read_text(encoding="utf-8") == "owned by user"


def test_validate_cache_checks_hashes_and_metadata(synthetic_h5, tmp_path):
    output = tmp_path / "cache"
    convert_dataset(
        ConversionConfig(
            input_path=synthetic_h5,
            output_dir=output,
            height=4,
            width=6,
            history=2,
            break_indices=(6,),
        )
    )
    metadata = validate_cache(output)
    assert metadata["n_transitions"] == 12
    assert metadata["n_trajectories"] == 4
    assert metadata["frame_shape"] == [4, 6, 3]
    assert set(metadata["sha256"]) == {
        "actions.npy",
        "episode_offsets.npy",
        "episode_splits.npy",
        "frames.npy",
        "rewards.npy",
        "source_transition_indices.npy",
    }
    card = (output / "README.md").read_text(encoding="utf-8")
    assert "action at transition t produces frame t+1" in card
```

- [ ] **Step 2: Run the new tests and verify missing conversion functions**

Run: `pytest tests/test_conversion.py -v`

Expected: collection fails because `convert_dataset` and `validate_cache` are not defined.

- [ ] **Step 3: Implement sequential conversion and artifact validation**

Add these private and public functions to `world_model/conversion.py`:

```python
import hashlib
import json
import os
import platform
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import cv2


ACTION_NAMES = ["NOOP", "right", "right+A", "right+B", "right+A+B", "A", "left"]
ARRAY_FILES = (
    "frames.npy",
    "actions.npy",
    "rewards.npy",
    "episode_offsets.npy",
    "episode_splits.npy",
    "source_transition_indices.npy",
)


def _resize_frame(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def _resize_batch(frames: np.ndarray, height: int, width: int, workers: int) -> np.ndarray:
    if workers <= 1:
        resized = [_resize_frame(frame, height, width) for frame in frames]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            resized = list(pool.map(lambda frame: _resize_frame(frame, height, width), frames))
    return np.stack(resized).astype(np.uint8, copy=False)


def _sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _dataset_card(metadata: dict) -> str:
    return (
        "---\n"
        "license: other\n"
        "task_categories:\n- video-prediction\n"
        "---\n\n"
        "# Mario 1-1 action-conditioned transitions\n\n"
        "This repository contains a NumPy memory-mapped cache derived from recorded "
        "Super Mario Bros. World 1-1 trajectories. The action at transition t produces "
        "frame t+1. Download the repository to local SSD before training.\n\n"
        f"- Transitions: {metadata['n_transitions']}\n"
        f"- Trajectories: {metadata['n_trajectories']}\n"
        f"- Frame shape: {metadata['frame_shape']}\n"
        f"- History: {metadata['history']}\n"
        f"- Explicit breaks: {metadata['break_indices']}\n\n"
        "The cache is not an official Nintendo dataset and contains gameplay imagery "
        "whose use remains subject to applicable rights and local law.\n"
    )


def _write_cache(config: ConversionConfig, temporary_dir: Path) -> dict:
    cv2.setNumThreads(1)
    with h5py.File(config.input_path, "r") as handle:
        schema = validate_source(handle)
        actions = handle["actions"][:].astype(np.uint8)
        rewards = handle["rewards"][:].astype(np.float32)
        dones = handle["dones"][:].astype(bool)
        offsets = build_episode_offsets(dones, config.break_indices)
        n_episodes = len(offsets) - 1
        splits = assign_episode_splits(n_episodes, config.split_seed, config.split_fractions)
        frame_count = schema.n_transitions + n_episodes
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
            np.arange(schema.n_transitions, dtype=np.int64),
        )
        observations = handle["observations"]
        next_obs = handle["next_obs"]
        source_chunk = int(observations.chunks[0]) if observations.chunks else 1024
        for block_start in range(0, schema.n_transitions, source_chunk):
            block_end = min(block_start + source_chunk, schema.n_transitions)
            source_frames = observations[block_start:block_end]
            resized = _resize_batch(source_frames, config.height, config.width, config.workers)
            first_episode = int(np.searchsorted(offsets[1:], block_start, side="right"))
            last_episode = int(np.searchsorted(offsets[1:], block_end - 1, side="right"))
            for episode in range(first_episode, last_episode + 1):
                start = max(block_start, int(offsets[episode]))
                end = min(block_end, int(offsets[episode + 1]))
                source_slice = slice(start - block_start, end - block_start)
                output_start = start + episode
                frames_out[output_start:output_start + (end - start)] = resized[source_slice]
            terminal_episodes = np.flatnonzero(
                (offsets[1:] - 1 >= block_start) & (offsets[1:] - 1 < block_end)
            )
            if terminal_episodes.size:
                terminal_indices = offsets[terminal_episodes + 1] - 1
                terminal_source = next_obs[terminal_indices.tolist()]
                terminal_frames = _resize_batch(
                    terminal_source, config.height, config.width, config.workers
                )
                for frame, episode, transition_end in zip(
                    terminal_frames, terminal_episodes, offsets[terminal_episodes + 1]
                ):
                    frames_out[int(transition_end) + int(episode)] = frame
        frames_out.flush()
    metadata = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_name": config.input_path.name,
        "source_size_bytes": config.input_path.stat().st_size,
        "n_transitions": schema.n_transitions,
        "n_trajectories": n_episodes,
        "n_frames": frame_count,
        "frame_shape": [config.height, config.width, 3],
        "history": config.history,
        "frame_skip": 4,
        "n_actions": schema.n_actions,
        "action_names": ACTION_NAMES[:schema.n_actions],
        "break_indices": sorted(int(index) for index in config.break_indices),
        "split_seed": config.split_seed,
        "split_fractions": list(config.split_fractions),
        "resize": "opencv.INTER_AREA",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "h5py_version": h5py.__version__,
        "opencv_version": cv2.__version__,
    }
    metadata["sha256"] = {name: _sha256(temporary_dir / name) for name in ARRAY_FILES}
    (temporary_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (temporary_dir / "README.md").write_text(_dataset_card(metadata), encoding="utf-8")
    return metadata


def validate_cache(path: Path) -> dict:
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    frames = np.load(path / "frames.npy", mmap_mode="r")
    actions = np.load(path / "actions.npy", mmap_mode="r")
    rewards = np.load(path / "rewards.npy", mmap_mode="r")
    offsets = np.load(path / "episode_offsets.npy")
    splits = np.load(path / "episode_splits.npy")
    source_indices = np.load(path / "source_transition_indices.npy", mmap_mode="r")
    expected_frames = metadata["n_transitions"] + metadata["n_trajectories"]
    if frames.shape != (expected_frames, *metadata["frame_shape"]):
        raise ValueError("frames.npy shape does not match metadata")
    if actions.shape != rewards.shape or actions.shape != source_indices.shape:
        raise ValueError("transition arrays have inconsistent shapes")
    if offsets[0] != 0 or offsets[-1] != metadata["n_transitions"]:
        raise ValueError("episode offsets do not cover all transitions")
    if len(splits) != metadata["n_trajectories"]:
        raise ValueError("episode splits do not match trajectory count")
    for name, expected in metadata["sha256"].items():
        if _sha256(path / name) != expected:
            raise ValueError(f"SHA-256 mismatch for {name}")
    return metadata


def convert_dataset(config: ConversionConfig) -> Path:
    output = config.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to replace non-empty output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent))
    _write_cache(config, temporary)
    validate_cache(temporary)
    if output.exists():
        output.rmdir()
    os.replace(temporary, output)
    return output
```

When implementing, replace h5py fancy indexing of unsorted terminal indices with sorted indices; the construction above already produces them in increasing order. Confirm that `frames_out[int(transition_end) + int(episode)]` is the final frame index for episode `episode`.

- [ ] **Step 4: Run cache tests**

Run: `pytest tests/test_conversion.py -v`

Expected: all conversion tests pass and no temporary cache remains after success.

- [ ] **Step 5: Commit the cache conversion**

```bash
git add world_model/conversion.py tests/test_conversion.py
git commit -m "feat: convert Mario trajectories to memmap cache"
```

---

### Task 3: Add the Conversion CLI and Hugging Face Upload

**Files:**
- Create: `prepare_world_model_data.py`
- Modify: `world_model/conversion.py`
- Create: `tests/test_huggingface.py`
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: `convert_dataset()` and a completed validated cache directory.
- Produces: `publish_cache(output_dir, repo_id, private, api=None) -> str` and the public conversion CLI.

- [ ] **Step 1: Add failing mocked Hub tests**

```python
# tests/test_huggingface.py
from unittest.mock import Mock

from world_model.conversion import publish_cache


def test_publish_cache_creates_dataset_repo_and_uploads_folder(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "metadata.json").write_text("{}", encoding="utf-8")
    api = Mock()
    api.create_repo.return_value = "https://huggingface.co/datasets/user/mario"
    api.upload_folder.return_value = Mock(repo_url="https://huggingface.co/datasets/user/mario")
    result = publish_cache(cache, "user/mario", private=True, api=api)
    api.create_repo.assert_called_once_with(
        repo_id="user/mario", repo_type="dataset", private=True, exist_ok=True
    )
    api.upload_folder.assert_called_once_with(
        folder_path=str(cache), repo_id="user/mario", repo_type="dataset"
    )
    assert result == "https://huggingface.co/datasets/user/mario"


def test_publish_cache_requires_valid_repo_id(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "metadata.json").write_text("{}", encoding="utf-8")
    try:
        publish_cache(cache, "missing-namespace", private=False, api=Mock())
    except ValueError as error:
        assert "namespace/name" in str(error)
    else:
        raise AssertionError("invalid repo id was accepted")
```

- [ ] **Step 2: Run the Hub tests and verify the missing publisher**

Run: `pytest tests/test_huggingface.py -v`

Expected: collection fails because `publish_cache` is not defined.

- [ ] **Step 3: Implement secure Hub publication**

Add to `world_model/conversion.py`:

```python
from typing import Any


def publish_cache(
    output_dir: Path,
    repo_id: str,
    private: bool,
    api: Any = None,
) -> str:
    if repo_id.count("/") != 1 or any(not part for part in repo_id.split("/")):
        raise ValueError("Hugging Face repo id must use namespace/name")
    if not (output_dir / "metadata.json").is_file():
        raise ValueError("cache must be validated before publication")
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
        folder_path=str(output_dir),
        repo_id=repo_id,
        repo_type="dataset",
    )
    return str(getattr(result, "repo_url", repo_url))
```

Do not add a token parameter. `HfApi()` must resolve authentication from the active Hugging Face login or `HF_TOKEN`.

- [ ] **Step 4: Implement the CLI with explicit, safe defaults**

```python
# prepare_world_model_data.py
from __future__ import annotations

import argparse
import os
from pathlib import Path

from world_model.conversion import ConversionConfig, convert_dataset, publish_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Mario HDF5 transitions into an H100-friendly NumPy cache"
    )
    parser.add_argument("input_h5", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--break-index", action="append", type=int, default=[])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--hf-repo", type=str)
    parser.add_argument("--hf-private", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ConversionConfig(
        input_path=args.input_h5,
        output_dir=args.output_dir,
        height=args.height,
        width=args.width,
        history=args.history,
        break_indices=tuple(args.break_index),
        split_seed=args.split_seed,
        workers=args.workers,
    )
    output = convert_dataset(config)
    print(f"Validated cache: {output}")
    if args.hf_repo:
        url = publish_cache(output, args.hf_repo, private=args.hf_private)
        print(f"Hugging Face dataset: {url}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Add dependencies and run tests/help**

Append to `requirements.txt`:

```text
huggingface_hub>=0.34.0
pytest>=8.0.0
```

Run: `pytest tests/test_conversion.py tests/test_huggingface.py -v`

Expected: all tests pass without making a network request.

Run: `python prepare_world_model_data.py --help`

Expected: help lists `--break-index`, `--workers`, `--hf-repo`, and `--hf-private`, with no token argument.

- [ ] **Step 6: Commit conversion CLI and publication support**

```bash
git add prepare_world_model_data.py world_model/conversion.py tests/test_huggingface.py requirements.txt
git commit -m "feat: publish prepared Mario dataset to Hugging Face"
```

---

### Task 4: Build the Boundary-Safe PyTorch Dataset

**Files:**
- Create: `world_model/data.py`
- Create: `tests/test_data.py`

**Interfaces:**
- Consumes: cache arrays and `metadata.json` from Task 2.
- Produces: `MarioWindowDataset(cache_dir, split, history)` returning `context`, `actions`, `target`, `transition_index`, and `episode_index`; `get_rollout(index, horizon)` for recorded-action recursion.

- [ ] **Step 1: Add failing alignment and boundary tests**

```python
# tests/test_data.py
import torch

from world_model.conversion import ConversionConfig, convert_dataset
from world_model.data import MarioWindowDataset


def _cache(synthetic_h5, tmp_path):
    output = tmp_path / "cache"
    convert_dataset(
        ConversionConfig(
            input_path=synthetic_h5,
            output_dir=output,
            height=4,
            width=6,
            history=2,
            break_indices=(6,),
            workers=1,
        )
    )
    return output


def test_dataset_returns_aligned_context_actions_and_target(synthetic_h5, tmp_path):
    dataset = MarioWindowDataset(_cache(synthetic_h5, tmp_path), split="all", history=2)
    sample = dataset[0]
    assert sample["context"].shape == (6, 4, 6)
    assert sample["actions"].shape == (2,)
    assert sample["target"].shape == (3, 4, 6)
    assert sample["context"].dtype == torch.float32
    assert torch.allclose(sample["context"][:3], torch.zeros(3, 4, 6))
    assert torch.allclose(sample["context"][3:], torch.full((3, 4, 6), 1 / 255))
    assert torch.allclose(sample["target"], torch.full((3, 4, 6), 2 / 255))
    torch.testing.assert_close(sample["actions"], torch.tensor([0, 1]))


def test_no_sample_crosses_episode_or_explicit_break(synthetic_h5, tmp_path):
    dataset = MarioWindowDataset(_cache(synthetic_h5, tmp_path), split="all", history=2)
    offsets = dataset.episode_offsets
    for transition, episode in zip(dataset.sample_transitions, dataset.sample_episodes):
        start, end = offsets[episode], offsets[episode + 1]
        assert transition - 1 >= start
        assert transition < end


def test_rollout_contains_recorded_future_actions_and_targets(synthetic_h5, tmp_path):
    dataset = MarioWindowDataset(_cache(synthetic_h5, tmp_path), split="all", history=2)
    rollout = dataset.get_rollout(0, horizon=2)
    assert rollout["initial_context"].shape == (2, 3, 4, 6)
    assert rollout["action_sequence"].shape == (3,)
    assert rollout["targets"].shape == (2, 3, 4, 6)
```

- [ ] **Step 2: Run data tests and verify the missing dataset**

Run: `pytest tests/test_data.py -v`

Expected: collection fails because `world_model.data` does not exist.

- [ ] **Step 3: Implement lazy memory maps and valid-window indexing**

Create `world_model/data.py` with this contract and logic:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset


SPLIT_IDS = {"train": 0, "validation": 1, "test": 2}


class MarioWindowDataset(Dataset):
    def __init__(
        self,
        cache_dir: str | Path,
        split: Literal["train", "validation", "test", "all"],
        history: int = 4,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.metadata = json.loads(
            (self.cache_dir / "metadata.json").read_text(encoding="utf-8")
        )
        self.history = int(history)
        if self.history < 1:
            raise ValueError("history must be positive")
        self.episode_offsets = np.load(self.cache_dir / "episode_offsets.npy")
        episode_splits = np.load(self.cache_dir / "episode_splits.npy")
        if split == "all":
            selected = np.arange(len(episode_splits), dtype=np.int64)
        else:
            selected = np.flatnonzero(episode_splits == SPLIT_IDS[split]).astype(np.int64)
        transitions: list[np.ndarray] = []
        episodes: list[np.ndarray] = []
        for episode in selected:
            start = int(self.episode_offsets[episode])
            end = int(self.episode_offsets[episode + 1])
            first = start + self.history - 1
            if first < end:
                values = np.arange(first, end, dtype=np.int64)
                transitions.append(values)
                episodes.append(np.full(len(values), episode, dtype=np.int64))
        self.sample_transitions = (
            np.concatenate(transitions) if transitions else np.empty(0, dtype=np.int64)
        )
        self.sample_episodes = (
            np.concatenate(episodes) if episodes else np.empty(0, dtype=np.int64)
        )
        self._frames = None
        self._actions = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_frames"] = None
        state["_actions"] = None
        return state

    def _open(self) -> None:
        if self._frames is None:
            self._frames = np.load(self.cache_dir / "frames.npy", mmap_mode="r")
            self._actions = np.load(self.cache_dir / "actions.npy", mmap_mode="r")

    def __len__(self) -> int:
        return len(self.sample_transitions)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        self._open()
        transition = int(self.sample_transitions[index])
        episode = int(self.sample_episodes[index])
        transition_start = int(self.episode_offsets[episode])
        local_t = transition - transition_start
        frame_start = transition_start + episode
        context_frames = np.array(
            self._frames[
                frame_start + local_t - self.history + 1:frame_start + local_t + 1
            ],
            copy=True,
        )
        target = np.array(self._frames[frame_start + local_t + 1], copy=True)
        actions = np.array(
            self._actions[transition - self.history + 1:transition + 1], copy=True
        )
        context = torch.from_numpy(context_frames).permute(0, 3, 1, 2).float().div_(255)
        return {
            "context": context.reshape(-1, context.shape[-2], context.shape[-1]),
            "actions": torch.from_numpy(actions.astype(np.int64, copy=False)),
            "target": torch.from_numpy(target).permute(2, 0, 1).float().div_(255),
            "transition_index": torch.tensor(transition, dtype=torch.int64),
            "episode_index": torch.tensor(episode, dtype=torch.int64),
        }

    def get_rollout(self, index: int, horizon: int) -> dict[str, torch.Tensor]:
        self._open()
        transition = int(self.sample_transitions[index])
        episode = int(self.sample_episodes[index])
        transition_start = int(self.episode_offsets[episode])
        transition_end = int(self.episode_offsets[episode + 1])
        if transition + horizon > transition_end:
            raise ValueError("rollout horizon crosses trajectory boundary")
        local_t = transition - transition_start
        frame_start = transition_start + episode
        initial = np.array(
            self._frames[
                frame_start + local_t - self.history + 1:frame_start + local_t + 1
            ],
            copy=True,
        )
        action_sequence = np.array(
            self._actions[
                transition - self.history + 1:transition + horizon
            ],
            copy=True,
        )
        targets = np.array(
            self._frames[
                frame_start + local_t + 1:frame_start + local_t + horizon + 1
            ],
            copy=True,
        )
        return {
            "initial_context": torch.from_numpy(initial).permute(0, 3, 1, 2).float().div_(255),
            "action_sequence": torch.from_numpy(action_sequence.astype(np.int64, copy=False)),
            "targets": torch.from_numpy(targets).permute(0, 3, 1, 2).float().div_(255),
        }
```

- [ ] **Step 4: Run dataset tests**

Run: `pytest tests/test_data.py -v`

Expected: all 3 tests pass.

- [ ] **Step 5: Commit the dataset loader**

```bash
git add world_model/data.py tests/test_data.py
git commit -m "feat: load boundary-safe Mario training windows"
```

---

### Task 5: Implement the FiLM-Conditioned Deterministic U-Net

**Files:**
- Create: `world_model/model.py`
- Modify: `world_model/__init__.py`
- Create: `tests/test_model.py`

**Interfaces:**
- Consumes: context `(B, history*3, H, W)` and actions `(B, history)`.
- Produces: predicted RGB `(B, 3, H, W)` in `[0, 1]` through `ActionConditionedUNet.forward()`.

- [ ] **Step 1: Add failing model behavior tests**

```python
# tests/test_model.py
import torch

from world_model.model import ActionConditionedUNet


def tiny_model():
    torch.manual_seed(0)
    return ActionConditionedUNet(
        history=4,
        n_actions=7,
        base_channels=8,
        channel_multipliers=(1, 2),
        blocks_per_level=1,
        action_embed_dim=8,
        cond_dim=16,
    )


def test_model_output_shape_and_range():
    model = tiny_model().eval()
    context = torch.rand(2, 12, 16, 24)
    actions = torch.randint(0, 7, (2, 4))
    with torch.no_grad():
        output = model(context, actions)
    assert output.shape == (2, 3, 16, 24)
    assert torch.all((0 <= output) & (output <= 1))


def test_model_is_deterministic_in_eval_mode():
    model = tiny_model().eval()
    context = torch.rand(1, 12, 16, 24)
    actions = torch.tensor([[0, 1, 2, 3]])
    with torch.no_grad():
        first = model(context, actions)
        second = model(context, actions)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_actions_reach_prediction_path():
    model = tiny_model().eval()
    context = torch.rand(1, 12, 16, 24)
    first_actions = torch.tensor([[0, 0, 0, 0]])
    second_actions = torch.tensor([[0, 0, 0, 4]])
    with torch.no_grad():
        first = model(context, first_actions)
        second = model(context, second_actions)
    assert not torch.equal(first, second)


def test_model_rejects_wrong_history_shape():
    model = tiny_model()
    try:
        model(torch.rand(1, 9, 16, 24), torch.zeros(1, 4, dtype=torch.long))
    except ValueError as error:
        assert "12 context channels" in str(error)
    else:
        raise AssertionError("invalid context channels were accepted")
```

- [ ] **Step 2: Run model tests and verify the missing module**

Run: `pytest tests/test_model.py -v`

Expected: collection fails because `world_model.model` does not exist.

- [ ] **Step 3: Implement action conditioning and residual blocks**

Create `world_model/model.py` with:

```python
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def _groups(channels: int) -> int:
    for value in (32, 16, 8, 4, 2, 1):
        if channels % value == 0:
            return value
    return 1


class ActionConditioner(nn.Module):
    def __init__(
        self,
        n_actions: int,
        history: int,
        embed_dim: int,
        cond_dim: int,
    ) -> None:
        super().__init__()
        self.history = history
        self.action_embedding = nn.Embedding(n_actions, embed_dim)
        self.position_embedding = nn.Embedding(history, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(history * embed_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim != 2 or actions.shape[1] != self.history:
            raise ValueError(f"actions must have shape (B, {self.history})")
        positions = torch.arange(self.history, device=actions.device)
        embedded = self.action_embedding(actions) + self.position_embedding(positions)[None]
        return self.mlp(embedded.flatten(1))


class FiLMResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.film = nn.Linear(cond_dim, 2 * out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        hidden = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.film(condition).chunk(2, dim=1)
        hidden = self.norm2(hidden)
        hidden = hidden * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        hidden = self.conv2(F.silu(hidden))
        return residual + hidden


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))
```

- [ ] **Step 4: Implement the U-Net encoder, bottleneck, decoder, and validation**

Append to `world_model/model.py`:

```python
class ActionConditionedUNet(nn.Module):
    def __init__(
        self,
        history: int = 4,
        n_actions: int = 7,
        base_channels: int = 64,
        channel_multipliers: Sequence[int] = (1, 2, 3, 4),
        blocks_per_level: int = 2,
        action_embed_dim: int = 64,
        cond_dim: int = 256,
    ) -> None:
        super().__init__()
        self.history = history
        self.n_actions = n_actions
        channels = [base_channels * int(value) for value in channel_multipliers]
        self.conditioner = ActionConditioner(
            n_actions, history, action_embed_dim, cond_dim
        )
        self.input_conv = nn.Conv2d(history * 3, channels[0], 3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        current = channels[0]
        for level, level_channels in enumerate(channels):
            blocks = nn.ModuleList()
            for block_index in range(blocks_per_level):
                in_channels = current if block_index == 0 else level_channels
                blocks.append(FiLMResBlock(in_channels, level_channels, cond_dim))
                current = level_channels
            self.down_blocks.append(blocks)
            if level < len(channels) - 1:
                self.downsamples.append(Downsample(current, channels[level + 1]))
                current = channels[level + 1]
        self.mid_blocks = nn.ModuleList(
            [FiLMResBlock(current, current, cond_dim), FiLMResBlock(current, current, cond_dim)]
        )
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for level in reversed(range(len(channels))):
            level_channels = channels[level]
            blocks = nn.ModuleList(
                [FiLMResBlock(current + level_channels, level_channels, cond_dim)]
            )
            blocks.extend(
                FiLMResBlock(level_channels, level_channels, cond_dim)
                for _ in range(blocks_per_level - 1)
            )
            self.up_blocks.append(blocks)
            current = level_channels
            if level > 0:
                self.upsamples.append(Upsample(current, channels[level - 1]))
                current = channels[level - 1]
        self.output_norm = nn.GroupNorm(_groups(current), current)
        self.output_conv = nn.Conv2d(current, 3, 3, padding=1)

    def forward(self, context: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        expected_channels = self.history * 3
        if context.ndim != 4 or context.shape[1] != expected_channels:
            raise ValueError(f"expected {expected_channels} context channels")
        factor = 2 ** (len(self.down_blocks) - 1)
        if context.shape[-2] % factor or context.shape[-1] % factor:
            raise ValueError(f"height and width must be divisible by {factor}")
        condition = self.conditioner(actions)
        hidden = self.input_conv(context)
        skips = []
        for level, blocks in enumerate(self.down_blocks):
            for block in blocks:
                hidden = block(hidden, condition)
            skips.append(hidden)
            if level < len(self.downsamples):
                hidden = self.downsamples[level](hidden)
        for block in self.mid_blocks:
            hidden = block(hidden, condition)
        for decoder_index, blocks in enumerate(self.up_blocks):
            hidden = torch.cat([hidden, skips.pop()], dim=1)
            for block in blocks:
                hidden = block(hidden, condition)
            if decoder_index < len(self.upsamples):
                hidden = self.upsamples[decoder_index](hidden)
        return torch.sigmoid(self.output_conv(F.silu(self.output_norm(hidden))))
```

Update `world_model/__init__.py` to export `ActionConditionedUNet` exactly as shown in Task 1.

- [ ] **Step 5: Run model tests and inspect parameter count**

Run: `pytest tests/test_model.py -v`

Expected: all 4 tests pass.

Run: `.venv/bin/python -c 'from world_model.model import ActionConditionedUNet; m=ActionConditionedUNet(); print(sum(p.numel() for p in m.parameters()))'`

Expected: a finite parameter count in the intended compact range; record the exact count in `README.md` during Task 8.

- [ ] **Step 6: Commit the model**

```bash
git add world_model/model.py world_model/__init__.py tests/test_model.py
git commit -m "feat: add action-conditioned deterministic U-Net"
```

---

### Task 6: Add Validated Configuration and Exact Checkpoint Resume

**Files:**
- Create: `world_model/config.py`
- Create: `world_model/checkpointing.py`
- Create: `configs/deterministic_unet.yaml`
- Create: `tests/test_checkpointing.py`

**Interfaces:**
- Consumes: YAML configuration and live PyTorch training state.
- Produces: `ExperimentConfig.from_yaml()`, `save_checkpoint()`, `load_checkpoint()`, `capture_rng_state()`, and `restore_rng_state()`.

- [ ] **Step 1: Write failing checkpoint round-trip test**

```python
# tests/test_checkpointing.py
import random

import numpy as np
import torch

from world_model.checkpointing import load_checkpoint, save_checkpoint


def test_checkpoint_restores_model_optimizer_step_and_rng(tmp_path):
    torch.manual_seed(7)
    np.random.seed(7)
    random.seed(7)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    prediction = model(torch.ones(1, 3)).sum()
    prediction.backward()
    optimizer.step()
    checkpoint = tmp_path / "latest.pt"
    save_checkpoint(
        checkpoint,
        model=model,
        ema_state={name: value.detach().clone() for name, value in model.state_dict().items()},
        optimizer=optimizer,
        scheduler=None,
        epoch=3,
        global_step=17,
        config={"seed": 7},
    )
    expected_torch = torch.rand(1)
    expected_numpy = np.random.rand()
    expected_python = random.random()
    restored = torch.nn.Linear(3, 2)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    state = load_checkpoint(
        checkpoint,
        model=restored,
        optimizer=restored_optimizer,
        scheduler=None,
        restore_rng=True,
    )
    assert state["epoch"] == 3
    assert state["global_step"] == 17
    assert torch.equal(torch.rand(1), expected_torch)
    assert np.random.rand() == expected_numpy
    assert random.random() == expected_python
    for first, second in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(first, second)
```

- [ ] **Step 2: Run checkpoint tests and verify missing module**

Run: `pytest tests/test_checkpointing.py -v`

Expected: collection fails because `world_model.checkpointing` does not exist.

- [ ] **Step 3: Implement atomic checkpoints and RNG restoration**

Create `world_model/checkpointing.py` with:

```python
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def capture_rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    ema_state: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    global_step: int,
    config: dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema_state,
            "optimizer": optimizer.state_dict(),
            "scheduler": None if scheduler is None else scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "config": config,
            "rng": capture_rng_state(),
        },
        temporary,
    )
    os.replace(temporary, path)


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    restore_rng: bool,
) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state["scheduler"] is not None:
        scheduler.load_state_dict(state["scheduler"])
    if restore_rng:
        restore_rng_state(state["rng"])
    return state
```

- [ ] **Step 4: Implement validated YAML configuration**

Create `world_model/config.py` with these frozen dataclasses and loader:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DataConfig:
    cache_dir: str
    history: int
    batch_size: int
    validation_batch_size: int
    workers: int
    prefetch_factor: int


@dataclass(frozen=True)
class ModelConfig:
    n_actions: int
    base_channels: int
    channel_multipliers: tuple[int, ...]
    blocks_per_level: int
    action_embed_dim: int
    condition_dim: int


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float
    warmup_steps: int
    gradient_clip: float


@dataclass(frozen=True)
class RuntimeConfig:
    output_dir: str
    seed: int
    epochs: int
    device: str
    compile: bool
    use_bfloat16: bool
    use_tf32: bool
    channels_last: bool
    ema_decay: float
    validation_every_epochs: int
    rollout_horizon: int


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    runtime: RuntimeConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        required = {"data", "model", "optimizer", "runtime"}
        if not isinstance(raw, dict) or set(raw) != required:
            raise ValueError(f"configuration sections must be exactly {sorted(required)}")
        model_values = dict(raw["model"])
        model_values["channel_multipliers"] = tuple(model_values["channel_multipliers"])
        config = cls(
            data=DataConfig(**raw["data"]),
            model=ModelConfig(**model_values),
            optimizer=OptimizerConfig(**raw["optimizer"]),
            runtime=RuntimeConfig(**raw["runtime"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.data.history < 1 or self.data.batch_size < 1:
            raise ValueError("history and batch size must be positive")
        if self.data.validation_batch_size < 1 or self.data.workers < 0:
            raise ValueError("validation batch size must be positive and workers non-negative")
        if self.data.prefetch_factor < 1:
            raise ValueError("prefetch factor must be positive")
        if self.runtime.epochs < 1 or self.runtime.validation_every_epochs < 1:
            raise ValueError("epoch counts must be positive")
        if self.runtime.rollout_horizon < 1 or not 0 < self.runtime.ema_decay < 1:
            raise ValueError("rollout horizon and EMA decay are invalid")
        if self.optimizer.learning_rate <= 0 or self.optimizer.warmup_steps < 0:
            raise ValueError("learning rate must be positive and warmup non-negative")
        if self.optimizer.gradient_clip <= 0:
            raise ValueError("gradient clip must be positive")
        if len(self.model.channel_multipliers) < 2:
            raise ValueError("at least two channel multipliers are required")
        if any(value < 1 for value in self.model.channel_multipliers):
            raise ValueError("channel multipliers must be positive")
        if self.model.n_actions < 1 or self.model.base_channels < 1:
            raise ValueError("model action and channel counts must be positive")

    def to_dict(self) -> dict:
        return asdict(self)
```

Create `configs/deterministic_unet.yaml`:

```yaml
data:
  cache_dir: dataset/mario-1-1-120x128
  history: 4
  batch_size: 64
  validation_batch_size: 64
  workers: 16
  prefetch_factor: 4
model:
  n_actions: 7
  base_channels: 64
  channel_multipliers: [1, 2, 3, 4]
  blocks_per_level: 2
  action_embed_dim: 64
  condition_dim: 256
optimizer:
  learning_rate: 0.0002
  weight_decay: 0.0001
  beta1: 0.9
  beta2: 0.999
  warmup_steps: 2000
  gradient_clip: 1.0
runtime:
  output_dir: runs/deterministic-unet
  seed: 42
  epochs: 30
  device: cuda
  compile: true
  use_bfloat16: true
  use_tf32: true
  channels_last: true
  ema_decay: 0.999
  validation_every_epochs: 1
  rollout_horizon: 16
```

- [ ] **Step 5: Run checkpoint tests and config parse smoke test**

Run: `pytest tests/test_checkpointing.py -v`

Expected: test passes.

Run: `.venv/bin/python -c 'from world_model.config import ExperimentConfig; print(ExperimentConfig.from_yaml("configs/deterministic_unet.yaml").runtime.device)'`

Expected output: `cuda`.

- [ ] **Step 6: Commit configuration and checkpoint support**

```bash
git add world_model/config.py world_model/checkpointing.py configs/deterministic_unet.yaml tests/test_checkpointing.py
git commit -m "feat: add world model configuration and checkpoints"
```

---

### Task 7: Implement the H100 Training Engine and CLI

**Files:**
- Create: `world_model/training.py`
- Create: `train_world_model.py`
- Create: `tests/test_training.py`

**Interfaces:**
- Consumes: `ExperimentConfig`, `MarioWindowDataset`, and `ActionConditionedUNet`.
- Produces: `EMA`, `warmup_cosine_factor()`, `train_one_epoch()`, `evaluate()`, `autoregressive_rollout()`, prediction grids, best/latest checkpoints, and the training CLI.

- [ ] **Step 1: Write failing CPU training and rollout tests**

```python
# tests/test_training.py
import torch
from torch.utils.data import DataLoader, Dataset

from world_model.model import ActionConditionedUNet
from world_model.training import EMA, autoregressive_rollout, evaluate, train_one_epoch


class TinyDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(index)
        return {
            "context": torch.rand(12, 16, 24, generator=generator),
            "actions": torch.tensor([0, 1, 2, index % 7]),
            "target": torch.rand(3, 16, 24, generator=generator),
            "transition_index": torch.tensor(index),
            "episode_index": torch.tensor(0),
        }


def _model():
    return ActionConditionedUNet(
        base_channels=8,
        channel_multipliers=(1, 2),
        blocks_per_level=1,
        action_embed_dim=8,
        cond_dim=16,
    )


def test_one_cpu_epoch_returns_finite_loss_and_updates_ema():
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loader = DataLoader(TinyDataset(), batch_size=2)
    ema = EMA(model, decay=0.9)
    before = {name: value.clone() for name, value in ema.state_dict().items()}
    metrics, steps = train_one_epoch(
        model=model,
        loader=loader,
        optimizer=optimizer,
        scheduler=None,
        ema=ema,
        device=torch.device("cpu"),
        use_bfloat16=False,
        channels_last=False,
        gradient_clip=1.0,
        global_step=0,
    )
    assert torch.isfinite(torch.tensor(metrics["l1"]))
    assert steps == 2
    assert any(not torch.equal(before[name], value) for name, value in ema.state_dict().items())


def test_evaluate_reports_per_action_metrics():
    metrics = evaluate(
        model=_model().eval(),
        loader=DataLoader(TinyDataset(), batch_size=2),
        device=torch.device("cpu"),
        use_bfloat16=False,
        channels_last=False,
        n_actions=7,
    )
    assert metrics["l1"] >= 0
    assert metrics["mse"] >= 0
    assert set(metrics["per_action_l1"]) == set(range(7))


def test_autoregressive_rollout_shape():
    model = _model().eval()
    initial = torch.rand(1, 4, 3, 16, 24)
    actions = torch.tensor([[0, 1, 2, 3, 4, 5]])
    with torch.no_grad():
        predictions = autoregressive_rollout(model, initial, actions, horizon=3)
    assert predictions.shape == (1, 3, 3, 16, 24)
```

- [ ] **Step 2: Run training tests and verify the missing engine**

Run: `pytest tests/test_training.py -v`

Expected: collection fails because `world_model.training` does not exist.

- [ ] **Step 3: Implement EMA, schedule, one-epoch training, and evaluation**

Create `world_model/training.py` with:

```python
from __future__ import annotations

import math
from contextlib import nullcontext

import torch
from torch.nn import functional as F


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, value in model.state_dict().items():
            if value.is_floating_point():
                self.shadow[name].lerp_(value.detach(), 1 - self.decay)
            else:
                self.shadow[name].copy_(value)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: value.clone() for name, value in self.shadow.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.shadow = {name: value.clone() for name, value in state.items()}


def warmup_cosine_factor(step: int, warmup_steps: int, total_steps: int) -> float:
    if warmup_steps and step < warmup_steps:
        return max(1, step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1 + math.cos(math.pi * progress))


def _autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def _move(batch: dict, device: torch.device, channels_last: bool):
    context = batch["context"].to(device, non_blocking=True)
    actions = batch["actions"].to(device, non_blocking=True)
    target = batch["target"].to(device, non_blocking=True)
    if channels_last:
        context = context.contiguous(memory_format=torch.channels_last)
        target = target.contiguous(memory_format=torch.channels_last)
    return context, actions, target


def train_one_epoch(
    *, model, loader, optimizer, scheduler, ema, device, use_bfloat16,
    channels_last, gradient_clip, global_step,
):
    model.train()
    total_loss = 0.0
    total_items = 0
    for batch in loader:
        context, actions, target = _move(batch, device, channels_last)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, use_bfloat16):
            prediction = model(context, actions)
            loss = F.l1_loss(prediction, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        ema.update(model)
        batch_size = context.shape[0]
        total_loss += float(loss.detach()) * batch_size
        total_items += batch_size
        global_step += 1
    return {"l1": total_loss / max(1, total_items)}, global_step


@torch.inference_mode()
def evaluate(*, model, loader, device, use_bfloat16, channels_last, n_actions):
    model.eval()
    l1_sum = mse_sum = 0.0
    pixels = 0
    action_sums = torch.zeros(n_actions, dtype=torch.float64)
    action_counts = torch.zeros(n_actions, dtype=torch.int64)
    for batch in loader:
        context, actions, target = _move(batch, device, channels_last)
        with _autocast(device, use_bfloat16):
            prediction = model(context, actions)
        absolute = (prediction.float() - target.float()).abs()
        squared = (prediction.float() - target.float()).square()
        l1_sum += float(absolute.sum())
        mse_sum += float(squared.sum())
        pixels += absolute.numel()
        sample_l1 = absolute.flatten(1).mean(1).cpu()
        current_actions = actions[:, -1].cpu()
        for action in range(n_actions):
            mask = current_actions == action
            action_sums[action] += sample_l1[mask].sum().double()
            action_counts[action] += mask.sum()
    per_action = {
        action: float(action_sums[action] / action_counts[action])
        if action_counts[action]
        else float("nan")
        for action in range(n_actions)
    }
    return {"l1": l1_sum / pixels, "mse": mse_sum / pixels, "per_action_l1": per_action}


@torch.inference_mode()
def autoregressive_rollout(model, initial_context, action_sequence, horizon):
    context_frames = list(initial_context.unbind(dim=1))
    history = len(context_frames)
    predictions = []
    for step in range(horizon):
        action_history = action_sequence[:, step:step + history]
        stacked = torch.cat(context_frames[-history:], dim=1)
        prediction = model(stacked, action_history)
        predictions.append(prediction)
        context_frames.append(prediction)
    return torch.stack(predictions, dim=1)
```

- [ ] **Step 4: Implement prediction grids and the training CLI orchestration**

Add `save_prediction_grid()` and `save_rollout_grid()` to `world_model/training.py` using `torchvision.utils.make_grid` and `torchvision.utils.save_image`. The one-step grid rows must contain `last context | target | prediction | absolute error`. The rollout grid rows must contain `ground truth | prediction | absolute error` for each horizon step.

Create `train_world_model.py` with these exact orchestration stages:

```python
def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    set_seed(config.runtime.seed)
    configure_torch(config.runtime)
    train_dataset = MarioWindowDataset(config.data.cache_dir, "train", config.data.history)
    validation_dataset = MarioWindowDataset(
        config.data.cache_dir, "validation", config.data.history
    )
    if args.overfit_batches:
        limit = min(len(train_dataset), args.overfit_batches * config.data.batch_size)
        fixed_indices = list(range(limit))
        train_dataset = torch.utils.data.Subset(train_dataset, fixed_indices)
        validation_dataset = train_dataset
    train_loader = build_loader(train_dataset, config.data, training=True)
    validation_loader = build_loader(validation_dataset, config.data, training=False)
    raw_model = build_model(config).to(args.device or config.runtime.device)
    if config.runtime.channels_last:
        raw_model.to(memory_format=torch.channels_last)
    optimizer = build_optimizer(raw_model, config.optimizer)
    total_steps = config.runtime.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: warmup_cosine_factor(
            step, config.optimizer.warmup_steps, total_steps
        ),
    )
    ema = EMA(raw_model, config.runtime.ema_decay)
    start_epoch = global_step = 0
    if args.resume:
        state = load_checkpoint(
            args.resume,
            model=raw_model,
            optimizer=optimizer,
            scheduler=scheduler,
            restore_rng=True,
        )
        ema.load_state_dict(state["ema"])
        start_epoch = state["epoch"] + 1
        global_step = state["global_step"]
    train_model = torch.compile(raw_model) if config.runtime.compile and not args.no_compile else raw_model
    run_training_loop(
        raw_model=raw_model,
        train_model=train_model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        config=config,
        start_epoch=start_epoch,
        global_step=global_step,
    )
```

Implement the named helpers in the same file without global mutable state:

- `parse_args()` accepts `--config`, `--resume`, `--overfit-batches`, `--device`, and `--no-compile`.
- `set_seed()` seeds Python, NumPy, CPU torch, and all CUDA devices.
- `configure_torch()` enables TF32 only when requested and available.
- `build_loader()` sets `pin_memory=True` for CUDA, applies persistent workers only when workers are positive, and omits `prefetch_factor` when workers are zero.
- `build_model()` maps every `ModelConfig` field to `ActionConditionedUNet`.
- `build_optimizer()` uses fused AdamW only when CUDA supports the `fused` keyword.
- `run_training_loop()` logs train/validation metrics and learning rate, evaluates with EMA weights loaded temporarily, saves prediction and rollout grids, writes `latest.pt` every epoch, and replaces `best.pt` only when validation L1 improves.
- Always save checkpoints from `raw_model`, not the compiled wrapper.

- [ ] **Step 5: Run CPU training tests**

Run: `pytest tests/test_training.py -v`

Expected: all 3 tests pass with finite metrics.

- [ ] **Step 6: Run the complete unit suite**

Run: `pytest -v`

Expected: every conversion, Hub, data, model, checkpoint, and training test passes.

- [ ] **Step 7: Commit the training engine**

```bash
git add world_model/training.py train_world_model.py tests/test_training.py
git commit -m "feat: train deterministic Mario model on H100"
```

---

### Task 8: Document and Verify the End-to-End Workflow

**Files:**
- Modify: `README.md`
- Modify: `.gitignore`
- Modify: `inspect_dataset.py`

**Interfaces:**
- Consumes: all implemented CLIs and the real `mario_1-1_live.h5` file.
- Produces: reproducible operator commands, ignored generated artifacts, and a read-only real-data inspection report.

- [ ] **Step 1: Update generated-artifact ignores without hiding source code or configuration**

Append to `.gitignore`:

```text
runs/
*.partial-*/
*.pt
```

Keep the existing `dataset/` and `checkpoints/` ignores. Do not ignore `configs/`, `tests/`, or `docs/`.

- [ ] **Step 2: Extend the inspector with read-only storage and boundary reporting**

Add a `--stats-only` flag to `inspect_dataset.py`. In stats-only mode it must not create PNG or GIF files. Add output for dataset chunks, compression, per-dataset storage bytes, complete episode length min/median/mean/p95/max, trailing partial length, and explicit continuity checks supplied through repeatable `--break-index` arguments. Ensure trajectory GIF selection samples a valid episode window and never crosses a done or explicit break.

Add a regression test to `tests/test_conversion.py` that invokes the inspector's statistics function on `synthetic_h5` and confirms it returns without creating image files.

- [ ] **Step 3: Document exact conversion, Hub, download, overfit, train, and resume commands**

Add these sections to `README.md`:

```markdown
## Prepare world-model data

```bash
python prepare_world_model_data.py \
  mario_1-1_live.h5 \
  dataset/mario-1-1-120x128 \
  --break-index 10000 \
  --workers 32
```

The source file remains unchanged. The output is an approximately 11.6 GiB
uncompressed NumPy cache designed to be copied to local SSD before training.

## Publish to Hugging Face Hub

```bash
export HF_TOKEN=<write-token>
python prepare_world_model_data.py \
  mario_1-1_live.h5 \
  dataset/mario-1-1-120x128 \
  --break-index 10000 \
  --workers 32 \
  --hf-repo <namespace>/mario-1-1-world-model \
  --hf-private
```

Download on the H100 instance:

```bash
hf download <namespace>/mario-1-1-world-model \
  --repo-type dataset \
  --local-dir dataset/mario-1-1-120x128
```

## Validate with a tiny overfit

```bash
python train_world_model.py \
  --config configs/deterministic_unet.yaml \
  --overfit-batches 4
```

Inspect the TensorBoard prediction grid before starting a full run.

## Train on one H100

```bash
python train_world_model.py --config configs/deterministic_unet.yaml
```

Resume exactly from the latest checkpoint:

```bash
python train_world_model.py \
  --config configs/deterministic_unet.yaml \
  --resume runs/deterministic-unet/latest.pt
```
```

Also document the exact default model parameter count measured in Task 5 and explain that deterministic inference remains interactive because actions condition every predicted next frame.

- [ ] **Step 4: Run static and unit verification**

Run: `python -m compileall world_model prepare_world_model_data.py train_world_model.py inspect_dataset.py`

Expected: every source file compiles without syntax errors.

Run: `pytest -v`

Expected: all tests pass.

Run: `git diff --check`

Expected: no output.

- [ ] **Step 5: Run real-data read-only validation and CLI checks**

Run: `python inspect_dataset.py mario_1-1_live.h5 --stats-only --break-index 10000`

Expected: reports 267,536 transitions, 240×256 RGB frames, seven actions, 1,535 done flags, one trailing partial trajectory, and the explicit new trajectory at 10,000 without writing images.

Run: `python prepare_world_model_data.py --help`

Expected: safe conversion and Hub arguments are displayed.

Run: `python train_world_model.py --help`

Expected: config, resume, overfit, device, and compile controls are displayed.

- [ ] **Step 6: Commit documentation and integration checks**

```bash
git add README.md .gitignore inspect_dataset.py tests/test_conversion.py
git commit -m "docs: add deterministic world model workflow"
```

---

### Task 9: Perform a Small Real Conversion Before the Full 11.6 GiB Cache

**Files:**
- Modify only if verification exposes a defect: conversion, data, or documentation files from Tasks 1–8.

**Interfaces:**
- Consumes: the real HDF5 source and completed conversion CLI.
- Produces: evidence that the converter works on real source chunks without committing generated data.

- [ ] **Step 1: Add a conversion limit intended only for smoke testing**

Add `limit_transitions: int | None = None` to `ConversionConfig` and `--limit-transitions` to the CLI. When set, validate that the value is at least the history length, slice all one-dimensional source arrays to the limit, discard break indices at or beyond the limit with a printed explanation, and construct a final partial trajectory at the limit. Record the limit in metadata. The default remains `None`, which converts the complete file.

Add a synthetic test proving `--limit-transitions 8` writes eight actions and offsets ending at eight.

- [ ] **Step 2: Run a 4,096-transition real-data conversion into a temporary directory**

Run:

```bash
python prepare_world_model_data.py \
  mario_1-1_live.h5 \
  /tmp/mario-world-model-smoke \
  --limit-transitions 4096 \
  --workers 8
```

Expected: validated cache contains 4,096 transitions and can be opened by `MarioWindowDataset`.

- [ ] **Step 3: Benchmark cache window reads**

Run:

```bash
.venv/bin/python -c 'import time; from world_model.data import MarioWindowDataset; d=MarioWindowDataset("/tmp/mario-world-model-smoke", "all", 4); t=time.perf_counter(); [d[i] for i in range(min(1000,len(d)))]; dt=time.perf_counter()-t; print(f"samples_per_second={min(1000,len(d))/dt:.1f}")'
```

Expected: reports a finite positive sample rate and completes without HDF5 access.

- [ ] **Step 4: Run a CPU model forward pass on a real converted sample**

Run:

```bash
.venv/bin/python -c 'from world_model.data import MarioWindowDataset; from world_model.model import ActionConditionedUNet; d=MarioWindowDataset("/tmp/mario-world-model-smoke", "all", 4); s=d[0]; m=ActionConditionedUNet(base_channels=8,channel_multipliers=(1,2),blocks_per_level=1,action_embed_dim=8,cond_dim=16); y=m(s["context"].unsqueeze(0),s["actions"].unsqueeze(0)); print(tuple(y.shape),float(y.min()),float(y.max()))'`
```

Expected: shape `(1, 3, 120, 128)` and values between zero and one.

- [ ] **Step 5: Re-run the full verification suite after any smoke-test corrections**

Run: `pytest -v && python -m compileall world_model prepare_world_model_data.py train_world_model.py inspect_dataset.py && git diff --check`

Expected: tests and compilation pass; diff check is silent.

- [ ] **Step 6: Commit only if the smoke test required code corrections**

```bash
git add world_model prepare_world_model_data.py train_world_model.py inspect_dataset.py tests README.md requirements.txt configs .gitignore
git commit -m "fix: harden real Mario data conversion"
```

Do not commit `/tmp/mario-world-model-smoke`, the original HDF5 file, the full converted cache, checkpoints, or TensorBoard runs.
