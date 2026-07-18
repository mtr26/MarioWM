import numpy as np
import pytest
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
    dataset = MarioWindowDataset(
        _cache(synthetic_h5, tmp_path), split="all", history=2
    )

    sample = dataset[0]

    assert sample["context"].shape == (6, 4, 6)
    assert sample["actions"].shape == (2,)
    assert sample["target"].shape == (3, 4, 6)
    assert sample["context"].dtype == torch.float32
    assert torch.allclose(sample["context"][:3], torch.zeros(3, 4, 6))
    assert torch.allclose(
        sample["context"][3:], torch.full((3, 4, 6), 1 / 255)
    )
    assert torch.allclose(sample["target"], torch.full((3, 4, 6), 2 / 255))
    torch.testing.assert_close(sample["actions"], torch.tensor([0, 1]))


def test_no_sample_crosses_episode_or_explicit_break(synthetic_h5, tmp_path):
    dataset = MarioWindowDataset(
        _cache(synthetic_h5, tmp_path), split="all", history=2
    )

    offsets = dataset.episode_offsets
    for transition, episode in zip(
        dataset.sample_transitions, dataset.sample_episodes
    ):
        start, end = offsets[episode], offsets[episode + 1]
        assert transition - 1 >= start
        assert transition < end


def test_rollout_contains_recorded_future_actions_and_targets(
    synthetic_h5, tmp_path
):
    dataset = MarioWindowDataset(
        _cache(synthetic_h5, tmp_path), split="all", history=2
    )

    rollout = dataset.get_rollout(0, horizon=2)

    assert rollout["initial_context"].shape == (2, 3, 4, 6)
    assert rollout["action_sequence"].shape == (3,)
    assert rollout["targets"].shape == (2, 3, 4, 6)
    torch.testing.assert_close(rollout["action_sequence"], torch.tensor([0, 1, 2]))


def test_rollout_rejects_boundary_crossing(synthetic_h5, tmp_path):
    dataset = MarioWindowDataset(
        _cache(synthetic_h5, tmp_path), split="all", history=2
    )

    with pytest.raises(ValueError, match="trajectory boundary"):
        dataset.get_rollout(0, horizon=4)


def test_split_selection_contains_only_matching_episodes(synthetic_h5, tmp_path):
    cache = _cache(synthetic_h5, tmp_path)
    train = MarioWindowDataset(cache, split="train", history=2)
    split_labels = torch.from_numpy(np.load(cache / "episode_splits.npy"))

    assert len(train) > 0
    for episode in train.sample_episodes:
        assert int(split_labels[int(episode)]) == 0
