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

    np.testing.assert_array_equal(
        offsets, np.array([0, 4, 6, 10, 12], dtype=np.int64)
    )


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


def test_validate_source_rejects_wrong_transition_dtype(synthetic_h5):
    with h5py.File(synthetic_h5, "a") as handle:
        values = handle["actions"][:].astype(np.float32)
        del handle["actions"]
        handle.create_dataset("actions", data=values)

    with h5py.File(synthetic_h5, "r") as handle:
        with pytest.raises(SourceValidationError, match="integer dtype"):
            validate_source(handle)
