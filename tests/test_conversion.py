import json
import subprocess
import sys

import h5py
import numpy as np
import pytest

from inspect_dataset import inspect
from world_model.conversion import (
    CacheValidationError,
    ConversionConfig,
    SourceValidationError,
    assign_episode_splits,
    build_episode_offsets,
    convert_dataset,
    validate_cache,
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

    assert result == output.resolve()
    frames = np.load(output / "frames.npy", mmap_mode="r")
    actions = np.load(output / "actions.npy", mmap_mode="r")
    offsets = np.load(output / "episode_offsets.npy")
    splits = np.load(output / "episode_splits.npy")
    np.testing.assert_array_equal(
        offsets, np.array([0, 4, 6, 10, 12], dtype=np.int64)
    )
    assert frames.shape == (16, 4, 6, 3)
    assert actions.shape == (12,)
    assert splits.shape == (4,)
    assert int(frames[0, 0, 0, 0]) == 0
    assert int(frames[4, 0, 0, 0]) == 4
    assert int(frames[7, 0, 0, 0]) == 89
    assert int(frames[8, 0, 0, 0]) == 90


def test_convert_dataset_rejects_unmarked_temporal_discontinuity(
    synthetic_h5, tmp_path
):
    with h5py.File(synthetic_h5, "a") as handle:
        handle["next_obs"][7] = np.full(
            handle["next_obs"].shape[1:], 77, dtype=np.uint8
        )

    with pytest.raises(SourceValidationError, match="--break-index 8"):
        convert_dataset(
            ConversionConfig(
                input_path=synthetic_h5,
                output_dir=tmp_path / "cache",
                height=4,
                width=6,
                history=2,
                break_indices=(6,),
                workers=1,
            )
        )


def test_convert_dataset_reports_conversion_and_hashing_progress(
    synthetic_h5, tmp_path, capsys
):
    convert_dataset(
        ConversionConfig(
            input_path=synthetic_h5,
            output_dir=tmp_path / "cache",
            height=4,
            width=6,
            history=2,
            break_indices=(6,),
            workers=1,
        )
    )

    stderr = capsys.readouterr().err
    assert "Converting transitions" in stderr
    assert "Hashing cache" in stderr
    assert stderr.count("100%") >= 2


def test_convert_dataset_refuses_nonempty_output(synthetic_h5, tmp_path):
    output = tmp_path / "cache"
    output.mkdir()
    owned_file = output / "keep.txt"
    owned_file.write_text("owned by user", encoding="utf-8")
    config = ConversionConfig(
        input_path=synthetic_h5,
        output_dir=output,
        height=4,
        width=6,
    )

    with pytest.raises(FileExistsError, match="non-empty"):
        convert_dataset(config)

    assert owned_file.read_text(encoding="utf-8") == "owned by user"


def test_convert_dataset_can_limit_transitions_for_smoke_test(
    synthetic_h5, tmp_path, capsys
):
    output = tmp_path / "limited-cache"

    convert_dataset(
        ConversionConfig(
            input_path=synthetic_h5,
            output_dir=output,
            height=4,
            width=6,
            history=2,
            break_indices=(6, 10),
            workers=1,
            limit_transitions=8,
        )
    )

    actions = np.load(output / "actions.npy")
    offsets = np.load(output / "episode_offsets.npy")
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert actions.shape == (8,)
    np.testing.assert_array_equal(offsets, np.array([0, 4, 6, 8]))
    assert metadata["limit_transitions"] == 8
    assert metadata["ignored_break_indices"] == [10]
    assert "Ignoring break indices outside converted prefix: [10]" in capsys.readouterr().out


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
            workers=1,
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


def test_validate_cache_detects_tampered_array(synthetic_h5, tmp_path):
    output = tmp_path / "cache"
    convert_dataset(
        ConversionConfig(
            input_path=synthetic_h5,
            output_dir=output,
            height=4,
            width=6,
            break_indices=(6,),
            workers=1,
        )
    )
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    metadata["sha256"]["actions.npy"] = "0" * 64
    (output / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(CacheValidationError, match="SHA-256 mismatch"):
        validate_cache(output)


def test_inspector_stats_only_reports_boundaries_without_writing_images(
    synthetic_h5,
):
    stats = inspect(
        str(synthetic_h5),
        n_preview=4,
        traj_len=2,
        stats_only=True,
        break_indices=(6,),
    )

    assert stats["n_transitions"] == 12
    assert stats["n_trajectories"] == 4
    assert stats["trailing_partial_length"] == 2
    assert stats["datasets"]["observations"]["chunks"] == [4, 8, 10, 3]
    assert stats["explicit_breaks"][0]["index"] == 6
    assert stats["explicit_breaks"][0]["continuous"] is False
    assert not (synthetic_h5.parent / "preview.png").exists()
    assert not (synthetic_h5.parent / "sample_trajectory.gif").exists()


def test_inspector_does_not_import_visual_stack_for_stats_only_usage():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import inspect_dataset; "
                "assert 'matplotlib.pyplot' not in sys.modules; "
                "assert 'imageio.v3' not in sys.modules"
            ),
        ],
        text=True,
        capture_output=True,
        timeout=15,
    )

    assert result.returncode == 0, result.stderr
