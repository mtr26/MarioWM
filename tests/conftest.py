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

    # Index 6 starts a new collector trajectory without a done at index 5.
    observations[6].fill(90)
    next_obs[5].fill(89)

    actions = np.arange(n, dtype=np.int32) % 7
    rewards = np.arange(n, dtype=np.float32)
    dones = np.zeros(n, dtype=bool)
    dones[[3, 9]] = True

    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "observations", data=observations, chunks=(4, height, width, 3)
        )
        handle.create_dataset("next_obs", data=next_obs, chunks=(4, height, width, 3))
        handle.create_dataset("actions", data=actions)
        handle.create_dataset("rewards", data=rewards)
        handle.create_dataset("dones", data=dones)

    return path
