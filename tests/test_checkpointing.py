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
        ema_state={
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        },
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


def test_checkpoint_write_is_atomic_and_leaves_no_temporary_file(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    checkpoint = tmp_path / "model.pt"

    save_checkpoint(
        checkpoint,
        model=model,
        ema_state=model.state_dict(),
        optimizer=optimizer,
        scheduler=None,
        epoch=0,
        global_step=0,
        config={},
    )

    assert checkpoint.is_file()
    assert not checkpoint.with_suffix(".pt.tmp").exists()
