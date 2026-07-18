import math

import torch
from torch.utils.data import DataLoader, Dataset

from world_model.model import ActionConditionedUNet
from world_model.training import (
    EMA,
    autoregressive_rollout,
    evaluate,
    train_one_epoch,
    warmup_cosine_factor,
)


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
    assert any(
        not torch.equal(before[name], value)
        for name, value in ema.state_dict().items()
    )


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
    assert math.isnan(metrics["per_action_l1"][6])


def test_autoregressive_rollout_shape():
    model = _model().eval()
    initial = torch.rand(1, 4, 3, 16, 24)
    actions = torch.tensor([[0, 1, 2, 3, 4, 5]])

    with torch.no_grad():
        predictions = autoregressive_rollout(
            model, initial, actions, horizon=3
        )

    assert predictions.shape == (1, 3, 3, 16, 24)


def test_warmup_cosine_factor_warms_up_then_decays_to_zero():
    assert warmup_cosine_factor(0, warmup_steps=4, total_steps=12) == 0.25
    assert warmup_cosine_factor(3, warmup_steps=4, total_steps=12) == 1.0
    assert warmup_cosine_factor(12, warmup_steps=4, total_steps=12) == 0.0


def test_ema_context_uses_shadow_parameters_then_restores_model():
    model = torch.nn.Linear(2, 1)
    ema = EMA(model, decay=0.9)
    original = {name: value.clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1)

    with ema.average_parameters(model):
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, original[name])

    assert any(
        not torch.equal(value, original[name])
        for name, value in model.state_dict().items()
    )


def test_ema_load_normalizes_checkpoint_tensors_to_model_dtype():
    model = torch.nn.Linear(2, 1).float()
    ema = EMA(model, decay=0.9)
    checkpoint_state = {
        name: value.double() if value.is_floating_point() else value
        for name, value in ema.state_dict().items()
    }

    ema.load_state_dict(checkpoint_state)

    for name, value in ema.state_dict().items():
        assert value.dtype == model.state_dict()[name].dtype
