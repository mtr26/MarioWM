import pytest
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

    with pytest.raises(ValueError, match="12 context channels"):
        model(
            torch.rand(1, 9, 16, 24),
            torch.zeros(1, 4, dtype=torch.long),
        )


def test_model_rejects_spatial_shape_not_divisible_by_downsampling_factor():
    model = tiny_model()

    with pytest.raises(ValueError, match="divisible by 2"):
        model(
            torch.rand(1, 12, 15, 24),
            torch.zeros(1, 4, dtype=torch.long),
        )


def test_model_rejects_action_outside_vocabulary():
    model = tiny_model()

    with pytest.raises(ValueError, match="action indices"):
        model(
            torch.rand(1, 12, 16, 24),
            torch.tensor([[0, 1, 2, 7]]),
        )
