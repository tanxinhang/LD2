import torch

from uav_isac.agents.trainer import apply_cacsr_sensing_residual


def test_cacsr_gate_off_is_exact_noop_and_preserves_budget():
    weights = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    output, delta = apply_cacsr_sensing_residual(
        weights,
        torch.zeros_like(weights, dtype=torch.bool),
        torch.tensor([[0.2, 0.3, 0.4, 0.5]]),
        torch.tensor([[1.0, 2.0, 2.0, 3.0]]),
        torch.ones_like(weights),
        gain=0.5,
    )
    torch.testing.assert_close(output, weights, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(delta, torch.zeros_like(delta))
    torch.testing.assert_close(output.sum(dim=-1), torch.ones(1))


def test_cacsr_increases_underloaded_capable_target_share():
    weights = torch.full((1, 4), 0.25)
    output, _ = apply_cacsr_sensing_residual(
        weights,
        torch.tensor([[True, False, False, False]]),
        torch.tensor([[0.0, 0.9, 0.9, 0.9]]),
        torch.tensor([[1.0, 2.0, 2.0, 2.0]]),
        torch.tensor([[1.0, 0.5, 0.5, 0.5]]),
        gain=0.5,
    )
    assert output[0, 0] > weights[0, 0]
    torch.testing.assert_close(output.sum(dim=-1), torch.ones(1))
