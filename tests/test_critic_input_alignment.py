from __future__ import annotations

import numpy as np
import torch

from uav_isac.agents.trainer import build_aligned_critic_input


def test_centralized_critic_matches_rollout_field_order() -> None:
    # Stored rows are [global base (2), communication summary (2)].
    stored = torch.tensor([
        [10.0, 11.0, 100.0, 101.0],
        [20.0, 21.0, 200.0, 201.0],
    ])
    local = torch.zeros(3, 3)
    result = build_aligned_critic_input(
        stored, local, np.array([0, 1, 2]), num_agents=2,
        centralized=True, base_state_dim=2, comm_dim=2)
    expected = torch.tensor([
        [10.0, 11.0, 1.0, 0.0, 100.0, 101.0],
        [10.0, 11.0, 0.0, 1.0, 100.0, 101.0],
        [20.0, 21.0, 1.0, 0.0, 200.0, 201.0],
    ])
    assert torch.equal(result, expected)


def test_ippo_uses_local_base_and_recorded_communication() -> None:
    stored = torch.tensor([
        [10.0, 11.0, 100.0, 101.0],
        [20.0, 21.0, 200.0, 201.0],
    ])
    local = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])
    result = build_aligned_critic_input(
        stored, local, np.array([1, 2]), num_agents=2,
        centralized=False, base_state_dim=3, comm_dim=2)
    expected = torch.tensor([
        [1.0, 2.0, 3.0, 0.0, 1.0, 100.0, 101.0],
        [4.0, 5.0, 6.0, 1.0, 0.0, 200.0, 201.0],
    ])
    assert torch.equal(result, expected)
