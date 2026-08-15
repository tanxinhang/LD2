from pathlib import Path

import pytest
import torch

from uav_isac.agents.equivariant_movement_plan import (
    DoubleSetEquivariantResidualMovementPlan,
    FrozenEquivariantMovementPlanner,
    EquivariantResidualMovementPlan,
)


def _inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(7)
    return (
        torch.rand(2, 4, 2) * 800.0,
        torch.rand(2, 6, 2) * 800.0,
        torch.rand(2, 4, 2) - 0.5,
        torch.rand(2, 4) * 0.2,
        torch.softmax(torch.rand(2, 4, 6), dim=-1) * 0.8,
        torch.tensor([1, 2]),
    )


def test_zero_initialization_is_exact_zoh_with_speed_bound() -> None:
    model = EquivariantResidualMovementPlan(
        horizon_steps=3,
        movement_decision_interval=2,
        region_size_m=(800.0, 800.0),
        maximum_displacement_m=2.5,
    )
    inputs = _inputs()
    output = model(*inputs)
    expected = inputs[2][:, None].expand(-1, 3, -1, -1)
    assert torch.equal(output, expected)
    assert torch.all(torch.linalg.vector_norm(output, dim=-1) <= 2.5)


def test_target_permutation_invariance_and_uav_equivariance() -> None:
    model = EquivariantResidualMovementPlan(
        horizon_steps=3,
        movement_decision_interval=2,
        region_size_m=(800.0, 800.0),
        maximum_displacement_m=2.5,
    )
    torch.nn.init.normal_(model.plan_output.weight, std=0.02)
    inputs = _inputs()
    reference = model(*inputs)
    target_permutation = torch.tensor([3, 1, 5, 0, 2, 4])
    target_inputs = list(inputs)
    target_inputs[1] = target_inputs[1][:, target_permutation]
    target_inputs[4] = target_inputs[4][:, :, target_permutation]
    assert torch.allclose(model(*target_inputs), reference, atol=1.0e-6)

    uav_permutation = torch.tensor([2, 0, 3, 1])
    uav_inputs = list(inputs)
    for index in (0, 2, 3, 4):
        uav_inputs[index] = uav_inputs[index][:, uav_permutation]
    assert torch.allclose(
        model(*uav_inputs), reference[:, :, uav_permutation], atol=1.0e-6)


def test_known_movement_hold_phase_cannot_be_overridden() -> None:
    model = EquivariantResidualMovementPlan(
        horizon_steps=3,
        movement_decision_interval=2,
        region_size_m=(800.0, 800.0),
        maximum_displacement_m=2.5,
    )
    torch.nn.init.constant_(model.plan_output.weight, 1.0)
    torch.nn.init.constant_(model.plan_output.bias, 1.0)
    inputs = _inputs()
    output = model(*inputs)
    assert torch.allclose(output[0, 0], inputs[2][0], atol=1.0e-7)
    assert not torch.allclose(output[1, 0], inputs[2][1])


def test_one_parameterization_supports_different_cardinalities() -> None:
    model = EquivariantResidualMovementPlan(
        horizon_steps=3,
        movement_decision_interval=2,
        region_size_m=(800.0, 800.0),
        maximum_displacement_m=2.5,
    )
    for num_uavs, num_targets in ((4, 4), (6, 6), (8, 5)):
        output = model(
            torch.rand(1, num_uavs, 2) * 800.0,
            torch.rand(1, num_targets, 2) * 800.0,
            torch.rand(1, num_uavs, 2),
            torch.full((1, num_uavs), 0.2),
            torch.full(
                (1, num_uavs, num_targets), 0.8 / num_targets),
            torch.tensor([2]),
        )
        assert output.shape == (1, 3, num_uavs, 2)


def test_double_set_head_is_target_invariant_and_uav_equivariant() -> None:
    model = DoubleSetEquivariantResidualMovementPlan(
        horizon_steps=3,
        movement_decision_interval=2,
        region_size_m=(800.0, 800.0),
        maximum_displacement_m=2.5,
    )
    torch.nn.init.normal_(model.plan_output.weight, std=0.02)
    inputs = _inputs()
    reference = model(*inputs)
    target_permutation = torch.tensor([3, 1, 5, 0, 2, 4])
    target_inputs = list(inputs)
    target_inputs[1] = target_inputs[1][:, target_permutation]
    target_inputs[4] = target_inputs[4][:, :, target_permutation]
    assert torch.allclose(model(*target_inputs), reference, atol=1.0e-6)
    uav_permutation = torch.tensor([2, 0, 3, 1])
    uav_inputs = list(inputs)
    for index in (0, 2, 3, 4):
        uav_inputs[index] = uav_inputs[index][:, uav_permutation]
    assert torch.allclose(
        model(*uav_inputs), reference[:, :, uav_permutation], atol=1.0e-6)


def test_double_set_head_is_uniform_region_scale_covariant() -> None:
    source = DoubleSetEquivariantResidualMovementPlan(
        horizon_steps=3,
        movement_decision_interval=2,
        region_size_m=(800.0, 800.0),
        maximum_displacement_m=2.5,
    )
    torch.manual_seed(19)
    for parameter in source.parameters():
        torch.nn.init.normal_(parameter, std=0.02)
    scaled = DoubleSetEquivariantResidualMovementPlan(
        horizon_steps=3,
        movement_decision_interval=2,
        region_size_m=(1200.0, 1200.0),
        maximum_displacement_m=2.5,
    )
    scaled.load_state_dict(source.state_dict())
    scaled.region_size_m.copy_(torch.tensor([1200.0, 1200.0]))
    inputs = _inputs()
    scaled_inputs = list(inputs)
    scaled_inputs[0] = 1.5 * scaled_inputs[0]
    scaled_inputs[1] = 1.5 * scaled_inputs[1]
    assert torch.allclose(
        source(*inputs), scaled(*scaled_inputs), rtol=0.0, atol=2.0e-7)


def _checkpoint(path: Path, model: EquivariantResidualMovementPlan) -> None:
    torch.save({
        "model_state_dict": model.state_dict(),
        "hidden_dim": 64,
        "metadata": {
            "horizon_steps": 3,
            "movement_decision_interval": 2,
            "region_size_m": [800.0, 800.0],
            "maximum_displacement_m": 2.5,
            "residual_limit_fraction": 0.75,
            "training_episode_ids": [1, 2],
            "selection_episode_ids": [3],
            "selection_admitted": True,
        },
    }, path)


def test_uniform_region_reparameterization_is_exactly_scale_covariant(
    tmp_path: Path,
) -> None:
    source = EquivariantResidualMovementPlan(
        horizon_steps=3,
        movement_decision_interval=2,
        region_size_m=(800.0, 800.0),
        maximum_displacement_m=2.5,
    )
    torch.manual_seed(13)
    for parameter in source.parameters():
        torch.nn.init.normal_(parameter, std=0.02)
    checkpoint = tmp_path / "movement.pt"
    _checkpoint(checkpoint, source)
    planner = FrozenEquivariantMovementPlanner.from_checkpoint(
        checkpoint,
        validation_episode_ids=(10,),
        horizon_steps=3,
        movement_decision_interval=2,
        region_size_m=(1200.0, 1200.0),
        maximum_displacement_m=2.5,
        allow_uniform_region_scaling=True,
        validation_domain_key="k4q4",
    )
    inputs = _inputs()
    reference = source(*inputs)
    scaled_inputs = list(inputs)
    scaled_inputs[0] = 1.5 * scaled_inputs[0]
    scaled_inputs[1] = 1.5 * scaled_inputs[1]
    adapted = planner.model(*scaled_inputs)
    assert planner.uniform_region_scaling_applied
    assert planner.uniform_region_scale == pytest.approx(1.5)
    assert torch.allclose(adapted, reference, rtol=0.0, atol=2.0e-7)


def test_region_migration_is_opt_in_and_rejects_anisotropic_scaling(
    tmp_path: Path,
) -> None:
    model = EquivariantResidualMovementPlan(
        horizon_steps=3,
        movement_decision_interval=2,
        region_size_m=(800.0, 800.0),
        maximum_displacement_m=2.5,
    )
    checkpoint = tmp_path / "movement.pt"
    _checkpoint(checkpoint, model)
    common = dict(
        validation_episode_ids=(10,),
        horizon_steps=3,
        movement_decision_interval=2,
        maximum_displacement_m=2.5,
    )
    with pytest.raises(ValueError, match="physical contract"):
        FrozenEquivariantMovementPlanner.from_checkpoint(
            checkpoint, region_size_m=(1200.0, 1200.0), **common)
    with pytest.raises(ValueError, match="physical contract"):
        FrozenEquivariantMovementPlanner.from_checkpoint(
            checkpoint,
            region_size_m=(1200.0, 1000.0),
            allow_uniform_region_scaling=True,
            **common,
        )


def test_domain_key_distinguishes_equal_numeric_seeds_across_scales(
    tmp_path: Path,
) -> None:
    model = EquivariantResidualMovementPlan(
        horizon_steps=3,
        movement_decision_interval=2,
        region_size_m=(800.0, 800.0),
        maximum_displacement_m=2.5,
    )
    checkpoint = tmp_path / "movement.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "hidden_dim": 64,
        "metadata": {
            "architecture": "shared-uav-target-set-invariant-zoh-residual-v1",
            "horizon_steps": 3,
            "movement_decision_interval": 2,
            "region_size_m": [800.0, 800.0],
            "maximum_displacement_m": 2.5,
            "residual_limit_fraction": 0.75,
            "training_episode_ids": [7],
            "selection_episode_ids": [8],
            "training_episode_keys": ["k4q4:7"],
            "selection_episode_keys": ["k4q4:8"],
            "selection_admitted": True,
        },
    }, checkpoint)
    common = dict(
        validation_episode_ids=(7,),
        horizon_steps=3,
        movement_decision_interval=2,
        region_size_m=(800.0, 800.0),
        maximum_displacement_m=2.5,
    )
    FrozenEquivariantMovementPlanner.from_checkpoint(
        checkpoint, validation_domain_key="k8q8", **common)
    with pytest.raises(ValueError, match="development/validation overlap"):
        FrozenEquivariantMovementPlanner.from_checkpoint(
            checkpoint, validation_domain_key="k4q4", **common)
