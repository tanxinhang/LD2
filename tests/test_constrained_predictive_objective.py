import numpy as np
import torch

from uav_isac.physical.detection import compute_detection_probabilities
from uav_isac.prediction.certified_gnn import CertifiedBipartiteGNN
from uav_isac.prediction.constrained_objective import (
    constrained_detection_objective,
    detection_probability_from_deflection,
    project_power_to_row_budget,
)


def test_power_projection_enforces_native_row_constraints():
    logits = torch.tensor([
        [[2.0, 1.0, -3.0], [4.0, 5.0, 6.0], [0.0, 0.0, 0.0]]
    ])
    budget = torch.tensor([[0.4, 0.7, 0.2]])
    feasible = torch.tensor([
        [[True, True, False], [False, True, True], [False, False, False]]
    ])

    power = project_power_to_row_budget(logits, budget, feasible)

    assert torch.all(power >= 0.0)
    torch.testing.assert_close(power[0, :2].sum(dim=-1), budget[0, :2])
    torch.testing.assert_close(power[0, 2], torch.zeros(3))
    assert power[0, 0, 2] == 0.0
    assert power[0, 1, 0] == 0.0


def test_torch_detector_matches_canonical_numpy_detector():
    deflection = torch.tensor(
        [0.0, 1.0, 5.0, 11.179523266319096, 20.0],
        dtype=torch.float64,
    )
    actual = detection_probability_from_deflection(deflection, p_fa=0.001)
    expected = compute_detection_probabilities(
        deflection.numpy(), P_FA=0.001)
    np.testing.assert_allclose(actual.numpy(), expected, atol=1e-12, rtol=1e-12)


def test_joint_objective_is_finite_differentiable_and_budget_feasible():
    torch.manual_seed(9)
    batch, nodes, targets = 2, 4, 3
    model = CertifiedBipartiteGNN(
        feature_dim=9, hidden_dim=12, message_rounds=1)
    features = torch.randn(batch, nodes, targets, 9)
    visible = torch.ones(batch, nodes, targets, dtype=torch.bool)
    prediction = model(features, visible)
    gain = torch.rand(batch, nodes, targets) * 1000.0
    budget = torch.full((batch, nodes), 0.0251)
    teacher = torch.full(
        (batch, nodes, targets), 0.0251 / targets)

    result = constrained_detection_objective(
        prediction,
        gain,
        budget,
        visible,
        teacher,
        teacher,
        p_fa=0.001,
        qos_floor=0.60,
    )
    result.total.backward()

    assert torch.isfinite(result.total)
    assert result.budget_violation_w == 0.0
    assert model.power_head.weight.grad is not None
    assert torch.all(torch.isfinite(model.power_head.weight.grad))


def test_qos_penalty_orders_feasible_above_infeasible_physics():
    torch.manual_seed(11)
    model = CertifiedBipartiteGNN(
        feature_dim=9, hidden_dim=8, message_rounds=1)
    features = torch.zeros(1, 2, 2, 9)
    visible = torch.ones(1, 2, 2, dtype=torch.bool)
    prediction = model(features, visible)
    budget = torch.full((1, 2), 0.0251)
    teacher = torch.full((1, 2, 2), 0.01255)
    weak = constrained_detection_objective(
        prediction, torch.zeros(1, 2, 2), budget, visible,
        teacher, teacher, p_fa=0.001, qos_floor=0.60)
    strong = constrained_detection_objective(
        prediction, torch.full((1, 2, 2), 1000.0), budget, visible,
        teacher, teacher, p_fa=0.001, qos_floor=0.60)

    assert weak.qos_violation > strong.qos_violation
    assert weak.total > strong.total


def test_nonnegative_dual_weight_penalizes_native_qos_residual():
    torch.manual_seed(13)
    model = CertifiedBipartiteGNN(
        feature_dim=9, hidden_dim=8, message_rounds=1)
    visible = torch.ones(1, 2, 2, dtype=torch.bool)
    prediction = model(torch.zeros(1, 2, 2, 9), visible)
    budget = torch.full((1, 2), 0.0251)
    teacher = torch.full((1, 2, 2), 0.01255)
    base = constrained_detection_objective(
        prediction, torch.zeros(1, 2, 2), budget, visible,
        teacher, teacher, p_fa=0.001, qos_floor=0.60,
        qos_dual_weight=0.0)
    priced = constrained_detection_objective(
        prediction, torch.zeros(1, 2, 2), budget, visible,
        teacher, teacher, p_fa=0.001, qos_floor=0.60,
        qos_dual_weight=3.0)

    torch.testing.assert_close(
        priced.total - base.total, 3.0 * base.qos_residual)


def test_qos_constraint_mask_excludes_infeasible_training_rows():
    torch.manual_seed(17)
    model = CertifiedBipartiteGNN(
        feature_dim=9, hidden_dim=8, message_rounds=1)
    visible = torch.ones(2, 2, 2, dtype=torch.bool)
    prediction = model(torch.zeros(2, 2, 2, 9), visible)
    budget = torch.full((2, 2), 0.0251)
    teacher = torch.full((2, 2, 2), 0.01255)
    gains = torch.stack([
        torch.zeros(2, 2),
        torch.full((2, 2), 1000.0),
    ])
    mask = torch.tensor([[False, False], [True, True]])

    masked = constrained_detection_objective(
        prediction, gains, budget, visible, teacher, teacher,
        p_fa=0.001, qos_floor=0.60, qos_constraint_mask=mask)
    strong_only = constrained_detection_objective(
        replace_prediction_batch(prediction, 1), gains[1:], budget[1:],
        visible[1:], teacher[1:], teacher[1:], p_fa=0.001,
        qos_floor=0.60)

    torch.testing.assert_close(masked.qos_residual, strong_only.qos_residual)


def replace_prediction_batch(prediction, index):
    """Slice every prediction tensor for a focused constraint test."""
    from dataclasses import replace

    return replace(
        prediction,
        owner_logits=prediction.owner_logits[index:],
        transmitter_logits=prediction.transmitter_logits[index:],
        power_logits=prediction.power_logits[index:],
        dual_warm_start=prediction.dual_warm_start[index:],
        risk_radius=prediction.risk_radius[index:],
        detection_logits=prediction.detection_logits[index:],
        detection_delta=prediction.detection_delta[index:],
        endpoint_embedding=prediction.endpoint_embedding[index:],
    )
