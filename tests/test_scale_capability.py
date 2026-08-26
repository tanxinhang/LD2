import numpy as np
import pytest

from uav_isac.coordination.scale_capability import (
    cap_aware_sensing_budget,
    characterize_bistatic_scale_capability,
    owner_consistent_bistatic_capability,
    route_capability_shadow,
    route_task_capability_shadow,
    task_detection_metrics,
)


def test_cap_aware_budget_uses_tighter_of_joint_headroom_and_pa_cap():
    budget = cap_aware_sensing_budget(
        np.ones(3), np.array([0.0, 0.98, 1.2]), 0.0251)
    np.testing.assert_allclose(budget, [0.0251, 0.02, 0.0])


def test_bistatic_capability_ignores_monostatic_diagonal_and_reports_concentration():
    coefficient = np.zeros((2, 2, 2))
    coefficient[0, 0] = 1000.0
    coefficient[0, 1] = [2.0, 1.0]
    coefficient[1, 0] = [1.0, 3.0]
    result = characterize_bistatic_scale_capability(
        coefficient, np.array([0.5, 0.25]),
        deployed_target_power_w=np.array([0.9, 0.1]))
    np.testing.assert_allclose(result.best_single_pair_capability, [1.0, 0.75])
    np.testing.assert_allclose(result.owner_consistent_target_capability, [1.0, 0.75])
    np.testing.assert_allclose(result.relaxed_target_ceiling, [1.25, 1.25])
    assert result.geometry_bottleneck == pytest.approx(0.75)
    assert result.power_concentration_max == pytest.approx(0.9)
    assert result.power_concentration_hhi == pytest.approx(0.82)


def test_owner_consistent_capability_is_between_single_pair_and_free_owner_relaxation():
    coefficient = np.zeros((3, 3, 1))
    coefficient[0, 2, 0] = 4.0
    coefficient[1, 2, 0] = 3.0
    coefficient[2, 0, 0] = 5.0
    budget = np.array([0.5, 0.5, 0.5])
    result = characterize_bistatic_scale_capability(coefficient, budget)
    owner = owner_consistent_bistatic_capability(coefficient, budget)
    assert result.best_single_pair_capability[0] == pytest.approx(2.5)
    assert owner[0] == pytest.approx(3.5)
    assert owner[0] <= result.relaxed_target_ceiling[0]


@pytest.mark.parametrize(
    "fixed,joint,relaxed,region,action,certified",
    [
        ([1.1, 1.0], None, [1.2, 1.2], "I", "hold", True),
        ([0.8, 0.9], [1.1, 1.0], [1.2, 1.2], "II", "l2_structure_repair", True),
        ([0.4, 0.5], [0.6, 0.7], [0.8, 0.9], "III", "l3_geometry_repair", True),
        ([0.4, 0.5], [0.6, 0.7], [1.2, 1.2], "U", "unresolved_shadow", False),
    ],
)
def test_shadow_router_requires_positive_feasibility_or_upper_infeasibility_certificate(
    fixed, joint, relaxed, region, action, certified,
):
    route = route_capability_shadow(
        np.asarray(fixed), np.asarray(relaxed), np.ones(2),
        joint_feasible_deflection=(None if joint is None else np.asarray(joint)))
    assert (route.region, route.action, route.certified) == (
        region, action, certified)


def test_task_metrics_preserve_worst_bottom_k_and_average_floors():
    metrics = task_detection_metrics(
        np.array([0.60, 0.72, 0.80, 0.92]), (0.60, 0.65, 0.75, 2))
    assert metrics["worst"] == pytest.approx(0.60)
    assert metrics["bottom_k"] == pytest.approx(0.66)
    assert metrics["average"] == pytest.approx(0.76)
    assert metrics["feasibility_ratio"] <= 1.0


def test_task_metrics_zero_detection_has_infinite_gauge():
    assert np.isinf(task_detection_metrics(
        np.zeros(3), (0.60, 0.70, 0.80, 2))["feasibility_ratio"])


@pytest.mark.parametrize(
    "fixed,joint,relaxed,region",
    [
        ([0.61, 0.81, 0.90], None, [0.8, 0.9, 0.95], "I"),
        ([0.59, 0.80, 0.90], [0.62, 0.82, 0.91], [0.8, 0.9, 0.95], "II"),
        ([0.40, 0.60, 0.70], [0.5, 0.6, 0.7], [0.59, 0.80, 0.90], "III"),
        ([0.40, 0.60, 0.70], [0.5, 0.6, 0.7], [0.62, 0.82, 0.91], "U"),
    ],
)
def test_task_router_uses_all_three_aggregate_constraints(
    fixed, joint, relaxed, region,
):
    route = route_task_capability_shadow(
        np.asarray(fixed), np.asarray(relaxed), (0.60, 0.65, 0.75, 2),
        joint_feasible_pd=None if joint is None else np.asarray(joint))
    assert route.region == region
