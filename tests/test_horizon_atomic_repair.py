import numpy as np
import pytest

from uav_isac.coordination.horizon_atomic_repair import (
    HorizonCandidateResources,
    HorizonCoefficientEnvelope,
    HorizonLagrangePrices,
    HorizonPowerProtocol,
    HorizonResourceLimits,
    horizon_common_mix_proxy,
    paired_deflection_difference_lower,
    rank_atomic_horizon_repairs,
)
from uav_isac.coordination.local_exchange_oracle import LocalMove
from uav_isac.coordination.maxmin_power import fixed_owner_gain_matrix
from uav_isac.physical.detection import compute_detection_probabilities


def _structures():
    initial = np.zeros((3, 3, 1), dtype=bool)
    initial[0, 2, 0] = True
    proposal = np.zeros_like(initial)
    proposal[1, 2, 0] = True
    move = LocalMove(
        kind="N5",
        selected=proposal,
        role=np.asarray([1, 1, 0], dtype=np.int8),
        owner=np.asarray([2], dtype=np.int64),
    )
    return initial, move


def _envelope(*, candidate_second_step: float = 2.0):
    lower = np.zeros((2, 3, 3, 1), dtype=np.float64)
    upper = np.zeros_like(lower)
    lower[:, 0, 2, 0] = [1.0, 1.0]
    upper[:, 0, 2, 0] = [1.1, 1.1]
    lower[:, 1, 2, 0] = [2.0, candidate_second_step]
    upper[:, 1, 2, 0] = [2.2, max(candidate_second_step, 0.0)]
    return HorizonCoefficientEnvelope(
        lower=lower,
        upper=upper,
        source="physics",
        miscoverage=0.02,
    )


def _resources(*, latency: float = 0.01):
    return HorizonCandidateResources(
        comm_power_w=np.full((2, 3), 0.1),
        over_air_bits=200,
        protocol_latency_s=latency,
        control_energy_j=0.001,
    )


def _protocol():
    return HorizonPowerProtocol(
        rounds=4,
        price_bits=6,
        feedback_bits=16,
    )


def _rank(
    envelope=None,
    resources=None,
    prices=None,
    *,
    common_power_across_horizon=False,
):
    initial, move = _structures()
    return rank_atomic_horizon_repairs(
        initial,
        [move],
        envelope or _envelope(),
        noop_resources=HorizonCandidateResources(
            comm_power_w=np.full((2, 3), 0.1)),
        candidate_resources=[resources or _resources()],
        limits=HorizonResourceLimits(
            control_period_s=0.1,
            total_power_w=1.0,
        ),
        false_alarm_probability=0.1,
        qos_floor=0.6,
        discount=0.95,
        prices=prices,
        power_protocol=_protocol(),
        common_power_across_horizon=common_power_across_horizon,
    )


def test_horizon_atomic_repair_accepts_strict_targetwise_improvement():
    result = _rank()

    assert result.accept
    assert result.selected_index == 0
    evaluation = result.evaluations[0]
    assert evaluation.gate.accept
    assert np.all(evaluation.gate.target_safe)
    assert evaluation.net_discounted_gain_lower > 0.0
    assert evaluation.rollout.max_power_balance_error_w < 1.0e-12
    assert np.allclose(
        evaluation.rollout.comm_power_w
        + np.sum(evaluation.rollout.sensing_power_w, axis=2),
        1.0,
    )


def test_interval_and_detector_monotonicity_bracket_realized_rollout():
    envelope = _envelope()
    result = _rank(envelope=envelope)
    initial, move = _structures()
    evaluation = result.evaluations[0]
    actual = 0.4 * envelope.lower + 0.6 * envelope.upper

    candidate_edges = [tuple(edge) for edge in np.argwhere(move.selected)]
    noop_edges = [tuple(edge) for edge in np.argwhere(initial)]
    for step in range(envelope.horizon):
        candidate_gain, _ = fixed_owner_gain_matrix(
            actual[step], candidate_edges)
        candidate_pd = compute_detection_probabilities(
            np.sum(
                candidate_gain
                * evaluation.rollout.sensing_power_w[step],
                axis=0,
            ),
            0.1,
        )
        noop_gain, _ = fixed_owner_gain_matrix(actual[step], noop_edges)
        noop_pd = compute_detection_probabilities(
            np.sum(
                noop_gain * result.noop.sensing_power_w[step], axis=0),
            0.1,
        )
        assert np.all(
            candidate_pd + 1.0e-12 >= evaluation.rollout.lower_pd[step])
        assert np.all(noop_pd <= result.noop.upper_pd[step] + 1.0e-12)


def test_late_horizon_degradation_fails_closed():
    result = _rank(envelope=_envelope(candidate_second_step=0.5))

    assert not result.accept
    assert not result.evaluations[0].gate.target_safe[1, 0]


def test_hard_deadline_cannot_be_bought_with_detection_gain():
    result = _rank(resources=_resources(latency=0.11))

    assert not result.accept
    assert "deadline" in result.evaluations[0].resource_reasons
    assert not result.evaluations[0].gate.hard_feasible


def test_nonnegative_lagrange_cost_can_reject_marginal_switch():
    result = _rank(prices=HorizonLagrangePrices(per_switched_edge=1.0))

    assert not result.accept
    assert result.evaluations[0].objective_penalty == 2.0
    assert result.evaluations[0].net_discounted_gain_lower < 0.0


def test_invalid_coefficient_interval_is_rejected():
    lower = np.zeros((1, 3, 3, 1))
    upper = np.zeros_like(lower)
    lower[0, 0, 2, 0] = 2.0
    upper[0, 0, 2, 0] = 1.0

    with pytest.raises(ValueError, match="lower <= upper"):
        HorizonCoefficientEnvelope(
            lower=lower,
            upper=upper,
            source="physics",
            miscoverage=0.02,
        )


def test_h1_is_the_first_step_degenerate_case_of_stationary_horizon():
    initial, move = _structures()
    full = _envelope()
    one = HorizonCoefficientEnvelope(
        lower=full.lower[:1],
        upper=full.upper[:1],
        source=full.source,
        miscoverage=full.miscoverage,
    )
    one_resource = HorizonCandidateResources(
        comm_power_w=np.full((1, 3), 0.1),
        over_air_bits=200,
        protocol_latency_s=0.01,
        control_energy_j=0.001,
    )
    one_result = rank_atomic_horizon_repairs(
        initial,
        [move],
        one,
        noop_resources=HorizonCandidateResources(
            comm_power_w=np.full((1, 3), 0.1)),
        candidate_resources=[one_resource],
        limits=HorizonResourceLimits(control_period_s=0.1),
        false_alarm_probability=0.1,
        qos_floor=0.6,
        discount=0.95,
        power_protocol=_protocol(),
    )
    full_result = _rank()

    assert one_result.accept == full_result.accept
    assert np.allclose(
        one_result.selected_sensing_power_w[0],
        full_result.selected_sensing_power_w[0],
    )


def test_common_horizon_power_plan_shares_weights_and_preserves_power():
    resources = HorizonCandidateResources(
        comm_power_w=np.asarray([
            [0.1, 0.2, 0.3],
            [0.2, 0.1, 0.25],
        ]),
    )
    initial, move = _structures()
    result = rank_atomic_horizon_repairs(
        initial,
        [move],
        _envelope(),
        noop_resources=resources,
        candidate_resources=[resources],
        limits=HorizonResourceLimits(control_period_s=0.1),
        false_alarm_probability=0.1,
        qos_floor=0.6,
        discount=0.95,
        power_protocol=_protocol(),
        common_power_across_horizon=True,
    )

    assert result.common_power_across_horizon
    for rollout in (result.noop, result.evaluations[0].rollout):
        budgets = 1.0 - rollout.comm_power_w
        weights = rollout.sensing_power_w / budgets[:, :, None]
        assert np.allclose(weights[0], weights[1])
        assert np.allclose(
            rollout.comm_power_w
            + np.sum(rollout.sensing_power_w, axis=2),
            1.0,
        )
        assert np.all(np.isfinite(rollout.lower_pd))


def test_common_mix_proxy_is_below_finite_protocol_rollout():
    result = _rank()
    _, move = _structures()
    evaluation = result.evaluations[0]
    proxy = horizon_common_mix_proxy(
        move.selected,
        _envelope(),
        evaluation.rollout.comm_power_w,
        total_power_w=1.0,
        false_alarm_probability=0.1,
        feedback_bits=16,
        discount=0.95,
    )

    assert np.all(
        proxy.worst_pd_lower
        <= np.min(evaluation.rollout.lower_pd, axis=1) + 1.0e-12
    )


def test_common_mix_binary16_saturation_remains_a_finite_lower_bound():
    initial, _ = _structures()
    lower = np.zeros((1, 3, 3, 1), dtype=np.float64)
    lower[0, 0, 2, 0] = 1.0e9
    envelope = HorizonCoefficientEnvelope(
        lower=lower,
        upper=lower,
        source="physics",
        miscoverage=0.02,
    )
    proxy = horizon_common_mix_proxy(
        initial,
        envelope,
        np.full((1, 3), 0.1),
        total_power_w=1.0,
        false_alarm_probability=0.1,
        feedback_bits=16,
    )

    raw_capacity = 0.9e9
    raw_pd = compute_detection_probabilities(
        np.asarray([raw_capacity]), 0.1)[0]
    assert np.isfinite(proxy.target_capacity_lower[0, 0])
    assert proxy.target_capacity_lower[0, 0] == np.finfo(np.float16).max
    assert proxy.target_capacity_lower[0, 0] <= raw_capacity
    assert proxy.worst_pd_lower[0] <= raw_pd + 1.0e-12


def test_paired_deflection_box_bound_contains_every_common_realization():
    initial, move = _structures()
    envelope = _envelope()
    noop_power = np.zeros((2, 3, 1), dtype=np.float64)
    candidate_power = np.zeros_like(noop_power)
    noop_power[:, 0, 0] = 0.7
    candidate_power[:, 1, 0] = 0.8
    lower = paired_deflection_difference_lower(
        initial,
        move.selected,
        noop_power,
        candidate_power,
        envelope,
    )
    rng = np.random.default_rng(20260811)
    for _ in range(100):
        actual = envelope.lower + rng.random(envelope.lower.shape) * (
            envelope.upper - envelope.lower)
        candidate = np.sum(
            move.selected[None, ...]
            * candidate_power[:, :, None, :]
            * actual,
            axis=(1, 2),
        )
        noop = np.sum(
            initial[None, ...]
            * noop_power[:, :, None, :]
            * actual,
            axis=(1, 2),
        )
        assert np.all(lower <= candidate - noop + 1.0e-12)
