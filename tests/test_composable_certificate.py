import numpy as np
import pytest

from uav_isac.coordination.composable_certificate import (
    RobustPrimalDualBounds,
    aggregate_target_responsibility_certificate,
    conservative_row_contribution,
    quantize_lower_log,
    quantize_upper_log,
    optimal_simplex_dual_upper,
    robust_movement_dominates,
    robust_primal_dual_bounds,
    targetwise_price_row_dual_upper,
    uniform_price_row_dual_upper,
)
from uav_isac.coordination.hyperedge import (
    reconstruct_bistatic_coefficient_from_public_state,
    reconstruct_bistatic_coefficient_dd_upper_from_public_state,
    reconstruct_bistatic_coefficient_upper_from_public_state,
)
from uav_isac.coordination.maxmin_power import (
    solve_fixed_structure_maxmin_power_lp,
)


def test_downward_log_quantization_never_overstates_contribution():
    values = np.geomspace(1.0e-10, 1.0e4, 1000)
    decoded = quantize_lower_log(
        values, bits=12, scale=1.0e-8, maximum=256.0)
    assert np.all(decoded >= 0.0)
    assert np.all(decoded <= values)
    assert decoded[-1] == pytest.approx(256.0)


def test_upward_log_quantization_never_understates_and_rejects_overflow():
    values = np.geomspace(1.0e-10, 200.0, 1000)
    decoded = quantize_upper_log(
        values, bits=12, scale=1.0e-8, maximum=256.0)
    assert np.all(decoded >= values)
    with pytest.raises(OverflowError):
        quantize_upper_log(
            np.asarray([257.0]), bits=12, scale=1.0e-8, maximum=256.0)


def test_uniform_dual_row_terms_bound_exact_maxmin_optimum():
    gain = np.asarray([[2.0, 0.4], [0.3, 1.5]])
    budget = np.asarray([0.8, 0.7])
    row_upper = uniform_price_row_dual_upper(gain, budget)
    # Analytic/LP optimum cannot exceed any feasible simplex-price dual value.
    exact = solve_fixed_structure_maxmin_power_lp(gain, budget)
    assert exact.worst_deflection <= np.sum(row_upper) + 1.0e-12


def test_targetwise_dual_family_bounds_exact_and_never_loosens_uniform():
    rng = np.random.default_rng(20260908)
    for _ in range(100):
        gain = rng.lognormal(size=(5, 4))
        budget = rng.uniform(0.05, 1.0, size=5)
        exact = solve_fixed_structure_maxmin_power_lp(gain, budget)
        uniform = float(np.sum(uniform_price_row_dual_upper(gain, budget)))
        targetwise = float(np.min(np.sum(
            targetwise_price_row_dual_upper(gain, budget), axis=0)))
        combined = min(uniform, targetwise)
        assert exact.worst_deflection <= combined + 1.0e-9
        assert combined <= uniform + 1.0e-12


def test_optimal_simplex_dual_matches_exact_and_tightens_fixed_prices():
    rng = np.random.default_rng(20260909)
    for _ in range(50):
        gain = rng.lognormal(mean=0.0, sigma=2.0, size=(5, 4))
        budget = rng.uniform(0.05, 1.0, size=5)
        exact = solve_fixed_structure_maxmin_power_lp(gain, budget)
        optimal, prices = optimal_simplex_dual_upper(gain, budget)
        uniform = float(np.sum(uniform_price_row_dual_upper(gain, budget)))
        targetwise = float(np.min(np.sum(
            targetwise_price_row_dual_upper(gain, budget), axis=0)))
        assert np.sum(prices) == pytest.approx(1.0)
        assert optimal == pytest.approx(
            exact.worst_deflection, rel=2.0e-7, abs=1.0e-10)
        assert optimal <= min(uniform, targetwise) + 1.0e-9


def test_robust_primal_dual_sandwich_contains_sampled_optima() -> None:
    rng = np.random.default_rng(20260831)
    K, Q = 4, 3
    lower = rng.uniform(0.1, 1.0, size=(K, Q))
    upper = lower + rng.uniform(0.0, 0.5, size=(K, Q))
    budget = rng.uniform(0.2, 1.0, size=K)
    power = np.broadcast_to(budget[:, None] / Q, (K, Q)).copy()
    prices = np.full(Q, 1.0 / Q)
    certificate = robust_primal_dual_bounds(
        lower, upper, power, budget, prices)
    for _ in range(30):
        truth = lower + rng.uniform(size=(K, Q)) * (upper - lower)
        exact = solve_fixed_structure_maxmin_power_lp(truth, budget)
        assert certificate.primal_witness_lower <= (
            exact.worst_deflection + 1.0e-10)
        assert exact.worst_deflection <= certificate.dual_upper + 1.0e-10


def test_robust_strong_dominance_uses_crossed_bounds() -> None:
    incumbent = RobustPrimalDualBounds(1.0, 2.0)
    candidate = RobustPrimalDualBounds(2.1, 3.0)
    assert robust_movement_dominates(candidate, incumbent)
    assert not robust_movement_dominates(
        RobustPrimalDualBounds(2.0, 2.5), incumbent)
    assert not robust_movement_dominates(candidate, incumbent, margin=0.2)


def test_robust_bounds_reject_nonfeasible_stitched_witness() -> None:
    with pytest.raises(ValueError, match="row-feasible"):
        robust_primal_dual_bounds(
            np.ones((2, 2)),
            np.ones((2, 2)),
            np.asarray([[0.8, 0.8], [0.5, 0.5]]),
            np.ones(2),
            np.full(2, 0.5),
        )


def test_row_contribution_composes_under_different_private_views():
    power = np.asarray([[0.2, 0.8], [0.7, 0.3]])
    lower = np.asarray([[2.0, 0.5], [0.4, 3.0]])
    truth = lower + np.asarray([[0.1, 0.3], [0.2, 0.4]])
    contribution = conservative_row_contribution(power, lower)

    assert np.all(np.sum(contribution, axis=0)
                  <= np.sum(power * truth, axis=0))


def test_target_owner_requires_complete_fresh_same_frame_reports():
    # owner x sender x target
    reports = np.zeros((2, 2, 2), dtype=np.float64)
    reports[0, :, 0] = [0.3, 0.4]
    reports[1, :, 1] = [0.2, 0.5]
    frames = np.asarray([[7, 7], [7, 6]])
    certificate = aggregate_target_responsibility_certificate(
        reports,
        frames,
        target_owner=np.asarray([0, 1]),
        max_age_frames=2,
        current_frame=8,
        row_dual_upper=np.asarray([[0.5, 0.6], [0.5, 0.6]]),
    )

    assert certificate.target_complete.tolist() == [True, False]
    np.testing.assert_allclose(certificate.target_deflection_lower, [0.7, 0.0])
    assert certificate.worst_complete_lower == pytest.approx(0.7)
    assert np.isinf(certificate.global_optimum_upper)


def test_stale_complete_report_fails_closed():
    certificate = aggregate_target_responsibility_certificate(
        np.ones((2, 2, 1)),
        np.full((2, 2), 3),
        target_owner=np.asarray([0]),
        max_age_frames=1,
        current_frame=5,
    )
    assert certificate.complete_fraction == 0.0
    assert certificate.worst_complete_lower == 0.0


def test_complete_reports_compose_global_dual_upper_and_ratio():
    reports = np.asarray([
        [[0.3, 0.2], [0.4, 0.5]],
        [[0.3, 0.2], [0.4, 0.5]],
    ])
    certificate = aggregate_target_responsibility_certificate(
        reports,
        np.full((2, 2), 9),
        target_owner=np.asarray([0, 1]),
        max_age_frames=1,
        current_frame=9,
        row_dual_upper=np.asarray([[0.5, 0.6], [0.5, 0.6]]),
    )
    assert certificate.global_optimum_upper == pytest.approx(1.1)
    assert certificate.joint_approximation_ratio_lower == pytest.approx(
        0.7 / 1.1)


def test_targetwise_reports_tighten_composable_upper_without_changing_lower():
    reports = np.asarray([
        [[0.3, 0.2], [0.4, 0.5]],
        [[0.3, 0.2], [0.4, 0.5]],
    ])
    targetwise = np.asarray([
        [[0.40, 9.0], [0.40, 8.0]],
        [[7.00, 0.45], [6.00, 0.45]],
    ])
    certificate = aggregate_target_responsibility_certificate(
        reports,
        np.full((2, 2), 9),
        target_owner=np.asarray([0, 1]),
        max_age_frames=1,
        current_frame=9,
        row_dual_upper=np.asarray([[0.5, 0.6], [0.5, 0.6]]),
        row_targetwise_upper=targetwise,
    )
    assert certificate.target_deflection_lower.tolist() == pytest.approx(
        [0.7, 0.7])
    assert certificate.global_optimum_upper == pytest.approx(0.8)
    assert certificate.joint_approximation_ratio_lower == pytest.approx(
        0.7 / 0.8)


def test_composable_certificate_rejects_primal_above_targetwise_dual():
    with pytest.raises(ValueError, match="primal lower exceeds"):
        aggregate_target_responsibility_certificate(
            np.full((2, 2, 1), 0.4),
            np.full((2, 2), 4),
            target_owner=np.asarray([0]),
            max_age_frames=1,
            current_frame=4,
            row_targetwise_upper=np.full((2, 2, 1), 0.1),
        )


def test_robust_geometry_dd_coefficient_is_lower_than_sampled_set():
    rng = np.random.default_rng(20260828)
    position = np.asarray([
        [[100.0, 150.0]],
        [[420.0, 310.0]],
    ])
    velocity = np.asarray([
        [[4.0, -2.0]],
        [[-3.0, 1.0]],
    ])
    target = np.asarray([[260.0, 230.0, 0.0]])
    target_velocity = np.asarray([[2.0, -1.0, 0.0]])
    seen = np.ones((2, 1), dtype=bool)
    kwargs = dict(
        uav_height_m=100.0,
        fc_hz=28.0e9,
        rcs_m2=1.0,
        delta_f_hz=15.0e3,
        symbol_period_s=1.0 / 15.0e3,
        delay_bins=64,
        doppler_bins=16,
        dd_gate_min=0.5,
        coefficient_scale=1.0,
        dd_gain_mode="continuous",
    )
    robust = reconstruct_bistatic_coefficient_from_public_state(
        position,
        velocity,
        target,
        target_velocity,
        seen,
        position_uncertainty_m=np.asarray([2.0, 3.0]),
        target_position_uncertainty_m=np.asarray([4.0]),
        velocity_uncertainty_mps=np.asarray([0.3, 0.4]),
        target_velocity_uncertainty_mps=np.asarray([0.5]),
        robust_dd_uncertainty=True,
        **kwargs,
    )
    upper = reconstruct_bistatic_coefficient_upper_from_public_state(
        position,
        target,
        seen,
        uav_height_m=kwargs["uav_height_m"],
        fc_hz=kwargs["fc_hz"],
        rcs_m2=kwargs["rcs_m2"],
        coefficient_scale=kwargs["coefficient_scale"],
        position_uncertainty_m=np.asarray([2.0, 3.0]),
        target_position_uncertainty_m=np.asarray([4.0]),
    )
    dd_upper = reconstruct_bistatic_coefficient_dd_upper_from_public_state(
        position,
        velocity,
        target,
        target_velocity,
        seen,
        position_uncertainty_m=np.asarray([2.0, 3.0]),
        target_position_uncertainty_m=np.asarray([4.0]),
        velocity_uncertainty_mps=np.asarray([0.3, 0.4]),
        target_velocity_uncertainty_mps=np.asarray([0.5]),
        **kwargs,
    )
    assert np.all(dd_upper <= upper + 1.0e-30)

    def ball(radius, shape):
        direction = rng.normal(size=shape)
        direction /= np.maximum(
            np.linalg.norm(direction, axis=-1, keepdims=True), 1.0e-15)
        return direction * rng.uniform(0.0, radius, size=shape[:-1])[..., None]

    for _ in range(200):
        sampled_position = position + np.asarray([
            [ball(2.0, (1, 2))[0]],
            [ball(3.0, (1, 2))[0]],
        ])
        sampled_velocity = velocity + np.asarray([
            [ball(0.3, (1, 2))[0]],
            [ball(0.4, (1, 2))[0]],
        ])
        sampled_target = target.copy()
        sampled_target[0, :2] += ball(4.0, (1, 2))[0]
        sampled_target_velocity = target_velocity.copy()
        sampled_target_velocity[0, :2] += ball(0.5, (1, 2))[0]
        point = reconstruct_bistatic_coefficient_from_public_state(
            sampled_position,
            sampled_velocity,
            sampled_target,
            sampled_target_velocity,
            seen,
            robust_dd_uncertainty=False,
            **kwargs,
        )
        assert np.all(robust <= point + 1.0e-30)
        assert np.all(point <= dd_upper + 1.0e-30)
        assert np.all(point <= upper + 1.0e-30)
