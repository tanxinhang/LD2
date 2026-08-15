import numpy as np

from uav_isac.coordination.owner_local_physics import (
    OwnerLocalKinematicState,
    owner_local_dd_effectiveness,
    owner_local_dd_effectiveness_bounds,
    owner_local_horizon_coefficient_bounds,
    propagate_owner_local_horizon,
)
from uav_isac.physical.geometry import (
    C_LIGHT,
    compute_bistatic_range,
    compute_delay,
    compute_doppler,
)
from uav_isac.physical.otfs import compute_dd_effectiveness


def _state(*, covariance=None):
    target = np.asarray([[[5.0, 5.0, 1.0, -1.0]]] * 2)
    cov = (
        np.zeros((2, 1, 4), dtype=np.float64)
        if covariance is None
        else np.broadcast_to(
            np.asarray(covariance, dtype=np.float64), (2, 1, 4)
        ).copy()
    )
    return OwnerLocalKinematicState(
        uav_position_m=np.asarray([
            [0.0, 0.0, 10.0],
            [10.0, 0.0, 10.0],
        ]),
        uav_velocity_mps=np.zeros((2, 3)),
        target_mean_by_owner=target,
        target_cov_diag_by_owner=cov,
        target_aoi_frames_by_owner=np.zeros((2, 1)),
    )


def _bounds(states, *, beta=0.0, threshold=0.1):
    support = np.zeros((2, 2, 1), dtype=bool)
    support[0, 1, 0] = True
    return owner_local_horizon_coefficient_bounds(
        states,
        np.asarray([100.0]),
        np.asarray([110.0]),
        support,
        carrier_hz=1.0e9,
        delta_f_hz=1.0e3,
        symbol_period_s=1.0e-4,
        delay_bins=8,
        doppler_bins=8,
        covariance_radius=beta,
        dd_support_threshold=threshold,
    )


def test_minkowski_diagonal_covariance_propagation_is_conservative():
    initial = _state(covariance=[4.0, 9.0, 1.0, 4.0])
    states = propagate_owner_local_horizon(
        initial,
        np.zeros((1, 2, 2)),
        dt_s=1.0,
        max_speed_mps=5.0,
        area_size_m=(100.0, 100.0),
        advance_targets=True,
        target_acceleration_std_mps2=2.0,
    )

    assert len(states) == 2
    assert np.allclose(
        states[1].target_cov_diag_by_owner[0, 0],
        [16.0, 36.0, 9.0, 16.0],
    )
    assert np.allclose(
        states[1].target_mean_by_owner[0, 0, :2], [6.0, 4.0])
    assert states[1].target_aoi_frames_by_owner[0, 0] == 1.0


def test_inverse_range_interval_contains_every_point_in_position_ball():
    state = _state(covariance=[1.0, 1.0, 0.0, 0.0])
    result = _bounds((state,), beta=2.0)
    lower = result.lower[0, 0, 1, 0]
    upper = result.upper[0, 0, 1, 0]
    assert result.dd_certified_support[0, 0, 1, 0]

    owner_mean = state.target_mean_by_owner[1, 0]
    actual_target = np.asarray([
        owner_mean[0] + 1.2, owner_mean[1] - 1.0, 0.0,
    ])
    assert np.linalg.norm(actual_target[:2] - owner_mean[:2]) <= 2.0
    tx_range = np.linalg.norm(
        state.uav_position_m[0] - actual_target)
    rx_range = np.linalg.norm(
        state.uav_position_m[1] - actual_target)
    actual = 105.0 / (tx_range ** 2 * rx_range ** 2)

    assert lower <= actual <= upper


def test_uncertainty_widens_interval_and_uncertified_dd_fails_closed():
    state = _state(covariance=[1.0, 1.0, 0.0, 0.0])
    point = _bounds((state,), beta=0.0)
    uncertain = _bounds((state,), beta=2.0)

    assert uncertain.lower[0, 0, 1, 0] < point.lower[0, 0, 1, 0]
    assert uncertain.upper[0, 0, 1, 0] > point.upper[0, 0, 1, 0]

    unsupported = _bounds((state,), beta=2.0, threshold=1.0)
    assert unsupported.lower[0, 0, 1, 0] == 0.0
    assert unsupported.upper[0, 0, 1, 0] > 0.0
    assert not unsupported.dd_certified_support[0, 0, 1, 0]


def test_uav_reachable_set_contains_moved_platform_dd_outcome():
    nominal = _state()
    radius_position = 2.5
    radius_velocity = 25.0
    bound = owner_local_dd_effectiveness_bounds(
        nominal,
        carrier_hz=28.0e9,
        delta_f_hz=15625.0,
        symbol_period_s=6.4e-5,
        delay_bins=64,
        doppler_bins=16,
        covariance_radius=0.0,
        uav_position_radius_m=radius_position,
        uav_velocity_radius_mps=radius_velocity,
    )
    moved = OwnerLocalKinematicState(
        uav_position_m=nominal.uav_position_m + np.asarray([
            [2.0, -1.0, 0.0],
            [-1.5, 1.5, 0.0],
        ]),
        uav_velocity_mps=np.asarray([
            [20.0, 10.0, 0.0],
            [-15.0, 15.0, 0.0],
        ]),
        target_mean_by_owner=nominal.target_mean_by_owner,
        target_cov_diag_by_owner=nominal.target_cov_diag_by_owner,
        target_aoi_frames_by_owner=nominal.target_aoi_frames_by_owner,
    )
    actual = owner_local_dd_effectiveness(
        moved,
        carrier_hz=28.0e9,
        delta_f_hz=15625.0,
        symbol_period_s=6.4e-5,
        delay_bins=64,
        doppler_bins=16,
    )

    assert bound.lower[0, 1, 0] <= actual[0, 1, 0] + 1.0e-12
    assert bound.delay_bin_radius[0, 1, 0] > 0.0
    assert bound.doppler_bin_radius[0, 1, 0] > 0.0


def test_horizon_inverse_range_includes_uav_reachable_position_ball():
    nominal = _state()
    result = owner_local_horizon_coefficient_bounds(
        (nominal,),
        np.asarray([100.0]),
        np.asarray([110.0]),
        np.asarray([[[False], [True]], [[False], [False]]]),
        carrier_hz=1.0e9,
        delta_f_hz=1.0e3,
        symbol_period_s=1.0e-4,
        delay_bins=8,
        doppler_bins=8,
        covariance_radius=0.0,
        dd_support_threshold=0.1,
        uav_position_radius_m=2.0,
        uav_velocity_radius_mps=0.0,
    )
    target = np.asarray([5.0, 5.0, 0.0])
    moved_tx = nominal.uav_position_m[0] + np.asarray([1.0, -1.0, 0.0])
    moved_rx = nominal.uav_position_m[1] + np.asarray([-1.0, 1.0, 0.0])
    actual = 105.0 / (
        np.linalg.norm(moved_tx - target) ** 2
        * np.linalg.norm(moved_rx - target) ** 2
    )

    assert result.lower[0, 0, 1, 0] <= actual
    assert actual <= result.upper[0, 0, 1, 0]


def test_vectorized_dd_bound_matches_scalar_equations_on_random_state():
    rng = np.random.default_rng(20260811)
    K, Q = 3, 2
    position = rng.uniform(5.0, 95.0, size=(K, 3))
    position[:, 2] = rng.uniform(8.0, 20.0, size=K)
    velocity = rng.uniform(-8.0, 8.0, size=(K, 3))
    target = rng.uniform(10.0, 90.0, size=(K, Q, 4))
    target[:, :, 2:4] = rng.uniform(-3.0, 3.0, size=(K, Q, 2))
    covariance = rng.uniform(0.01, 4.0, size=(K, Q, 4))
    state = OwnerLocalKinematicState(
        uav_position_m=position,
        uav_velocity_mps=velocity,
        target_mean_by_owner=target,
        target_cov_diag_by_owner=covariance,
        target_aoi_frames_by_owner=np.zeros((K, Q)),
    )
    fc = 3.5e9
    delta_f = 15.0e3
    symbol_period = 1.0 / delta_f
    M, N = 32, 16
    beta = 1.7
    uav_position_radius = np.asarray([0.2, 0.4, 0.7])
    uav_velocity_radius = np.asarray([1.0, 2.0, 3.0])
    result = owner_local_dd_effectiveness_bounds(
        state,
        carrier_hz=fc,
        delta_f_hz=delta_f,
        symbol_period_s=symbol_period,
        delay_bins=M,
        doppler_bins=N,
        covariance_radius=beta,
        uav_position_radius_m=uav_position_radius,
        uav_velocity_radius_mps=uav_velocity_radius,
    )
    scalar_point = owner_local_dd_effectiveness(
        state,
        carrier_hz=fc,
        delta_f_hz=delta_f,
        symbol_period_s=symbol_period,
        delay_bins=M,
        doppler_bins=N,
    )
    np.testing.assert_allclose(result.point, scalar_point, atol=2.0e-15)

    def scalar_sinc_lower(center, radius):
        if radius >= 0.5:
            distance = 0.5
        else:
            left, right = center - radius, center + radius
            crosses = np.ceil(left - 0.5) <= np.floor(right - 0.5)
            distance = 0.5 if crosses else min(0.5, max(
                abs(left - np.round(left)),
                abs(right - np.round(right)),
            ))
        return abs(np.sinc(distance))

    for transmitter in range(K):
        for receiver in range(K):
            if transmitter == receiver:
                continue
            for q in range(Q):
                mean = target[receiver, q]
                target_position = np.asarray([mean[0], mean[1], 0.0])
                target_velocity = np.asarray([mean[2], mean[3], 0.0])
                tx_range = max(float(np.linalg.norm(
                    target_position - position[transmitter])), 1.0e-9)
                rx_range = max(float(np.linalg.norm(
                    position[receiver] - target_position)), 1.0e-9)
                tau = (tx_range + rx_range) / C_LIGHT
                nu = compute_doppler(
                    position[transmitter], velocity[transmitter],
                    position[receiver], velocity[receiver],
                    target_position, target_velocity, fc,
                )
                scalar_effectiveness = compute_dd_effectiveness(
                    compute_delay(compute_bistatic_range(
                        position[transmitter],
                        position[receiver],
                        target_position,
                    )),
                    nu,
                    delta_f,
                    symbol_period,
                    M,
                    N,
                )
                np.testing.assert_allclose(
                    scalar_point[transmitter, receiver, q],
                    scalar_effectiveness,
                    rtol=1.0e-14,
                    atol=2.0e-15,
                )
                position_radius = beta * float(np.sqrt(np.max(
                    covariance[receiver, q, :2])))
                velocity_radius = beta * float(np.sqrt(np.max(
                    covariance[receiver, q, 2:4])))
                delay_radius = (
                    2.0 * position_radius
                    + uav_position_radius[transmitter]
                    + uav_position_radius[receiver]
                ) / C_LIGHT * M * delta_f
                tx_relative_radius = (
                    position_radius + uav_position_radius[transmitter])
                rx_relative_radius = (
                    position_radius + uav_position_radius[receiver])
                tx_unit_radius = min(
                    2.0,
                    2.0 * tx_relative_radius
                    / max(tx_range - tx_relative_radius, 1.0e-9),
                )
                rx_unit_radius = min(
                    2.0,
                    2.0 * rx_relative_radius
                    / max(rx_range - rx_relative_radius, 1.0e-9),
                )
                doppler_hz_radius = fc / C_LIGHT * (
                    uav_velocity_radius[transmitter]
                    + uav_velocity_radius[receiver]
                    + 2.0 * velocity_radius
                    + (
                        np.linalg.norm(
                            velocity[transmitter] - target_velocity)
                        + uav_velocity_radius[transmitter]
                        + velocity_radius
                    ) * tx_unit_radius
                    + (
                        np.linalg.norm(
                            target_velocity + velocity[receiver])
                        + uav_velocity_radius[receiver]
                        + velocity_radius
                    ) * rx_unit_radius
                )
                doppler_radius = doppler_hz_radius * N * symbol_period
                expected_lower = (
                    scalar_sinc_lower(tau * M * delta_f, delay_radius)
                    * scalar_sinc_lower(
                        nu * N * symbol_period, doppler_radius))
                np.testing.assert_allclose(
                    result.delay_bin_radius[transmitter, receiver, q],
                    delay_radius, rtol=1.0e-14, atol=1.0e-15)
                np.testing.assert_allclose(
                    result.doppler_bin_radius[transmitter, receiver, q],
                    doppler_radius, rtol=1.0e-14, atol=1.0e-15)
                np.testing.assert_allclose(
                    result.lower[transmitter, receiver, q],
                    min(expected_lower, scalar_point[
                        transmitter, receiver, q]),
                    rtol=1.0e-14, atol=1.0e-15)
