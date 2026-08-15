import numpy as np

from uav_isac.coordination.owner_local_physics import (
    OwnerLocalKinematicState,
    OwnerTargetInvariantCache,
    advance_owner_local_kinematics,
    decode_owner_local_kinematics,
    owner_local_dd_effectiveness_bounds,
    predict_coefficients_from_lagged_feedback,
    reciprocal_token_candidate_mask,
    update_target_invariant_cache,
)
from uav_isac.environment.observation_slices import ObservationSlices


def _state(position: np.ndarray, target_xy: tuple[float, float]):
    K = position.shape[0]
    target = np.zeros((K, 1, 4), dtype=np.float64)
    target[:, 0, :2] = np.asarray(target_xy, dtype=np.float64)
    return OwnerLocalKinematicState(
        uav_position_m=np.asarray(position, dtype=np.float64),
        uav_velocity_mps=np.zeros((K, 3), dtype=np.float64),
        target_mean_by_owner=target,
        target_cov_diag_by_owner=np.zeros((K, 1, 4), dtype=np.float64),
        target_aoi_frames_by_owner=np.zeros((K, 1), dtype=np.float64),
    )


def test_decode_owner_local_kinematics_reverses_normalization():
    slices = ObservationSlices.from_config(K=2, Q=1, use_rel_features=True)
    obs = np.zeros((2, slices.total_dim), dtype=np.float64)
    obs[0, :6] = [0.25, 0.5, 1.0, 0.4, -0.2, 0.1]
    belief = slices.extract_beliefs(obs)
    belief[0, 0] = [0.5, 0.25, 0.2, -0.4, 0.01, 0.04, 0.09, 0.16, 0.3]

    state = decode_owner_local_kinematics(
        obs, slices, area_size_m=(800.0, 400.0), height_m=20.0)

    assert np.allclose(state.uav_position_m[0], [200.0, 200.0, 20.0])
    assert np.allclose(state.uav_velocity_mps[0], [10.0, -5.0, 2.5])
    assert np.allclose(
        state.target_mean_by_owner[0, 0], [400.0, 100.0, 5.0, -10.0])
    assert np.allclose(
        state.target_cov_diag_by_owner[0, 0], [6400.0, 6400.0, 56.25, 100.0])
    assert np.isclose(state.target_aoi_frames_by_owner[0, 0], 30.0)


def test_reciprocal_token_candidates_require_both_directions():
    visible = np.zeros((3, 3, 2), dtype=bool)
    visible[0, 1, 0] = True
    visible[1, 0, 1] = True
    visible[0, 2, 0] = True

    candidate = reciprocal_token_candidate_mask(visible)

    assert np.all(candidate[0, 1])
    assert np.all(candidate[1, 0])
    assert not np.any(candidate[0, 2])
    assert not np.any(candidate[2, 0])
    assert not np.any(np.diagonal(candidate, axis1=0, axis2=1))


def test_lagged_feedback_uses_target_invariant_and_fails_closed():
    previous_position = np.asarray([
        [0.0, 0.0, 10.0],
        [20.0, 0.0, 10.0],
    ])
    current_position = np.asarray([
        [0.0, 10.0, 10.0],
        [20.0, 10.0, 10.0],
    ])
    previous_state = _state(previous_position, (10.0, 100.0))
    current_state = _state(current_position, (10.0, 100.0))
    previous = np.zeros((2, 2, 1), dtype=np.float64)
    previous[0, 1, 0] = 2.0
    observed = previous > 0.0
    support = np.ones_like(observed)

    result = predict_coefficients_from_lagged_feedback(
        previous, observed, previous_state, current_state,
        current_support=support)

    previous_range_sq = 10.0 ** 2 + 100.0 ** 2 + 10.0 ** 2
    current_range_sq = 10.0 ** 2 + 90.0 ** 2 + 10.0 ** 2
    expected = 2.0 * previous_range_sq ** 2 / current_range_sq ** 2
    assert np.isclose(result.coefficient_per_watt[0, 1, 0], expected)
    assert np.isclose(result.coefficient_per_watt[1, 0, 0], expected)
    assert result.direct_edge_count == 1
    assert result.target_fallback_edge_count == 1

    unavailable = predict_coefficients_from_lagged_feedback(
        np.zeros_like(previous), np.zeros_like(observed),
        previous_state, current_state, current_support=support)
    assert np.all(unavailable.coefficient_per_watt == 0.0)
    assert unavailable.unavailable_edge_count == 2


def test_target_invariant_cache_retains_ages_and_expires_fail_closed():
    position = np.asarray([
        [0.0, 0.0, 10.0],
        [20.0, 0.0, 10.0],
    ])
    state = _state(position, (10.0, 100.0))
    coefficient = np.zeros((2, 2, 1), dtype=np.float64)
    coefficient[0, 1, 0] = 2.0
    observed = coefficient > 0.0

    fresh = update_target_invariant_cache(
        coefficient,
        observed,
        state,
        prior_cache=None,
        elapsed_frames=5,
        max_age_frames=20,
    )
    assert fresh.target_invariant[0] > 0.0
    assert fresh.age_frames.tolist() == [5]
    assert fresh.version.tolist() == [1]

    retained = update_target_invariant_cache(
        np.zeros_like(coefficient),
        np.zeros_like(observed),
        state,
        prior_cache=fresh,
        elapsed_frames=5,
        max_age_frames=20,
    )
    assert np.allclose(retained.target_invariant, fresh.target_invariant)
    assert retained.age_frames.tolist() == [10]
    assert retained.version.tolist() == [1]

    expired = update_target_invariant_cache(
        np.zeros_like(coefficient),
        np.zeros_like(observed),
        state,
        prior_cache=retained,
        elapsed_frames=11,
        max_age_frames=20,
    )
    assert expired.target_invariant.tolist() == [0.0]
    assert expired.age_frames.tolist() == [21]
    assert expired.version.tolist() == [1]

    refreshed = update_target_invariant_cache(
        coefficient,
        observed,
        state,
        prior_cache=expired,
        elapsed_frames=5,
        max_age_frames=20,
    )
    assert refreshed.target_invariant[0] > 0.0
    assert refreshed.age_frames.tolist() == [5]
    assert refreshed.version.tolist() == [2]


def test_lagged_feedback_uses_only_same_target_cache_fallback():
    position = np.asarray([
        [0.0, 0.0, 10.0],
        [20.0, 0.0, 10.0],
    ])
    state = _state(position, (10.0, 100.0))
    coefficient = np.zeros((2, 2, 1), dtype=np.float64)
    support = np.ones_like(coefficient, dtype=bool)
    cache = OwnerTargetInvariantCache(
        target_invariant=np.asarray([2.0e8]),
        age_frames=np.asarray([10], dtype=np.int64),
        version=np.asarray([3], dtype=np.int64),
    )

    result = predict_coefficients_from_lagged_feedback(
        coefficient,
        np.zeros_like(coefficient, dtype=bool),
        state,
        state,
        current_support=support,
        target_invariant_cache=cache,
    )

    assert np.all(result.coefficient_per_watt[[0, 1], [1, 0], 0] > 0.0)
    assert result.direct_edge_count == 0
    assert result.target_fallback_edge_count == 2
    assert result.cached_target_fallback_edge_count == 2
    assert result.target_invariant_age_frames.tolist() == [10]
    assert result.target_invariant_version.tolist() == [3]


def test_owner_local_dd_bound_is_exact_at_zero_covariance():
    position = np.asarray([
        [0.0, 0.0, 10.0],
        [20.0, 0.0, 10.0],
    ])
    state = _state(position, (10.0, 100.0))

    result = owner_local_dd_effectiveness_bounds(
        state,
        carrier_hz=3.5e9,
        delta_f_hz=15.0e3,
        symbol_period_s=1.0 / 15.0e3,
        delay_bins=32,
        doppler_bins=32,
        covariance_radius=3.0,
    )

    assert np.allclose(result.lower, result.point)
    assert np.all(result.delay_bin_radius == 0.0)
    assert np.all(result.doppler_bin_radius == 0.0)


def test_owner_local_dd_bound_widens_monotonically_with_covariance_radius():
    position = np.asarray([
        [0.0, 0.0, 10.0],
        [20.0, 0.0, 10.0],
    ])
    base = _state(position, (10.0, 100.0))
    covariance = np.zeros((2, 1, 4), dtype=np.float64)
    covariance[:, 0] = [100.0, 64.0, 4.0, 1.0]
    uncertain = OwnerLocalKinematicState(
        uav_position_m=base.uav_position_m,
        uav_velocity_mps=base.uav_velocity_mps,
        target_mean_by_owner=base.target_mean_by_owner,
        target_cov_diag_by_owner=covariance,
        target_aoi_frames_by_owner=base.target_aoi_frames_by_owner,
    )

    small = owner_local_dd_effectiveness_bounds(
        uncertain,
        carrier_hz=3.5e9,
        delta_f_hz=15.0e3,
        symbol_period_s=1.0 / 15.0e3,
        delay_bins=32,
        doppler_bins=32,
        covariance_radius=1.0,
    )
    large = owner_local_dd_effectiveness_bounds(
        uncertain,
        carrier_hz=3.5e9,
        delta_f_hz=15.0e3,
        symbol_period_s=1.0 / 15.0e3,
        delay_bins=32,
        doppler_bins=32,
        covariance_radius=3.0,
    )

    off_diagonal = ~np.eye(2, dtype=bool)[:, :, None]
    assert np.all(large.lower[off_diagonal] <= small.lower[off_diagonal])
    assert np.all(
        large.delay_bin_radius[off_diagonal]
        >= small.delay_bin_radius[off_diagonal])
    assert np.all(
        large.doppler_bin_radius[off_diagonal]
        >= small.doppler_bin_radius[off_diagonal])
    assert np.all((large.lower >= 0.0) & (large.lower <= 1.0))


def test_action_alignment_matches_speed_clip_bounce_and_target_cv():
    position = np.asarray([
        [99.8, 0.2, 10.0],
        [20.0, 20.0, 10.0],
    ])
    base = _state(position, (10.0, 100.0))
    target = base.target_mean_by_owner.copy()
    target[:, 0, 2:4] = [2.0, -1.0]
    moving = OwnerLocalKinematicState(
        uav_position_m=base.uav_position_m,
        uav_velocity_mps=base.uav_velocity_mps,
        target_mean_by_owner=target,
        target_cov_diag_by_owner=base.target_cov_diag_by_owner,
        target_aoi_frames_by_owner=base.target_aoi_frames_by_owner,
    )

    result = advance_owner_local_kinematics(
        moving,
        np.asarray([[3.0, -4.0], [0.3, 0.4]]),
        dt_s=0.1,
        max_speed_mps=10.0,
        area_size_m=(100.0, 100.0),
        advance_targets=True,
    )

    # First displacement is clipped from norm 5 to norm 1: [0.6,-0.8].
    assert np.allclose(result.uav_position_m[0], [99.6, 0.6, 10.0])
    assert np.allclose(result.uav_velocity_mps[0], [6.0, -8.0, 0.0])
    assert np.allclose(result.uav_position_m[1], [20.3, 20.4, 10.0])
    assert np.allclose(
        result.target_mean_by_owner[:, 0, :2], [[10.2, 99.9], [10.2, 99.9]])
    assert np.all(result.target_aoi_frames_by_owner == 1.0)
