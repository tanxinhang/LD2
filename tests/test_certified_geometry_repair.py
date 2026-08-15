import numpy as np
import pytest
from dataclasses import replace

from uav_isac.coordination.certified_geometry_repair import (
    CertifiedGeometryRepairConfig,
    _maxmin_bundle_direction,
    _minimum_common_scale_pd_gain,
    _minimum_common_scale_clipped_pd_gain,
    _minimum_common_scale_target_safety_margin,
    _maximum_three_step_return_scale,
    _joint_box_maxmin_direction,
    _minimum_swept_separation,
    _optimized_recovery_plan,
    _possible_worst_targets,
    _certified_candidate_rank,
    certified_trust_region_geometry_repair,
    inverse_range_deflection_gradient,
)
from uav_isac.coordination.owner_local_physics import (
    OwnerLocalKinematicState,
    OwnerTargetInvariantCache,
)
from uav_isac.coordination.target_invariant_transport import (
    TargetInvariantWireLayout,
)
from uav_isac.environment.communication import InterUAVCommunicationModel
from uav_isac.physical.detection import compute_detection_probabilities


def _model():
    return InterUAVCommunicationModel(
        rate_bits_per_dim=[0, 4], header_bits=64,
        bandwidth_hz=100_000.0, deadline_s=0.005,
        processing_delay_s=0.0002, snr_threshold_db=0.0,
        antenna_gain_dbi=0.0, carrier_hz=28.0e9,
        tx_power_w=0.25, kT=4.0e-21, noise_figure_db=4.0,
        dt=0.1,
    )


def _state(positions):
    target = np.zeros((2, 1, 4), dtype=np.float64)
    target[:, 0, :2] = np.asarray([200.0, 200.0])
    return OwnerLocalKinematicState(
        uav_position_m=np.asarray(positions, dtype=np.float64),
        uav_velocity_mps=np.zeros((2, 3), dtype=np.float64),
        target_mean_by_owner=target,
        target_cov_diag_by_owner=np.zeros((2, 1, 4), dtype=np.float64),
        target_aoi_frames_by_owner=np.zeros((2, 1), dtype=np.float64),
    )


def _case(
    positions=None,
    safe_separation=20.0,
    proposal_plan=None,
    learned_verification_top_m=0,
):
    state = _state(positions or (
        (100.0, 200.0, 20.0),
        (300.0, 200.0, 20.0),
    ))
    selected = np.zeros((2, 2, 1), dtype=bool)
    selected[0, 1, 0] = True
    coefficient = np.zeros((2, 2, 1), dtype=np.float64)
    coefficient[0, 1, 0] = 0.05
    tx_range_squared = float(np.sum(
        (state.uav_position_m[0] - np.asarray([200.0, 200.0, 0.0])) ** 2))
    rx_range_squared = float(np.sum(
        (state.uav_position_m[1] - np.asarray([200.0, 200.0, 0.0])) ** 2))
    exact_invariant = 0.05 * tx_range_squared * rx_range_squared
    layout = TargetInvariantWireLayout(2, 1)
    encoded_invariant = layout.quantize_invariant_lower(exact_invariant)
    cache = OwnerTargetInvariantCache(
        target_invariant=np.asarray([encoded_invariant]),
        age_frames=np.zeros(1, dtype=np.int64),
        version=np.ones(1, dtype=np.int64),
    )
    decision = certified_trust_region_geometry_repair(
        coefficient,
        coefficient,
        state,
        cache,
        selected,
        selected,
        np.asarray([1]),
        np.asarray([[0.5], [0.0]]),
        np.full(2, 0.25),
        np.full(2, 1000.0),
        communication_model=_model(),
        control_period_s=0.1,
        p_fa=0.1,
        qos_floor=0.6,
        dt_s=0.1,
        max_speed_mps=25.0,
        area_size_m=(400.0, 400.0),
        safe_separation_m=safe_separation,
        static_flight_power_w=80.0,
        quadratic_flight_power_coeff=0.05,
        carrier_hz=28.0e9,
        delta_f_hz=15_625.0,
        symbol_period_s=64.0e-6,
        delay_bins=64,
        doppler_bins=16,
        covariance_radius=0.0,
        dd_support_threshold=0.0,
        dd_additive_margin=0.0,
        residual_log_margin=0.0,
        config=CertifiedGeometryRepairConfig(
            minimum_worst_pd_improvement=1.0e-7,
            learned_verification_top_m=learned_verification_top_m),
        invariant_layout=layout,
        proposal_movement_plan_m=proposal_plan,
    )
    return decision, state, coefficient, selected


def test_inverse_range_gradient_points_each_endpoint_toward_target():
    _, state, coefficient, selected = _case()
    gradient = inverse_range_deflection_gradient(
        coefficient,
        selected,
        np.asarray([[0.5], [0.0]]),
        state,
        np.asarray([1]),
    )
    assert gradient[0, 0, 0] > 0.0
    assert gradient[1, 0, 0] < 0.0
    assert abs(gradient[0, 0, 1]) < 1.0e-15
    assert abs(gradient[1, 0, 1]) < 1.0e-15


def test_planar_bundle_solver_finds_common_maxmin_ascent():
    direction = _maxmin_bundle_direction(np.asarray([
        [1.0, 0.0],
        [0.0, 1.0],
    ]))
    assert direction is not None
    assert np.allclose(direction, np.ones(2) / np.sqrt(2.0))
    assert _maxmin_bundle_direction(np.asarray([
        [1.0, 0.0],
        [-1.0, 0.0],
    ])) is None


def test_common_scale_detector_gain_matches_dense_search():
    baseline = 2.0e-8
    candidate = 2.2e-8
    lower = 1.0e6
    upper = 2.0e6
    certified = _minimum_common_scale_pd_gain(
        baseline, candidate, lower, upper, 0.1)
    scale = np.linspace(lower, upper, 100_001)
    brute = np.min(
        np.asarray([
            compute_detection_probabilities(
                np.asarray([baseline * value, candidate * value]), 0.1,
            )[1]
            - compute_detection_probabilities(
                np.asarray([baseline * value, candidate * value]), 0.1,
            )[0]
            for value in scale
        ])
    )
    assert certified <= brute + 1.0e-10
    assert certified >= brute - 3.0e-7


def test_joint_box_lp_coordinates_multiple_movers():
    direction = _joint_box_maxmin_direction(np.asarray([
        [[1.0, 0.0], [0.0, 1.0]],
        [[0.0, 1.0], [1.0, 0.0]],
    ]))
    assert direction is not None
    assert np.max(np.linalg.norm(direction, axis=1)) <= 1.0 + 1.0e-12
    flattened = np.asarray([
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0, 0.0],
    ])
    assert np.min(flattened @ direction.reshape(-1)) > 0.0


def test_three_step_scale_solves_both_speed_disks():
    maximum = 2.0
    direction = np.asarray([[1.0, 1.0]]) / np.sqrt(2.0)
    baseline = np.zeros((3, 1, 2), dtype=np.float64)
    baseline[0, 0] = [-maximum, 0.0]
    baseline[1, 0] = [0.0, maximum]
    scale = _maximum_three_step_return_scale(
        baseline, (0,), direction, maximum)
    correction = scale * direction[0]

    assert scale > 0.0
    assert np.linalg.norm(baseline[0, 0] + correction) <= maximum + 1.0e-12
    assert np.linalg.norm(baseline[1, 0] - correction) <= maximum + 1.0e-12
    assert np.isclose(scale, maximum)


def test_common_scale_safety_allows_surplus_above_qos():
    margin = _minimum_common_scale_target_safety_margin(
        baseline_factor=2.0,
        candidate_factor=1.9,
        scale_lower=10.0,
        scale_upper=11.0,
        p_fa=0.1,
        qos_floor=0.6,
    )
    assert margin > 0.0


def test_common_scale_clipped_gain_matches_dense_search():
    baseline = 2.0
    candidate = 1.9
    lower = 0.1
    upper = 10.0
    qos = 0.6
    certified = _minimum_common_scale_clipped_pd_gain(
        baseline, candidate, lower, upper, 0.1, qos)
    scale = np.linspace(lower, upper, 100_001)
    brute = float(np.min([
        min(compute_detection_probabilities(
            np.asarray([candidate * value]), 0.1)[0], qos)
        - min(compute_detection_probabilities(
            np.asarray([baseline * value]), 0.1)[0], qos)
        for value in scale
    ]))
    assert certified <= brute + 1.0e-10
    assert certified >= brute - 3.0e-7


def test_active_set_certificate_protects_global_worst():
    baseline_lower = np.asarray([0.20, 0.70, 0.80])
    baseline_upper = np.asarray([0.30, 0.75, 0.85])
    possible_worst = baseline_lower <= np.min(baseline_upper)
    candidate_lower = np.asarray([0.21, 0.40, 0.31])
    coupled_gain = np.asarray([0.01, -0.35, -0.49])
    safety = np.where(
        possible_worst,
        coupled_gain,
        candidate_lower - np.min(baseline_upper),
    )
    assert np.all(safety > 0.0)


def test_possible_worst_active_set_tracks_framewise_target_switch():
    possible = _possible_worst_targets(
        np.asarray([[0.20, 0.70], [0.70, 0.20]]),
        np.asarray([[0.30, 0.80], [0.80, 0.30]]),
        1.0e-9,
    )

    assert np.array_equal(possible, np.asarray([
        [True, False],
        [False, True],
    ]))


def test_recovery_lp_closes_terminal_position_and_velocity():
    baseline = np.zeros((4, 2, 2), dtype=np.float64)
    first = baseline[0].copy()
    first[0, 0] = 0.5
    gradient = np.zeros((4, 2, 1, 2), dtype=np.float64)
    gradient[1:3, 0, 0, 0] = 1.0
    plan = _optimized_recovery_plan(
        baseline,
        first,
        (0,),
        gradient,
        np.asarray([True]),
        1.0,
    )
    assert plan is not None
    assert np.allclose(np.sum(plan, axis=0), np.sum(baseline, axis=0))
    assert np.allclose(plan[-1], baseline[-1])




def test_geometry_repair_accepts_only_exactly_reverified_improvement():
    decision, _, _, _ = _case()
    assert decision.accepted
    assert decision.best_candidate is not None
    assert np.max(np.linalg.norm(
        decision.displacement_m, axis=1)) <= 2.5 + 1.0e-12
    assert decision.certified_worst_pd_improvement > 0.0
    assert decision.best_candidate.transport.feasible
    assert decision.best_candidate.incremental_flight_energy_j > 0.0
    assert decision.feasible_candidate_count <= 4
    assert decision.best_candidate.certificate_horizon_steps == 3
    assert decision.candidate_lower_pd.shape == (3, 1)
    step_improvement = (
        np.min(decision.candidate_lower_pd, axis=1)
        - np.min(decision.baseline_upper_pd, axis=1))
    assert step_improvement[0] > 0.0
    baseline_terminal = np.sum(
        decision.best_candidate.baseline_movement_plan_m, axis=0)
    repaired_terminal = np.sum(
        decision.best_candidate.movement_plan_m, axis=0)
    assert np.allclose(repaired_terminal, baseline_terminal)
    assert np.allclose(
        decision.best_candidate.movement_plan_m[-1],
        decision.best_candidate.baseline_movement_plan_m[-1],
    )


def test_learned_candidate_slots_cannot_replace_analytic_success():
    analytic, _, _, _ = _case()
    learned_intent = np.zeros((3, 2, 2), dtype=np.float64)
    learned_intent[:, 0, 1] = np.asarray([1.0, -1.0, 0.0])
    augmented, _, _, _ = _case(
        proposal_plan=learned_intent,
        learned_verification_top_m=4,
    )

    assert analytic.accepted and augmented.accepted
    assert augmented.best_candidate is not None
    assert augmented.best_candidate.proposal_source == (
        "analytic_maxmin_gradient")
    assert np.array_equal(
        augmented.best_candidate.movement_plan_m,
        analytic.best_candidate.movement_plan_m,
    )
    assert augmented.best_candidate.transport.total_over_air_bits == (
        analytic.best_candidate.transport.total_over_air_bits)


def test_candidate_rank_uses_theorem_utility_before_proposal_source():
    decision, _, _, _ = _case()
    assert decision.best_candidate is not None
    analytic = decision.best_candidate
    learned_better = replace(
        analytic,
        proposal_source="learned_equivariant_intent",
        coupled_window_worst_pd_gain=(
            analytic.coupled_window_worst_pd_gain + 1.0e-4),
    )
    learned_tied = replace(
        analytic,
        proposal_source="learned_equivariant_intent",
    )

    assert _certified_candidate_rank(learned_better) > (
        _certified_candidate_rank(analytic))
    assert _certified_candidate_rank(analytic) > (
        _certified_candidate_rank(learned_tied))


def test_geometry_repair_rejects_collision_unsafe_trust_region():
    decision, _, _, _ = _case(
        positions=(
            (190.0, 200.0, 20.0),
            (210.0, 200.0, 20.0),
        ),
        safe_separation=20.0,
    )
    assert not decision.accepted
    assert decision.reason == "no_certified_trust_region_move"


def test_geometry_repair_checks_the_complete_swept_path():
    minimum = _minimum_swept_separation(
        np.asarray([
            [199.0, 200.0, 20.0],
            [201.0, 200.0, 20.0],
        ]),
        np.asarray([
            [2.0, 0.0],
            [-2.0, 0.0],
        ]),
    )
    assert minimum == 0.0


def test_geometry_repair_rejects_uncalibrated_ground_report_link():
    decision, state, coefficient, selected = _case()
    layout = TargetInvariantWireLayout(2, 1)
    cache = decision.best_candidate
    assert cache is not None
    with pytest.raises(ValueError, match="ground-report"):
        certified_trust_region_geometry_repair(
            coefficient, coefficient, state,
            OwnerTargetInvariantCache(
                target_invariant=np.asarray([1.0e6]),
                age_frames=np.zeros(1, dtype=np.int64),
                version=np.ones(1, dtype=np.int64),
            ),
            selected, selected, np.asarray([1]),
            np.asarray([[0.5], [0.0]]), np.full(2, 0.25),
            np.full(2, 1000.0), communication_model=_model(),
            control_period_s=0.1, p_fa=0.1, qos_floor=0.6,
            dt_s=0.1, max_speed_mps=25.0,
            area_size_m=(400.0, 400.0), safe_separation_m=20.0,
            static_flight_power_w=80.0,
            quadratic_flight_power_coeff=0.05,
            carrier_hz=28.0e9, delta_f_hz=15_625.0,
            symbol_period_s=64.0e-6, delay_bins=64, doppler_bins=16,
            covariance_radius=0.0, dd_support_threshold=0.0,
            dd_additive_margin=0.0, residual_log_margin=0.0,
            ground_communication_enabled=True,
            invariant_layout=layout,
        )
