import hashlib
import json

import numpy as np
import pytest

from uav_isac.coordination.causal_hierarchical_controller import (
    FrozenCausalEnvelopeCalibration,
    build_causal_coefficient_envelope,
)
from uav_isac.coordination.owner_local_physics import OwnerLocalKinematicState
from uav_isac.environment.communication import InterUAVCommunicationModel


def _state(position):
    target = np.zeros((2, 1, 4), dtype=np.float64)
    target[:, 0, :2] = [10.0, 10.0]
    return OwnerLocalKinematicState(
        uav_position_m=np.asarray(position, dtype=np.float64),
        uav_velocity_mps=np.zeros((2, 3), dtype=np.float64),
        target_mean_by_owner=target,
        target_cov_diag_by_owner=np.zeros_like(target),
        target_aoi_frames_by_owner=np.zeros((2, 1)),
    )


def _model():
    return InterUAVCommunicationModel(
        rate_bits_per_dim=[0, 4], header_bits=64,
        bandwidth_hz=100_000.0, deadline_s=0.005,
        processing_delay_s=0.0002, snr_threshold_db=0.0,
        antenna_gain_dbi=0.0, carrier_hz=28.0e9,
        tx_power_w=0.25, kT=4.0e-21, noise_figure_db=4.0,
        dt=0.1,
    )


def _calibration():
    return FrozenCausalEnvelopeCalibration(
        alpha=0.02,
        coverage_floor=60.0 / 61.0,
        calibration_episode_count=60,
        base_current_log_margin=1.0e-5,
        transition_residual_log_margin=0.0,
        horizon_steps=3,
        source_paths=("a",),
        source_sha256=("0" * 64,),
        envelope_calibration_ready=True,
        system_certificate_ready=False,
        observability_mode="synthetic_test",
        target_invariant_token_transport_enabled=True,
        action_aligned_owner_state_enabled=True,
        target_invariant_cache_max_age_frames=150,
        calibrated_support_policy=(
            "reciprocal_target_token_edges_or_excited_persistent_edges"),
    )


def test_frozen_calibration_verifies_source_hash(tmp_path):
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    source_payload = {
        "schema_version": 6,
        "seed_order": [1, 2],
        "config": "config.yaml",
        "observability_mode": "owner_local_conformal",
        "target_invariant_token_transport_enabled": True,
        "action_aligned_owner_state_enabled": True,
        "target_invariant_cache_max_age_frames": 150,
        "owner_local_dd_covariance_radius": 0.0,
        "frozen_owner_local_dd_additive_margin": 0.0,
        "frozen_unknown_edge_lower_log_margin": 1e-5,
        "frozen_unknown_edge_upper_log_margin": 1e-5,
        "horizon_future_calibration_diagnostic": {
            "score": "episode max over current-provenance edges",
            "future_information_used_by_controller": False,
        },
    }
    source.write_text(json.dumps(source_payload), encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    payload = {
        "method": "episode_level_split_conformal_simultaneous_log_envelope",
        "alpha": 0.02,
        "target_coverage": 0.98,
        "finite_sample_coverage_floor": 60.0 / 61.0,
        "calibration_episode_count": 60,
        "frozen_transition_residual_log_margin": 0.0,
        "envelope_calibration_ready": True,
        "system_certificate_ready": False,
        "horizon_design": {"steps": 3, "base_current_log_margin": 1e-5},
        "sources": [{"path": "source.json", "sha256": digest}],
    }
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = FrozenCausalEnvelopeCalibration.from_json(
        path, source_root=tmp_path)
    assert result.current_log_margin == 1e-5


def test_causal_envelope_contains_exact_static_radar_law():
    previous_state = _state([[0.0, 0.0, 20.0], [20.0, 0.0, 20.0]])
    current_state = _state([[1.0, 0.0, 20.0], [19.0, 0.0, 20.0]])
    previous = np.zeros((2, 2, 1), dtype=np.float64)
    previous[0, 1, 0] = 2.0
    observed = previous > 0.0
    support = ~np.eye(2, dtype=bool)[:, :, None]
    result = build_causal_coefficient_envelope(
        previous,
        observed,
        previous_state,
        current_state,
        np.asarray([1]),
        support,
        current_state.uav_position_m,
        np.full(2, 0.25),
        current_frame=2,
        elapsed_frames=1,
        max_age_frames=150,
        calibration=_calibration(),
        communication_model=_model(),
        carrier_hz=28.0e9,
        delta_f_hz=1.0e-6,
        symbol_period_s=1.0e-6,
        delay_bins=1,
        doppler_bins=1,
        dd_support_threshold=0.0,
    )
    previous_range_sq = 10.0 ** 2 + 10.0 ** 2 + 20.0 ** 2
    current_range_sq = 9.0 ** 2 + 10.0 ** 2 + 20.0 ** 2
    actual = 2.0 * previous_range_sq ** 2 / current_range_sq ** 2
    assert result.available
    assert result.lower[0, 1, 0] <= actual <= result.upper[0, 1, 0]
    assert result.transport.feasible


def test_causal_envelope_rejects_uncalibrated_swerling_domain():
    state = _state([[0.0, 0.0, 20.0], [20.0, 0.0, 20.0]])
    coefficient = np.zeros((2, 2, 1), dtype=np.float64)
    coefficient[0, 1, 0] = 1.0
    with pytest.raises(ValueError, match="non-Swerling"):
        build_causal_coefficient_envelope(
            coefficient,
            coefficient > 0.0,
            state,
            state,
            np.asarray([1]),
            ~np.eye(2, dtype=bool)[:, :, None],
            state.uav_position_m,
            np.full(2, 0.25),
            current_frame=2,
            elapsed_frames=1,
            max_age_frames=150,
            calibration=_calibration(),
            communication_model=_model(),
            carrier_hz=28.0e9,
            delta_f_hz=1.0e-6,
            symbol_period_s=1.0e-6,
            delay_bins=1,
            doppler_bins=1,
            dd_support_threshold=0.0,
            use_swerling=True,
        )
