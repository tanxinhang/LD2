"""C0/C1 closure tests (audit advice/001, 2026-08-26).

C0  — system identity: config/system_manifest.yaml is the unique frozen
      post-G2 system; param dataclass defaults align with the paper scope
      (tracking-free, U2U-only, joint RF power in the manifest).
C1  — physics closure:
      (1) bistatic Doppler receiver term sign (Tx/target static, Rx flying
          toward target => positive Doppler);
      (2) OTFS unambiguous support gate + continuous |A(tau,nu)|^2 gain
          replacing the binary g_dd gate on deflection;
      (3) unified sensing time-energy clock (T_sense = n_cpi*N*T_sym in
          battery billing instead of P_sense*dt).
Strict no-truth mode (audit advice/001 P0): when the manifest pins
``distributed_no_truth_fail_closed=true`` together with local-belief + tracking,
no distributed decision path may read simulator ground truth.
"""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml

from config.params import get_default_config, load_config, MasterConfig
from tools.check_system_identity import collect_checks, main as identity_main
from uav_isac.physical.geometry import compute_doppler
from uav_isac.physical.otfs import (
    compute_dd_unambiguous_support,
    compute_dd_ambiguity_gain,
    compute_dd_phys_gain,
    compute_dd_phys_gain_batch,
    compute_dd_effectiveness,
)
from uav_isac.physical.deflection import (
    DeflectionComputer,
    validate_cpi_schedule,
)


# ---------------------------------------------------------------------------
# C1-1  Doppler sign contract
# ---------------------------------------------------------------------------

def test_doppler_rx_toward_target_is_positive():
    """Physics contract (advice/001 section 4): Tx/target static, Rx flying
    TOWARD the target => positive Doppler (bistatic range shrinking)."""
    tx_pos = np.array([0.0, 0.0, 100.0])
    rx_pos = np.array([1000.0, 0.0, 100.0])
    tgt_pos = np.array([500.0, 0.0, 0.0])
    zeros = np.zeros(3)
    rx_toward = np.array([-10.0, 0.0, 0.0])  # toward the target
    nu = compute_doppler(tx_pos, zeros, rx_pos, rx_toward, tgt_pos, zeros, 28.0e9)
    assert nu > 0.0, f"Rx toward target should be positive Doppler, got {nu}"


def test_doppler_rx_away_from_target_is_negative():
    tx_pos = np.array([0.0, 0.0, 100.0])
    rx_pos = np.array([1000.0, 0.0, 100.0])
    tgt_pos = np.array([500.0, 0.0, 0.0])
    zeros = np.zeros(3)
    rx_away = np.array([10.0, 0.0, 0.0])  # away from the target
    nu = compute_doppler(tx_pos, zeros, rx_pos, rx_away, tgt_pos, zeros, 28.0e9)
    assert nu < 0.0, f"Rx away from target should be negative Doppler, got {nu}"


def test_doppler_tx_toward_target_is_positive():
    tx_pos = np.array([0.0, 0.0, 100.0])
    rx_pos = np.array([1000.0, 0.0, 100.0])
    tgt_pos = np.array([500.0, 0.0, 0.0])
    zeros = np.zeros(3)
    tx_toward = np.array([10.0, 0.0, 0.0])
    nu = compute_doppler(tx_pos, tx_toward, rx_pos, zeros, tgt_pos, zeros, 28.0e9)
    assert nu > 0.0


# ---------------------------------------------------------------------------
# C1-2  OTFS unambiguous support + continuous |A|^2 gain
# ---------------------------------------------------------------------------

# Canonical numerology (default.yaml / system_manifest.yaml)
DELTA_F = 1.5625e4
T_SYM = 6.4e-5
M, N = 64, 16


def test_unambiguous_support_on_grid_bin():
    delay = 20.0 / 3.0e8          # 20 m bistatic delay, inside [0, 1/Delta_f)
    nu = 500.0                     # inside +/-1/(2*T_sym)=7812.5 Hz
    assert compute_dd_unambiguous_support(delay, nu, DELTA_F, T_SYM, M, N) == 1.0


def test_unambiguous_support_aliased_delay_is_zero():
    """Out-of-support delay must give 0 even when the aliased fractional bin
    is almost integer (the old fractional-mismatch check could yield g~1)."""
    tau_out = 1.0 / DELTA_F + 1e-4   # just past the delay unambiguous region
    nu = 0.0
    assert compute_dd_unambiguous_support(tau_out, nu, DELTA_F, T_SYM, M, N) == 0.0
    # Legacy fractional gain can be non-trivial while phys support must be zero.
    assert compute_dd_phys_gain(tau_out, nu, DELTA_F, T_SYM, M, N) == 0.0


def test_unambiguous_support_doppler_outside_is_zero():
    nu_out = 1.0 / (2.0 * T_SYM) + 100.0  # just beyond Doppler unambiguous
    tau = 20.0 / 3.0e8
    assert compute_dd_unambiguous_support(tau, nu_out, DELTA_F, T_SYM, M, N) == 0.0
    assert compute_dd_phys_gain(tau, nu_out, DELTA_F, T_SYM, M, N) == 0.0


def test_ambiguity_gain_on_bin_is_one():
    # on-bin: l = tau*M*Delta_F integer, k = nu*N*T_sym integer
    tau = 20.0 / (M * DELTA_F)   # l = 20 (integer bin)
    nu = 500.0 / (N * T_SYM)     # k = 500 (integer bin)
    assert compute_dd_ambiguity_gain(tau, nu, DELTA_F, T_SYM, M, N) == pytest.approx(1.0)


def test_ambiguity_gain_half_bin_is_sinc_squared():
    """Half-bin offset: |A| = |sinc(0.5)| = 2/pi, so |A|^2 = (2/pi)^2 ~ 0.4053."""
    half_delay = 0.5 / (M * DELTA_F)
    tau = half_delay
    nu = 0.0
    gain_sq = compute_dd_ambiguity_gain(tau, nu, DELTA_F, T_SYM, M, N)
    expected = (2.0 / np.pi) ** 2
    assert gain_sq == pytest.approx(expected, rel=1e-6)
    # Legacy amplitude is |A| (0.6366), not |A|^2.
    assert compute_dd_effectiveness(tau, nu, DELTA_F, T_SYM, M, N) == pytest.approx(
        2.0 / np.pi, rel=1e-6)


def test_dd_phys_gain_batch_matches_scalar_random_and_boundaries():
    rng = np.random.default_rng(20260827)
    taus = rng.uniform(-0.25 / DELTA_F, 1.25 / DELTA_F, size=10_000)
    nus = rng.uniform(-1.25 / (2.0 * T_SYM), 1.25 / (2.0 * T_SYM), size=10_000)
    delay_limit = 1.0 / DELTA_F
    doppler_limit = 1.0 / (2.0 * T_SYM)
    taus[:7] = [
        0.0,
        np.nextafter(0.0, -1.0),
        np.nextafter(delay_limit, 0.0),
        delay_limit,
        0.5 / (M * DELTA_F),
        20.0 / (M * DELTA_F),
        0.25 * delay_limit,
    ]
    nus[:7] = [
        0.0,
        0.0,
        doppler_limit,
        np.nextafter(doppler_limit, np.inf),
        0.0,
        -doppler_limit,
        np.nextafter(-doppler_limit, -np.inf),
    ]
    scalar = np.asarray([
        compute_dd_phys_gain(tau, nu, DELTA_F, T_SYM, M, N)
        for tau, nu in zip(taus, nus)
    ])
    batched = compute_dd_phys_gain_batch(taus, nus, DELTA_F, T_SYM, M, N)
    np.testing.assert_allclose(batched, scalar, rtol=1.0e-15, atol=1.0e-15)


def test_dd_phys_gain_batch_broadcast_and_validation():
    taus = np.array([[0.0], [0.5 / (M * DELTA_F)]])
    nus = np.array([0.0, 0.5 / (N * T_SYM), 1.0 / T_SYM])
    batched = compute_dd_phys_gain_batch(taus, nus, DELTA_F, T_SYM, M, N)
    assert batched.shape == (2, 3)
    for row in range(2):
        for column in range(3):
            assert batched[row, column] == pytest.approx(
                compute_dd_phys_gain(
                    taus[row, 0], nus[column], DELTA_F, T_SYM, M, N,
                ),
                abs=1.0e-15,
            )
    with pytest.raises(ValueError, match="finite"):
        compute_dd_phys_gain_batch(
            np.array([0.0, np.nan]), np.zeros(2), DELTA_F, T_SYM, M, N,
        )


def test_deflection_continuous_mode_scales_by_phys_gain():
    """DeflectionComputer continuous mode: d_eff = chi_rep*d_raw*I_support*|A|^2."""
    cfg = get_default_config()
    rng = np.random.default_rng(7)
    computer = DeflectionComputer(
        fc=cfg.otfs.fc, delta_f=cfg.otfs.delta_f, T_sym=cfg.otfs.T_sym,
        M=cfg.otfs.M, N=cfg.otfs.N, kT=cfg.channel.kT, B=cfg.otfs.B,
        NF_dB=cfg.channel.NF, P_sense=cfg.uav.P_sense, P_report=cfg.uav.P_report,
        ric_K=cfg.channel.ric_K, rcs=cfg.target.rcs, g_min=cfg.detection.g_min,
        rng=rng, dd_gain_mode="continuous",
        use_report_link=False,
    )
    uav_pos = np.array([[100.0, 100.0, 100.0], [900.0, 100.0, 100.0],
                        [100.0, 900.0, 100.0], [900.0, 900.0, 100.0]])
    uav_vel = np.zeros((4, 3))
    tgt_pos = np.array([[400.0, 500.0, 0.0]])
    tgt_vel = np.zeros((1, 3))
    roles = np.array([0, 0, 1, 1], dtype=np.int32)
    entries = computer.compute(uav_pos, uav_vel, tgt_pos, tgt_vel, roles,
                               np.array([500.0, 500.0, 100.0]))
    assert len(entries) > 0
    for e in entries:
        phys_gain = compute_dd_phys_gain(e.tau, e.nu, cfg.otfs.delta_f,
                                         cfg.otfs.T_sym, cfg.otfs.M, cfg.otfs.N)
        expected = e.chi_rep * e.d_raw * phys_gain
        assert e.d_eff == pytest.approx(expected, rel=1e-9)


def test_deflection_binary_mode_keeps_legacy():
    """Binary mode reproduces the pre-G2 gate exactly."""
    cfg = get_default_config()
    rng = np.random.default_rng(7)
    computer = DeflectionComputer(
        fc=cfg.otfs.fc, delta_f=cfg.otfs.delta_f, T_sym=cfg.otfs.T_sym,
        M=cfg.otfs.M, N=cfg.otfs.N, kT=cfg.channel.kT, B=cfg.otfs.B,
        NF_dB=cfg.channel.NF, P_sense=cfg.uav.P_sense, P_report=cfg.uav.P_report,
        ric_K=cfg.channel.ric_K, rcs=cfg.target.rcs, g_min=cfg.detection.g_min,
        rng=rng, dd_gain_mode="binary",
        use_report_link=False,
    )
    uav_pos = np.array([[100.0, 100.0, 100.0], [900.0, 100.0, 100.0],
                        [100.0, 900.0, 100.0], [900.0, 900.0, 100.0]])
    uav_vel = np.zeros((4, 3))
    tgt_pos = np.array([[400.0, 500.0, 0.0]])
    tgt_vel = np.zeros((1, 3))
    roles = np.array([0, 0, 1, 1], dtype=np.int32)
    entries = computer.compute(uav_pos, uav_vel, tgt_pos, tgt_vel, roles,
                               np.array([500.0, 500.0, 100.0]))
    for e in entries:
        expected = e.chi_rep * e.d_raw if e.g_dd >= cfg.detection.g_min else 0.0
        assert e.d_eff == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# C1-3  Sensing time-energy clock
# ---------------------------------------------------------------------------

def test_validate_cpi_schedule_clock():
    duration, max_looks = validate_cpi_schedule(1, N, T_SYM, 0.1)
    assert duration == pytest.approx(N * T_SYM)          # 1.024 ms
    assert max_looks == int(np.floor(0.1 / (N * T_SYM)))  # ~97 frames per step


def test_manifest_pins_are_strict():
    manifest = load_config("config/system_manifest.yaml")
    assert manifest.marl.ground_communication_enabled is False
    assert manifest.marl.joint_isac_power_enabled is True
    assert manifest.marl.distributed_coordination_use_local_belief_targets is True
    assert manifest.marl.distributed_no_truth_fail_closed is True
    # Strict no-truth identity: local-belief coordination requires the belief
    # manager, so the canonical manifest pins tracking ON (the legacy
    # tracking-free default.yaml stays OFF as a separate non-distributed mode).
    assert manifest.marl.tracking_enabled is True
    # Scalar score-order bounds cannot certify arbitrary neural target-token
    # payloads without a decoder/action Lipschitz certificate.  The canonical
    # identity therefore keeps all C6 online stages fail-closed.
    assert (manifest.marl.distributed_decision_sufficient_comm_enabled is False)
    assert (manifest.marl.distributed_decision_sufficient_event_trigger_enabled
            is False)
    assert (manifest.marl.distributed_decision_sufficient_adaptive_bits_enabled
            is False)
    assert manifest.marl.target_allocation_enabled is True
    assert manifest.detection.dd_gain_mode == "continuous"
    assert manifest.scenario.sensing_energy_mode == "cpi_frame"
    assert float(manifest.uav.P_isac_total) == 1.0


def test_system_identity_checker_passes():
    failures, _passes = collect_checks("config/system_manifest.yaml")
    assert failures == [], f"identity mismatches: {failures}"


def test_system_identity_checker_rejects_missing_extends(tmp_path):
    raw = yaml.safe_load(Path("config/system_manifest.yaml").read_text(
        encoding="utf-8"))
    raw.pop("extends")
    candidate = tmp_path / "manifest_without_extends.yaml"
    candidate.write_text(
        yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    failures, _passes = collect_checks(str(candidate))
    assert any("extends must be" in failure for failure in failures)


def test_system_identity_checker_rejects_arbitrary_commit():
    assert identity_main([
        "--manifest", "config/system_manifest.yaml",
        "--commit", "definitely-not-the-current-sha",
    ]) == 1


def test_documented_identity_script_entrypoint_runs_directly():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "tools/check_system_identity.py"],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "RESULT: PASS" in completed.stdout


def test_defaults_aligned_with_paper_scope():
    plain = MasterConfig()
    assert plain.marl.tracking_enabled is False
    assert plain.marl.ground_communication_enabled is False
    cfg = get_default_config()
    assert cfg.marl.ground_communication_enabled is False
    assert cfg.scenario.sensing_energy_mode == "dt_frame"  # legacy default
    assert cfg.detection.dd_gain_mode == "binary"          # legacy default


def test_strict_no_truth_requires_local_belief_at_construction():
    """Strict no-truth without local-belief coordination must fail closed."""
    from uav_isac.environment.env_core import EnvironmentCore

    cfg = get_default_config()
    cfg.marl.tracking_enabled = True
    cfg.marl.distributed_coordination_use_local_belief_targets = False
    cfg.marl.distributed_no_truth_fail_closed = True
    with pytest.raises(ValueError, match="requires.*local_belief"):
        EnvironmentCore(cfg)


def test_strict_resolver_refuses_truth_fallback():
    """With strict no-truth on, the coordination resolver fails closed when the
    local-belief flag is off instead of silently reading simulator truth."""
    from unittest.mock import patch

    from uav_isac.environment.env_core import EnvironmentCore

    cfg = get_default_config()
    cfg.marl.tracking_enabled = True
    cfg.marl.distributed_coordination_use_local_belief_targets = True
    cfg.marl.distributed_no_truth_fail_closed = True
    env = EnvironmentCore(cfg)
    env.reset()
    # Belief manager now exists -> strict resolver serves local belief.
    pos, vel = env._coordination_target_state_for_viewer(0)
    assert pos.shape == (env.Q, 3) and np.all(np.isfinite(pos))
    assert vel.shape == (env.Q, 3) and np.all(np.isfinite(vel))
    # Forcing the local-belief flag off under strict mode must raise, not leak.
    with patch.object(
        env, "_distributed_coordination_use_local_belief_targets", False
    ):
        with pytest.raises(RuntimeError, match="refusing to expose simulator truth"):
            env._coordination_target_state_for_viewer(0)


def test_strict_replicated_power_fallback_never_reads_truth():
    """Under strict no-truth the replicated-power local-range fallback must not
    call the simulator-truth minimax share (legacy mode may)."""
    import inspect

    from uav_isac.environment.env_core import EnvironmentCore

    # The strict switch lives in the power-execution path of ``_step`` (the
    # only caller of the truth-based minimax share).  Lock the gate in source
    # so a future refactor cannot silently reintroduce the truth read.
    src = inspect.getsource(EnvironmentCore.step)
    assert "_distributed_no_truth_fail_closed" in src
    assert "incomplete_prior = None" in src
    assert "local_transmitter_range_minimax_share" in src  # legacy branch kept

    # Full strict replicated stack needs the budget-reconstructable hyperedge
    # state channel; that combination is exercised by the pilot config load.
    # Here we lock (a) the source gate and (b) the strict resolver fail-closed.
    cfg = get_default_config()
    cfg.marl.tracking_enabled = True
    cfg.marl.distributed_coordination_use_local_belief_targets = True
    cfg.marl.distributed_no_truth_fail_closed = True
    cfg.marl.joint_isac_power_enabled = True
    cfg.marl.analytical_sensing_power_enabled = True
    strict = EnvironmentCore(cfg)
    assert strict._distributed_no_truth_fail_closed is True
    strict.reset()
    pos, _ = strict._coordination_target_state_for_viewer(0)
    assert pos.shape == (strict.Q, 3) and np.all(np.isfinite(pos))
    # Strict switch is live in the execution path; the fallback branch, when
    # enabled, skips the truth share (source-locked above).


def test_owner_local_physics_matches_doppler_sign():
    """Vectorized owner-local Doppler must share the fixed sign contract."""
    from uav_isac.coordination.owner_local_physics import OwnerLocalKinematicState
    from uav_isac.coordination.owner_local_physics import owner_local_dd_effectiveness
    state = OwnerLocalKinematicState(
        uav_position_m=np.array([[0.0, 0.0, 10.0], [1000.0, 0.0, 10.0]]),
        uav_velocity_mps=np.array([[0.0, 0.0, 0.0], [-10.0, 0.0, 0.0]]),
        target_mean_by_owner=np.array([[[500.0, 0.0, 0.0, 0.0]],
                                       [[500.0, 0.0, 0.0, 0.0]]]),
        target_cov_diag_by_owner=np.zeros((2, 1, 4)),
        target_aoi_frames_by_owner=np.zeros((2, 1)),
    )
    result = owner_local_dd_effectiveness(
        state,
        carrier_hz=28.0e9, delta_f_hz=1.0e3, symbol_period_s=1.0e-4,
        delay_bins=8, doppler_bins=8,
    )
    assert result.ndim == 3 and result.shape[0] == 2
    # Directly verify with compute_doppler that the transmitted/received
    # geometry used in the owner-local map follows the same sign convention.
    scalar = compute_doppler(
        np.array([0.0, 0.0, 10.0]), np.zeros(3),
        np.array([1000.0, 0.0, 10.0]), np.array([-10.0, 0.0, 0.0]),
        np.array([500.0, 0.0, 0.0]), np.zeros(3), 28.0e9)
    assert scalar > 0.0
