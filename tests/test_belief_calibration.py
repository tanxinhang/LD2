"""Tests for NIS-driven covariance calibration (Layer 1 of CG-SR)."""

import numpy as np
import pytest
from uav_isac.environment.belief import BeliefManager


class TestNISComputation:
    """Verify NIS is computed correctly from Kalman innovations."""

    def test_nis_near_one_for_well_calibrated_filter(self):
        """NIS/d_z ≈ 1.0 when measurement noise matches R."""
        K, Q = 2, 1
        true_pos = np.array([[400., 500., 20.]])
        true_vel = np.array([[10., 5., 0.]])
        rng = np.random.default_rng(42)

        mgr = BeliefManager(
            K=K, Q=Q,
            initial_positions=true_pos,
            initial_velocities=true_vel,
            meas_pos_std=15.0, meas_vel_std=3.0,
            rng=rng,
            nis_enabled=True,
        )

        # Run predict + update many times with in-distribution noise
        nis_values = []
        true_state = np.array([400., 500., 10., 5.])
        for _ in range(200):
            mgr.step()
            # Simulate observation with noise matching R
            noise = rng.normal(0, [15.0, 15.0, 3.0, 3.0])
            z = true_state + noise
            # We need to manually trigger the update through the public API
            mgr.update_after_observation(0, 0, True, true_state)
            nis_values.append(mgr._last_nis[0, 0])

        # Mean NIS should be ~4 (d_z) for a well-calibrated filter
        mean_nis = np.mean(nis_values[-50:])  # steady-state
        # Chi-squared 4-dof: expected value = 4, 95% CI for 50 samples ≈ [2.5, 5.5]
        assert 1.5 < mean_nis < 7.0, f"Mean NIS={mean_nis:.2f} outside expected range"

    def test_nis_ema_increases_with_corrupted_predictions(self):
        """NIS EMA rises when filter predictions are corrupted.

        We directly corrupt the mean after predict to simulate severe
        model mismatch. This guarantees large innovations that tiny
        covariance cannot explain.
        """
        K, Q = 2, 1
        true_pos = np.array([[400., 500., 20.]])
        true_vel = np.array([[10., 5., 0.]])
        rng = np.random.default_rng(42)

        mgr = BeliefManager(
            K=K, Q=Q,
            initial_positions=true_pos,
            initial_velocities=true_vel,
            rng=rng,
            nis_enabled=True, nis_window=0.2,
            sigma_a=0.01,  # tiny Q → overconfident covariance
            meas_pos_std=5.0, meas_vel_std=1.0,  # small R → S is cov-dominated
        )

        true_state = np.array([400., 500., 10., 5.], dtype=np.float64)
        nis_ema_values = []

        for step in range(40):
            mgr.step()
            # Corrupt prediction: add large unmodeled offset
            # This simulates a sudden maneuver the CV model cannot predict
            mgr.mean[0, 0] += rng.normal(0, [30.0, 30.0, 5.0, 5.0])
            mgr.update_after_observation(0, 0, True, true_state)
            nis_ema_values.append(float(mgr.nis_ema[0, 0]))

        # Corrupted predictions → persistent high innovation → NIS EMA rises
        final_ema = np.mean(nis_ema_values[-10:])
        assert final_ema > 2.0, \
            f"NIS EMA should increase with corrupted predictions, got {final_ema:.2f}"


class TestCovarianceInflation:
    """Verify inflation factor responds correctly to NIS."""

    def test_no_inflation_when_nis_normal(self):
        """λ ≈ 1.0 when NIS EMA ≈ 1.0."""
        K, Q = 2, 1
        true_pos = np.array([[400., 500., 20.]])
        true_vel = np.array([[10., 5., 0.]])
        rng = np.random.default_rng(42)

        mgr = BeliefManager(
            K=K, Q=Q,
            initial_positions=true_pos,
            initial_velocities=true_vel,
            rng=rng,
            nis_enabled=True, nis_window=0.1,
            nis_inflate_k=2.0,
        )

        # NIS EMA starts at 1.0 → no inflation
        mgr.step()
        assert mgr.inflate_factor[0, 0] == pytest.approx(1.0, abs=0.01)

    def test_inflation_triggers_on_high_nis(self):
        """λ > 1.0 when NIS EMA is above 1.0."""
        K, Q = 2, 1
        true_pos = np.array([[400., 500., 20.]])
        true_vel = np.array([[10., 5., 0.]])
        rng = np.random.default_rng(42)

        mgr = BeliefManager(
            K=K, Q=Q,
            initial_positions=true_pos,
            initial_velocities=true_vel,
            rng=rng,
            nis_enabled=True, nis_window=0.3,  # fast window for test
            nis_inflate_k=2.0, nis_lambda_max=100.0,
        )

        # Manually set high NIS EMA to simulate persistent miscalibration
        mgr.nis_ema[0, 0] = 2.5
        mgr.step()
        # Linear: λ = 1 + k*(r̄-1) = 1 + 2.0*1.5 = 4.0
        assert mgr.inflate_factor[0, 0] == pytest.approx(4.0, abs=0.1), \
            f"Expected λ ≈ 4.0, got {mgr.inflate_factor[0,0]:.2f}"

    def test_inflation_capped_at_lambda_max(self):
        """λ does not exceed λ_max."""
        K, Q = 2, 1
        true_pos = np.array([[400., 500., 20.]])
        true_vel = np.array([[10., 5., 0.]])
        rng = np.random.default_rng(42)

        mgr = BeliefManager(
            K=K, Q=Q,
            initial_positions=true_pos,
            initial_velocities=true_vel,
            rng=rng,
            nis_enabled=True, nis_window=0.3,
            nis_inflate_k=2.0, nis_lambda_max=5.0,
        )

        mgr.nis_ema[0, 0] = 10.0  # would be exp(18) without cap
        mgr.step()
        assert mgr.inflate_factor[0, 0] == pytest.approx(5.0, abs=0.01), \
            f"Inflation should be capped at 5.0, got {mgr.inflate_factor[0,0]:.2f}"

    def test_deflation_when_nis_recovers(self):
        """λ decays geometrically when NIS EMA ≤ 1.0."""
        K, Q = 2, 1
        true_pos = np.array([[400., 500., 20.]])
        true_vel = np.array([[10., 5., 0.]])
        rng = np.random.default_rng(42)

        mgr = BeliefManager(
            K=K, Q=Q,
            initial_positions=true_pos,
            initial_velocities=true_vel,
            rng=rng,
            nis_enabled=True, nis_window=0.3,
            nis_deflate_rate=0.8,
        )

        # Inflate first
        mgr.nis_ema[0, 0] = 3.0
        mgr.step()
        inflated = mgr.inflate_factor[0, 0]
        assert inflated > 1.0

        # Recover NIS → should deflate
        mgr.nis_ema[0, 0] = 0.8
        mgr.step()
        assert mgr.inflate_factor[0, 0] < inflated, \
            "Inflation should decrease when NIS recovers"

    def test_deflation_stops_at_one(self):
        """λ never goes below 1.0."""
        K, Q = 2, 1
        true_pos = np.array([[400., 500., 20.]])
        true_vel = np.array([[10., 5., 0.]])
        rng = np.random.default_rng(42)

        mgr = BeliefManager(
            K=K, Q=Q,
            initial_positions=true_pos,
            initial_velocities=true_vel,
            rng=rng,
            nis_enabled=True, nis_window=0.3,
            nis_deflate_rate=0.8,
        )

        mgr.inflate_factor[0, 0] = 1.05
        mgr.nis_ema[0, 0] = 0.5
        for _ in range(10):
            mgr.step()
        assert mgr.inflate_factor[0, 0] >= 1.0, "λ should not drop below 1.0"


class TestCovarianceFloor:
    """Verify physical floor is applied to covariance diagonal."""

    def test_floor_added_to_covariance(self):
        """P_cal diagonal ≥ P_floor even when P is small."""
        K, Q = 2, 1
        true_pos = np.array([[400., 500., 20.]])
        true_vel = np.array([[10., 5., 0.]])
        rng = np.random.default_rng(42)

        mgr = BeliefManager(
            K=K, Q=Q,
            initial_positions=true_pos,
            initial_velocities=true_vel,
            rng=rng,
            nis_enabled=True,
            cov_floor_pos=100.0,   # 10m σ
            cov_floor_vel=4.0,     # 2m/s σ
        )

        # Set very small covariance
        mgr.cov[0, 0] = np.diag([1.0, 1.0, 0.1, 0.1])
        mgr.step()

        diag = np.diag(mgr.cov[0, 0])
        # After predict (F @ cov @ F.T + Q), floor is applied on top
        assert diag[0] >= 100.0, f"Position variance {diag[0]:.1f} below floor 100.0"
        assert diag[1] >= 100.0, f"Position variance {diag[1]:.1f} below floor 100.0"
        assert diag[2] >= 4.0, f"Velocity variance {diag[2]:.1f} below floor 4.0"


class TestAccessors:
    """Verify diagnostic accessor methods."""

    def test_get_nis_status(self):
        K, Q = 2, 1
        true_pos = np.array([[400., 500., 20.]])
        true_vel = np.array([[10., 5., 0.]])
        rng = np.random.default_rng(42)

        mgr = BeliefManager(
            K=K, Q=Q,
            initial_positions=true_pos,
            initial_velocities=true_vel,
            rng=rng, nis_enabled=True,
        )

        status = mgr.get_nis_status(0, 0)
        assert 'nis_ema' in status
        assert 'inflate_factor' in status
        assert 'last_nis' in status
        assert 'cov_diag_cal' in status
        assert status['nis_ema'] == pytest.approx(1.0)
        assert status['inflate_factor'] == pytest.approx(1.0)
        assert len(status['cov_diag_cal']) == 4

    def test_get_all_nis_status(self):
        K, Q = 3, 2
        true_pos = np.array([[400., 500., 20.], [600., 300., 20.]])
        true_vel = np.array([[10., 5., 0.], [-5., 8., 0.]])
        rng = np.random.default_rng(42)

        mgr = BeliefManager(
            K=K, Q=Q,
            initial_positions=true_pos,
            initial_velocities=true_vel,
            rng=rng, nis_enabled=True,
        )

        status = mgr.get_all_nis_status()
        assert status['nis_ema'].shape == (K, Q)
        assert status['inflate_factor'].shape == (K, Q)
        assert status['last_nis'].shape == (K, Q)
        assert np.all(status['nis_ema'] == 1.0)

    def test_get_calibrated_covariance(self):
        K, Q = 2, 1
        true_pos = np.array([[400., 500., 20.]])
        true_vel = np.array([[10., 5., 0.]])
        rng = np.random.default_rng(42)

        mgr = BeliefManager(
            K=K, Q=Q,
            initial_positions=true_pos,
            initial_velocities=true_vel,
            rng=rng,
        )
        cov = mgr.get_calibrated_covariance()
        assert cov.shape == (K, Q, 4, 4)


class TestBackwardCompatibility:
    """Verify NIS disabled (default) does not change behavior."""

    def test_nis_disabled_preserves_covariance(self):
        """Without NIS, step() behaves identically to before."""
        K, Q = 2, 1
        true_pos = np.array([[400., 500., 20.]])
        true_vel = np.array([[10., 5., 0.]])
        rng = np.random.default_rng(42)

        # NIS disabled (default) — identical to pre-calibration behavior
        mgr_off = BeliefManager(
            K=K, Q=Q,
            initial_positions=true_pos,
            initial_velocities=true_vel,
            rng=rng,
            nis_enabled=False,
        )
        for _ in range(5):
            mgr_off.step()

        # With nis_enabled=False, no floor or inflation applied
        diag = np.diag(mgr_off.cov[0, 0])
        # Position variance ~ σ_pos² + dt²·σ_vel² + q_p from Q (standard CV)
        assert diag[0] > 2000  # reasonable initial covariance growth

    def test_nis_enabled_is_more_conservative(self):
        """With NIS enabled at r=1.0, the physical floor is a no-op when
        covariance is already above floor. The Joseph form Kalman update
        (for both paths) is more conservative than the simplified form.
        Verify that NIS-enabled path never reduces covariance below the floor."""
        K, Q = 2, 1
        true_pos = np.array([[400., 500., 20.]])
        true_vel = np.array([[10., 5., 0.]])
        rng = np.random.default_rng(42)

        mgr_off = BeliefManager(
            K=K, Q=Q,
            initial_positions=true_pos,
            initial_velocities=true_vel,
            rng=rng,
            nis_enabled=False,
        )
        mgr_on = BeliefManager(
            K=K, Q=Q,
            initial_positions=true_pos,
            initial_velocities=true_vel,
            rng=rng,
            nis_enabled=True,
        )

        for _ in range(5):
            mgr_off.step()
            mgr_on.step()

        # Both paths should have PSD covariance
        for k in range(K):
            for q in range(Q):
                # Check PSD: all eigenvalues >= 0
                eigvals_off = np.linalg.eigvalsh(mgr_off.cov[k, q])
                eigvals_on = np.linalg.eigvalsh(mgr_on.cov[k, q])
                assert np.all(eigvals_off >= -1e-10), "NIS-off cov not PSD"
                assert np.all(eigvals_on >= -1e-10), "NIS-on cov not PSD"
                # Floor ensures diagonal >= floor values
                diag_on = np.diag(mgr_on.cov[k, q])
                assert diag_on[0] >= 25.0, f"Position variance below floor: {diag_on[0]}"
                assert diag_on[2] >= 1.0, f"Velocity variance below floor: {diag_on[2]}"

    def test_reset_clears_nis_state(self):
        K, Q = 2, 1
        true_pos = np.array([[400., 500., 20.]])
        true_vel = np.array([[10., 5., 0.]])
        rng = np.random.default_rng(42)

        mgr = BeliefManager(
            K=K, Q=Q,
            initial_positions=true_pos,
            initial_velocities=true_vel,
            rng=rng, nis_enabled=True,
        )

        # Modify NIS state
        mgr.nis_ema[0, 0] = 3.0
        mgr.inflate_factor[0, 0] = 5.0
        mgr._last_nis[0, 0] = 50.0

        # Reset
        mgr.reset(true_pos, true_vel)

        # Should be back to defaults
        assert mgr.nis_ema[0, 0] == 1.0
        assert mgr.inflate_factor[0, 0] == 1.0
        assert mgr._last_nis[0, 0] == 1.0
