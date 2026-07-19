"""Tests for TrustManager: disagreement-gated belief fusion (Layer 2)."""

import numpy as np
import pytest
from uav_isac.environment.trust_manager import TrustManager


class TestGateComputation:
    """Verify trust scores and gating logic."""

    @pytest.fixture
    def sample_data(self):
        """Create sample belief data for K=3, Q=2."""
        K, Q = 3, 2
        rng = np.random.default_rng(42)
        mean = np.zeros((K, Q, 4))
        cov = np.zeros((K, Q, 4, 4))
        aoi = np.zeros((K, Q))
        nis_ema = np.ones((K, Q))

        for k in range(K):
            for q in range(Q):
                mean[k, q] = [400 + k * 50, 500 + q * 50, 10, 5]
                cov[k, q] = np.diag([100., 100., 4., 4.])
                aoi[k, q] = k + q
                nis_ema[k, q] = 1.0
        return K, Q, mean, cov, aoi, nis_ema

    def test_initial_trust_is_one(self, sample_data):
        K, Q, mean, cov, aoi, nis_ema = sample_data
        tm = TrustManager(K=K, Q=Q, ema_rho=1.0)  # no smoothing
        trust, raw_w = tm.compute_gate_weights(mean, cov, aoi, nis_ema)

        # With well-calibrated filters and similar beliefs, trust should be high
        for k in range(K):
            for j in range(K):
                if j != k:
                    for q in range(Q):
                        assert 0.0 <= trust[k, j, q] <= 1.0

    def test_self_trust_skipped(self, sample_data):
        K, Q, mean, cov, aoi, nis_ema = sample_data
        tm = TrustManager(K=K, Q=Q, ema_rho=1.0)
        trust, raw_w = tm.compute_gate_weights(mean, cov, aoi, nis_ema)

        # Self-trust (diagonal) is skipped — never used in fusion
        # Remains 0 from zero-init (not 1.0 since it's meaningless)
        for k in range(K):
            for q in range(Q):
                assert trust[k, k, q] == 0.0, \
                    "Self-trust should remain at 0 (never computed)"

    def test_large_disagreement_reduces_trust(self, sample_data):
        K, Q, mean, cov, aoi, nis_ema = sample_data
        tm = TrustManager(K=K, Q=Q, ema_rho=1.0)

        # Make agent 1's belief very different from agent 0's
        mean[1, 0] = [800., 800., 10., 5.]  # 400m away → huge disagreement
        trust, raw_w = tm.compute_gate_weights(mean, cov, aoi, nis_ema)

        # Trust from agent 0 → agent 1 should be very low
        assert trust[0, 1, 0] < 0.1, \
            f"Trust should be low with large disagreement, got {trust[0,1,0]:.4f}"

    def test_high_nis_reduces_trust(self, sample_data):
        K, Q, mean, cov, aoi, nis_ema = sample_data
        tm = TrustManager(K=K, Q=Q, ema_rho=1.0)

        # Make agent 1's NIS very high (overconfident filter)
        nis_ema[1, 0] = 5.0
        trust, raw_w = tm.compute_gate_weights(mean, cov, aoi, nis_ema)

        # Trust from agent 0 → agent 1 should be reduced by g_nis penalty
        assert trust[0, 1, 0] < 0.5, \
            f"Trust should be reduced by NIS penalty, got {trust[0,1,0]:.4f}"

    def test_old_aoi_reduces_trust(self, sample_data):
        K, Q, mean, cov, aoi, nis_ema = sample_data
        tm = TrustManager(K=K, Q=Q, ema_rho=1.0, aoi_max=50.0)

        # Make agent 1's AoI very old
        aoi[1, 0] = 50.0  # at threshold → g_age = 0
        trust, raw_w = tm.compute_gate_weights(mean, cov, aoi, nis_ema)

        # Trust from agent 0 → agent 1 should be zero (g_age=0)
        assert trust[0, 1, 0] < 0.01, \
            f"Trust should be ~0 at max AoI, got {trust[0,1,0]:.4f}"

    def test_trust_bounded_zero_one(self, sample_data):
        K, Q, mean, cov, aoi, nis_ema = sample_data
        tm = TrustManager(K=K, Q=Q, ema_rho=1.0)

        # Extreme values
        mean[1, 0] = [1e6, 1e6, 1000., 1000.]  # huge disagreement
        nis_ema[1, 0] = 100.0  # terrible NIS
        aoi[1, 0] = 1000.0  # way past max

        trust, raw_w = tm.compute_gate_weights(mean, cov, aoi, nis_ema)

        for k in range(K):
            for j in range(K):
                for q in range(Q):
                    assert 0.0 <= trust[k, j, q] <= 1.0, \
                        f"trust[{k},{j},{q}] = {trust[k,j,q]:.4f} out of [0,1]"


class TestWeightGating:
    """Verify weight bounds and renormalization."""

    def test_weight_capped(self):
        K, Q = 3, 2
        tm = TrustManager(K=K, Q=Q, weight_max=0.6)

        # Create attention weights where one neighbor dominates
        attn_w = np.zeros((1, Q, 2))  # B=1, Q=2, N=2 (K-1)
        attn_w[0, 0, 0] = 0.95  # neighbor 0 dominates
        attn_w[0, 0, 1] = 0.05

        # High trust for both
        trust = np.ones((1, Q, 2))

        gated = tm.apply_gate_to_weights(attn_w, trust)

        # No weight should exceed weight_max
        assert np.all(gated <= tm.weight_max + 1e-10), \
            f"Weights exceed cap: max={gated.max():.4f}"

    def test_zero_trust_kills_weight(self):
        K, Q = 3, 2
        tm = TrustManager(K=K, Q=Q, weight_max=0.6)

        attn_w = np.ones((1, Q, 2)) * 0.5
        trust = np.zeros((1, Q, 2))  # no trust at all

        gated = tm.apply_gate_to_weights(attn_w, trust)

        assert np.all(gated == 0.0), \
            "Zero trust should produce zero weights"

    def test_renormalization_preserves_local_budget(self):
        K, Q = 3, 2
        tm = TrustManager(K=K, Q=Q, weight_max=1.0, local_weight_min=0.3)

        attn_w = np.ones((1, Q, 2)) * 0.5
        trust = np.ones((1, Q, 2))

        gated = tm.apply_gate_to_weights(attn_w, trust)

        # Sum of gated weights should be <= 1 - local_weight_min
        assert np.all(gated.sum(axis=-1) <= (1.0 - tm.local_weight_min) + 1e-10), \
            f"Gated weights exceed budget: sum={gated.sum(axis=-1)}"


class TestQuarantine:
    """Verify quarantine mechanism."""

    def test_quarantine_blocks_trust(self):
        K, Q = 3, 2
        mean = np.zeros((K, Q, 4))
        cov = np.zeros((K, Q, 4, 4))
        for k in range(K):
            for q in range(Q):
                cov[k, q] = np.diag([100., 100., 4., 4.])
        aoi = np.zeros((K, Q))
        nis_ema = np.ones((K, Q))

        tm = TrustManager(K=K, Q=Q, ema_rho=1.0,
                          quarantine_nis_ratio=1.5,
                          quarantine_duration=10)

        # Set up NIS mismatch to trigger quarantine
        tm.nis_fused_ema[0, 1, 0] = 5.0
        tm.nis_local_ema[0, 0] = 1.0  # local is fine
        tm.check_quarantine(0, 1, 0)  # should trigger (5.0 > 1.5*1.0)

        assert tm.quarantine_counter[0, 1, 0] == 10

        # Gated weights should kill quarantined neighbor
        trust, raw_w = tm.compute_gate_weights(mean, cov, aoi, nis_ema)
        assert raw_w[0, 1, 0] == 0.0, "Quarantined neighbor should have zero weight"

    def test_quarantine_decays(self):
        K, Q = 3, 2
        tm = TrustManager(K=K, Q=Q, quarantine_duration=5)

        tm.quarantine_counter[0, 1, 0] = 5
        for _ in range(5):
            tm.decay_quarantine()
        assert tm.quarantine_counter[0, 1, 0] == 0

    def test_quarantine_does_not_go_negative(self):
        K, Q = 3, 2
        tm = TrustManager(K=K, Q=Q)
        tm.quarantine_counter[0, 1, 0] = 1
        for _ in range(5):
            tm.decay_quarantine()
        assert tm.quarantine_counter[0, 1, 0] == 0


class TestTrustFeedback:
    """Verify post-measurement trust updates."""

    def test_good_fused_prediction_increases_trust(self):
        K, Q = 3, 2
        tm = TrustManager(K=K, Q=Q, ema_rho=1.0)

        tm.trust_score[0, 1, 0] = 0.5
        # Good fused prediction: NIS_fused = 2 (near expected 4)
        tm.update_trust_from_nis(0, 1, 0, nis_fused=2.0, nis_local=4.0)

        # exp(-2/8) ≈ 0.78 → trust should increase from 0.5
        assert tm.trust_score[0, 1, 0] > 0.7, \
            f"Trust should increase with good fused prediction, got {tm.trust_score[0,1,0]:.4f}"

    def test_bad_fused_prediction_decreases_trust(self):
        K, Q = 3, 2
        tm = TrustManager(K=K, Q=Q, ema_rho=1.0)

        tm.trust_score[0, 1, 0] = 0.9
        # Bad fused prediction: NIS_fused = 40 (10x expected)
        tm.update_trust_from_nis(0, 1, 0, nis_fused=40.0, nis_local=4.0)

        # exp(-40/8) ≈ 0.007 → trust should decrease
        assert tm.trust_score[0, 1, 0] < 0.1, \
            f"Trust should decrease with bad fused prediction, got {tm.trust_score[0,1,0]:.4f}"


class TestDiagnostics:
    """Verify diagnostic output."""

    def test_get_trust_summary(self):
        K, Q = 3, 2
        tm = TrustManager(K=K, Q=Q)
        summary = tm.get_trust_summary()

        assert 'fusion_trust_mean' in summary
        assert 'fusion_trust_min' in summary
        assert 'fusion_trust_max' in summary
        assert 'fusion_disagreement_mean' in summary
        assert 'quarantine_count' in summary
        assert 'fusion_rejection_rate' in summary
        assert 0.0 <= summary['fusion_trust_mean'] <= 1.0

    def test_get_trust_matrix(self):
        K, Q = 3, 2
        tm = TrustManager(K=K, Q=Q)
        mat = tm.get_trust_matrix(0)
        assert mat.shape == (K, K)

    def test_reset_clears_state(self):
        K, Q = 3, 2
        tm = TrustManager(K=K, Q=Q)

        tm.trust_score[0, 1, 0] = 0.1
        tm.quarantine_counter[0, 1, 0] = 5
        tm.nis_fused_ema[0, 1, 0] = 10.0

        tm.reset()

        assert tm.trust_score[0, 1, 0] == 1.0
        assert tm.quarantine_counter[0, 1, 0] == 0
        assert tm.nis_fused_ema[0, 1, 0] == 1.0
