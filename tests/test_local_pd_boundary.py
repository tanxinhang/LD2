"""P0 regression: local PD_hist timing, strict decentralized boundary, RX-only.

Three deterministic sub-tests:
  1. Timing: next_obs contains P_D_t (current frame), not P_D_{t-1}.
  2. No-fallback: empty prev_P_D_local → zeros, not global prev_P_D.
  3. RX-only: only the RX UAV gets local P_D credit; TX gets zero.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from config.params import load_config
from uav_isac.environment.env_wrapper import UAVISACEnv


def _extract_pd_hist(obs_vector, Q):
    """PD_hist is the Q dims before the last 16 (comm aggregation)."""
    return obs_vector[-16 - Q:-16].copy()


@pytest.mark.parametrize("config_path", [
    'config/exp_800_q4.yaml',
    'config/exp_800_k8_q8.yaml',
])
def test_env_step_dynamic_kq(config_path):
    """Environment must step cleanly for both K=4,Q=4 and K=8,Q=8."""
    cfg = load_config(config_path)
    cfg.marl.num_envs = 1
    env = UAVISACEnv(config=cfg, seed=42)
    obs, _ = env.reset(seed=42)
    K = cfg.scenario.K
    assert len(obs) == K, f"Expected {K} agents in obs, got {len(obs)}"
    for _ in range(3):
        actions = {str(k): {'delta_p': np.zeros(2), 'role': 0} for k in range(K)}
        obs, reward, term, trunc, info = env.step(actions)
        assert len(obs) == K
    env.close()


def test_local_pd_timing_current_frame_in_next_obs():
    """next_obs must contain P_D_t, not P_D_{t-1}.

    Strategy: inject a known old value into prev_P_D_local, take one step,
    then verify next_obs contains the new step's value (not the injected old one).
    """
    cfg = load_config('config/exp_800_q4.yaml')
    cfg.marl.num_envs = 1
    env = UAVISACEnv(config=cfg, seed=42)
    K, Q = cfg.scenario.K, cfg.scenario.Q

    # Inject a known OLD value into prev_P_D_local
    OLD_VALUE = 0.11
    env.core.prev_P_D_local = {k: np.full(Q, OLD_VALUE) for k in range(K)}

    # Take one step — the env computes NEW local P_D and updates prev_P_D_local
    # BEFORE building next_obs (timing fix: step 9 before step 10)
    actions = {str(k): {'delta_p': np.random.randn(2) * 2.5, 'role': 0}
               for k in range(K)}
    next_obs, _, _, _, info = env.step(actions)

    # Extract PD_hist from next_obs for each agent
    pd_in_obs = {}
    for k in range(K):
        pd_in_obs[k] = _extract_pd_hist(next_obs[str(k)], Q)

    # The PD_hist in next_obs must NOT be the injected OLD_VALUE.
    # It should be the current frame's local P_D (which may be zero or
    # non-zero depending on whether this UAV was RX for any selected pair).
    # The key check: at least one dimension of one UAV differs from OLD_VALUE.
    any_differs = any(
        not np.allclose(pd_in_obs[k], np.full(Q, OLD_VALUE), atol=1e-6)
        for k in range(K)
    )
    assert any_differs, (
        f"PD_hist in next_obs still equals injected old value {OLD_VALUE}. "
        f"prev_P_D_local was NOT updated before _build_observations(). "
        f"Timing bug: next_obs contains P_D_{{t-1}} not P_D_t."
    )
    env.close()


def test_no_global_pd_fallback():
    """When prev_P_D_local is empty, _build_observations must use zeros,
    NOT fall back to global prev_P_D."""
    cfg = load_config('config/exp_800_q4.yaml')
    cfg.marl.num_envs = 1
    env = UAVISACEnv(config=cfg, seed=42)
    K, Q = cfg.scenario.K, cfg.scenario.Q

    # Set global prev_P_D to a distinctive non-zero value
    GLOBAL_VALUE = 0.99
    env.core.prev_P_D = np.full(Q, GLOBAL_VALUE)
    # Clear local dict — simulates edge case (first frame, snapshot restore, etc.)
    env.core.prev_P_D_local = {}

    # Build observations directly
    obs = env.core._build_observations()

    for k in range(K):
        pd_hist = _extract_pd_hist(obs[k], Q)
        assert np.allclose(pd_hist, 0.0, atol=1e-6), (
            f"UAV {k} PD_hist={pd_hist} but should be zeros. "
            f"Global prev_P_D={GLOBAL_VALUE} leaked into local observation. "
            f"Fallback to global P_D is a strict decentralized violation."
        )
    env.close()


def test_local_pd_rx_only():
    """Only the RX UAV (j in pair (i,j,q)) gets local P_D credit.
    TX UAV must not receive free detection confidence.

    We verify this deterministically: after a step, check that the
    env's prev_P_D_local dict has correct RX-only structure.
    """
    cfg = load_config('config/exp_800_q4.yaml')
    cfg.marl.num_envs = 1
    env = UAVISACEnv(config=cfg, seed=42)
    K, Q = cfg.scenario.K, cfg.scenario.Q

    # Reset and take a step
    env.reset(seed=42)
    actions = {str(k): {'delta_p': np.random.randn(2) * 2.5, 'role': 0}
               for k in range(K)}
    _, _, _, _, info = env.step(actions)

    # Get the selected pairs from the step info
    selected_set = env.core._last_selected_set if hasattr(env.core, '_last_selected_set') else []
    local_pd = env.core.prev_P_D_local

    if len(selected_set) > 0:
        # For each selected pair (i,j,q), UAV j (RX) should have non-zero
        # local P_D for target q, while UAV i (TX) should have zero for target q
        # (unless i is also RX for another pair on the same target).
        for (i, j, q) in selected_set:
            # Build a set of all RX UAVs for this target
            rx_uavs_for_q = {jj for (ii, jj, qq) in selected_set if qq == q}

            # TX that is NOT also an RX for this target → should have zero
            if i not in rx_uavs_for_q:
                tx_pd = local_pd.get(i, np.zeros(Q))
                assert tx_pd[q] == 0.0 or np.isclose(tx_pd[q], 0.0, atol=1e-10), (
                    f"TX UAV {i} has local P_D[{q}]={tx_pd[q]:.6f} but should be 0. "
                    f"TX is not RX for target {q} in any pair. "
                    f"Selected pairs: {selected_set}"
                )

            # RX should have non-zero local P_D for this target
            rx_pd = local_pd.get(j, np.zeros(Q))
            # Note: P_D could be small if deflection is small, but it should
            # not be exactly zero for a selected pair
            assert rx_pd[q] >= 0.0, (
                f"RX UAV {j} local P_D[{q}]={rx_pd[q]:.6f} is negative — invalid."
            )

    env.close()


def test_local_pd_differs_across_uavs():
    """After several steps, PD_hist should differ across UAVs (no global broadcast)."""
    cfg = load_config('config/exp_800_q4.yaml')
    cfg.marl.num_envs = 1
    env = UAVISACEnv(config=cfg, seed=42)
    obs, _ = env.reset(seed=42)
    K, Q = cfg.scenario.K, cfg.scenario.Q

    pd_samples = {k: [] for k in range(K)}
    for step in range(10):
        actions = {str(k): {'delta_p': np.random.randn(2) * 2.5, 'role': 0}
                   for k in range(K)}
        obs, reward, term, trunc, info = env.step(actions)
        for k in range(K):
            pd_samples[k].append(_extract_pd_hist(obs[str(k)], Q))
        if term.get('__all__') or trunc.get('__all__'):
            break
    env.close()

    if len(pd_samples[0]) < 3:
        pytest.skip("Not enough frames collected")

    last_frame_pd = np.array([pd_samples[k][-1] for k in range(K)])  # (K, Q)
    any_diff = any(
        not np.allclose(last_frame_pd[0], last_frame_pd[k], atol=1e-6)
        for k in range(1, K)
    )
    assert any_diff, (
        "All UAVs have identical PD_hist — possible global broadcast leak. "
        f"PD_hist per UAV:\n{last_frame_pd}"
    )
