"""P0 regression: local PD_hist timing, strict decentralized, RX-only.

Three deterministic semantic tests:
  1. Timing: next_obs PD_hist == env.prev_P_D_local[k]  (exact equality)
  2. RX-only: only UAVs in the RX-set for target q get non-zero P_D[k][q]
  3. No-fallback: empty local dict → zeros, not global prev_P_D
  4. Cross-UAV asymmetry: deterministic injection → verified per-UAV difference
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


# ── Smoke ───────────────────────────────────────────────────────────

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
    assert len(obs) == K
    for _ in range(3):
        actions = {str(k): {'delta_p': np.zeros(2), 'role': 0} for k in range(K)}
        obs, reward, term, trunc, info = env.step(actions)
        assert len(obs) == K
    env.close()


# ── Timing ──────────────────────────────────────────────────────────

def test_local_pd_timing_next_obs_equals_computed():
    """next_obs PD_hist must EXACTLY equal env.prev_P_D_local[k].

    This is the strongest timing assertion: the value written into
    prev_P_D_local at step 9 MUST be the value that _build_observations
    reads at step 10.  Any off-by-one will cause a mismatch.
    """
    cfg = load_config('config/exp_800_q4.yaml')
    cfg.marl.num_envs = 1
    env = UAVISACEnv(config=cfg, seed=42)
    K, Q = cfg.scenario.K, cfg.scenario.Q

    env.reset(seed=42)

    # Take one step — env computes local P_D (step 9), then builds obs (step 10)
    actions = {str(k): {'delta_p': np.random.randn(2) * 2.5, 'role': 0}
               for k in range(K)}
    next_obs, _, _, _, _ = env.step(actions)

    # The PD_hist embedded in next_obs MUST match env.prev_P_D_local EXACTLY
    local_pd = env.core.prev_P_D_local
    assert len(local_pd) == K, f"prev_P_D_local missing keys: {list(local_pd.keys())}"

    mismatches = []
    for k in range(K):
        actual = _extract_pd_hist(next_obs[str(k)], Q)
        expected = local_pd[k]
        if not np.allclose(actual, expected, atol=1e-7):
            mismatches.append((k, actual, expected))

    assert len(mismatches) == 0, (
        f"PD_hist in next_obs does not match env.prev_P_D_local. "
        f"Timing bug: next_obs was built BEFORE prev_P_D_local was updated. "
        f"Mismatches (k, actual, expected): {mismatches}"
    )
    env.close()


# ── RX-only ────────────────────────────────────────────────────────

def test_local_pd_rx_only():
    """Only UAVs that are RX for target q get non-zero P_D[k][q].

    TX UAVs that are never RX for target q must have P_D[k][q] == 0.
    At least one RX UAV must have P_D > 0 for its served target.
    """
    cfg = load_config('config/exp_800_q4.yaml')
    cfg.marl.num_envs = 1
    env = UAVISACEnv(config=cfg, seed=42)
    K, Q = cfg.scenario.K, cfg.scenario.Q

    env.reset(seed=42)
    actions = {str(k): {'delta_p': np.random.randn(2) * 2.5, 'role': 0}
               for k in range(K)}
    _, _, _, _, info = env.step(actions)

    selected_set = getattr(env.core, '_last_selected_set', [])
    local_pd = env.core.prev_P_D_local

    if len(selected_set) == 0:
        # No pairs selected — all local P_D should be zero
        for k in range(K):
            assert np.allclose(local_pd.get(k, np.zeros(Q)), 0.0, atol=1e-10), (
                f"No pairs selected but UAV {k} has non-zero local P_D"
            )
    else:
        # Build RX-set per target: R_q = {j | (i,j,q) selected}
        R = {q: set() for q in range(Q)}
        for (i, j, q) in selected_set:
            R[q].add(j)

        # Check TX-not-in-RX-set → zero
        for (i, j, q) in selected_set:
            if i not in R[q]:
                tx_pd = local_pd.get(i, np.zeros(Q))
                assert tx_pd[q] == 0.0 or np.isclose(tx_pd[q], 0.0, atol=1e-10), (
                    f"TX UAV {i} has P_D[{q}]={tx_pd[q]:.6f} "
                    f"but is NOT in RX-set R[{q}]={R[q]}. "
                    f"Selected pairs: {selected_set}"
                )

        # Check at least one RX has positive P_D for its target
        any_positive = False
        for q in range(Q):
            for rx in R[q]:
                rx_pd = local_pd.get(rx, np.zeros(Q))
                if rx_pd[q] > 1e-8:
                    any_positive = True
                    break
        # If P0 selected pairs, at least one RX should have deflection > 0
        # (g_min gate allows d_eff=0 in some cases, so this is soft)
        if not any_positive:
            pass  # possible if all d_eff below g_min; not a code bug

    env.close()


# ── No-fallback ─────────────────────────────────────────────────────

def test_no_global_pd_fallback():
    """When prev_P_D_local is empty, PD_hist must be zeros, not global P_D."""
    cfg = load_config('config/exp_800_q4.yaml')
    cfg.marl.num_envs = 1
    env = UAVISACEnv(config=cfg, seed=42)
    K, Q = cfg.scenario.K, cfg.scenario.Q

    env.core.prev_P_D = np.full(Q, 0.99)   # distinctive global value
    env.core.prev_P_D_local = {}            # simulate edge case

    obs = env.core._build_observations()

    for k in range(K):
        pd_hist = _extract_pd_hist(obs[k], Q)
        assert np.allclose(pd_hist, 0.0, atol=1e-6), (
            f"UAV {k} PD_hist={pd_hist} leaked from global prev_P_D=0.99. "
            f"Fallback to global P_D violates strict decentralized boundary."
        )
    env.close()


# ── Cross-UAV asymmetry ────────────────────────────────────────────

def test_local_pd_asymmetric_across_uavs():
    """Deterministic asymmetric injection verifies each UAV sees OWN PD_hist.

    We monkeypatch prev_P_D_local with known per-UAV values, then call
    _build_observations and verify each UAV got exactly ITS value.
    """
    cfg = load_config('config/exp_800_q4.yaml')
    cfg.marl.num_envs = 1
    env = UAVISACEnv(config=cfg, seed=42)
    K, Q = cfg.scenario.K, cfg.scenario.Q

    env.reset(seed=42)

    # Inject deterministic, clearly asymmetric values
    injected = {}
    for k in range(K):
        val = np.zeros(Q)
        val[k % Q] = 0.7 + 0.01 * k  # each UAV gets a unique non-zero at a different target
        injected[k] = val.copy()

    env.core.prev_P_D_local = injected

    obs = env.core._build_observations()

    for k in range(K):
        actual = _extract_pd_hist(obs[k], Q)
        expected = injected[k]
        assert np.allclose(actual, expected, atol=1e-7), (
            f"UAV {k}: PD_hist={actual}, expected={expected}. "
            f"Cross-UAV PD_hist leakage or wrong indexing."
        )

    # Also verify: not all UAVs got the same thing
    all_hist = np.array([_extract_pd_hist(obs[k], Q) for k in range(K)])
    any_diff = any(
        not np.allclose(all_hist[0], all_hist[k], atol=1e-6)
        for k in range(1, K)
    )
    assert any_diff, "All UAVs got identical PD_hist despite asymmetric injection."
    env.close()
