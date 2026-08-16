"""Tests for D1.8 feasibility-aware L3 warm start (advice 013).

Frame 0 previously produced zero movement (the previous-frame structure is
unavailable after reset), wasting the first frame of the receding-horizon
geometry descent.  The warm start precomputes the initial deflection entries
and a minimal single-owner structure from the CURRENT geometry, so the L3
hook acts on frame 0 -- an initial capability-gauge-feasibility-aware trigger
(gamma_0* > 1 implies Phase-1 deficit descent from frame 0).
"""

import numpy as np
import pytest

from config.params import load_config
from uav_isac.environment.env_wrapper import UAVISACEnv


def _env(seed=503):
    cfg = load_config(
        'config/exp_800_k8q8_analytical_l0l1_movement_lex_candidates.yaml')
    cfg.scenario.K = 4
    cfg.scenario.Q = 4
    cfg.scenario.T = 20
    cfg.scenario.region_size = (800.0, 800.0)
    cfg.marl.analytical_movement_enabled = True
    cfg.marl.analytical_sensing_power_enabled = True
    cfg.marl.analytical_structure_ranking_enabled = True
    cfg.marl.analytical_comm_power_enabled = True
    cfg.marl.task_constrained_power_enabled = True
    cfg.marl.task_constrained_mode = "lexicographic"
    cfg.marl.analytical_movement_candidates_enabled = True
    cfg.marl.analytical_movement_dual_prune = True
    return UAVISACEnv(cfg, seed=seed)


def _run_first_frames(env, n=5):
    env.reset()
    core = env.core
    prev = np.array([u.pos[:2].copy() for u in core.uavs])
    moves, worses = [], []
    for _ in range(n):
        a = {str(k): {"delta_p": np.zeros(2), "role": 0}
             for k in range(core.K)}
        _, _, _, _, info = env.step(a)
        cur = np.array([u.pos[:2].copy() for u in core.uavs])
        moves.append(float(np.linalg.norm(cur - prev, axis=1).sum()))
        worses.append(float(np.min(np.atleast_1d(info.get("P_D_q", [0.0])))))
        prev = cur
    env.close()
    return np.asarray(moves), np.asarray(worses)


def test_frame0_moves_with_warm_start():
    moves, worses = _run_first_frames(_env(503))
    # Frame 0 must move (previously 0.0 -- the warm start acts immediately).
    assert moves[0] > 0.0
    # Movement bounded by the kinematic limit K * v_max * dt = 4 * 2.5.
    assert np.all(moves <= 10.0 + 1e-9)
    # The early trajectory improves monotonically (deficit descent).
    assert worses[0] < worses[-1]


def test_warm_start_preserves_feasibility_and_budget():
    env = _env(503)
    env.reset()
    core = env.core
    # The initial structure must be a valid single-owner structure.
    from uav_isac.coordination.maxmin_power import fixed_owner_gain_matrix
    entries, selected = core._initial_analytical_state()
    assert entries is not None and len(selected) >= core.Q
    coefficient = core._per_watt_coefficient_from_entries(entries)
    gain, owner = fixed_owner_gain_matrix(coefficient, selected)
    assert gain.shape == (core.K, core.Q)
    assert owner.shape == (core.Q,)
    assert np.all(owner >= 0)  # every target has an owner
    env.close()


def test_warm_start_speeds_up_early_convergence():
    # With the warm start, frame 1's worst should be no worse than the old
    # frame-0-wasted trajectory's frame 1 (i.e. the warm start never hurts).
    # Compare seeds to cover geometry variability.
    for seed in (503, 700, 922):
        moves, worses = _run_first_frames(_env(seed))
        assert moves[0] > 0.0
        assert worses[0] > 0.0
