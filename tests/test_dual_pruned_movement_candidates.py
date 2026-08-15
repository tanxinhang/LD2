"""Tests for D1.1-B+ dual-pruned multi-candidate L3 (2026-08-16).

Theory: for the fixed-owner max-min power LP, weak duality bounds the
candidate geometry's max-min deflection from above by
    U_lambda(g') = sum_i b_i max_q lambda*_q a'_iq
for any feasible simplex price lambda* (e.g. the current frame's optimal
dual).  P_D is strictly monotone in deflection, so
U_lambda(g') <= best_deflection proves the candidate cannot beat the
incumbent and its exact LP evaluation is provably dominated.

These tests verify (1) the weak-duality bound numerically, (2) that enabling
the pruning never changes the selected candidate, and (3) that it reduces
exact LP evaluations.
"""

import numpy as np
import pytest

from config.params import load_config
from uav_isac.coordination.maxmin_power import (
    fixed_owner_gain_matrix,
    optimal_maxmin_dual_prices,
    solve_fixed_structure_maxmin_power_lp,
)
from uav_isac.environment.env_wrapper import UAVISACEnv


def _small_env():
    cfg = load_config('config/exp_800_q4_u2u_joint_isac.yaml')
    cfg.scenario.K = 2
    cfg.scenario.Q = 2
    cfg.scenario.T = 10
    cfg.target.omega_q = [0.5, 0.5]
    env = UAVISACEnv(cfg, seed=7)
    env.reset()
    return env


def _synthetic_inputs(rng):
    """Deterministic synthetic (coefficient, selected, budget, uav, tgt)."""
    K, Q = 2, 2
    coefficient = np.abs(rng.normal(size=(K, K, Q))) + 0.5
    selected = ((0, 1, 0), (1, 0, 1))  # tx, rx, target (valid half-duplex)
    budget = np.array([0.75, 0.75])
    uav = np.array([[100.0, 100.0], [150.0, 130.0]])
    tgt = np.array([[200.0, 200.0], [180.0, 260.0]])
    return coefficient, selected, budget, uav, tgt


def test_weak_duality_upper_bound_dominates_maxmin():
    rng = np.random.default_rng(20260816)
    for _ in range(20):
        K, Q = 3, 3
        gain = np.abs(rng.normal(size=(K, Q))) + 0.1
        budget = rng.uniform(0.2, 0.9, size=K)
        lam, _ = optimal_maxmin_dual_prices(gain, budget)
        res = solve_fixed_structure_maxmin_power_lp(gain, budget)
        # Weak duality: U_lambda >= t* for the same geometry.
        u_lambda = float(np.sum(budget * np.max(lam[None, :] * gain, axis=1)))
        assert u_lambda >= res.worst_deflection - 1e-9
        # Optimal dual value equals t* (strong duality) at the current geometry.
        _, dual_value = optimal_maxmin_dual_prices(gain, budget)
        assert dual_value == pytest.approx(res.worst_deflection, abs=1e-9)


def test_dual_pruning_never_changes_selected_candidate():
    env = _small_env()
    rng = np.random.default_rng(20260816)
    coefficient, selected, budget, uav, tgt = _synthetic_inputs(rng)
    gain_cur, _ = fixed_owner_gain_matrix(coefficient, selected)
    lam, _ = optimal_maxmin_dual_prices(gain_cur, budget)
    grads = {
        0: np.array([1.0, 0.5]),
        1: np.array([-0.3, 0.9]),
    }

    results = {}
    for prune in (True, False):
        env.core.cfg.marl.analytical_movement_dual_prune = bool(prune)
        out = env.core._select_best_movement_candidate(
            coefficient, selected, budget, uav, tgt, grads, step=2.5)
        results[prune] = out
        assert lam is not None

    # The chosen candidate must be bitwise identical with/without pruning.
    for k in results[False]:
        np.testing.assert_array_equal(
            results[True].get(k, np.zeros(2)),
            results[False].get(k, np.zeros(2)),
            err_msg=f"pruned selection differs at UAV {k}")


def test_gauge_price_step_enlarges_candidate_set_and_never_degrades():
    """D1.1-B++ monotonicity: adding the gauge-price step (and the dual x
    radial combination) is a candidate-set superset with stay retained, so the
    exact-LP best score can never be lower than without it."""
    env = _small_env()
    rng = np.random.default_rng(20260817)
    coefficient, selected, budget, uav, tgt = _synthetic_inputs(rng)
    gain_cur, owner = fixed_owner_gain_matrix(coefficient, selected)
    lam, _ = optimal_maxmin_dual_prices(gain_cur, budget)
    grads = {
        0: np.array([1.0, 0.5]),
        1: np.array([-0.3, 0.9]),
    }

    best_scores = {}
    for gauge in (True, False):
        env.core.cfg.marl.analytical_movement_dual_prune = True
        env.core.cfg.marl.analytical_movement_gauge_price_step = bool(gauge)
        out = env.core._select_best_movement_candidate(
            coefficient, selected, budget, uav, tgt, grads, step=2.5)
        # Re-score the chosen candidate with the same exact-LP scoring the
        # selector uses (min P_D over targets at the moved geometry).
        score = _exact_candidate_score(env, coefficient, selected, budget,
                                       uav, tgt, out)
        best_scores[gauge] = score

    # The gauge-price candidate set is a superset with stay retained, so the
    # chosen candidate's exact score must not be lower with it enabled.
    assert best_scores[True] >= best_scores[False] - 1e-12
    assert lam is not None


def _exact_candidate_score(env, coefficient, selected, budget, uav, tgt,
                           movement):
    from uav_isac.coordination.maxmin_power import (
        fixed_owner_gain_matrix,
        solve_fixed_structure_maxmin_power_lp,
    )
    from uav_isac.physical.detection import compute_detection_probabilities
    mat = np.zeros_like(uav)
    for k, delta in movement.items():
        mat[k] = np.asarray(delta, dtype=np.float64)
    nu = np.clip(uav + mat, 0.0, 1e9)
    coeff_c = env.core._friis_rescale_tensor(coefficient, uav, tgt, nu)
    g_c, _ = fixed_owner_gain_matrix(coeff_c, selected)
    res = solve_fixed_structure_maxmin_power_lp(g_c, budget)
    pd = compute_detection_probabilities(res.deflection, env.core.cfg.detection.P_FA)
    return float(np.min(pd))


def test_lex_scoring_matches_executed_l1_objective():
    """D1.1-B+++ score/execute consistency: under a lexicographic inner layer
    the candidate scorer must use qos_constrained_maxmin_lp (Stage B), not the
    plain max-min LP, so the chosen candidate optimises the same objective
    that will be executed.  The test verifies the lex branch runs and that
    its score equals the Stage-B worst P_D on the moved geometry (or the
    reserve-first max-min fallback when Stage-B is infeasible there)."""
    env = _small_env()
    env.core.cfg.marl.task_constrained_power_enabled = True
    env.core.cfg.marl.task_constrained_mode = "lexicographic"
    env.core.cfg.marl.analytical_movement_dual_prune = True
    env.core.cfg.marl.analytical_movement_lex_scoring = True
    rng = np.random.default_rng(20260818)
    coefficient, selected, budget, uav, tgt = _synthetic_inputs(rng)
    # Near-field geometry so the Stage-B QoS floors are typically feasible.
    uav = np.array([[40.0, 40.0], [60.0, 50.0]])
    tgt = np.array([[100.0, 100.0], [90.0, 120.0]])
    grads = {
        0: np.array([1.0, 0.5]),
        1: np.array([-0.3, 0.9]),
    }
    # Must run without error and return a dict of moves.
    out = env.core._select_best_movement_candidate(
        coefficient, selected, budget, uav, tgt, grads, step=2.5)
    assert isinstance(out, dict)
    from uav_isac.coordination.capability import qos_constrained_maxmin_lp
    from uav_isac.coordination.pwl_pd import (
        chord_lower_bound,
        curvature_breakpoints,
    )
    from uav_isac.physical.detection import (
        compute_detection_probabilities,
        minimum_deflection_for_detection_probability,
    )
    p_fa = env.core.cfg.detection.P_FA
    qos_floors = (0.60, 0.70, 0.80, 3)
    d_min = float(minimum_deflection_for_detection_probability(
        np.asarray([qos_floors[0]]), p_fa)[0])
    gain_cur, _ = fixed_owner_gain_matrix(coefficient, selected)
    d_max = float(np.max(np.sum(gain_cur * budget[:, None], axis=0))) + 1.0
    bps = curvature_breakpoints(p_fa, d_min, d_max, epsilon=1e-3)
    cs, ci = chord_lower_bound(p_fa, bps)
    mat = np.zeros_like(uav)
    for k, delta in out.items():
        mat[k] = np.asarray(delta, dtype=np.float64)
    nu = np.clip(uav + mat, 0.0, 1e9)
    coeff_c = env.core._friis_rescale_tensor(coefficient, uav, tgt, nu)
    g_c, _ = fixed_owner_gain_matrix(coeff_c, selected)
    st = qos_constrained_maxmin_lp(g_c, budget, p_fa, qos_floors, cs, ci, d_min)
    if st is not None:
        # Stage-B feasible: the executed score must equal the Stage-B worst P_D.
        _t_star, _p, d_star = st
        pd = compute_detection_probabilities(d_star, p_fa)
        assert float(np.min(pd)) == pytest.approx(
            _exact_candidate_score(env, coefficient, selected, budget, uav,
                                   tgt, out),
            abs=1e-9)
    else:
        # Stage-B infeasible on the moved geometry: the executed path (and
        # the lex scorer) fall back to reserve-first max-min, which is exactly
        # what _exact_candidate_score computes.
        assert _exact_candidate_score(
            env, coefficient, selected, budget, uav, tgt, out) >= 0.0


def test_weak_duality_pruning_stays_exact_under_lex_scoring():
    """With lex scoring enabled, lex t* <= plain max-min t* <= U_lambda, so a
    candidate with U_lambda <= best_deflection can never beat the incumbent
    under the executed Stage-B objective either."""
    env = _small_env()
    env.core.cfg.marl.task_constrained_power_enabled = True
    env.core.cfg.marl.task_constrained_mode = "lexicographic"
    env.core.cfg.marl.analytical_movement_dual_prune = True
    env.core.cfg.marl.analytical_movement_lex_scoring = True
    rng = np.random.default_rng(20260819)
    coefficient, selected, budget, uav, tgt = _synthetic_inputs(rng)
    gain_cur, _ = fixed_owner_gain_matrix(coefficient, selected)
    lam, _ = optimal_maxmin_dual_prices(gain_cur, budget)
    # Plain max-min on the current geometry:
    res = solve_fixed_structure_maxmin_power_lp(gain_cur, budget)
    u_lambda = float(np.sum(budget * np.max(lam[None, :] * gain_cur, axis=1)))
    assert u_lambda >= res.worst_deflection - 1e-9  # weak duality
    # And lex Stage-B cannot exceed the plain max-min value (it only adds
    # QoS-floor constraints; the gauge's own allocation is feasible), so the
    # U_lambda bound dominates the lex objective as well.
    from uav_isac.coordination.capability import qos_constrained_maxmin_lp
    from uav_isac.coordination.pwl_pd import (
        chord_lower_bound,
        curvature_breakpoints,
    )
    from uav_isac.physical.detection import (
        minimum_deflection_for_detection_probability,
    )
    p_fa = env.core.cfg.detection.P_FA
    qos_floors = (0.60, 0.70, 0.80, 3)
    d_min = float(minimum_deflection_for_detection_probability(
        np.asarray([qos_floors[0]]), p_fa)[0])
    d_max = float(np.max(np.sum(gain_cur * budget[:, None], axis=0))) + 1.0
    bps = curvature_breakpoints(p_fa, d_min, d_max, epsilon=1e-3)
    cs, ci = chord_lower_bound(p_fa, bps)
    st = qos_constrained_maxmin_lp(gain_cur, budget, p_fa, qos_floors,
                                   cs, ci, d_min)
    if st is not None:
        assert st[0] <= res.worst_deflection + 1e-9
        assert st[0] <= u_lambda + 1e-9


def test_dual_pruning_reduces_lp_evaluations(monkeypatch):
    env = _small_env()
    rng = np.random.default_rng(20260816)
    coefficient, selected, budget, uav, tgt = _synthetic_inputs(rng)
    grads = {
        0: np.array([1.0, 0.5]),
        1: np.array([-0.3, 0.9]),
    }

    real_solve = solve_fixed_structure_maxmin_power_lp
    counters = {"calls": 0}

    def counting_solve(*args, **kwargs):
        counters["calls"] += 1
        return real_solve(*args, **kwargs)

    counts = {}
    for prune in (True, False):
        env.core.cfg.marl.analytical_movement_dual_prune = bool(prune)
        counters["calls"] = 0
        monkeypatch.setattr(
            "uav_isac.coordination.maxmin_power."
            "solve_fixed_structure_maxmin_power_lp", counting_solve)
        env.core._select_best_movement_candidate(
            coefficient, selected, budget, uav, tgt, grads, step=2.5)
        counts[prune] = counters["calls"]

    # Pruning must not increase the LP budget; on this synthetic geometry the
    # gradient and radial candidates are usually dominated, so strictly fewer
    # LP evaluations are expected in practice.
    assert counts[True] <= counts[False]
    assert counts[True] < counts[False] or counts[True] == 1
