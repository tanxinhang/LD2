"""Regression tests for the 2026-08-25 deep-audit fixes.

Locked behaviors:
1. feasibility_oracle K==1 / empty-partition degrade paths return a valid
   OracleSolution (previously NameError / AssertionError) and use the module's
   "no deflection => P_D = P_FA" convention instead of a hard-coded 0.0.
2. minimum_intervention_repair validates the task-floors tuple BEFORE unpacking
   (ValueError, not IndexError) and accepts the (0,1] domain that
   scale_capability.task_detection_metrics also accepts.
3. env_core.reseed() rebinds the cost-aware InterUAVCommunicationModel RNG so
   reset(seed=...) replays channel shadowing/burst identically (previously the
   stale constructor stream made same-seed episodes diverge).
"""

import numpy as np
import pytest

from config.params import load_config
from uav_isac.environment.env_wrapper import UAVISACEnv
from uav_isac.physical.feasibility_oracle import (
    solve_joint_pair_power_oracle,
    solve_pair_only_oracle,
)
from uav_isac.coordination.minimum_intervention_repair import (
    solve_minimum_intervention_task_repair_milp,
)


# ---------------------------------------------------------------------------
# 1. K==1 degenerate oracle paths
# ---------------------------------------------------------------------------


def test_k1_joint_oracle_degrades_gracefully():
    coeff = np.zeros((1, 1, 2))
    solution = solve_joint_pair_power_oracle(
        coefficient=coeff, P_FA=0.001,
        target_pair_limit=3, reports_per_receiver=4,
        fusion_mode="central_oracle",
    )
    assert solution.mode == "degenerate"
    assert solution.tx_indices == () and solution.rx_indices == ()
    assert np.all(np.isfinite(solution.P_D_q))


def test_k1_pair_oracle_degrades_gracefully():
    coeff = np.zeros((1, 1, 2))
    solution = solve_pair_only_oracle(
        coeff, np.zeros((1, 2)) + 0.0251, P_FA=0.001,
        target_pair_limit=3, reports_per_receiver=4,
        fusion_mode="central_oracle",
    )
    assert solution.mode == "pair_only_degenerate"
    assert np.all(np.isfinite(solution.P_D_q))


# ---------------------------------------------------------------------------
# 2. minimum_intervention_repair task-floors contract
# ---------------------------------------------------------------------------


def _repair_inputs(k: int = 4, q: int = 2):
    gain = np.zeros((k, k, q))
    gain[0, 2, 0] = 20.0
    gain[1, 3, 1] = 20.0
    selected = gain > 0
    is_tx = np.zeros(k, dtype=bool)
    is_rx = np.zeros(k, dtype=bool)
    is_tx[::2] = True
    is_rx[1::2] = True
    owners = np.arange(q, dtype=np.int64) * 2 + 1
    owners = np.clip(owners, 0, k - 1)
    return gain, np.ones(k), selected, is_tx, is_rx, owners


def test_task_floors_validated_before_unpack():
    args = _repair_inputs()
    with pytest.raises(ValueError, match="task_floors"):
        solve_minimum_intervention_task_repair_milp(
            *args, p_fa=0.1, task_floors=(0.6, 0.6),  # short tuple
            target_pair_limit=1, reports_per_receiver=2,
        )


def test_task_floors_unit_floor_fails_closed_with_explanation():
    args = _repair_inputs()
    # A unit floor cannot be inverted to finite deflection in this Gaussian
    # detector (scale_capability accepts (0,1] only as a feasibility-check
    # floor, not as an inversion target).  The repair must therefore raise a
    # ValueError with the reason instead of silently rejecting or crashing.
    with pytest.raises(ValueError, match="cannot be inverted"):
        solve_minimum_intervention_task_repair_milp(
            *args, p_fa=0.1, task_floors=(1.0, 1.0, 1.0, 1),
            target_pair_limit=1, reports_per_receiver=2,
        )


# ---------------------------------------------------------------------------
# 3. reseed rebinds the cost-aware comm RNG
# ---------------------------------------------------------------------------


def _roll(env, frames: int) -> None:
    for _ in range(frames):
        obs = env.current_obs if hasattr(env, "current_obs") else None
        if obs is None:
            break
        actions = {k: {"delta_p": np.zeros(2), "role": 0} for k in obs}
        env.step(actions)


def test_reseed_rebinds_cost_aware_comm_rng():
    cfg = load_config(
        "config/exp_800_k12q12_distributed_v2_fbl_shadow4ms.yaml")
    env = UAVISACEnv(cfg)
    core = env.core
    comm = core._inter_uav_comm
    if comm is None:
        pytest.skip("cost-aware comm model not enabled by this config")

    env.reset(seed=91)
    _roll(env, 3)
    shadow_a = comm._snr_shadowing_db.copy()
    burst_a = comm._burst_bad_state.copy()

    env.reset(seed=91)
    _roll(env, 3)
    shadow_b = comm._snr_shadowing_db.copy()
    burst_b = comm._burst_bad_state.copy()

    assert comm.rng is core.rng  # reseed rebinds the comm stream
    assert np.array_equal(shadow_a, shadow_b)
    assert np.array_equal(burst_a, burst_b)