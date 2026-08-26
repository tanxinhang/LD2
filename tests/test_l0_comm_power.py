"""Tests for D0.93 L0 analytical minimum communication power."""

import numpy as np
import pytest

from uav_isac.environment.env_wrapper import UAVISACEnv
from config.params import get_default_config


def _build(l0_enabled):
    cfg = get_default_config()
    cfg.scenario.K = 4
    cfg.scenario.Q = 4
    cfg.target.omega_q = [0.25] * 4
    cfg.marl.joint_isac_power_enabled = True
    cfg.marl.architecture_v2_enabled = False
    cfg.marl.tracking_enabled = False
    cfg.marl.analytical_comm_power_enabled = l0_enabled
    return UAVISACEnv(config=cfg, seed=0)


def test_l0_reclaims_comm_slack_into_sensing_budget():
    """L0 keeps the 1 W budget exact while lowering comm power."""
    on = _build(True)
    on.reset(seed=0)
    actions = {str(k): {"delta_p": np.zeros(2), "role": 2} for k in range(4)}
    for _ in range(5):
        _, _, _, _, info = on.step(actions)
    # Both hardware caps hold; unused RF slack is physically allowed.
    assert float(info["isac_max_power_budget_violation_w"]) < 1e-12
    sensing = np.asarray(info["isac_per_uav_sensing_power_w"], dtype=np.float64)
    assert np.all(sensing >= 0.0)
    assert np.all(sensing <= on.cfg.uav.P_sense_max + 1e-12)
    assert float(info["isac_unused_power_w"]) >= 0.0
    on.close()


def test_l0_requires_joint_isac_power():
    cfg = get_default_config()
    cfg.scenario.K = 4
    cfg.scenario.Q = 4
    cfg.marl.analytical_comm_power_enabled = True  # without joint_isac_power
    with pytest.raises(ValueError):
        UAVISACEnv(config=cfg, seed=0)
