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
    # Power balance must remain exact (1 W per UAV).
    assert float(info["isac_max_power_balance_error_w"]) < 1e-12
    # Sensing power is strictly positive and comm power bounded by 1 W.
    sensing = np.asarray(info["isac_per_uav_sensing_power_w"], dtype=np.float64)
    assert np.all(sensing >= 0.0)
    on.close()


def test_l0_requires_joint_isac_power():
    cfg = get_default_config()
    cfg.scenario.K = 4
    cfg.scenario.Q = 4
    cfg.marl.analytical_comm_power_enabled = True  # without joint_isac_power
    with pytest.raises(ValueError):
        UAVISACEnv(config=cfg, seed=0)
