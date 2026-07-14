"""Shared test fixtures for UAV-ISAC tests."""

import pytest
import numpy as np
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.params import get_default_config, MasterConfig


@pytest.fixture
def default_config() -> MasterConfig:
    """Full default configuration."""
    return get_default_config()


@pytest.fixture
def small_config() -> MasterConfig:
    """Reduced config for fast smoke tests."""
    cfg = get_default_config()
    small = cfg.to_small_config()
    return small


@pytest.fixture
def seeded_rng():
    """Seeded NumPy random generator (seed=42)."""
    return np.random.default_rng(42)


@pytest.fixture
def sample_uav_positions():
    """K=4 UAVs in known positions (1000x1000 area, H=100m)."""
    return np.array([
        [100.0, 100.0, 100.0],
        [900.0, 100.0, 100.0],
        [100.0, 900.0, 100.0],
        [900.0, 900.0, 100.0],
    ], dtype=np.float64)


@pytest.fixture
def sample_uav_velocities():
    """Zero initial velocities for UAVs."""
    return np.zeros((4, 3), dtype=np.float64)


@pytest.fixture
def sample_target_positions():
    """Q=2 targets at known positions."""
    return np.array([
        [400.0, 500.0, 0.0],
        [600.0, 500.0, 0.0],
    ], dtype=np.float64)


@pytest.fixture
def sample_target_velocities():
    """Q=2 targets with moderate velocities."""
    return np.array([
        [10.0, 5.0, 0.0],
        [-8.0, 12.0, 0.0],
    ], dtype=np.float64)


@pytest.fixture
def sample_roles():
    """2 tx, 2 rx UAVs."""
    return np.array([0, 0, 1, 1], dtype=np.int32)  # tx, tx, rx, rx


@pytest.fixture
def fc_position():
    """Fusion center at center of area."""
    return np.array([500.0, 500.0, 100.0], dtype=np.float64)
