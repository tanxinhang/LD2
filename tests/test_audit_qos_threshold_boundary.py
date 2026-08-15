from pathlib import Path

import numpy as np

from tools.audit_qos_threshold_boundary import _episode_bootstrap


def test_episode_bootstrap_is_deterministic_and_ordered():
    first = _episode_bootstrap(
        np.asarray([0.1, 0.2, 0.3]), samples=200, seed=7)
    second = _episode_bootstrap(
        np.asarray([0.1, 0.2, 0.3]), samples=200, seed=7)

    assert first == second
    assert first[0] <= first[1]


def test_audit_module_is_importable_from_workspace():
    assert Path(__file__).resolve().parents[1].name == "LD3"
