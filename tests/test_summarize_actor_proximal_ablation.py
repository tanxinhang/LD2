from __future__ import annotations

import numpy as np

from tools.summarize_actor_proximal_ablation import (
    bootstrap_mean_ci,
    exact_one_sided_sign_p,
)


def test_cluster_bootstrap_is_deterministic() -> None:
    values = [0.0, 0.0, 0.0, 0.0, 0.005]
    first = bootstrap_mean_ci(values, samples=5000, seed=9308)
    second = bootstrap_mean_ci(values, samples=5000, seed=9308)
    assert first == second
    assert first[0] == 0.0
    assert first[1] > 0.0


def test_sign_test_excludes_numerical_ties() -> None:
    values = [2e-4, 3e-4, 5e-5, 1e-12, -1e-12]
    assert np.isclose(exact_one_sided_sign_p(values, 1e-8), 0.125)
    assert exact_one_sided_sign_p([0.0, 1e-12], 1e-8) == 1.0
