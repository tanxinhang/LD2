from __future__ import annotations

import numpy as np

from tools.summarize_algorithm_baselines import (
    bootstrap_mean_ci,
    exact_mcnemar_p,
    qos_mask,
)


def test_bootstrap_constant_paired_delta_is_exact() -> None:
    interval = bootstrap_mean_ci(
        np.full(100, 0.125), 500, np.random.default_rng(7))
    assert np.allclose(interval, [0.125, 0.125])


def test_exact_mcnemar_two_sided_binomial() -> None:
    assert exact_mcnemar_p(0, 0) == 1.0
    assert np.isclose(exact_mcnemar_p(0, 5), 0.0625)
    assert np.isclose(exact_mcnemar_p(1, 6), 0.125)


def test_qos_mask_requires_all_three_medium_floors() -> None:
    arrays = {
        "steady": np.asarray([0.80, 0.79, 0.95, 0.95]),
        "weak3": np.asarray([0.70, 0.90, 0.69, 0.90]),
        "worst": np.asarray([0.60, 0.90, 0.90, 0.59]),
    }
    assert qos_mask(arrays).tolist() == [True, False, False, False]
