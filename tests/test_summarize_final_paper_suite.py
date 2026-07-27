from __future__ import annotations

import numpy as np

from tools.summarize_final_paper_suite import (
    _exact_mcnemar_p,
    bootstrap_mean_ci,
    qos_mask,
    wilson_lower,
)


def test_qos_mask_requires_all_three_medium_thresholds() -> None:
    arrays = {
        "steady": np.array([0.81, 0.79, 0.90, 0.90]),
        "weak3": np.array([0.71, 0.80, 0.69, 0.80]),
        "worst": np.array([0.61, 0.80, 0.80, 0.59]),
    }
    assert qos_mask(arrays).tolist() == [True, False, False, False]


def test_wilson_lower_is_below_observed_rate() -> None:
    bound = wilson_lower(83, 100)
    assert 0.70 < bound < 0.83


def test_paired_bootstrap_constant_delta_is_exact() -> None:
    ci = bootstrap_mean_ci(
        np.full(20, 0.125), 1_000, np.random.default_rng(7))
    assert np.allclose(ci, [0.125, 0.125])


def test_exact_mcnemar_is_symmetric() -> None:
    assert _exact_mcnemar_p(6, 1) == _exact_mcnemar_p(1, 6)
    assert np.isclose(_exact_mcnemar_p(6, 1), 0.125)
    assert _exact_mcnemar_p(0, 0) == 1.0
