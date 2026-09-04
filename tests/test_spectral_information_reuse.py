"""Mathematical contracts for spectrally certified information reuse."""

import numpy as np
import pytest

from uav_isac.coordination.spectral_information_reuse import (
    information_factor,
    logdet_information_gain,
    rank_trace_logdet_upper_bound,
    spectrally_certified_information,
    trace_logdet_upper_bound,
    woodbury_covariance_update,
)


def test_independent_targets_receive_exact_information_credit():
    blocks = (np.asarray([[2.0]]), np.asarray([[3.0]]))
    certificate = spectrally_certified_information(blocks, np.eye(2))

    assert certificate.coupling_norm == pytest.approx(0.0)
    assert certificate.information_discount == pytest.approx(1.0)
    np.testing.assert_allclose(
        certificate.lower_information,
        certificate.actual_information,
    )


def test_correlated_returns_are_discounted_by_a_psd_safe_bound():
    blocks = (np.asarray([[1.0]]), np.asarray([[1.0]]))
    covariance = np.asarray([[1.0, 0.5], [0.5, 1.0]])
    certificate = spectrally_certified_information(blocks, covariance)

    assert certificate.coupling_norm == pytest.approx(0.5)
    assert certificate.information_discount == pytest.approx(2.0 / 3.0)
    slack = (
        certificate.actual_information - certificate.lower_information)
    assert np.min(np.linalg.eigvalsh(slack)) >= -1.0e-12


def test_stronger_coupling_never_creates_free_information_credit():
    blocks = (np.asarray([[1.0]]), np.asarray([[1.0]]))
    weak = spectrally_certified_information(
        blocks, np.asarray([[1.0, 0.1], [0.1, 1.0]]))
    strong = spectrally_certified_information(
        blocks, np.asarray([[1.0, 0.8], [0.8, 1.0]]))

    assert strong.information_discount < weak.information_discount
    assert np.trace(strong.lower_information) < np.trace(
        weak.lower_information)


def test_low_rank_logdet_and_woodbury_match_dense_algebra():
    prior_information = np.diag([2.0, 3.0, 4.0])
    increment = np.asarray([
        [0.8, 0.1, 0.0],
        [0.1, 0.5, 0.0],
        [0.0, 0.0, 0.0],
    ])
    factor = information_factor(increment)
    gain = logdet_information_gain(prior_information, factor)
    dense_gain = (
        np.linalg.slogdet(prior_information + increment)[1]
        - np.linalg.slogdet(prior_information)[1]
    )
    assert gain == pytest.approx(dense_gain)
    prior_covariance = np.linalg.inv(prior_information)
    assert trace_logdet_upper_bound(
        prior_covariance, factor) >= gain - 1.0e-12
    rank_bound = rank_trace_logdet_upper_bound(prior_covariance, factor)
    assert gain <= rank_bound + 1.0e-12
    assert rank_bound <= trace_logdet_upper_bound(
        prior_covariance, factor) + 1.0e-12

    updated = woodbury_covariance_update(prior_covariance, factor)
    np.testing.assert_allclose(
        updated,
        np.linalg.inv(prior_information + increment),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_non_positive_noise_covariance_fails_closed():
    with pytest.raises(ValueError, match="positive definite"):
        spectrally_certified_information(
            (np.asarray([[1.0]]), np.asarray([[1.0]])),
            np.asarray([[1.0, 1.0], [1.0, 1.0]]),
        )
