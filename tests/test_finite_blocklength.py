import numpy as np

from uav_isac.physical.finite_blocklength import (
    achievable_information_bits,
    awgn_capacity_bits_per_use,
    awgn_dispersion_bits2_per_use,
    minimum_blocklength_normal_approximation,
    minimum_snr_normal_approximation,
    normal_approximation_packet_error_probability,
)


def test_awgn_capacity_and_dispersion_have_correct_limits():
    assert awgn_capacity_bits_per_use(0.0) == 0.0
    assert awgn_dispersion_bits2_per_use(0.0) == 0.0
    assert awgn_capacity_bits_per_use(10.0) == np.log2(11.0)
    assert 0.0 < awgn_dispersion_bits2_per_use(10.0) < np.log2(np.e) ** 2


def test_minimum_blocklength_is_minimal_and_meets_bler():
    snr, bits, epsilon = 10.0, 512, 1.0e-5
    n = minimum_blocklength_normal_approximation(snr, bits, epsilon)
    assert n is not None and n > 0
    assert achievable_information_bits(snr, n, epsilon) >= bits
    assert achievable_information_bits(snr, n - 1, epsilon) < bits
    assert normal_approximation_packet_error_probability(snr, n, bits) <= epsilon


def test_snr_and_blocklength_inversions_are_consistent():
    n, bits, epsilon = 300, 512, 1.0e-4
    snr = minimum_snr_normal_approximation(n, bits, epsilon)
    assert snr is not None
    assert normal_approximation_packet_error_probability(
        snr, n, bits) <= epsilon * (1.0 + 1.0e-8)
    assert minimum_blocklength_normal_approximation(
        snr, bits, epsilon) <= n


def test_zero_snr_cannot_carry_positive_information():
    assert minimum_blocklength_normal_approximation(
        0.0, 1, 1.0e-3, max_blocklength=1000) is None
