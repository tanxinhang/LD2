"""Tests for detection probability computation."""

import numpy as np
import pytest
from uav_isac.physical.detection import (
    compute_detection_probabilities,
    gaussian_shift_parameters,
    minimum_deflection_for_detection_probability,
    monte_carlo_gaussian_shift_roc,
    compute_target_utilities,
    compute_weighted_utility,
    compute_team_reward,
)
from uav_isac.utils.math_utils import compute_PD, Q_function
from tools.audit_detector_normalization import (
    certification_readiness,
    link_budget_sanity,
)


class TestPDProperties:
    def test_pd_approaches_pfa_when_d_zero(self):
        """P_D → P_FA as D_q → 0."""
        P_FA = 0.001
        D_q = np.array([0.0])
        P_D = compute_PD(D_q, P_FA)
        assert P_D[0] == pytest.approx(P_FA, rel=0.1)

    def test_pd_approaches_one_when_d_large(self):
        """P_D → 1 as D_q → ∞."""
        P_FA = 0.001
        D_q = np.array([1e6])
        P_D = compute_PD(D_q, P_FA)
        assert P_D[0] > 0.9999

    def test_pd_monotonic_in_d(self):
        """P_D strictly increases with D_q."""
        P_FA = 0.001
        D_small = np.array([1.0])
        D_large = np.array([10.0])
        pd_small = compute_PD(D_small, P_FA)
        pd_large = compute_PD(D_large, P_FA)
        assert pd_large[0] > pd_small[0]

    def test_pd_in_unit_interval(self):
        """P_D is always in [0, 1]."""
        P_FA = 0.001
        for d in [0.0, 0.1, 1.0, 10.0, 100.0, 1000.0]:
            pd = compute_PD(np.array([d]), P_FA)
            assert 0.0 <= pd[0] <= 1.0, f"P_D={pd[0]} for D={d}"


class TestDetectionProbabilities:
    def test_link_budget_sanity_has_inverse_fourth_power_range_law(self):
        audit = link_budget_sanity([100.0, 300.0, 500.0, 800.0])
        rows = audit["rows"]
        assert all(
            rows[index]["after_n_cpi_deflection"]
            > rows[index + 1]["after_n_cpi_deflection"]
            for index in range(len(rows) - 1)
        )
        assert rows[0]["received_snr"] / rows[1]["received_snr"] \
            == pytest.approx(3.0**4)
        assert audit["cpi_exceeds_control_frame"] == (
            audit["cpi_duration_s"] > audit["control_frame_s"])
        assert audit["n_cpi"] == 1
        assert audit["phantom_look_free"]
        assert audit["max_time_feasible_looks"] == 97
        assert audit["power_cap_respected"]
        assert audit["sensing_power_w"] == audit["sensing_power_cap_w"]

    def test_declared_gaussian_shift_has_deflection_definition(self):
        values = np.asarray([0.0, 5.0, 10.0, 20.0])
        mu0, mu1, var0, var1 = gaussian_shift_parameters(values)
        np.testing.assert_allclose(np.square(mu1 - mu0) / var0, values)
        np.testing.assert_allclose(var0, var1)

    def test_link_budget_audit_rejects_sensing_power_above_hardware_cap(self):
        with pytest.raises(ValueError, match="P_sense_max"):
            link_budget_sanity([100.0], sensing_power_w=0.0252)

    def test_certification_gate_blocks_saturated_overlong_cpi(self):
        result = {
            "link_budget": {
                "saturated_all": True,
                "cpi_exceeds_control_frame": True,
                "phantom_look_free": True,
                "power_cap_respected": True,
            },
            "roc_monte_carlo": {
                "empirical_pd": [0.2, 0.5],
                "analytical_pd": [0.201, 0.499],
            },
        }
        gate = certification_readiness(result)
        assert not gate["ready_for_g2_1"]
        assert gate["blockers"] == [
            "link_budget_saturated_at_all_audit_ranges",
            "declared_cpi_exceeds_control_frame",
        ]

    def test_monte_carlo_roc_matches_analytical_mapping(self):
        audit = monte_carlo_gaussian_shift_roc(
            np.asarray([5.0, 10.0, 15.0, 20.0]),
            0.001,
            samples=300_000,
            seed=20260820,
        )
        assert audit["empirical_pfa"] == pytest.approx(0.001, abs=2.5e-4)
        np.testing.assert_allclose(
            audit["empirical_pd"], audit["analytical_pd"], atol=3e-3)
        np.testing.assert_allclose(
            audit["empirical_deflection"],
            audit["analytical_deflection"],
            atol=0.10,
        )

    def test_output_shape(self):
        D_q = np.array([5.0, 10.0])
        P_D = compute_detection_probabilities(D_q, P_FA=0.001)
        assert P_D.shape == (2,)
        assert np.all(P_D >= 0)
        assert np.all(P_D <= 1)

    def test_multi_target_ordering(self):
        """Target with higher Deflection has higher P_D."""
        D_q = np.array([5.0, 50.0])
        P_D = compute_detection_probabilities(D_q, P_FA=0.001)
        assert P_D[1] > P_D[0]

    def test_probability_inverse_recovers_minimum_deflection(self):
        requested = np.asarray([0.2, 0.6, 0.9])
        deflection = minimum_deflection_for_detection_probability(
            requested, P_FA=0.001)
        reconstructed = compute_detection_probabilities(
            deflection, P_FA=0.001)
        np.testing.assert_allclose(reconstructed, requested, atol=1e-10)


class TestTargetUtilities:
    def test_utility_monotonic(self):
        """Utility increases with D_q."""
        D_small = np.array([1.0])
        D_large = np.array([10.0])
        U_small = compute_target_utilities(D_small, P_FA=0.001)
        U_large = compute_target_utilities(D_large, P_FA=0.001)
        assert U_large[0] > U_small[0]

    def test_utility_positive(self):
        """Utility is always positive."""
        D_q = np.array([0.0, 5.0, 50.0])
        U = compute_target_utilities(D_q, P_FA=0.001)
        assert np.all(U >= 0)


class TestTeamReward:
    def test_reward_decreases_with_communication(self):
        """Higher communication cost → lower team reward."""
        D_q = np.array([10.0, 10.0])
        omega = np.array([0.5, 0.5])
        r_low_bits = compute_team_reward(D_q, 0.001, omega, total_bits=0, lambda_report=0.01)
        r_high_bits = compute_team_reward(D_q, 0.001, omega, total_bits=500, lambda_report=0.01)
        assert r_high_bits < r_low_bits
