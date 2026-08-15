from types import SimpleNamespace

import numpy as np
import pytest

from uav_isac.evaluation.physical_oracle_audit import (
    per_watt_deflection_tensor,
    per_watt_deflection_tensor_from_observables,
    summarize_physical_feasibility_oracles,
)


def test_per_watt_deflection_recovers_linear_gain():
    entries = [
        SimpleNamespace(i=0, j=1, q=0, d_eff=4.0),
        SimpleNamespace(i=1, j=0, q=0, d_eff=3.0),
    ]
    power = np.array([[0.5], [0.25]])
    coefficient = per_watt_deflection_tensor(
        entries, power, num_uavs=2, num_targets=1)
    assert coefficient[0, 1, 0] == 8.0
    assert coefficient[1, 0, 0] == 12.0
    assert coefficient[0, 0, 0] == 0.0


def test_observable_reconstruction_covers_zero_power_counterfactual_edges():
    alpha = np.zeros((2, 2, 2), dtype=np.float64)
    alpha[0, 1] = [2.0e-7, 3.0e-7]
    g_dd = np.ones_like(alpha)
    chi_rep = np.ones_like(alpha)
    coefficient = per_watt_deflection_tensor_from_observables(
        alpha,
        g_dd,
        chi_rep,
        T_sym=1.0e-4,
        M=4,
        N=2,
        kT=4.0e-21,
        bandwidth_hz=1.0e6,
        noise_figure_db=0.0,
        g_tx_dbi=0.0,
        g_rx_dbi=0.0,
        n_cpi=1,
        g_min=0.5,
    )
    scale = 1.0e-4 * 4 * 2 / (4.0e-21 * 1.0e6)
    np.testing.assert_allclose(
        coefficient[0, 1], alpha[0, 1] ** 2 * scale)
    assert coefficient[1, 0, 0] == 0.0


def test_observable_reconstruction_applies_dd_gate_and_rejects_swerling():
    alpha = np.ones((2, 2, 1), dtype=np.float64) * 1.0e-7
    g_dd = np.ones_like(alpha)
    g_dd[0, 1, 0] = 0.4
    chi_rep = np.ones_like(alpha)
    kwargs = dict(
        T_sym=1.0e-4, M=4, N=2, kT=4.0e-21,
        bandwidth_hz=1.0e6, noise_figure_db=0.0,
        g_tx_dbi=0.0, g_rx_dbi=0.0, n_cpi=1, g_min=0.5,
    )
    coefficient = per_watt_deflection_tensor_from_observables(
        alpha, g_dd, chi_rep, **kwargs)
    assert coefficient[0, 1, 0] == 0.0
    with pytest.raises(ValueError, match="Swerling"):
        per_watt_deflection_tensor_from_observables(
            alpha, g_dd, chi_rep, use_swerling=True, **kwargs)


def test_physical_oracle_summary_separates_pair_and_duplex_gaps():
    row = {
        "fusion_mode": "local_only",
        "deployed_worst": 0.40,
        "deployed_weak3": 0.60,
        "deployed_steady": 0.70,
        "pair_only_worst": 0.52,
        "pair_only_weak3": 0.68,
        "pair_only_steady": 0.75,
        "power_only_worst": 0.48,
        "power_only_weak3": 0.65,
        "power_only_steady": 0.73,
        "single_worst": 0.65,
        "single_weak3": 0.75,
        "single_steady": 0.80,
        "duplex_worst": 0.75,
        "duplex_weak3": 0.82,
        "duplex_steady": 0.86,
        "single_worst_gap": 0.25,
        "pair_only_worst_gap": 0.12,
        "power_only_worst_gap": 0.08,
        "joint_over_best_isolated_worst_gap": 0.13,
        "duplex_worst_gap": 0.35,
        "duplex_over_single_worst_gap": 0.10,
    }
    summary = summarize_physical_feasibility_oracles(
        [[dict(row)] for _ in range(10)],
        bootstrap_samples=20,
    )
    assert summary["eval_physical_oracle_single_gate_pass"] is True
    assert summary["eval_physical_oracle_pair_only_gate_pass"] is True
    assert summary["eval_physical_oracle_duplex_gate_pass"] is True
    assert summary["eval_physical_oracle_single_feasible_rate"] == 1.0
    assert summary["eval_physical_oracle_fusion_modes"] == ["local_only"]
