"""Critical tests for the P0 inner solver.

Validates:
1. Greedy solution satisfies all constraints
2. Greedy vs exhaustive gap ≤ 30% for small scale
3. Marginal gain is non-negative and correctly computed
4. Empty input produces valid empty output
"""

import numpy as np
import pytest
from typing import Optional
from uav_isac.physical.inner_solver import InnerSolver
from uav_isac.utils.types import DeflectionEntry


def make_deflection_entries(
    K: int, Q: int,
    d_eff_values: Optional[list] = None,
    rng: Optional[np.random.Generator] = None
) -> list:
    """Create synthetic DeflectionEntry list for testing.

    Creates entries for: tx={0}, rx={1..K-1}, all targets.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    entries = []
    idx = 0
    for i in range(K):
        if roles_check(i, K):  # only tx UAVs
            for j in range(K):
                if i != j:  # cannot be same UAV
                    for q in range(Q):
                        if d_eff_values is not None and idx < len(d_eff_values):
                            d_eff = d_eff_values[idx]
                        else:
                            d_eff = rng.uniform(0.5, 5.0)
                        idx += 1
                        entries.append(DeflectionEntry(
                            i=i, j=j, q=q,
                            tau=1e-6, nu=100.0, alpha=1e-6,
                            d_raw=d_eff, g_dd=1.0, chi_rep=1.0,
                            d_eff=d_eff
                        ))
    return entries


def roles_check(i: int, K: int) -> bool:
    """Simple role assignment: first half tx, second half rx."""
    return i < K // 2


class TestInnerSolverConstraints:
    """Verify that the greedy solution satisfies all constraints."""

    def test_capacity_constraint(self):
        """No rx UAV receives more than capacity_per_rx bits."""
        K, Q = 4, 2
        solver = InnerSolver(K_q_max=3, B_q=64, capacity_per_rx=200,
                             latency_max=0.1, P_FA=0.001)

        entries = make_deflection_entries(K, Q)

        sol = solver.solve(entries, Q=Q, K=K)

        # Check capacity per rx
        bits_per_rx = {}
        for (i, j, q) in sol.selected_set:
            bits_per_rx[j] = bits_per_rx.get(j, 0) + 64  # B_q=64

        for j, bits in bits_per_rx.items():
            assert bits <= 200, f"Rx UAV {j} exceeded capacity: {bits} > 200"

    def test_cardinality_constraint(self):
        """No target has more than K_q_max reporting sources."""
        K, Q = 4, 2
        K_q_max = 2
        solver = InnerSolver(K_q_max=K_q_max, B_q=64, capacity_per_rx=1000,
                             latency_max=0.1, P_FA=0.001)

        entries = make_deflection_entries(K, Q)
        sol = solver.solve(entries, Q=Q, K=K)

        target_counts = {}
        for (i, j, q) in sol.selected_set:
            target_counts[q] = target_counts.get(q, 0) + 1

        for q, count in target_counts.items():
            assert count <= K_q_max, f"Target {q} exceeded cardinality: {count} > {K_q_max}"

    def test_no_self_pair_selection(self):
        """UAV should not select itself as both tx and rx."""
        K, Q = 4, 2
        solver = InnerSolver(K_q_max=3, B_q=64, capacity_per_rx=1000,
                             latency_max=0.1, P_FA=0.001)

        entries = make_deflection_entries(K, Q)
        sol = solver.solve(entries, Q=Q, K=K)

        for (i, j, q) in sol.selected_set:
            assert i != j, f"Self-pair selected: i={i}, j={j}"


class TestGreedyVsExhaustive:
    """Compare greedy against brute-force exhaustive search."""

    def test_small_scale_gap(self):
        """For K=3, Q=1 (small), greedy gap should be ≤ 30%."""
        K, Q = 3, 1

        # Create explicit deflection values for reproducibility
        # tx UAVs: 0,1  rx UAVs: 2
        # Entries: (0,2,0) and (1,2,0)
        entries = [
            DeflectionEntry(i=0, j=2, q=0, tau=1e-6, nu=100.0, alpha=1e-6,
                           d_raw=3.0, g_dd=1.0, chi_rep=1.0, d_eff=3.0),
            DeflectionEntry(i=1, j=2, q=0, tau=1e-6, nu=100.0, alpha=1e-6,
                           d_raw=5.0, g_dd=1.0, chi_rep=1.0, d_eff=5.0),
        ]

        solver = InnerSolver(K_q_max=3, B_q=64, capacity_per_rx=500,
                             latency_max=0.1, P_FA=0.001)

        result = solver.compute_greedy_gap(entries, Q=Q, K=K)

        # Both entries should be selected (no constraint violation)
        assert result['greedy_is_optimal'] or result['relative_gap'] <= 0.30, \
            f"Gap too large: {result['relative_gap']:.4f}"

    def test_with_binding_capacity(self):
        """When capacity binds, greedy should still be near-optimal."""
        entries = [
            DeflectionEntry(i=0, j=2, q=0, tau=1e-6, nu=100.0, alpha=1e-6,
                           d_raw=3.0, g_dd=1.0, chi_rep=1.0, d_eff=3.0),
            DeflectionEntry(i=0, j=3, q=0, tau=1e-6, nu=100.0, alpha=1e-6,
                           d_raw=4.0, g_dd=1.0, chi_rep=1.0, d_eff=4.0),
            DeflectionEntry(i=1, j=2, q=0, tau=1e-6, nu=100.0, alpha=1e-6,
                           d_raw=5.0, g_dd=1.0, chi_rep=1.0, d_eff=5.0),
        ]

        # Capacity only allows 1 entry per rx
        solver = InnerSolver(K_q_max=3, B_q=64, capacity_per_rx=80,
                             latency_max=0.1, P_FA=0.001)

        result = solver.compute_greedy_gap(entries, Q=1, K=4)

        assert result['relative_gap'] <= 0.30, \
            f"Gap too large with binding capacity: {result['relative_gap']:.4f}"

    def test_gap_for_multiple_configs(self):
        """Test greedy vs exhaustive for various random configs."""
        rng = np.random.default_rng(42)
        solver = InnerSolver(K_q_max=2, B_q=64, capacity_per_rx=200,
                             latency_max=0.1, P_FA=0.001)

        max_gap = 0.0
        for seed in range(20):
            rng_s = np.random.default_rng(seed)
            # K=3, Q=2: max entries = 2 tx * 2 rx * 2 Q = 8 entries
            entries = make_deflection_entries(3, 2, rng=rng_s)
            result = solver.compute_greedy_gap(entries, Q=2, K=3)
            max_gap = max(max_gap, result['relative_gap'])

        # Max gap across all configs should be ≤ 30%
        assert max_gap <= 0.30, f"Max gap {max_gap:.4f} exceeds 30%"


class TestEmptyInput:
    """Verify behavior with empty/no-valid entries."""

    def test_empty_entries(self):
        solver = InnerSolver(K_q_max=3, B_q=64, capacity_per_rx=200,
                             latency_max=0.1, P_FA=0.001)

        sol = solver.solve([], Q=2, K=4)

        assert len(sol.selected_set) == 0
        assert np.all(sol.D_q_star == 0)
        assert sol.total_bits == 0

    def test_zero_effective_deflection(self):
        """Entries with d_eff=0 should be ignored."""
        entries = [
            DeflectionEntry(i=0, j=2, q=0, tau=1e-6, nu=100.0, alpha=1e-6,
                           d_raw=5.0, g_dd=0.3, chi_rep=1.0, d_eff=0.0),
        ]
        solver = InnerSolver(K_q_max=3, B_q=64, capacity_per_rx=200,
                             latency_max=0.1, P_FA=0.001)

        sol = solver.solve(entries, Q=2, K=4)
        assert len(sol.selected_set) == 0


class TestMarginalGain:
    """Verify marginal gain computation properties."""

    def test_gain_positive_for_positive_deflection(self):
        """Adding a positive deflection should increase utility."""
        from uav_isac.utils.math_utils import marginal_utility_gain

        gain = marginal_utility_gain(D_q_current=0.0, d_eff_new=5.0, P_FA=0.001)
        assert gain > 0

    def test_gain_diminishing(self):
        """Marginal gain eventually diminishes at sufficiently high D_q.

        Note: -log(1-P_D) is S-shaped — convex at low D_q, concave at high D_q.
        With very low P_FA (0.001), the concave regime starts at high D_q.
        We test at D_q values where the function is definitely concave.
        """
        from uav_isac.utils.math_utils import marginal_utility_gain

        # At very high D_q (P_D close to 1), diminishing returns must hold
        gain_at_100 = marginal_utility_gain(D_q_current=100.0, d_eff_new=10.0, P_FA=0.001)
        gain_at_200 = marginal_utility_gain(D_q_current=200.0, d_eff_new=10.0, P_FA=0.001)

        assert gain_at_200 < gain_at_100, \
            f"gain(100→110)={gain_at_100:.6f}, gain(200→210)={gain_at_200:.6f}"

    def test_gain_zero_for_zero_deflection(self):
        """Adding zero deflection gives zero gain."""
        from uav_isac.utils.math_utils import marginal_utility_gain

        gain = marginal_utility_gain(D_q_current=5.0, d_eff_new=0.0, P_FA=0.001)
        assert gain == pytest.approx(0.0, abs=1e-12)
