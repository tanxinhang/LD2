"""Heuristic marginal-utility greedy P0 inner solver.

(Historically labelled "monotone submodular" — that is INCORRECT under the
current utility; see the utility note below and docs/KNOWN_ISSUES.md B8.)

Solves the inner-layer problem: given geometry (UAV/target positions,
roles, Deflection entries), select bistatic reporting assignments
z_ijq ∈ {0, 1} to maximize weighted detection utility subject to:

  (C1) Capacity:  sum_{i,q} z_{ij,q} * B_q <= C_j_report   per rx UAV j
  (C2) Latency:   sum_{i,q} z_{ij,q} * (B_q / R_j) <= T_max   per rx UAV j
  (C3) Cardinality: |S_q| <= K_q_max   per target q

Utility: U_q(D_q) = -log(1 - P_D(D_q)), monotone INCREASING but NOT concave in
D_q (see docs/KNOWN_ISSUES.md B8: convex in P_D; U''(D)>0 over ~99.6% of range).
Hence F(S) = sum_q ω_q * U_q(Σ_{e∈S_q} d_eff_e) is NOT submodular and the greedy
below is a HEURISTIC with no (1-1/e) guarantee. A saturating utility concave in
D (e.g. 1-exp(-kD)) would restore submodularity.

The greedy algorithm (iteratively add the element with the highest marginal
gain) achieves at least (1 - 1/e) ≈ 63% of the optimal value.
"""

import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from itertools import combinations, product

from uav_isac.utils.types import DeflectionEntry, P0Solution
from uav_isac.utils.math_utils import marginal_utility_gain


class InnerSolver:
    """P0 inner-layer solver: heuristic marginal-utility greedy + exhaustive verifier.
    (Not submodular under the current non-concave utility; see KNOWN_ISSUES B8.)"""

    def __init__(
        self,
        K_q_max: int = 3,
        B_q: int = 64,
        capacity_per_rx: int = 256,
        latency_max: float = 0.005,
        omega_q: Optional[np.ndarray] = None,
        P_FA: float = 0.001,
        P_D_min: float = 0.8,
    ):
        """
        Args:
            K_q_max: Max reporting UAVs per target (cardinality constraint)
            B_q: Soft information bits per report
            capacity_per_rx: Max bits per frame per receiving UAV
            latency_max: Max reporting latency per rx UAV (s)
            omega_q: Target priority weights, shape (Q,); defaults to uniform
            P_FA: False alarm probability
            P_D_min: Minimum detection probability (fairness constraint)
        """
        self.K_q_max = K_q_max
        self.B_q = B_q
        self.capacity_per_rx = capacity_per_rx
        self.latency_max = latency_max
        self.P_FA = P_FA
        self.P_D_min = P_D_min
        self._omega_q = omega_q

    def _get_omega(self, Q: int) -> np.ndarray:
        if self._omega_q is not None and len(self._omega_q) == Q:
            return np.asarray(self._omega_q, dtype=np.float64)
        return np.ones(Q, dtype=np.float64) / Q

    def solve(
        self,
        deflection_entries: List[DeflectionEntry],
        reporting_rates: Optional[np.ndarray] = None,  # (K,) bps per rx UAV
        Q: Optional[int] = None,
        K: Optional[int] = None,
        enforce_single_role: bool = False,
        # B3: Uncertainty-aware scoring (all optional, default off)
        belief_cov_diag: Optional[np.ndarray] = None,  # (Q, 4) fused cov diagonal
        belief_aoi: Optional[np.ndarray] = None,        # (Q,) age of information
        beta_uncertainty: float = 0.0,                   # uncertainty penalty weight
        eta_aoi: float = 0.0,                            # AoI urgency weight
        # ── Layer 3: Safe P0 with bounded fusion correction ──
        fusion_confidence: Optional[np.ndarray] = None,  # (Q,) per-target trust [0,1]
        fusion_confidence_min: float = 0.3,              # min confidence for B3
        # ── DU-P0: Decision-Uncertainty-aware scheduling ──
        du_enabled: bool = False,                        # enable DU-P0
        du_ambiguity_threshold: float = 3.0,             # A_q above this → probe mode
        du_ambiguity_bonus: float = 0.1,                 # γ: bonus weight for ambiguous targets
    ) -> P0Solution:
        """HEURISTIC greedy assignment by marginal detection utility.

        NOTE: this is NOT a submodular maximization with a (1-1/e) guarantee.
        That guarantee would require each per-target utility f_q(D) to be
        monotone AND concave in D. The current utility U(D) = -log(1-P_D(D))
        is monotone increasing but EMPIRICALLY NON-CONCAVE in D (U''>0 over
        ~99.6% of the relevant range; marginal gain is increasing, not
        diminishing). So treat this as a heuristic. To recover the guarantee,
        switch to a saturating utility concave in D, e.g. 1-exp(-kD).
        See docs/KNOWN_ISSUES.md B8.

        Algorithm:
        1. Filter to valid entries (d_eff > 0)
        2. Initialize S = {}, D_q = 0 for all q, capacity/latency budgets
        3. Greedy loop:
           a. For each candidate e not in S:
              - Compute marginal gain Δ = Σ_q ω_q [U_q(D_q + d_eff_e) - U_q(D_q)]
              - Check all constraints if e is added
              - Keep e if feasible and Δ > 0
           b. Select e* = argmax Δ
           c. If no feasible positive gain: break
           d. Add e* to S, update D_q, budgets
        4. Return P0Solution

        Args:
            deflection_entries: List of DeflectionEntry objects
            reporting_rates: (K,) data rate per rx UAV in bps; if None, use infinite
            Q: Number of targets (inferred if None)
            K: Number of UAVs (inferred if None)

        Returns:
            P0Solution with selected assignments and per-target Deflection
        """
        # Filter valid entries
        valid = [e for e in deflection_entries if e.d_eff > 0]

        if not valid:
            if Q is None:
                Q = 0
            return P0Solution(
                z_selected=np.zeros((K or 0, K or 0, Q), dtype=np.int32),
                D_q_star=np.zeros(Q, dtype=np.float64),
                U_q=np.zeros(Q, dtype=np.float64),
                selected_set=[],
                total_bits=0.0,
                total_latency=0.0
            )

        # Infer dimensions
        if Q is None:
            Q = max(e.q for e in valid) + 1
        if K is None:
            K = max(max(e.i for e in valid), max(e.j for e in valid)) + 1

        omega = self._get_omega(Q)

        # Initialize
        selected: Set[Tuple[int, int, int]] = set()
        D_q = np.zeros(Q, dtype=np.float64)
        remaining_capacity = {j: float(self.capacity_per_rx) for j in range(K)}
        remaining_latency = {j: float(self.latency_max) for j in range(K)}
        target_counts = {q: 0 for q in range(Q)}
        # One-role-per-UAV bookkeeping (used when roles are assigned by P0 rather
        # than the policy): a UAV may be tx for several targets or rx for several,
        # but never both tx and rx in the same frame.
        uav_role: Dict[int, str] = {}

        # Build candidate index for fast lookup
        candidates = list(valid)

        # ── DU-P0: compute per-target ambiguity A_q ──
        du_ambiguity = np.zeros(Q, dtype=np.float64)
        if du_enabled and belief_cov_diag is not None and len(candidates) > 0:
            for q in range(Q):
                q_gains = []
                for e in candidates:
                    if e.q == q:
                        gain = marginal_utility_gain(0.0, e.d_eff, self.P_FA)
                        q_gains.append((gain, e.d_eff, e))
                q_gains.sort(key=lambda x: -x[0])
                if len(q_gains) >= 2:
                    delta_mu = abs(q_gains[0][0] - q_gains[1][0])
                    cov_trace = float(np.mean(np.abs(belief_cov_diag[q, :4])))
                    sigma_pos = np.sqrt(max(cov_trace, 1e-8))
                    sigma_u = sigma_pos * (q_gains[0][0] / (q_gains[0][1] + 1e-6))
                    du_ambiguity[q] = sigma_u / (delta_mu + 1e-8)

        while True:
            best_gain = -1.0
            best_entry = None

            for e in candidates:
                key = (e.i, e.j, e.q)
                if key in selected:
                    continue

                # Role-consistency: forbid a UAV being both tx and rx this frame.
                if enforce_single_role and (
                    uav_role.get(e.i) == 'rx' or uav_role.get(e.j) == 'tx'
                ):
                    continue

                # Check capacity constraint (C1)
                if self.B_q > remaining_capacity.get(e.j, 0):
                    continue

                # Check cardinality constraint (C3)
                if target_counts.get(e.q, 0) >= self.K_q_max:
                    continue

                # Check latency constraint (C2)
                if reporting_rates is not None and e.j < len(reporting_rates):
                    R_j = reporting_rates[e.j]
                    if R_j > 0:
                        latency_contribution = self.B_q / R_j
                        if latency_contribution > remaining_latency.get(e.j, 0):
                            continue

                # Compute marginal gain
                gain = marginal_utility_gain(D_q[e.q], e.d_eff, self.P_FA)
                weighted_gain = omega[e.q] * gain

                # B3: Uncertainty-aware scoring.
                # Covariance is in raw units (m² for position, (m/s)² for velocity).
                # Normalize by scenario scale so β works across different area sizes.
                # ── Layer 3: Safe P0 — only apply B3 when fusion is trusted ──
                apply_b3 = True
                if fusion_confidence is not None:
                    apply_b3 = float(fusion_confidence[e.q]) >= fusion_confidence_min

                if apply_b3:
                    if beta_uncertainty > 0 and belief_cov_diag is not None:
                        cov_mean = float(np.mean(np.abs(belief_cov_diag[e.q])))
                        # Use sqrt(cov_mean) normalized by a reference scale (~100m)
                        uncertainty = np.sqrt(max(cov_mean, 1e-8)) / 100.0
                        weighted_gain -= beta_uncertainty * uncertainty

                    if eta_aoi > 0 and belief_aoi is not None:
                        # Reward freshness proportional to AoI (higher AoI = more urgent)
                        aoi_val = float(belief_aoi[e.q])
                        weighted_gain += eta_aoi * aoi_val

                # ── DU-P0: ambiguity bonus for uncertain targets ──
                if du_enabled and du_ambiguity[e.q] > du_ambiguity_threshold:
                    weighted_gain += du_ambiguity_bonus * du_ambiguity[e.q] * e.d_eff

                if weighted_gain > best_gain:
                    best_gain = weighted_gain
                    best_entry = e

            # Stop if no positive gain
            if best_entry is None or best_gain <= 1e-12:
                break

            # Add best entry
            e = best_entry
            key = (e.i, e.j, e.q)
            selected.add(key)
            if enforce_single_role:
                uav_role[e.i] = 'tx'
                uav_role[e.j] = 'rx'
            D_q[e.q] += e.d_eff
            remaining_capacity[e.j] -= self.B_q
            if reporting_rates is not None and e.j < len(reporting_rates):
                R_j = reporting_rates[e.j]
                if R_j > 0:
                    remaining_latency[e.j] -= self.B_q / R_j
            target_counts[e.q] += 1

        # Build output
        z_selected = np.zeros((K, K, Q), dtype=np.int32)
        for (i, j, q) in selected:
            z_selected[i, j, q] = 1

        total_bits = len(selected) * self.B_q

        from uav_isac.physical.detection import compute_target_utilities
        U_q = compute_target_utilities(D_q, self.P_FA)

        return P0Solution(
            z_selected=z_selected,
            D_q_star=D_q,
            U_q=U_q,
            selected_set=list(selected),
            total_bits=float(total_bits),
            total_latency=0.0  # simplified; can be computed from rates
        )

    def solve_exhaustive(
        self,
        deflection_entries: List[DeflectionEntry],
        reporting_rates: Optional[np.ndarray] = None,
        Q: Optional[int] = None,
        K: Optional[int] = None,
    ) -> P0Solution:
        """Brute-force optimal solution for verification (small scale only).

        Enumerates all subsets of deflection entries and picks the one
        maximizing weighted utility subject to constraints.
        Only feasible for K ≤ 5, Q ≤ 3, entries ≤ 20.

        Args:
            deflection_entries: List of DeflectionEntry objects
            reporting_rates: (K,) data rate per rx UAV
            Q: Number of targets
            K: Number of UAVs

        Returns:
            Optimal P0Solution
        """
        valid = [e for e in deflection_entries if e.d_eff > 0]

        if not valid:
            if Q is None:
                Q = 0
            return P0Solution(
                z_selected=np.zeros((K or 0, K or 0, Q), dtype=np.int32),
                D_q_star=np.zeros(Q, dtype=np.float64),
                U_q=np.zeros(Q, dtype=np.float64),
                selected_set=[],
                total_bits=0.0,
                total_latency=0.0
            )

        if Q is None:
            Q = max(e.q for e in valid) + 1
        if K is None:
            K = max(max(e.i for e in valid), max(e.j for e in valid)) + 1

        omega = self._get_omega(Q)
        n = len(valid)

        # Safety check: 2^n grows fast
        if n > 20:
            raise ValueError(f"Exhaustive search infeasible: {n} entries → 2^{n} subsets")

        from uav_isac.physical.detection import compute_target_utilities

        best_utility = -np.inf
        best_subset = None
        best_D_q: np.ndarray = np.zeros(Q, dtype=np.float64)

        # Enumerate all subsets
        for r in range(n + 1):
            for indices in combinations(range(n), r):
                D_q = np.zeros(Q, dtype=np.float64)
                capacity_used = {j: 0 for j in range(K)}
                target_counts = {q: 0 for q in range(Q)}
                feasible = True

                for idx in indices:
                    e = valid[idx]

                    # Check constraints
                    if capacity_used.get(e.j, 0) + self.B_q > self.capacity_per_rx:
                        feasible = False
                        break
                    if target_counts.get(e.q, 0) >= self.K_q_max:
                        feasible = False
                        break

                    capacity_used[e.j] = capacity_used.get(e.j, 0) + self.B_q
                    target_counts[e.q] = target_counts.get(e.q, 0) + 1
                    D_q[e.q] += e.d_eff

                if not feasible:
                    continue

                U_q = compute_target_utilities(D_q, self.P_FA)
                utility = float(np.dot(omega, U_q))

                if utility > best_utility:
                    best_utility = utility
                    best_subset = set(valid[idx] for idx in indices)
                    best_D_q = D_q.copy()

        if best_subset is None:
            best_subset = set()
            best_D_q = np.zeros(Q, dtype=np.float64)

        z_selected = np.zeros((K, K, Q), dtype=np.int32)
        for e in best_subset:
            z_selected[e.i, e.j, e.q] = 1

        total_bits = len(best_subset) * self.B_q
        U_q = compute_target_utilities(best_D_q, self.P_FA)

        return P0Solution(
            z_selected=z_selected,
            D_q_star=best_D_q,
            U_q=U_q,
            selected_set=[(e.i, e.j, e.q) for e in best_subset],
            total_bits=float(total_bits),
            total_latency=0.0
        )

    def compute_greedy_gap(
        self,
        deflection_entries: List[DeflectionEntry],
        reporting_rates: Optional[np.ndarray] = None,
        Q: Optional[int] = None,
        K: Optional[int] = None,
    ) -> dict:
        """Compare greedy vs exhaustive and report the optimality gap.

        Returns:
            dict with keys: greedy_utility, optimal_utility,
            relative_gap (0 = optimal, 1 = greedy achieved nothing),
            greedy_is_optimal (bool)
        """
        greedy_sol = self.solve(deflection_entries, reporting_rates, Q, K)
        exhaust_sol = self.solve_exhaustive(deflection_entries, reporting_rates, Q, K)

        if exhaust_sol.U_q is None or len(exhaust_sol.U_q) == 0:
            return {
                'greedy_utility': 0.0,
                'optimal_utility': 0.0,
                'relative_gap': 0.0,
                'greedy_is_optimal': True
            }

        # Compute total weighted utility
        omega = self._get_omega(len(exhaust_sol.U_q)) if exhaust_sol.U_q is not None else np.array([])
        if len(omega) == 0:
            return {'greedy_utility': 0.0, 'optimal_utility': 0.0,
                    'relative_gap': 0.0, 'greedy_is_optimal': True}

        greedy_U = float(np.dot(omega, greedy_sol.U_q))
        optimal_U = float(np.dot(omega, exhaust_sol.U_q))

        if optimal_U <= 0:
            relative_gap = 0.0 if greedy_U == 0.0 else 1.0
        else:
            relative_gap = max(0.0, (optimal_U - greedy_U) / optimal_U)

        return {
            'greedy_utility': greedy_U,
            'optimal_utility': optimal_U,
            'relative_gap': relative_gap,
            'greedy_is_optimal': abs(greedy_U - optimal_U) < 1e-10
        }
