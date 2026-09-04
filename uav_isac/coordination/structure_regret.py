"""Closed-loop task-regret certificate for distributed structure proxies.

advice/016 (2026-08-17): the structure Student is calibrated not by imitation
accuracy but by the SENSING-TASK REGRET it induces.  For a fixed structure the
max-min sensing power LP has the dual

    t*(A) = min_{lambda in simplex(Q)}  sum_i b_i max_q lambda_q a_iq,

where ``A`` is the (K,Q) per-watt gain matrix of the deployed structure and
``b_i`` the sensing budget.  The structural approximation error of a Student
(``A_hat`` vs the teacher's ``A``) propagates to the max-min capability as

    |t*(A) - t*(A_hat)| <= sum_i b_i max_q |a_iq - a_hat_iq|

which follows from ``|min f - min g| <= sup|f - g|`` together with the
1-Lipschitzness of the max over a simplex weight: for every lambda,

    | sum_i b_i max_q lambda_q a_iq - sum_i b_i max_q lambda_q a_hat_iq |
      <= sum_i b_i max_q |lambda_q (a_iq - a_hat_iq)|
      <= sum_i b_i max_q |a_iq - a_hat_iq|            (lambda_q <= 1).

This closes the chain

    structure-proxy error -> per-watt gain error -> max-min detection loss,

so the Student can be trained against an *upper bound on the task regret*
instead of raw imitation.  A QoS floor is preserved when the teacher's margin
``m = t*(A) - d_req > 0`` strictly exceeds the structural error bound.

The second certificate implements the decision-preserving communication bound
of advice/016 §13-14: for two candidate scores ``s1 > s2`` with margin
``delta = s1 - s2`` and quantization dynamic range ``R``, B-bit uniform
quantization carries per-score error ``eps_B <= R / (2(2^B - 1))``, so the
decision (order) is preserved iff ``delta > 2 eps_B``, i.e.

    B > log2(1 + R / delta).

Because the order condition is strict, the minimum integer is
``floor(log2(1 + R / delta)) + 1``.  A plain ceiling is unsafe when the
logarithm is already an integer: it permits a quantized tie.

An event trigger then suppresses transmission while the stale-score drift
bound ``E_stale(h)`` keeps ``delta > 2 E_stale(h) + 2 eps_B`` -- the minimal
communication that preserves the downstream ISAC coordination decision.
"""

from __future__ import annotations

import numpy as np

from uav_isac.coordination.maxmin_power import (
    canonical_maxmin_dual_prices,
    optimal_maxmin_dual_prices,
    solve_fixed_structure_maxmin_power_lp,
)


def tstar_of(gain_per_watt: np.ndarray, sensing_budget_w: np.ndarray) -> float:
    """Max-min capability ``t*(A)`` of a fixed structure (dual value)."""
    gain = np.asarray(gain_per_watt, dtype=np.float64)
    budget = np.asarray(sensing_budget_w, dtype=np.float64).reshape(-1)
    if gain.ndim != 2 or gain.shape[0] != budget.size:
        raise ValueError("gain must be (K,Q) matching budget (K,)")
    if not np.any(gain > 0.0) or not np.any(budget > 0.0):
        return 0.0
    _pi, value = optimal_maxmin_dual_prices(gain, budget)
    return float(value)


def structure_error_bound(
    sensing_budget_w: np.ndarray,
    gain_err_per_watt: np.ndarray,
) -> float:
    """``sum_i b_i * max_q |a_iq - a_hat_iq|``: the structural error bound.

    ``gain_err_per_watt`` is the per-watt gain deviation ``(K,Q)`` of the
    Student structure vs the teacher structure (signed; the absolute value is
    taken per entry as required by the max-norm).
    """
    budget = np.asarray(sensing_budget_w, dtype=np.float64).reshape(-1)
    err = np.asarray(gain_err_per_watt, dtype=np.float64)
    if err.ndim != 2 or err.shape[0] != budget.size:
        raise ValueError("gain_err must be (K,Q) matching budget (K,)")
    if np.any(~np.isfinite(err)):
        raise ValueError("gain_err must be finite")
    per_uav = np.max(np.abs(err), axis=1)
    return float(np.dot(budget, per_uav))


def qos_floor_preserved(
    tstar_teacher: float,
    d_req: float,
    error_bound: float,
    tol: float = 1e-9,
) -> bool:
    """``error_bound < t*(A) - d_req`` guarantees the worst QoS floor holds.

    Let ``m = t*(A) - d_req > 0`` be the teacher margin.  If the structural
    error bound is strictly below ``m`` then
    ``t*(A_hat) >= t*(A) - error_bound > d_req``, i.e. the Student structure
    cannot push the max-min floor below the QoS threshold (advice/016 §5).
    """
    margin = float(tstar_teacher) - float(d_req)
    return bool(margin > float(error_bound) + tol)


# ---------------------------------------------------------------------------
# decision-preserving communication (advice/016 §13-14)
# ---------------------------------------------------------------------------


def decision_preserving_bits(
    margin: float,
    dynamic_range: float,
) -> int:
    """Minimum uniform-quantization bits preserving a score order.

    ``margin = s1 - s2 > 0`` (the decision margin), ``dynamic_range = R`` the
    quantization range.  B-bit uniform quantization has per-value error
    ``eps_B <= R / (2(2^B - 1))``; the order survives iff
    ``margin > 2 eps_B``, giving

        B > log2( 1 + R / margin ).

    Returns ``0`` when the margin is so large that even 1 bit would suffice is
    NOT used (we keep >= 1 for a non-trivial quantizer); the caller decides
    whether to transmit via ``should_transmit``.
    """
    margin = float(margin)
    dynamic_range = float(dynamic_range)
    if (
        not np.isfinite(margin)
        or not np.isfinite(dynamic_range)
        or margin <= 0.0
        or dynamic_range <= 0.0
    ):
        return 1  # degenerate: cannot certify order preservation
    ratio = dynamic_range / margin
    if not np.isfinite(ratio):
        raise OverflowError("finite precision cannot certify this margin")
    bits = max(1, int(np.floor(np.log2(1.0 + ratio))) + 1)
    # Verify the defining strict inequality directly.  This also protects the
    # exact power-of-two boundary against floating-point log rounding.
    while not margin > 2.0 * quantization_error_bound(bits, dynamic_range):
        bits += 1
    while (
        bits > 1
        and margin > 2.0 * quantization_error_bound(
            bits - 1, dynamic_range)
    ):
        bits -= 1
    return bits


def certified_decision_preserving_bits(
    margin: float,
    dynamic_range: float,
    stale_drift_bound: float = 0.0,
    physical_error_bound: float = 0.0,
) -> int | None:
    """Bits after reserving margin for stale and physical-score errors.

    Returns ``None`` when ``margin - 2(E_stale+E_phys) <= 0``: no finite
    quantizer can certify the decision, so the caller must transmit a richer
    message, hold the previous safe decision, or fail closed.
    """
    values = np.asarray([
        margin,
        dynamic_range,
        stale_drift_bound,
        physical_error_bound,
    ], dtype=np.float64)
    if np.any(~np.isfinite(values)):
        return None
    effective = float(margin) - 2.0 * (
        max(float(stale_drift_bound), 0.0)
        + max(float(physical_error_bound), 0.0))
    if effective <= 0.0 or float(dynamic_range) <= 0.0:
        return None
    return decision_preserving_bits(effective, dynamic_range)


def quantization_error_bound(bits: int, dynamic_range: float) -> float:
    """``eps_B = R / (2(2^B - 1))``: per-score uniform quantization error."""
    bits = max(1, int(bits))
    levels = 2 ** bits - 1
    return float(dynamic_range) / (2.0 * levels)


def should_transmit(
    margin: float,
    stale_drift_bound: float,
    dynamic_range: float,
    bits: int,
    tol: float = 1e-12,
    physical_error_bound: float = 0.0,
) -> bool:
    """Event trigger: transmit only when the decision may flip.

    The score margin ``delta`` is known up to the stale drift ``E_stale(h)``
    (motion/CSI/AoI) and the quantization error ``eps_B``.  If
    ``delta > 2 E_stale(h) + 2 eps_B`` the order is CERTAIN to persist, so the
    Token can be suppressed; otherwise send (decision not certified).
    """
    eps = quantization_error_bound(bits, dynamic_range)
    uncertainty = max(float(stale_drift_bound), 0.0) + max(
        float(physical_error_bound), 0.0)
    return not bool(float(margin) > 2.0 * uncertainty + 2.0 * eps + tol)


def preserved_without_transmission(
    margin: float,
    stale_drift_bound: float,
    dynamic_range: float,
    bits: int,
    physical_error_bound: float = 0.0,
) -> bool:
    """Whether the decision stays certified if we suppress the Token."""
    return not should_transmit(
        margin,
        stale_drift_bound,
        dynamic_range,
        bits,
        physical_error_bound=physical_error_bound,
    )


# ---------------------------------------------------------------------------
# decision-sufficient local plan (advice/001 §12-13 / C6 closure)
# ---------------------------------------------------------------------------


def local_decision_margin(
    scores: np.ndarray,
    top_k: int = 2,
) -> float:
    """Local decision margin ``Δ = s_(1) - s_(2)`` from one viewer's candidates.

    ``scores`` is a per-candidate score vector (e.g. a UAV's per-target
    local-belief capability or its top-k owner bids).  The margin is the gap
    between the best and second-best candidate --- the slack that compression
    or staleness may eat without flipping the argmax (advice/001 §12).

    Returns ``0.0`` when fewer than two finite candidates exist (cannot
    certify an order), matching the fail-closed convention of the
    decision-preserving primitives.
    """
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return 0.0
    ordered = -np.sort(-finite)
    return float(max(ordered[0] - ordered[1], 0.0))


def decision_sufficient_plan(
    scores: np.ndarray,
    dynamic_range: float,
    stale_drift_bound: float = 0.0,
    physical_error_bound: float = 0.0,
    max_bits: int = 32,
    reference_bits: int | None = None,
) -> tuple[bool, int]:
    """One decision-sufficient token plan for a viewer's local candidates.

    Returns ``(transmit: bool, bits: int)``:

      - ``transmit=False`` when the local top-1/top-2 margin ``Δ`` satisfies
        ``Δ > 2(E_stale + E_phys) + 2 eps_B`` at the precision of the token
        receivers actually retain, so the decision is CERTAIN to persist and
        the token may be suppressed (event-triggered silence, no airtime).
      - otherwise ``transmit=True`` with ``bits`` = the minimum uniform-quant
        precision that preserves the order under the same error budget, capped
        at ``max_bits``.

    Uses only locally computable inputs; nothing here reads simulator truth.
    For the strict distributed identity this is the C6 hook that turns the
    per-frame full-state broadcast into a decision-preserving minimal token.
    """
    margin = local_decision_margin(scores)
    dynamic_range = float(max(dynamic_range, 1e-12))
    max_bits = max(1, int(max_bits))
    held_bits = max_bits if reference_bits is None else int(reference_bits)
    if held_bits < 1:
        # No common receiver-side reference exists.  Silence cannot preserve a
        # decision that peers have never received, so bootstrap at max width.
        return True, max_bits
    held_bits = min(held_bits, max_bits)
    if preserved_without_transmission(
        margin,
        stale_drift_bound,
        dynamic_range,
        held_bits,
        physical_error_bound=physical_error_bound,
    ):
        return False, 0
    needed = certified_decision_preserving_bits(
        margin, dynamic_range,
        stale_drift_bound=stale_drift_bound,
        physical_error_bound=physical_error_bound,
    )
    if needed is None:
        return True, max_bits  # cannot certify -> send richest available
    return True, min(max_bits, needed)


# ---------------------------------------------------------------------------
# dual-weighted training loss (advice/016 §6): w_iq = pi_q * p_iq
# ---------------------------------------------------------------------------


def dual_edge_weights(
    gain_per_watt: np.ndarray,
    sensing_budget_w: np.ndarray,
    min_power_floor: float = 0.0,
) -> np.ndarray:
    """Edge-level task-sensitivity weights ``w_iq = pi_q * p_iq``.

    ``pi`` is the max-min dual price (target scarcity) and ``p_iq`` the
    optimal sensing power of the fixed-owner max-min LP.  The envelope
    theorem gives ``partial t* / partial a_iq = pi_q * p_iq`` (advice/016 §6),
    so ``w_iq`` is exactly the marginal value of edge ``(i,q)`` to the
    max-min detection capability -- the correct training weight for the
    structure Student: errors on bottleneck targets served by scarce,
    high-power transmitters cost more (dual-consistent learning).

    Sparsity note (real-trace audit 2026-08-17): under the max-min optimum the
    power ``p_iq`` is zero on non-supporting (non-binding) transmitter-target
    pairs, so the raw envelope weight ``pi_q * p_iq`` is SPARSE -- only the
    supporting edges of the bottleneck target receive gradient.  This is
    physically correct (those edges carry the max-min margin) but concentrates
    the loss on very few edges.  ``min_power_floor > 0`` adds a small uniform
    power floor so every edge of a scarce target keeps a non-zero (tiny)
    weight, avoiding a degenerate zero-gradient Student for non-supporting
    edges while keeping the bottleneck concentration.
    """
    gain = np.asarray(gain_per_watt, dtype=np.float64)
    budget = np.asarray(sensing_budget_w, dtype=np.float64).reshape(-1)
    if gain.ndim != 2 or gain.shape[0] != budget.size:
        raise ValueError("gain must be (K,Q) matching budget (K,)")
    # Canonical labels remove arbitrary solver choices on a non-unique dual
    # face while preserving the primary max-min optimum exactly.
    pi, _ = canonical_maxmin_dual_prices(gain, budget)
    res = solve_fixed_structure_maxmin_power_lp(gain, budget)
    p = np.asarray(res.power_w, dtype=np.float64)
    floor = float(min_power_floor)
    if floor > 0.0:
        p = np.maximum(p, floor * budget[:, None])
    return np.asarray(pi, dtype=np.float64)[None, :] * p


def task_regret_loss(
    gain_per_watt: np.ndarray,
    sensing_budget_w: np.ndarray,
    gain_hat_per_watt: np.ndarray,
    d_req: float,
) -> tuple[float, float, float]:
    """Lexicographic task-regret surrogate (advice/016 §4.1).

    Returns ``(R_gamma, R_t, feasible_student)`` with

      R_gamma = [gamma*(x, S_hat) - gamma*(x, S*)]_+   (capability-gauge regret)
      R_t     = [t*(A*) - t*(A_hat)]_+                 (max-min regret)

    where ``gamma*`` is approximated by the feasibility indicator
    ``t* >= d_req`` (the max-min value is monotone in the per-watt gains, so
    a structure that is feasible at the floor has gauge <= 1).  ``R_gamma``
    counts a structural miss that turns a feasible state infeasible; ``R_t``
    is the residual max-min loss once both are feasible (lexicographic:
    feasibility first).
    """
    tA = tstar_of(gain_per_watt, sensing_budget_w)
    tAh = tstar_of(gain_hat_per_watt, sensing_budget_w)
    feasible_teacher = float(tA) >= float(d_req) - 1e-9
    feasible_student = float(tAh) >= float(d_req) - 1e-9
    # capability-gauge regret: teacher feasible but student not (the harmful
    # miss); if the teacher itself is infeasible there is no gauge regret to
    # attribute to the structure (geometry-limited frame).
    R_gamma = float(
        max(0.0, float(feasible_teacher) - float(feasible_student)))
    R_t = float(max(0.0, float(tA) - float(tAh)))
    return R_gamma, R_t, float(feasible_student)
