"""Learning-free local-view negotiation for directed bistatic hyperedges.

One sensing decision is a directed triple ``(tx, rx, target)`` rather than
two independent node-target weights.  Every UAV can run the same deterministic
planner from its own offer plus offers that arrived through its local inbox.
The environment integration activates only endpoint-mutual plans; these pure
functions never inspect global simulator state.
"""

from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, Tuple

import numpy as np


Hyperedge = Tuple[int, int, int]


@dataclass(frozen=True)
class LocalHyperedgePlan:
    """One UAV's deterministic plan reconstructed from its local view."""

    selected: Tuple[Hyperedge, ...]
    proxy_target_value: np.ndarray
    proxy_scores: Dict[Hyperedge, float]
    role_mask: np.ndarray


def factorized_endpoint_capabilities(
    distance_m: np.ndarray,
    sensing_fraction: np.ndarray,
    *,
    distance_scale_m: float,
    mode: str = "exponential",
) -> Tuple[np.ndarray, np.ndarray]:
    """Encode factorized bistatic Tx/Rx capability in bounded coordinates."""
    distance = np.asarray(distance_m, dtype=np.float64)
    sensing = np.asarray(sensing_fraction, dtype=np.float64)
    if distance.shape != sensing.shape:
        raise ValueError("distance and sensing_fraction must have equal shape")
    if not (np.all(np.isfinite(distance)) and np.all(np.isfinite(sensing))):
        raise ValueError("capability inputs must be finite")
    scale = max(float(distance_scale_m), 1.0e-9)
    normalized_mode = str(mode).strip().lower()
    if normalized_mode == "exponential":
        receive = np.exp(-np.maximum(distance, 0.0) / scale)
    elif normalized_mode == "inverse_square":
        ratio = np.maximum(distance, 0.0) / scale
        receive = 1.0 / (1.0 + ratio ** 2)
    else:
        raise ValueError(
            "capability mode must be exponential or inverse_square")
    receive = np.clip(receive, 0.0, 1.0)
    transmit = np.clip(
        receive * np.sqrt(np.maximum(sensing, 0.0)),
        0.0,
        1.0,
    )
    return transmit, receive


def encode_offer_stream(
    tx_capability: np.ndarray,
    rx_capability: np.ndarray,
    deficit: np.ndarray,
) -> np.ndarray:
    """Encode three normalized protocol fields into the radio range [-1, 1]."""
    tx = np.asarray(tx_capability, dtype=np.float64)
    rx = np.asarray(rx_capability, dtype=np.float64)
    gap = np.asarray(deficit, dtype=np.float64)
    if not (tx.shape == rx.shape == gap.shape) or tx.ndim != 1:
        raise ValueError("offer fields must be same-shape one-dimensional arrays")
    fields = np.stack([tx, rx, gap], axis=-1)
    if not np.all(np.isfinite(fields)):
        raise ValueError("offer fields must be finite")
    return 2.0 * np.clip(fields, 0.0, 1.0) - 1.0


def decode_offer_stream(stream: np.ndarray) -> np.ndarray:
    """Decode a ``(..., 3)`` quantized offer stream into [0, 1]."""
    encoded = np.asarray(stream, dtype=np.float64)
    if encoded.shape[-1:] != (3,):
        raise ValueError("offer stream must have final dimension 3")
    if not np.all(np.isfinite(encoded)):
        raise ValueError("offer stream must be finite")
    return np.clip(0.5 * (encoded + 1.0), 0.0, 1.0)


def reconstruct_bistatic_pair_value(
    tx_capability: np.ndarray,
    node_positions_xy: np.ndarray,
    target_positions_xy: np.ndarray,
    visible: np.ndarray,
    *,
    distance_scale_m: float,
) -> np.ndarray:
    """Reconstruct normalized pair value from public, quantized local state.

    No realized global deflection entry is used.  Mission target positions are
    common knowledge; node positions and Tx capability must come from the
    viewer's own public offer or an actually delivered neighbor offer.
    """
    tx_value = np.asarray(tx_capability, dtype=np.float64)
    positions = np.asarray(node_positions_xy, dtype=np.float64)
    targets = np.asarray(target_positions_xy, dtype=np.float64)
    seen = np.asarray(visible, dtype=bool)
    if tx_value.ndim != 2:
        raise ValueError("tx_capability must have shape (K, Q)")
    k_count, q_count = tx_value.shape
    if positions.shape != (k_count, q_count, 2):
        raise ValueError("node_positions_xy must have shape (K, Q, 2)")
    if targets.shape != (q_count, 2):
        raise ValueError("target_positions_xy must have shape (Q, 2)")
    if seen.shape != (k_count, q_count):
        raise ValueError("visible must have shape (K, Q)")
    if not (np.all(np.isfinite(tx_value))
            and np.all(np.isfinite(positions))
            and np.all(np.isfinite(targets))):
        raise ValueError("public reconstruction inputs must be finite")

    scale = max(float(distance_scale_m), 1.0e-9)
    pair_value = np.zeros((k_count, k_count, q_count), dtype=np.float64)
    for target in range(q_count):
        ranges = np.linalg.norm(
            positions[:, target] - targets[target], axis=-1)
        for tx in range(k_count):
            if not seen[tx, target]:
                continue
            tx_range = max(float(ranges[tx]), 1.0)
            geometry_proxy = np.exp(-tx_range / scale)
            estimated_power = np.clip(
                float(tx_value[tx, target])
                / max(float(geometry_proxy), 1.0e-9),
                0.0,
                1.0,
            ) ** 2
            for rx in range(k_count):
                if tx == rx or not seen[rx, target]:
                    continue
                rx_range = max(float(ranges[rx]), 1.0)
                pair_value[tx, rx, target] = (
                    estimated_power
                    / (tx_range ** 2 * rx_range ** 2)
                )
        target_max = float(np.max(pair_value[:, :, target]))
        if target_max > 0.0:
            pair_value[:, :, target] /= target_max
    return pair_value


def _edge_score(
    tx_capability: np.ndarray,
    rx_capability: np.ndarray,
    target_deficit: np.ndarray,
    tx: int,
    rx: int,
    target: int,
    deficit_gain: float,
    pair_value: np.ndarray | None = None,
) -> float:
    # The geometric mean avoids collapsing a useful edge merely because one
    # role proxy is numerically sharper than the other, while still requiring
    # both directed endpoints to be capable.
    role_value = (
        float(pair_value[tx, rx, target])
        if pair_value is not None
        else np.sqrt(max(
            float(tx_capability[tx, target])
            * float(rx_capability[rx, target]),
            0.0,
        ))
    )
    return role_value * (
        1.0 + max(0.0, float(deficit_gain))
        * float(np.clip(target_deficit[target], 0.0, 1.0))
    )


def plan_local_hyperedges(
    tx_capability: np.ndarray,
    rx_capability: np.ndarray,
    visible: np.ndarray,
    target_deficit: np.ndarray,
    *,
    target_pair_limit: int,
    deficit_gain: float,
    proxy_floor: float,
    pair_value: np.ndarray | None = None,
) -> LocalHyperedgePlan:
    """Choose a lexicographic max-min directed plan from one local view.

    A role mask is enumerated exactly.  Once roles are fixed, targets are
    separable because an endpoint may sense multiple targets while retaining
    one Tx/Rx role.  This is inexpensive for the intended 4--8 UAV audit and
    avoids importing a centralized environment solver.
    """
    tx_value = np.asarray(tx_capability, dtype=np.float64)
    rx_value = np.asarray(rx_capability, dtype=np.float64)
    seen = np.asarray(visible, dtype=bool)
    deficit = np.asarray(target_deficit, dtype=np.float64)
    if tx_value.shape != rx_value.shape or tx_value.ndim != 2:
        raise ValueError("capability matrices must have shape (K, Q)")
    k_count, q_count = tx_value.shape
    if seen.shape != (k_count, q_count):
        raise ValueError("visible must have shape (K, Q)")
    if deficit.shape != (q_count,):
        raise ValueError("target_deficit must have shape (Q,)")
    pair_score = None
    if pair_value is not None:
        pair_score = np.asarray(pair_value, dtype=np.float64)
        if pair_score.shape != (k_count, k_count, q_count):
            raise ValueError("pair_value must have shape (K, K, Q)")
    if not (np.all(np.isfinite(tx_value))
            and np.all(np.isfinite(rx_value))
            and np.all(np.isfinite(deficit))
            and (pair_score is None or np.all(np.isfinite(pair_score)))):
        raise ValueError("planner inputs must be finite")

    limit = max(1, int(target_pair_limit))
    floor = max(float(proxy_floor), 1e-9)
    best_key = None
    best_edges: Tuple[Hyperedge, ...] = tuple()
    best_target_value = np.zeros(q_count, dtype=np.float64)
    best_scores: Dict[Hyperedge, float] = {}
    best_roles = np.zeros(k_count, dtype=bool)

    # False=Rx, True=Tx. Exclude the all-same partitions.
    for role_tuple in product((False, True), repeat=k_count):
        role = np.asarray(role_tuple, dtype=bool)
        if not np.any(role) or np.all(role):
            continue
        scores: Dict[Hyperedge, float] = {}
        selected = []
        target_value = np.zeros(q_count, dtype=np.float64)
        tx_nodes = np.flatnonzero(role)
        rx_nodes = np.flatnonzero(~role)
        for target in range(q_count):
            # The deployed local-only detector may coherently accumulate
            # several Tx echoes at one receiver, but cannot add statistics
            # held by different receivers.  Nominate exactly one receiver
            # owner for each target before choosing its incoming Tx edges.
            owner_candidates = []
            for rx in rx_nodes:
                if not seen[rx, target]:
                    continue
                incoming = []
                for tx in tx_nodes:
                    if tx == rx or not seen[tx, target]:
                        continue
                    edge = (int(tx), int(rx), int(target))
                    score = _edge_score(
                        tx_value, rx_value, deficit,
                        int(tx), int(rx), int(target), deficit_gain,
                        pair_score,
                    )
                    if score > 0.0:
                        incoming.append((score, edge))
                incoming.sort(key=lambda item: (-item[0], item[1]))
                retained = tuple(incoming[:limit])
                if retained:
                    owner_candidates.append((
                        float(sum(score for score, _ in retained)),
                        int(rx),
                        retained,
                    ))
            owner_candidates.sort(
                key=lambda item: (-item[0], item[1]))
            retained = (
                owner_candidates[0][2] if owner_candidates else tuple())
            for score, edge in retained:
                selected.append(edge)
                scores[edge] = float(score)
                target_value[target] += float(score)

        # Primary: max-min proxy. Secondary: deficit-weighted floor coverage.
        # The remaining fields are deterministic stabilizers.
        capped = np.minimum(target_value / floor, 1.0)
        weights = 1.0 + np.clip(deficit, 0.0, 1.0)
        coverage_value = float(np.dot(weights, capped))
        selected_tuple = tuple(sorted(selected))
        key = (
            float(np.min(target_value)) if q_count else 0.0,
            coverage_value,
            float(np.sum(target_value)),
            -len(selected_tuple),
            tuple(-value for edge in selected_tuple for value in edge),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_edges = selected_tuple
            best_target_value = target_value
            best_scores = scores
            best_roles = role

    return LocalHyperedgePlan(
        selected=best_edges,
        proxy_target_value=best_target_value,
        proxy_scores=best_scores,
        role_mask=best_roles,
    )


def mutual_endpoint_consensus(
    plans: Iterable[LocalHyperedgePlan],
    *,
    num_uavs: int,
    num_targets: int,
    target_pair_limit: int,
) -> Tuple[Hyperedge, ...]:
    """Keep only directed edges endorsed by both of their endpoints."""
    local = tuple(plans)
    if len(local) != int(num_uavs):
        raise ValueError("one local plan is required per UAV")
    selected_sets = [set(plan.selected) for plan in local]
    candidates = set()
    for tx, rx, target in set().union(*selected_sets):
        edge = (int(tx), int(rx), int(target))
        if not (0 <= tx < num_uavs and 0 <= rx < num_uavs
                and 0 <= target < num_targets and tx != rx):
            continue
        if edge in selected_sets[tx] and edge in selected_sets[rx]:
            candidates.add(edge)

    # Mutual endorsement already preserves each endpoint's local single-role
    # constraint. Retain a deterministic target-capacity safety projection for
    # disconnected local views at larger K.
    by_target = {q: [] for q in range(int(num_targets))}
    for edge in sorted(candidates):
        tx, rx, target = edge
        support = 0.5 * (
            float(local[tx].proxy_scores.get(edge, 0.0))
            + float(local[rx].proxy_scores.get(edge, 0.0))
        )
        by_target[target].append((support, edge))

    chosen = []
    limit = max(1, int(target_pair_limit))
    for target in range(int(num_targets)):
        ranked = sorted(
            by_target[target], key=lambda item: (-item[0], item[1]))
        chosen.extend(edge for _, edge in ranked[:limit])

    # A final explicit invariant check makes integration errors fail closed.
    tx_nodes = {tx for tx, _, _ in chosen}
    rx_nodes = {rx for _, rx, _ in chosen}
    if tx_nodes & rx_nodes:
        raise RuntimeError("mutual plans violated the single-role invariant")
    return tuple(sorted(chosen))


def update_consensus_streak(
    previous: np.ndarray,
    mutual_edges: Iterable[Hyperedge],
    *,
    consensus_rounds: int,
) -> Tuple[np.ndarray, Tuple[Hyperedge, ...]]:
    """Require the same reciprocal edge for consecutive communication rounds."""
    streak = np.asarray(previous, dtype=np.int64).copy()
    if streak.ndim != 3 or streak.shape[0] != streak.shape[1]:
        raise ValueError("previous streak must have shape (K, K, Q)")
    mutual_mask = np.zeros_like(streak, dtype=bool)
    for tx, rx, target in mutual_edges:
        mutual_mask[int(tx), int(rx), int(target)] = True
    streak = np.where(mutual_mask, streak + 1, 0)
    required = max(1, int(consensus_rounds))
    active = tuple(
        (int(tx), int(rx), int(target))
        for tx, rx, target in np.argwhere(streak >= required)
    )
    return streak, tuple(sorted(active))
