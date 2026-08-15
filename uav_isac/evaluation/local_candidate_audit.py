"""Local-information candidate sufficiency audit for bistatic hyperedges.

The functions in this module deliberately separate local candidate generation
from the privileged receiver-owner projection used only as an audit upper
bound.  Candidate generation consumes local observations and delivered Token
state; the centralized solver may only choose from the resulting sparse union.
"""

from __future__ import annotations

from math import ceil
from typing import Any

import numpy as np

from uav_isac.environment.observation_slices import ObservationSlices
from uav_isac.physical.detection import compute_detection_probabilities
from uav_isac.physical.feasibility_oracle import (
    solve_maxmin_single_role_pairs,
)
from uav_isac.utils.types import DeflectionEntry


def delivered_target_tokens(
    local_obs: np.ndarray,
    slices: ObservationSlices,
) -> tuple[np.ndarray, np.ndarray]:
    """Return delivered-token visibility and AoI as ``(F,K,K,Q)`` arrays."""
    obs = np.asarray(local_obs, dtype=np.float32)
    if obs.ndim != 3 or obs.shape[1] != slices.K:
        raise ValueError("local_obs must have shape (F,K,D)")
    if not slices.has_comm_tokens or slices.comm_tokens_per_sender != slices.Q:
        raise ValueError("target-token observations are required")
    F, K, _ = obs.shape
    Q = slices.Q
    raw_mask = slices.extract_comm_mask(obs).reshape(F, K, K - 1, Q)
    raw_token = slices.extract_comm_tokens(obs).reshape(
        F, K, K - 1, Q, slices.comm_token_per_sender)
    visible = np.zeros((F, K, K, Q), dtype=bool)
    age = np.full((F, K, K, Q), np.inf, dtype=np.float64)
    for viewer in range(K):
        peers = [peer for peer in range(K) if peer != viewer]
        visible[:, viewer, peers, :] = raw_mask[:, viewer] > 0.5
        # Target-token metadata ends in normalized AoI. Invalid entries retain
        # infinity so they cannot win a local neighbor ranking tie.
        peer_age = np.maximum(raw_token[:, viewer, :, :, -1], 0.0)
        age[:, viewer, peers, :] = np.where(
            raw_mask[:, viewer] > 0.5,
            peer_age,
            np.inf,
        )
    return visible, age


def select_local_neighbors(
    token_visible: np.ndarray,
    token_age: np.ndarray,
    *,
    neighbor_topk: int,
) -> np.ndarray:
    """Select peers using only locally delivered Token availability and AoI."""
    visible = np.asarray(token_visible, dtype=bool)
    age = np.asarray(token_age, dtype=np.float64)
    if visible.shape != age.shape or visible.ndim != 4:
        raise ValueError("token visibility/AoI must have shape (F,K,K,Q)")
    F, K, K2, _ = visible.shape
    if K != K2:
        raise ValueError("token visibility must have equal UAV axes")
    limit = int(np.clip(int(neighbor_topk), 0, max(K - 1, 0)))
    selected = np.zeros((F, K, K), dtype=bool)
    for frame in range(F):
        for viewer in range(K):
            candidates = []
            for peer in range(K):
                if peer == viewer:
                    continue
                delivered = visible[frame, viewer, peer]
                count = int(np.sum(delivered))
                if count == 0:
                    continue
                mean_age = float(np.mean(age[frame, viewer, peer, delivered]))
                candidates.append((-count, mean_age, peer))
            candidates.sort()
            for _, _, peer in candidates[:limit]:
                selected[frame, viewer, peer] = True
    return selected


def select_value_guided_neighbors(
    edge_values: np.ndarray,
    local_pd: np.ndarray,
    token_visible: np.ndarray,
    token_age: np.ndarray,
    *,
    neighbor_topk: int,
    qos_floor: float = 0.60,
    coverage_fraction: float = 0.30,
) -> np.ndarray:
    """Refine a local peer set using value and weak-target coverage.

    This selector remains deployment-local: a node ranks only peers from
    which it has actually received at least one target Token.  A fraction of
    the peer budget is reserved for the locally weakest targets, and the
    remainder maximizes the shared edge scorer.  The returned set can be fed
    back into :func:`build_structure_student_features` so all received-token
    aggregates are recomputed on the final local neighborhood.
    """
    values = np.asarray(edge_values, dtype=np.float64)
    pd = np.asarray(local_pd, dtype=np.float64)
    visible = np.asarray(token_visible, dtype=bool)
    age = np.asarray(token_age, dtype=np.float64)
    if values.ndim != 4 or values.shape[1] != values.shape[2]:
        raise ValueError("edge_values must have shape (F,K,K,Q)")
    F, K, _, Q = values.shape
    if pd.shape != (F, K, Q):
        raise ValueError("local_pd must have shape (F,K,Q)")
    if visible.shape != (F, K, K, Q) or age.shape != visible.shape:
        raise ValueError("token visibility/AoI must have shape (F,K,K,Q)")

    limit = int(np.clip(int(neighbor_topk), 0, max(K - 1, 0)))
    reserve = (
        min(limit, max(1, int(ceil(
            limit * float(np.clip(coverage_fraction, 0.0, 1.0))))))
        if limit else 0
    )
    selected = np.zeros((F, K, K), dtype=bool)
    for frame in range(F):
        for sender in range(K):
            available = [
                peer for peer in range(K)
                if peer != sender
                and bool(np.any(visible[frame, sender, peer]))
            ]
            if not available or limit == 0:
                continue
            deficit = np.clip(
                float(qos_floor) - pd[frame, sender], 0.0, 1.0)
            best_value = np.max(
                values[frame, sender, available, :], axis=0)
            target_order = np.lexsort((
                np.arange(Q), -best_value, -deficit))
            chosen: list[int] = []
            for target in target_order:
                remaining = [peer for peer in available if peer not in chosen]
                if not remaining or len(chosen) >= reserve:
                    break
                peer = min(
                    remaining,
                    key=lambda item: (
                        -float(values[frame, sender, item, target]),
                        float(np.mean(age[
                            frame, sender, item,
                            visible[frame, sender, item]])),
                        int(item),
                    ),
                )
                chosen.append(peer)

            weighted = 1.0 + deficit
            remaining = [peer for peer in available if peer not in chosen]
            remaining.sort(key=lambda peer: (
                -float(np.max(values[frame, sender, peer] * weighted)),
                float(np.mean(age[
                    frame, sender, peer,
                    visible[frame, sender, peer]])),
                int(peer),
            ))
            chosen.extend(remaining[:max(0, limit - len(chosen))])
            selected[frame, sender, chosen[:limit]] = True
    return selected


def build_local_candidate_mask(
    edge_values: np.ndarray,
    local_pd: np.ndarray,
    local_neighbors: np.ndarray,
    token_visible: np.ndarray,
    *,
    target_topk: int,
    qos_floor: float = 0.60,
    coverage_fraction: float = 0.30,
    require_reciprocal_link: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a sparse union of sender-local directed hyperedge proposals.

    Target slots are split between high local edge value and high local QoS
    deficit.  A peer link is available when at least one target Token has been
    delivered; candidate target choice does not inspect privileged physics.
    """
    values = np.asarray(edge_values, dtype=np.float64)
    pd = np.asarray(local_pd, dtype=np.float64)
    neighbor = np.asarray(local_neighbors, dtype=bool)
    visible = np.asarray(token_visible, dtype=bool)
    if values.ndim != 4 or values.shape[1] != values.shape[2]:
        raise ValueError("edge_values must have shape (F,K,K,Q)")
    F, K, _, Q = values.shape
    if pd.shape != (F, K, Q):
        raise ValueError("local_pd must have shape (F,K,Q)")
    if neighbor.shape != (F, K, K):
        raise ValueError("local_neighbors must have shape (F,K,K)")
    if visible.shape != (F, K, K, Q):
        raise ValueError("token_visible must have shape (F,K,K,Q)")

    target_limit = int(np.clip(int(target_topk), 0, Q))
    reserve = (
        min(target_limit, max(1, int(ceil(
            target_limit * float(np.clip(coverage_fraction, 0.0, 1.0))))))
        if target_limit else 0
    )
    target_selected = np.zeros((F, K, Q), dtype=bool)
    candidate = np.zeros((F, K, K, Q), dtype=bool)
    peer_link = np.any(visible, axis=-1)
    for frame in range(F):
        for sender in range(K):
            peers = np.flatnonzero(neighbor[frame, sender])
            if peers.size == 0 or target_limit == 0:
                continue
            local_value = np.max(values[frame, sender, peers, :], axis=0)
            deficit = np.clip(
                float(qos_floor) - pd[frame, sender], 0.0, 1.0)
            deficit_order = np.lexsort((
                np.arange(Q), -local_value, -deficit))
            chosen = list(deficit_order[:reserve])
            remaining = [q for q in range(Q) if q not in chosen]
            value_order = sorted(
                remaining,
                key=lambda q: (-float(local_value[q]),
                               -float(deficit[q]), int(q)),
            )
            chosen.extend(value_order[:target_limit - len(chosen)])
            target_selected[frame, sender, chosen] = True
            for receiver in peers:
                if not peer_link[frame, receiver, sender]:
                    if require_reciprocal_link:
                        continue
                candidate[frame, sender, receiver, chosen] = True
    return candidate, target_selected


def candidate_recall_metrics(
    candidate_mask: np.ndarray,
    physical_mask: np.ndarray,
    teacher_pair: np.ndarray,
    teacher_owner: np.ndarray,
    target_deficit: np.ndarray,
) -> dict[str, float]:
    """Compute edge, owner, coverage, and deficit-weighted recall metrics."""
    candidate = np.asarray(candidate_mask, dtype=bool)
    physical = np.asarray(physical_mask, dtype=bool)
    teacher = np.asarray(teacher_pair, dtype=bool)
    owner = np.asarray(teacher_owner, dtype=np.int64)
    deficit = np.asarray(target_deficit, dtype=np.float64)
    if not (candidate.shape == physical.shape == teacher.shape):
        raise ValueError("candidate, physical, and teacher masks must match")
    if candidate.ndim != 4:
        raise ValueError("edge masks must have shape (F,K,K,Q)")
    F, _, _, Q = candidate.shape
    if owner.shape != (F, Q) or deficit.shape != (F, Q):
        raise ValueError("owner/deficit shapes do not match edge masks")

    admitted = candidate & physical
    selected_count = int(np.sum(teacher))
    recalled_count = int(np.sum(admitted & teacher))
    valid_owner = owner >= 0
    owner_hit = np.zeros((F, Q), dtype=bool)
    for frame, target in np.argwhere(valid_owner):
        receiver = int(owner[frame, target])
        owner_hit[frame, target] = bool(np.any(
            admitted[frame, :, receiver, target]))

    target_available = np.any(admitted, axis=(1, 2))
    teacher_per_target = np.sum(teacher, axis=(1, 2)).astype(np.float64)
    recalled_per_target = np.sum(
        admitted & teacher, axis=(1, 2)).astype(np.float64)
    weights = np.maximum(deficit, 0.0)
    weighted_denominator = float(np.sum(weights * teacher_per_target))
    deficit_recall = (
        float(np.sum(weights * recalled_per_target) / weighted_denominator)
        if weighted_denominator > 1.0e-12
        else recalled_count / max(selected_count, 1)
    )
    return {
        "edge_recall": float(recalled_count / max(selected_count, 1)),
        "owner_recall": float(np.mean(owner_hit[valid_owner]))
        if np.any(valid_owner) else 0.0,
        "target_coverage": float(np.mean(target_available)),
        "target_empty_rate": float(np.mean(~target_available)),
        "any_target_empty_rate": float(np.mean(
            np.any(~target_available, axis=-1))),
        "deficit_weighted_edge_recall": deficit_recall,
        "candidate_physical_fraction": float(
            np.sum(admitted) / max(int(np.sum(physical)), 1)),
        "candidate_edges_mean": float(np.mean(np.sum(
            admitted, axis=(1, 2, 3)))),
    }


def candidate_equivalence_metrics(
    candidate_mask: np.ndarray,
    physical_mask: np.ndarray,
    teacher_pair: np.ndarray,
    teacher_owner: np.ndarray,
    realized_d_eff: np.ndarray,
    *,
    target_pair_limit: int,
    evidence_ratio: float = 0.95,
) -> dict[str, float]:
    """Measure whether sparse candidates retain equivalent target evidence.

    Exact edge identity is unnecessarily strict when several transmitters or
    receivers provide near-identical bistatic evidence.  This audit-only
    metric compares the strongest evidence supported by the candidate set
    with the evidence of the recorded teacher structure.  It reports both a
    same-owner comparison and an any-owner comparison.  Global role coupling
    is intentionally left to the candidate-restricted solver gate.
    """
    candidate = np.asarray(candidate_mask, dtype=bool)
    physical = np.asarray(physical_mask, dtype=bool)
    teacher = np.asarray(teacher_pair, dtype=bool)
    owner = np.asarray(teacher_owner, dtype=np.int64)
    d_eff = np.asarray(realized_d_eff, dtype=np.float64)
    if not (candidate.shape == physical.shape == teacher.shape == d_eff.shape):
        raise ValueError("candidate, physical, teacher and d_eff must match")
    if candidate.ndim != 4:
        raise ValueError("edge arrays must have shape (F,K,K,Q)")
    F, _, _, Q = candidate.shape
    if owner.shape != (F, Q):
        raise ValueError("teacher_owner must have shape (F,Q)")
    pair_limit = max(1, int(target_pair_limit))
    threshold = float(np.clip(evidence_ratio, 0.0, 1.0))
    admitted = candidate & physical

    teacher_evidence = np.sum(
        np.where(teacher, np.maximum(d_eff, 0.0), 0.0), axis=(1, 2))
    same_owner_evidence = np.zeros((F, Q), dtype=np.float64)
    any_owner_evidence = np.zeros((F, Q), dtype=np.float64)
    owner_available = np.zeros((F, Q), dtype=bool)
    for frame in range(F):
        for target in range(Q):
            receiver_values = []
            for receiver in range(admitted.shape[2]):
                incoming = d_eff[frame, :, receiver, target][
                    admitted[frame, :, receiver, target]]
                evidence = float(np.sum(np.sort(
                    np.maximum(incoming, 0.0))[-pair_limit:]))
                receiver_values.append(evidence)
            any_owner_evidence[frame, target] = max(
                receiver_values, default=0.0)
            teacher_receiver = int(owner[frame, target])
            if 0 <= teacher_receiver < admitted.shape[2]:
                same_owner_evidence[frame, target] = receiver_values[
                    teacher_receiver]
                owner_available[frame, target] = bool(np.any(
                    admitted[frame, :, teacher_receiver, target]))

    valid = teacher_evidence > 1.0e-12
    same_ratio = np.ones((F, Q), dtype=np.float64)
    any_ratio = np.ones((F, Q), dtype=np.float64)
    same_ratio[valid] = (
        same_owner_evidence[valid] / teacher_evidence[valid])
    any_ratio[valid] = any_owner_evidence[valid] / teacher_evidence[valid]
    same_clipped = np.clip(same_ratio, 0.0, 1.0)
    any_clipped = np.clip(any_ratio, 0.0, 1.0)
    return {
        "evidence_ratio_threshold": threshold,
        "same_owner_evidence_ratio_mean": float(np.mean(same_clipped)),
        "any_owner_evidence_ratio_mean": float(np.mean(any_clipped)),
        "same_owner_equivalent_recall": float(np.mean(
            same_ratio >= threshold)),
        "any_owner_equivalent_recall": float(np.mean(
            any_ratio >= threshold)),
        "teacher_owner_candidate_rate": float(np.mean(owner_available)),
    }


def receiver_owner_from_pairs(pair_sequence: np.ndarray) -> np.ndarray:
    """Recover the unique per-target receiver owner from directed pairs."""
    pairs = np.asarray(pair_sequence, dtype=bool)
    if pairs.ndim != 4 or pairs.shape[1] != pairs.shape[2]:
        raise ValueError("pair_sequence must have shape (F,K,K,Q)")
    F, _, _, Q = pairs.shape
    owners = np.full((F, Q), -1, dtype=np.int64)
    receiver_used = np.any(pairs, axis=1)
    for frame, target in np.argwhere(np.any(receiver_used, axis=1)):
        receivers = np.flatnonzero(receiver_used[frame, :, target])
        if receivers.size != 1:
            raise ValueError("pair sequence violates unique receiver ownership")
        owners[frame, target] = int(receivers[0])
    return owners


def target_priority(
    coord_pd_ema: np.ndarray,
    valid: bool,
    *,
    floor: float,
    gain: float,
) -> np.ndarray:
    values = np.asarray(coord_pd_ema, dtype=np.float64)
    if not valid:
        return np.ones_like(values)
    return np.exp(np.clip(float(gain) * (float(floor) - values), -6.0, 6.0))


def select_receiver_owner_pairs(
    ranking_values: np.ndarray,
    available: np.ndarray,
    *,
    num_uavs: int,
    num_targets: int,
    target_pair_limit: int,
    reports_per_receiver: int,
    p_fa: float,
    p_d_floor: float,
    priority: np.ndarray,
) -> np.ndarray:
    """Run the audit-only receiver-owner projection on an admitted graph."""
    ranking = np.asarray(ranking_values, dtype=np.float64)
    mask = np.asarray(available, dtype=bool)
    expected = (num_uavs, num_uavs, num_targets)
    if ranking.shape != expected or mask.shape != expected:
        raise ValueError("ranking and availability must have shape (K,K,Q)")
    entries = [
        DeflectionEntry(
            i=i, j=j, q=q, tau=0.0, nu=0.0, alpha=0.0,
            d_raw=float(ranking[i, j, q]), g_dd=1.0, chi_rep=1.0,
            d_eff=float(ranking[i, j, q]),
        )
        for i, j, q in np.argwhere(mask & (ranking > 0.0))
        if int(i) != int(j)
    ]
    selected, _ = solve_maxmin_single_role_pairs(
        entries,
        num_uavs=int(num_uavs),
        num_targets=int(num_targets),
        target_pair_limit=int(target_pair_limit),
        reports_per_receiver=int(reports_per_receiver),
        p_fa=float(p_fa),
        p_d_floor=float(p_d_floor),
        target_priority=np.asarray(priority, dtype=np.float64),
        fusion_mode="local_only",
    )
    pair = np.zeros(expected, dtype=bool)
    for tx, rx, target in selected:
        pair[int(tx), int(rx), int(target)] = True
    return pair


def held_pair_sequence(
    ranking_values: np.ndarray,
    admitted_mask: np.ndarray,
    data: dict[str, np.ndarray],
) -> np.ndarray:
    """Solve only on trace resolve frames and hold each feasible structure."""
    ranking = np.asarray(ranking_values, dtype=np.float64)
    admitted = np.asarray(admitted_mask, dtype=bool)
    if ranking.shape != admitted.shape or ranking.ndim != 4:
        raise ValueError("ranking/admitted arrays must have shape (F,K,K,Q)")
    F, K, _, Q = ranking.shape
    result = np.zeros_like(admitted)
    current = np.zeros((K, K, Q), dtype=bool)
    previous_episode: int | None = None
    floor = float(np.asarray(data["qos_floor"]).reshape(-1)[0])
    gain = float(np.asarray(data["deficit_priority_gain"]).reshape(-1)[0])
    pair_limit = int(np.asarray(data["target_pair_limit"]).reshape(-1)[0])
    reports = int(np.asarray(data["reports_per_receiver"]).reshape(-1)[0])
    p_fa = float(np.asarray(data["p_fa"]).reshape(-1)[0])
    for frame in range(F):
        episode = int(data["episode"][frame])
        if previous_episode != episode:
            current = np.zeros((K, K, Q), dtype=bool)
            previous_episode = episode
        if bool(data["p0_resolved"][frame]):
            priority = target_priority(
                data["coord_pd_ema"][frame],
                bool(data["coord_pd_ema_valid"][frame]),
                floor=floor,
                gain=gain,
            )
            current = select_receiver_owner_pairs(
                ranking[frame],
                admitted[frame],
                num_uavs=K,
                num_targets=Q,
                target_pair_limit=pair_limit,
                reports_per_receiver=reports,
                p_fa=p_fa,
                p_d_floor=floor,
                priority=priority,
            )
        result[frame] = current
    return result


def realized_pd_history(
    pair_sequence: np.ndarray,
    realized_d_eff: np.ndarray,
    p_fa: float,
) -> np.ndarray:
    """Evaluate selected edges with local-only receiver evidence fusion."""
    pair = np.asarray(pair_sequence, dtype=bool)
    d_eff = np.asarray(realized_d_eff, dtype=np.float64)
    if pair.shape != d_eff.shape or pair.ndim != 4:
        raise ValueError("pair and d_eff must have shape (F,K,K,Q)")
    receiver_d = np.sum(d_eff * pair, axis=1)
    target_d = np.max(receiver_d, axis=1)
    return compute_detection_probabilities(target_d, float(p_fa))


def episode_detection_summary(
    pd_history: np.ndarray,
    episode: np.ndarray,
    seed: np.ndarray,
    *,
    steady_window: int = 20,
    qos_thresholds: tuple[float, float, float] = (0.80, 0.70, 0.60),
    cvar_fraction: float = 0.20,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Summarize frame histories with the formal episode-wise metric order."""
    values = np.asarray(pd_history, dtype=np.float64)
    episodes = np.asarray(episode, dtype=np.int64)
    seeds = np.asarray(seed, dtype=np.int64)
    if values.ndim != 2 or episodes.shape != (len(values),):
        raise ValueError("pd_history/episode shapes are incompatible")
    rows: list[dict[str, Any]] = []
    for ep in np.unique(episodes):
        indices = np.flatnonzero(episodes == ep)
        window = indices[-min(int(steady_window), len(indices)):]
        per_target = np.mean(values[window], axis=0)
        ordered = np.sort(per_target)
        row = {
            "episode": int(ep),
            "seed": int(seeds[indices[0]]),
            "steady": float(np.mean(per_target)),
            "weak3": float(np.mean(ordered[:min(3, len(ordered))])),
            "worst": float(ordered[0]),
            "per_target": per_target.tolist(),
        }
        row["qos_feasible"] = bool(
            row["steady"] >= qos_thresholds[0]
            and row["weak3"] >= qos_thresholds[1]
            and row["worst"] >= qos_thresholds[2]
        )
        rows.append(row)
    worst = np.asarray([row["worst"] for row in rows], dtype=np.float64)
    tail = max(1, int(ceil(float(cvar_fraction) * len(worst))))
    summary: dict[str, Any] = {
        "episodes": len(rows),
        "steady": float(np.mean([row["steady"] for row in rows])),
        "weak3": float(np.mean([row["weak3"] for row in rows])),
        "worst": float(np.mean(worst)),
        "cvar": float(np.mean(np.sort(worst)[:tail])),
        "qos_feasible": float(np.mean([
            row["qos_feasible"] for row in rows])),
        "episode_worst": worst.tolist(),
    }
    return summary, rows
