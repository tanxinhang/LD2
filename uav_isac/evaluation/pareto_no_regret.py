"""Pareto no-regret gates shared by structure-policy audits."""

from __future__ import annotations

from typing import Any

import numpy as np

from uav_isac.evaluation.local_candidate_audit import realized_pd_history


def componentwise_hold_pareto_gate(
    data: dict[str, np.ndarray],
    baseline_pairs: np.ndarray,
    proposal_pairs: np.ndarray,
    *,
    tolerance: float = 1.0e-9,
    minimum_gain: float = 1.0e-6,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Accept only complete-hold, per-target Pareto improvements.

    This function consumes realized outcomes and is therefore an audit/label
    construction primitive, not a deployable inference gate.
    """
    baseline = np.asarray(baseline_pairs, dtype=bool)
    proposal = np.asarray(proposal_pairs, dtype=bool)
    if baseline.shape != proposal.shape or baseline.ndim != 4:
        raise ValueError("pair tensors must share shape (F,K,K,Q)")
    realized = np.asarray(data["privileged_d_eff"], dtype=np.float64)
    p_fa = float(np.asarray(data["p_fa"]).reshape(-1)[0])
    baseline_pd = realized_pd_history(baseline, realized, p_fa)
    proposal_pd = realized_pd_history(proposal, realized, p_fa)
    episodes = np.asarray(data["episode"], dtype=np.int64)
    resolved = np.asarray(data["p0_resolved"], dtype=bool)
    gated = baseline.copy()
    rows: list[dict[str, Any]] = []
    for episode in np.unique(episodes):
        episode_indices = np.flatnonzero(episodes == episode)
        resolve_indices = episode_indices[resolved[episode_indices]]
        for position, frame in enumerate(resolve_indices):
            stop = (
                int(resolve_indices[position + 1])
                if position + 1 < len(resolve_indices)
                else int(episode_indices[-1]) + 1
            )
            segment = np.arange(int(frame), stop, dtype=np.int64)
            delta = np.mean(
                proposal_pd[segment] - baseline_pd[segment], axis=0)
            accept = bool(
                np.min(delta) >= -float(tolerance)
                and np.max(delta) >= float(minimum_gain)
            )
            if accept:
                gated[segment] = proposal[segment]
            rows.append({
                "seed": int(data["seed"][frame]),
                "frame": int(data["frame"][frame]),
                "accepted": accept,
                "minimum_target_delta": float(np.min(delta)),
                "maximum_target_delta": float(np.max(delta)),
                "mean_target_delta": float(np.mean(delta)),
            })
    accepted = [row for row in rows if row["accepted"]]
    return gated, {
        "segments": int(len(rows)),
        "accepted_segments": int(len(accepted)),
        "acceptance_rate": float(len(accepted) / max(len(rows), 1)),
        "minimum_accepted_target_delta": (
            float(min(row["minimum_target_delta"] for row in accepted))
            if accepted else None
        ),
        "mean_accepted_target_delta": (
            float(np.mean([row["mean_target_delta"] for row in accepted]))
            if accepted else 0.0
        ),
        "accepted_by_seed": {
            str(seed): int(sum(
                row["accepted"] and row["seed"] == seed for row in rows))
            for seed in sorted({row["seed"] for row in rows})
        },
    }
