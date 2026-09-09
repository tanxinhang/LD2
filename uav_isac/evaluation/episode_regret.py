"""Episode-level QoS regret and candidate-support decomposition.

This module is deliberately an *audit metric*, not a training loss.  It keeps
the three QoS floors separate and reports a lexicographic vector instead of
inventing an arbitrary scalar reward.  A candidate miss is only credited when
the missing support is followed by a reference-feasible -> method-infeasible
episode flip; otherwise it is coverage loss without demonstrated task regret.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np


QOS_METRICS = ("steady", "weak3", "worst")


def candidate_support_by_episode(
    candidate_mask: np.ndarray,
    reference_pairs: np.ndarray,
    episode: np.ndarray,
    *,
    resolved: np.ndarray | None = None,
) -> dict[int, bool]:
    """Return whether every reference edge is present on resolved frames.

    ``reference_pairs`` is normally the oracle/teacher held-pair sequence and
    ``candidate_mask`` is the candidate set before ranking.  The result is
    intentionally strict: one missing reference edge marks the episode as
    unsupported.  This avoids silently turning partial recall into a claimed
    episode guarantee.
    """
    candidate = np.asarray(candidate_mask, dtype=bool)
    reference = np.asarray(reference_pairs, dtype=bool)
    episodes = np.asarray(episode, dtype=np.int64).reshape(-1)
    if candidate.shape != reference.shape or candidate.ndim != 4:
        raise ValueError("candidate_mask/reference_pairs must both be (F,K,K,Q)")
    if episodes.shape != (candidate.shape[0],):
        raise ValueError("episode must have one value per frame")
    if resolved is None:
        active = np.ones(candidate.shape[0], dtype=bool)
    else:
        active = np.asarray(resolved, dtype=bool).reshape(-1)
        if active.shape != (candidate.shape[0],):
            raise ValueError("resolved must have one value per frame")

    frame_supported = np.all(~reference | candidate, axis=(1, 2, 3))
    result: dict[int, bool] = {}
    for ep in np.unique(episodes):
        indices = (episodes == ep) & active
        # No resolved frame gives no evidence of candidate support.
        result[int(ep)] = bool(np.any(indices) and np.all(frame_supported[indices]))
    return result


def episode_qos_regret(
    reference_rows: Iterable[Mapping[str, Any]],
    method_rows: Iterable[Mapping[str, Any]],
    *,
    candidate_supported: Mapping[int, bool] | None = None,
) -> dict[str, Any]:
    """Compare method episode rows against a reference without scalarization.

    The returned per-episode regret is the positive degradation in each QoS
    metric.  ``qos_flip`` counts only episodes feasible for the reference but
    infeasible for the method.  If candidate support is supplied, flips are
    partitioned into ``candidate_miss_qos_regret`` and
    ``rank_or_execution_qos_regret``.  The latter is not proof that ranking is
    at fault; it is the residual after candidate support was observed.
    """
    reference = {int(row["episode"]): row for row in reference_rows}
    method = {int(row["episode"]): row for row in method_rows}
    if not reference or set(reference) != set(method):
        raise ValueError("reference and method must contain the same non-empty episodes")
    if candidate_supported is not None:
        missing = set(reference) - {int(key) for key in candidate_supported}
        if missing:
            raise ValueError(f"candidate_supported missing episodes: {sorted(missing)}")

    per_episode: list[dict[str, Any]] = []
    for ep in sorted(reference):
        ref = reference[ep]
        got = method[ep]
        drops = {
            metric: float(max(float(ref[metric]) - float(got[metric]), 0.0))
            for metric in QOS_METRICS
        }
        qos_flip = bool(ref["qos_feasible"]) and not bool(got["qos_feasible"])
        row: dict[str, Any] = {
            "episode": ep,
            "reference_feasible": bool(ref["qos_feasible"]),
            "method_feasible": bool(got["qos_feasible"]),
            "qos_flip": qos_flip,
            "regret": drops,
        }
        if candidate_supported is not None:
            supported = bool(candidate_supported[ep])
            row["candidate_supported"] = supported
            row["candidate_miss_qos_regret"] = bool(qos_flip and not supported)
            row["rank_or_execution_qos_regret"] = bool(qos_flip and supported)
        per_episode.append(row)

    count = float(len(per_episode))
    aggregate: dict[str, float] = {
        metric: float(np.mean([row["regret"][metric] for row in per_episode]))
        for metric in QOS_METRICS
    }
    aggregate["qos_flip_rate"] = float(
        np.mean([row["qos_flip"] for row in per_episode]))
    if candidate_supported is not None:
        aggregate["candidate_miss_qos_regret_rate"] = float(
            np.mean([row["candidate_miss_qos_regret"] for row in per_episode]))
        aggregate["rank_or_execution_qos_regret_rate"] = float(
            np.mean([row["rank_or_execution_qos_regret"] for row in per_episode]))
        aggregate["candidate_unsupported_rate"] = float(
            np.mean([not row["candidate_supported"] for row in per_episode]))
    # Keep an explicit count for consumers that need to recompute rates.
    aggregate["episodes"] = count
    return {
        "schema": "episode-qos-regret-v1",
        "regret_order": list(QOS_METRICS),
        "episodes": per_episode,
        "aggregate": aggregate,
    }

