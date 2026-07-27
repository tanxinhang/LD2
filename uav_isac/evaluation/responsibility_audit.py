"""Diagnostics for separating useful multistatic overlap from wasted motion.

The movement ``collision`` metric used by the trainer only says that two or
more UAV movement vectors point at the same target.  In a bistatic system that
can be either useful (the UAVs form complementary sensing endpoints) or wasteful
(one of the movers has zero marginal effect on the target detection
probability).  This module performs a node-removal counterfactual on the
*realized* P0 edges to distinguish those cases without changing the policy.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from uav_isac.utils.math_utils import compute_PD


def node_removal_marginal_pd(
    selected_set: Sequence[Tuple[int, int, int]],
    deflection_entries: Iterable,
    num_agents: int,
    num_targets: int,
    p_fa: float,
) -> Dict[str, np.ndarray]:
    """Compute the realized P_D loss caused by removing each sensing endpoint.

    Removing node ``k`` deletes every selected bistatic edge on which it is
    either transmitter or receiver.  The result is a node-criticality measure;
    it is intentionally not additive because both endpoints are necessary for
    a bistatic edge.
    """
    entry_map = {
        (int(entry.i), int(entry.j), int(entry.q)): float(entry.d_eff)
        for entry in deflection_entries
    }
    total_d = np.zeros(num_targets, dtype=np.float64)
    node_d = np.zeros((num_agents, num_targets), dtype=np.float64)
    tx_d = np.zeros_like(node_d)
    rx_d = np.zeros_like(node_d)
    tx_participation = np.zeros(
        (num_agents, num_targets), dtype=bool)
    rx_participation = np.zeros_like(tx_participation)

    for i_raw, j_raw, q_raw in selected_set:
        i, j, q = int(i_raw), int(j_raw), int(q_raw)
        value = entry_map.get((i, j, q), 0.0)
        total_d[q] += value
        node_d[i, q] += value
        node_d[j, q] += value
        tx_d[i, q] += value
        rx_d[j, q] += value
        tx_participation[i, q] = True
        rx_participation[j, q] = True

    full_pd = compute_PD(total_d, p_fa)
    node_without_pd = np.empty_like(node_d)
    tx_without_pd = np.empty_like(tx_d)
    rx_without_pd = np.empty_like(rx_d)
    for k in range(num_agents):
        node_without_pd[k] = compute_PD(
            np.maximum(total_d - node_d[k], 0.0), p_fa)
        tx_without_pd[k] = compute_PD(
            np.maximum(total_d - tx_d[k], 0.0), p_fa)
        rx_without_pd[k] = compute_PD(
            np.maximum(total_d - rx_d[k], 0.0), p_fa)

    return {
        "full_pd": full_pd,
        "node_marginal_pd": np.maximum(
            full_pd[None, :] - node_without_pd, 0.0),
        "tx_marginal_pd": np.maximum(
            full_pd[None, :] - tx_without_pd, 0.0),
        "rx_marginal_pd": np.maximum(
            full_pd[None, :] - rx_without_pd, 0.0),
        "tx_participation": tx_participation,
        "rx_participation": rx_participation,
    }


def classify_responsibility_frame(
    target_choices: np.ndarray,
    active_movement: np.ndarray,
    selected_set: Sequence[Tuple[int, int, int]],
    deflection_entries: Iterable,
    num_targets: int,
    p_fa: float,
    marginal_epsilon: float = 1.0e-4,
) -> Dict[str, float]:
    """Classify one frame of movement responsibility against realized sensing."""
    choices = np.asarray(target_choices, dtype=np.int64).reshape(-1)
    active = np.asarray(active_movement, dtype=bool).reshape(-1)
    if choices.shape != active.shape:
        raise ValueError("target choices and active mask must have equal shape")
    num_agents = int(choices.size)
    counterfactual = node_removal_marginal_pd(
        selected_set,
        deflection_entries,
        num_agents,
        num_targets,
        p_fa,
    )
    marginal = counterfactual["node_marginal_pd"]
    tx_participation = counterfactual["tx_participation"]
    rx_participation = counterfactual["rx_participation"]
    selected_targets = {
        int(q) for _, _, q in selected_set
    }

    load = np.bincount(choices[active], minlength=num_targets)
    duplicated_targets = load > 1
    uncovered_targets = load == 0
    duplicate_responsibilities = 0
    ineffective_duplicate_responsibilities = 0
    effective_overlap_targets = 0
    ineffective_duplicate_targets = 0
    same_role_overlap_targets = 0
    productive_duplicate_marginals: List[float] = []
    ineffective_duplicate_marginals: List[float] = []
    single_responsibility_marginals: List[float] = []

    for q in range(num_targets):
        movers = np.flatnonzero(active & (choices == q))
        q_marginals = marginal[movers, q]
        productive = q_marginals > marginal_epsilon
        if movers.size == 1:
            single_responsibility_marginals.extend(q_marginals.tolist())
        if movers.size <= 1:
            continue

        duplicate_responsibilities += int(movers.size)
        ineffective_count = int(np.sum(~productive))
        ineffective_duplicate_responsibilities += ineffective_count
        ineffective_duplicate_targets += int(ineffective_count > 0)
        productive_duplicate_marginals.extend(
            q_marginals[productive].tolist())
        ineffective_duplicate_marginals.extend(
            q_marginals[~productive].tolist())

        productive_nodes = movers[productive]
        if productive_nodes.size >= 2:
            effective_overlap_targets += 1
            only_tx = np.all(
                tx_participation[productive_nodes, q]
                & ~rx_participation[productive_nodes, q])
            only_rx = np.all(
                rx_participation[productive_nodes, q]
                & ~tx_participation[productive_nodes, q])
            same_role_overlap_targets += int(only_tx or only_rx)

    active_count = max(int(np.sum(active)), 1)
    duplicate_count = max(duplicate_responsibilities, 1)
    uncovered_but_sensed = sum(
        int(uncovered_targets[q] and q in selected_targets)
        for q in range(num_targets)
    )
    uncovered_and_unsensed = sum(
        int(uncovered_targets[q] and q not in selected_targets)
        for q in range(num_targets)
    )

    def _mean(values: List[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    return {
        "duplicate_target_rate": float(np.mean(duplicated_targets)),
        "duplicate_frame": float(np.any(duplicated_targets)),
        "duplicate_responsibility_rate": float(
            duplicate_responsibilities / active_count),
        "ineffective_duplicate_responsibility_rate": float(
            ineffective_duplicate_responsibilities / duplicate_count),
        "ineffective_duplicate_target_rate": float(
            ineffective_duplicate_targets / max(num_targets, 1)),
        "effective_overlap_target_rate": float(
            effective_overlap_targets / max(num_targets, 1)),
        "same_role_overlap_target_rate": float(
            same_role_overlap_targets / max(num_targets, 1)),
        "uncovered_target_rate": float(np.mean(uncovered_targets)),
        "uncovered_but_sensed_target_rate": float(
            uncovered_but_sensed / max(num_targets, 1)),
        "uncovered_and_unsensed_target_rate": float(
            uncovered_and_unsensed / max(num_targets, 1)),
        "productive_duplicate_marginal_pd": _mean(
            productive_duplicate_marginals),
        "ineffective_duplicate_marginal_pd": _mean(
            ineffective_duplicate_marginals),
        "single_responsibility_marginal_pd": _mean(
            single_responsibility_marginals),
        "mean_node_marginal_pd": float(np.mean(
            marginal[np.arange(num_agents), choices]
            if num_agents else np.zeros(1))),
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average-rank ties, sufficient for a dependency-free Spearman metric."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    x_rank = _rankdata(x)
    y_rank = _rankdata(y)
    if x_rank.size < 3:
        return float("nan")
    x_centered = x_rank - np.mean(x_rank)
    y_centered = y_rank - np.mean(y_rank)
    denominator = np.sqrt(
        np.sum(x_centered ** 2) * np.sum(y_centered ** 2))
    if denominator <= 1.0e-12:
        return float("nan")
    return float(np.sum(x_centered * y_centered) / denominator)


def _group_difference(x: np.ndarray, y: np.ndarray) -> float:
    positive = x > 0.0
    if not np.any(positive) or np.all(positive):
        return float("nan")
    return float(np.mean(y[positive]) - np.mean(y[~positive]))


def _finite_interval(values: Sequence[float]) -> Tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return (
        float(np.quantile(finite, 0.025)),
        float(np.quantile(finite, 0.975)),
    )


def summarize_responsibility_episodes(
    episodes: Sequence[Sequence[Dict[str, float]]],
    pd_histories: Sequence[np.ndarray],
    horizon: int = 5,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 20260723,
) -> Dict[str, object]:
    """Summarize movement-boundary records with an episode-cluster bootstrap."""
    if len(episodes) != len(pd_histories):
        raise ValueError("episode records and P_D histories must align")
    augmented: List[List[Dict[str, float]]] = []
    for records, pd_history_raw in zip(episodes, pd_histories):
        pd_history = np.asarray(pd_history_raw, dtype=np.float64)
        episode_rows: List[Dict[str, float]] = []
        for record in records:
            frame_index = int(record["frame_index"])
            stop = frame_index + int(horizon)
            if frame_index < 0 or stop > len(pd_history):
                continue
            future_per_target = np.mean(
                pd_history[frame_index:stop], axis=0)
            ordered = np.sort(future_per_target)
            pre_pd = np.asarray(record["pre_pd"], dtype=np.float64)
            if pre_pd.size != future_per_target.size:
                continue
            row = {
                key: float(value)
                for key, value in record.items()
                if key not in {"frame_index", "pre_pd"}
            }
            row["future5_worst"] = float(ordered[0])
            row["future5_weak3"] = float(
                np.mean(ordered[:min(3, ordered.size)]))
            pre_ordered = np.sort(pre_pd)
            row["future5_worst_delta"] = float(
                row["future5_worst"] - pre_ordered[0])
            row["future5_weak3_delta"] = float(
                row["future5_weak3"]
                - np.mean(pre_ordered[:min(3, pre_ordered.size)]))
            episode_rows.append(row)
        augmented.append(episode_rows)

    flat = [row for episode in augmented for row in episode]
    if not flat:
        return {
            "eval_resp_episode_count": int(len(episodes)),
            "eval_resp_boundary_count": 0,
            "eval_resp_gate_pass": False,
        }

    rate_keys = [
        "duplicate_target_rate",
        "duplicate_frame",
        "duplicate_responsibility_rate",
        "ineffective_duplicate_responsibility_rate",
        "ineffective_duplicate_target_rate",
        "effective_overlap_target_rate",
        "same_role_overlap_target_rate",
        "uncovered_target_rate",
        "uncovered_but_sensed_target_rate",
        "uncovered_and_unsensed_target_rate",
        "productive_duplicate_marginal_pd",
        "ineffective_duplicate_marginal_pd",
        "single_responsibility_marginal_pd",
        "mean_node_marginal_pd",
    ]
    summary: Dict[str, object] = {
        "eval_resp_episode_count": int(len(episodes)),
        "eval_resp_boundary_count": int(len(flat)),
    }
    for key in rate_keys:
        summary[f"eval_resp_{key}"] = float(np.mean([
            row[key] for row in flat]))

    x = np.asarray([
        row["ineffective_duplicate_responsibility_rate"]
        for row in flat
    ])
    worst_delta = np.asarray([
        row["future5_worst_delta"] for row in flat])
    weak3_delta = np.asarray([
        row["future5_weak3_delta"] for row in flat])
    statistics = {
        "worst_spearman": _spearman(x, worst_delta),
        "weak3_spearman": _spearman(x, weak3_delta),
        "worst_group_delta": _group_difference(x, worst_delta),
        "weak3_group_delta": _group_difference(x, weak3_delta),
    }
    summary.update({
        f"eval_resp_future5_{key}": float(value)
        for key, value in statistics.items()
    })

    rng = np.random.default_rng(int(bootstrap_seed))
    bootstrap = {key: [] for key in statistics}
    episode_count = len(augmented)
    if episode_count > 0:
        for _ in range(max(1, int(bootstrap_samples))):
            sampled_indices = rng.integers(
                0, episode_count, size=episode_count)
            sampled = [
                row
                for index in sampled_indices
                for row in augmented[int(index)]
            ]
            if not sampled:
                continue
            sampled_x = np.asarray([
                row["ineffective_duplicate_responsibility_rate"]
                for row in sampled])
            sampled_worst = np.asarray([
                row["future5_worst_delta"] for row in sampled])
            sampled_weak3 = np.asarray([
                row["future5_weak3_delta"] for row in sampled])
            bootstrap["worst_spearman"].append(
                _spearman(sampled_x, sampled_worst))
            bootstrap["weak3_spearman"].append(
                _spearman(sampled_x, sampled_weak3))
            bootstrap["worst_group_delta"].append(
                _group_difference(sampled_x, sampled_worst))
            bootstrap["weak3_group_delta"].append(
                _group_difference(sampled_x, sampled_weak3))

    for key, values in bootstrap.items():
        low, high = _finite_interval(values)
        summary[f"eval_resp_future5_{key}_ci95"] = [low, high]

    worst_corr_ci = summary[
        "eval_resp_future5_worst_spearman_ci95"]
    worst_group_ci = summary[
        "eval_resp_future5_worst_group_delta_ci95"]
    significant_harm = (
        (np.isfinite(worst_corr_ci[1]) and worst_corr_ci[1] < 0.0)
        or (np.isfinite(worst_group_ci[1])
            and worst_group_ci[1] < 0.0)
    )
    prevalence = float(summary[
        "eval_resp_ineffective_duplicate_responsibility_rate"])
    independent_episode_count = sum(bool(rows) for rows in augmented)
    summary["eval_resp_gate_pass"] = bool(
        independent_episode_count >= 10
        and prevalence >= 0.05
        and significant_harm)
    summary["eval_resp_gate_rule"] = (
        "at least 10 independent episodes, ineffective duplicate "
        "responsibility rate >= 0.05, and the "
        "episode-cluster-bootstrap 95% upper bound is < 0 for either its "
        "Spearman association or present-vs-absent difference with future "
        "5-frame worst improvement"
    )
    return summary
