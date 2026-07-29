"""Common-random-number interventions for movement target controllability."""

from __future__ import annotations

from copy import deepcopy
from itertools import combinations
from typing import Dict, List, Sequence

import numpy as np


def _pending_communication(core) -> Dict[str, Dict]:
    """Copy the already-decoded fast actions queued for the next frame."""
    return {
        "messages": {
            int(k): np.asarray(v).copy()
            for k, v in core._pending_comm_messages.items()
        },
        "rates": {
            int(k): int(v) for k, v in core._pending_comm_rates.items()
        },
        "fractions": {
            int(k): float(v)
            for k, v in core._pending_comm_power_fractions.items()
        },
        "sensing": {
            int(k): np.asarray(v).copy()
            for k, v in core._pending_sensing_weights.items()
        },
        "masks": {
            int(k): np.asarray(v).copy()
            for k, v in core._pending_comm_token_masks.items()
        },
    }


def evaluate_target_choice_intervention(
    env,
    actions: Dict[str, Dict],
    agent_index: int,
    actual_choice: int,
    num_targets: int,
    horizon: int,
) -> Dict[str, object]:
    """Enumerate one UAV's target direction while holding all else fixed.

    Each candidate starts from an identical deep-copied simulator state and
    receives the same communication, rate, sensing-power, and other-UAV
    movement actions.  The selected UAV keeps its original step magnitude but
    points toward candidate target ``q`` for the complete movement hold.
    """
    agent_index = int(agent_index)
    horizon = max(1, int(horizon))
    queued = _pending_communication(env.core)
    original_delta = np.asarray(
        actions[str(agent_index)]["delta_p"], dtype=np.float64)
    step_size = float(np.linalg.norm(original_delta))
    if step_size <= 1.0e-9:
        step_size = float(env.max_dp)

    uav_xy = np.asarray(env.core.uavs[agent_index].pos[:2])
    target_xy = np.stack([
        target.get_position_3d()[:2] for target in env.core.targets
    ])
    candidate_worst = np.zeros(num_targets, dtype=np.float64)
    candidate_weak3 = np.zeros(num_targets, dtype=np.float64)

    for q in range(num_targets):
        branch = deepcopy(env)
        branch_actions = {
            key: {
                "delta_p": np.asarray(value["delta_p"]).copy(),
                "role": int(value["role"]),
            }
            for key, value in actions.items()
        }
        direction = target_xy[q] - uav_xy
        distance = float(np.linalg.norm(direction))
        branch_actions[str(agent_index)]["delta_p"] = (
            direction / distance * min(step_size, distance)
            if distance > 1.0e-9 else np.zeros(2, dtype=np.float64)
        )
        pd_rows = []
        for _ in range(horizon):
            branch.core.submit_learned_communications(
                queued["messages"],
                queued["rates"],
                queued["fractions"],
                queued["sensing"],
                token_masks=queued["masks"],
            )
            _, _, terminated, truncated, info = branch.step(branch_actions)
            pd_rows.append(np.asarray(info["P_D_q"], dtype=np.float64))
            if (terminated.get("__all__", False)
                    or truncated.get("__all__", False)):
                break
        per_target = np.mean(np.asarray(pd_rows), axis=0)
        ordered = np.sort(per_target)
        candidate_worst[q] = ordered[0]
        candidate_weak3[q] = np.mean(
            ordered[:min(3, ordered.size)])
        branch.close()

    actual_choice = int(np.clip(actual_choice, 0, num_targets - 1))
    best_choice = int(np.argmax(candidate_worst))
    return {
        "agent_index": agent_index,
        "actual_choice": actual_choice,
        "best_choice": best_choice,
        "candidate_worst": candidate_worst.tolist(),
        "candidate_weak3": candidate_weak3.tolist(),
        "worst_value_std": float(np.std(candidate_worst)),
        "worst_value_range": float(np.ptp(candidate_worst)),
        "worst_oracle_gap": float(
            candidate_worst[best_choice] - candidate_worst[actual_choice]),
        "actual_is_best": float(actual_choice == best_choice),
        "weak3_value_range": float(np.ptp(candidate_weak3)),
        "weak3_oracle_gap": float(
            np.max(candidate_weak3) - candidate_weak3[actual_choice]),
    }


def blend_sensing_allocation(
    base_weights: np.ndarray,
    target_index: int,
    residual_blend: float,
) -> np.ndarray:
    """Move bounded sensing mass toward one target without changing total power."""
    base = np.asarray(base_weights, dtype=np.float64).reshape(-1)
    if base.size == 0:
        raise ValueError("sensing allocation cannot be empty")
    base = np.maximum(base, 0.0)
    base = base / max(float(np.sum(base)), 1.0e-12)
    target_index = int(np.clip(target_index, 0, base.size - 1))
    blend = float(np.clip(residual_blend, 0.0, 1.0))
    candidate = (1.0 - blend) * base
    candidate[target_index] += blend
    return candidate / max(float(np.sum(candidate)), 1.0e-12)


def apply_sensing_choice_intervention(
    env,
    agent_index: int,
    target_index: int,
    residual_blend: float,
) -> np.ndarray:
    """Replace one queued sensing allocation while preserving all other actions."""
    queued = _pending_communication(env.core)
    agent_index = int(agent_index)
    if agent_index not in queued["sensing"]:
        raise RuntimeError("sensing intervention requires queued ISAC resources")
    candidate = blend_sensing_allocation(
        queued["sensing"][agent_index],
        target_index,
        residual_blend,
    )
    queued["sensing"][agent_index] = candidate
    env.core.submit_learned_communications(
        queued["messages"],
        queued["rates"],
        queued["fractions"],
        queued["sensing"],
        token_masks=queued["masks"],
    )
    return candidate


def evaluate_sensing_choice_intervention(
    env,
    actions: Dict[str, Dict],
    agent_index: int,
    num_targets: int,
    horizon: int,
    residual_blend: float = 0.25,
) -> Dict[str, object]:
    """Enumerate a bounded single-UAV sensing residual under common randomness.

    Movement, roles, messages, rates, communication power and every other
    UAV's sensing allocation remain fixed. Only a bounded fraction of this
    UAV's normalized sensing mass is redirected to candidate target ``q``.
    """
    agent_index = int(agent_index)
    horizon = max(1, int(horizon))
    queued = _pending_communication(env.core)
    if agent_index not in queued["sensing"]:
        raise RuntimeError("sensing intervention requires queued ISAC resources")
    base_sensing = np.asarray(
        queued["sensing"][agent_index], dtype=np.float64).reshape(-1)
    if base_sensing.size != int(num_targets):
        raise ValueError("queued sensing allocation does not match target count")

    def rollout(sensing_allocations: Dict[int, np.ndarray]) -> np.ndarray:
        branch = deepcopy(env)
        branch_actions = {
            key: {
                "delta_p": np.asarray(value["delta_p"]).copy(),
                "role": int(value["role"]),
            }
            for key, value in actions.items()
        }
        pd_rows = []
        for _ in range(horizon):
            branch.core.submit_learned_communications(
                queued["messages"],
                queued["rates"],
                queued["fractions"],
                sensing_allocations,
                token_masks=queued["masks"],
            )
            _, _, terminated, truncated, info = branch.step(branch_actions)
            pd_rows.append(np.asarray(info["P_D_q"], dtype=np.float64))
            if (terminated.get("__all__", False)
                    or truncated.get("__all__", False)):
                break
        per_target = np.mean(np.asarray(pd_rows), axis=0)
        branch.close()
        return per_target

    baseline_per_target = rollout({
        int(k): np.asarray(value).copy()
        for k, value in queued["sensing"].items()
    })
    baseline_ordered = np.sort(baseline_per_target)
    baseline_worst = float(baseline_ordered[0])
    baseline_weak3 = float(np.mean(
        baseline_ordered[:min(3, baseline_ordered.size)]))
    baseline_steady = float(np.mean(baseline_per_target))

    candidate_worst = np.zeros(num_targets, dtype=np.float64)
    candidate_weak3 = np.zeros(num_targets, dtype=np.float64)
    candidate_steady = np.zeros(num_targets, dtype=np.float64)
    for q in range(num_targets):
        candidate_sensing = {
            int(k): np.asarray(value).copy()
            for k, value in queued["sensing"].items()
        }
        candidate_sensing[agent_index] = blend_sensing_allocation(
            base_sensing, q, residual_blend)
        per_target = rollout(candidate_sensing)
        ordered = np.sort(per_target)
        candidate_worst[q] = ordered[0]
        candidate_weak3[q] = np.mean(ordered[:min(3, ordered.size)])
        candidate_steady[q] = np.mean(per_target)

    # A verified residual is admissible only when it is Pareto-safe for the
    # two deployment constraints. No-op is an explicit member of the action
    # set, so a harmful residual is never forced into the oracle.
    safe_candidates = np.logical_and(
        candidate_weak3 >= baseline_weak3 - 1.0e-9,
        candidate_steady >= baseline_steady - 1.0e-9,
    )
    safe_worst = np.where(safe_candidates, candidate_worst, -np.inf)
    best_candidate = int(np.argmax(safe_worst))
    best_choice = (
        best_candidate
        if (safe_candidates[best_candidate]
            and candidate_worst[best_candidate] > baseline_worst + 1.0e-9)
        else -1
    )
    safe_best_worst = (
        float(candidate_worst[best_choice])
        if best_choice >= 0 else baseline_worst
    )
    safe_best_weak3 = max(
        [baseline_weak3] + [
            float(candidate_weak3[q])
            for q in range(num_targets) if safe_candidates[q]
        ])
    all_worst = np.concatenate([
        np.asarray([baseline_worst]), candidate_worst])
    all_weak3 = np.concatenate([
        np.asarray([baseline_weak3]), candidate_weak3])
    return {
        "agent_index": agent_index,
        "actual_choice": -1,
        "best_choice": best_choice,
        "baseline_worst": baseline_worst,
        "baseline_weak3": baseline_weak3,
        "baseline_steady": baseline_steady,
        "candidate_worst": candidate_worst.tolist(),
        "candidate_weak3": candidate_weak3.tolist(),
        "candidate_steady": candidate_steady.tolist(),
        "candidate_safe": safe_candidates.astype(float).tolist(),
        "worst_value_std": float(np.std(all_worst)),
        "worst_value_range": float(np.ptp(all_worst)),
        "worst_oracle_gap": float(safe_best_worst - baseline_worst),
        "actual_is_best": float(best_choice < 0),
        "weak3_value_range": float(np.ptp(all_weak3)),
        "weak3_oracle_gap": float(safe_best_weak3 - baseline_weak3),
        "residual_blend": float(np.clip(residual_blend, 0.0, 1.0)),
    }


def _cluster_mean_bootstrap(
    episodes: Sequence[Sequence[Dict[str, object]]],
    key: str,
    samples: int,
    rng: np.random.Generator,
) -> List[float]:
    values = []
    episode_count = len(episodes)
    for _ in range(max(1, int(samples))):
        indices = rng.integers(0, episode_count, size=episode_count)
        rows = [
            row for index in indices for row in episodes[int(index)]]
        if rows:
            values.append(float(np.mean([
                float(row[key]) for row in rows])))
    return values


def summarize_target_choice_interventions(
    episodes: Sequence[Sequence[Dict[str, object]]],
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 20260723,
) -> Dict[str, object]:
    """Aggregate intervention spreads and oracle gaps by episode clusters."""
    rows = [row for episode in episodes for row in episode]
    summary: Dict[str, object] = {
        "eval_target_cf_episode_count": int(len(episodes)),
        "eval_target_cf_intervention_count": int(len(rows)),
        "eval_target_cf_note": (
            "common-random-number physical intervention with other movement "
            "and fast communication/resource actions held fixed; critic "
            "ranking is unavailable because the historical formal checkpoint "
            "persisted actor only"),
    }
    if not rows:
        summary["eval_target_cf_gate_pass"] = False
        return summary

    keys = [
        "worst_value_std",
        "worst_value_range",
        "worst_oracle_gap",
        "actual_is_best",
        "weak3_value_range",
        "weak3_oracle_gap",
    ]
    rng = np.random.default_rng(int(bootstrap_seed))
    for key in keys:
        summary[f"eval_target_cf_{key}"] = float(np.mean([
            float(row[key]) for row in rows]))
        boot = _cluster_mean_bootstrap(
            episodes, key, bootstrap_samples, rng)
        summary[f"eval_target_cf_{key}_ci95"] = [
            float(np.quantile(boot, 0.025)),
            float(np.quantile(boot, 0.975)),
        ]
    summary["eval_target_cf_opportunity_gt_002_rate"] = float(np.mean([
        float(row["worst_oracle_gap"]) > 0.02 for row in rows]))
    if all("actual_is_ineffective_duplicate" in row for row in rows):
        inefficient_episodes = [[
            row for row in episode
            if float(row["actual_is_ineffective_duplicate"]) > 0.5
        ] for episode in episodes]
        inefficient_rows = [
            row for episode in inefficient_episodes for row in episode]
        summary["eval_target_cf_ineffective_duplicate_count"] = int(
            len(inefficient_rows))
        summary["eval_target_cf_ineffective_duplicate_rate"] = float(
            len(inefficient_rows) / max(len(rows), 1))
        summary["eval_target_cf_ineffective_duplicate_episode_count"] = int(
            sum(bool(episode) for episode in inefficient_episodes))
        if inefficient_rows:
            for key in (
                    "worst_value_range",
                    "worst_oracle_gap",
                    "weak3_oracle_gap"):
                name = f"eval_target_cf_ineffective_{key}"
                summary[name] = float(np.mean([
                    float(row[key]) for row in inefficient_rows]))
                boot = _cluster_mean_bootstrap(
                    inefficient_episodes,
                    key,
                    bootstrap_samples,
                    rng,
                )
                summary[f"{name}_ci95"] = (
                    [
                        float(np.quantile(boot, 0.025)),
                        float(np.quantile(boot, 0.975)),
                    ]
                    if boot
                    else [float("nan"), float("nan")]
                )
            summary[
                "eval_target_cf_ineffective_opportunity_gt_002_rate"
            ] = float(np.mean([
                float(row["worst_oracle_gap"]) > 0.02
                for row in inefficient_rows
            ]))
        else:
            summary[
                "eval_target_cf_ineffective_opportunity_gt_002_rate"
            ] = 0.0

    range_ci = summary["eval_target_cf_worst_value_range_ci95"]
    gap_ci = summary["eval_target_cf_worst_oracle_gap_ci95"]
    summary["eval_target_cf_gate_pass"] = bool(
        sum(bool(episode) for episode in episodes) >= 10
        and range_ci[0] > 0.05
        and gap_ci[0] > 0.02
    )
    summary["eval_target_cf_gate_rule"] = (
        "at least 10 independent episodes, mean candidate worst range 95% "
        "lower bound > 0.05, and mean actual-to-oracle gap lower bound > 0.02"
    )
    inefficient_count = int(summary.get(
        "eval_target_cf_ineffective_duplicate_count", 0))
    inefficient_episode_count = int(summary.get(
        "eval_target_cf_ineffective_duplicate_episode_count", 0))
    inefficient_gap_ci = summary.get(
        "eval_target_cf_ineffective_worst_oracle_gap_ci95",
        [float("nan"), float("nan")],
    )
    summary["eval_target_cf_ineffective_repair_gate_pass"] = bool(
        inefficient_count >= 20
        and inefficient_episode_count >= 10
        and np.isfinite(inefficient_gap_ci[0])
        and inefficient_gap_ci[0] > 0.02
    )
    summary["eval_target_cf_ineffective_repair_gate_rule"] = (
        "at least 20 ineffective-duplicate interventions from at least 10 "
        "episodes and their mean actual-to-oracle worst gap 95% lower bound "
        "> 0.02"
    )
    return summary


def summarize_sensing_choice_interventions(
    episodes: Sequence[Sequence[Dict[str, object]]],
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 20260723,
) -> Dict[str, object]:
    """Apply the clustered causal gate with sensing-specific metric names."""
    target_summary = summarize_target_choice_interventions(
        episodes,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    summary = {
        key.replace("eval_target_cf_", "eval_sensing_cf_"): value
        for key, value in target_summary.items()
    }
    summary["eval_sensing_cf_note"] = (
        "common-random-number bounded sensing-power intervention; movement, "
        "communication, total RF power and all other UAV allocations held fixed"
    )
    return summary


def evaluate_joint_sensing_pair_intervention(
    env,
    actions: Dict[str, Dict],
    num_targets: int,
    horizon: int,
    residual_blend: float = 0.25,
) -> Dict[str, object]:
    """Enumerate two-UAV residual options for the baseline weakest target.

    This intervention matches the bistatic structure: both endpoints redirect
    bounded sensing mass to the same target. No-op remains an explicit option,
    and candidates are admitted only when steady and weak3 do not decrease.
    """
    horizon = max(1, int(horizon))
    queued = _pending_communication(env.core)
    agent_indices = sorted(int(k) for k in queued["sensing"])
    pairs = list(combinations(agent_indices, 2))
    if not pairs:
        raise RuntimeError("joint sensing audit requires at least two UAVs")

    def rollout(sensing_allocations: Dict[int, np.ndarray]) -> np.ndarray:
        branch = deepcopy(env)
        branch_actions = {
            key: {
                "delta_p": np.asarray(value["delta_p"]).copy(),
                "role": int(value["role"]),
            }
            for key, value in actions.items()
        }
        pd_rows = []
        for _ in range(horizon):
            branch.core.submit_learned_communications(
                queued["messages"],
                queued["rates"],
                queued["fractions"],
                sensing_allocations,
                token_masks=queued["masks"],
            )
            _, _, terminated, truncated, info = branch.step(branch_actions)
            pd_rows.append(np.asarray(info["P_D_q"], dtype=np.float64))
            if (terminated.get("__all__", False)
                    or truncated.get("__all__", False)):
                break
        per_target = np.mean(np.asarray(pd_rows), axis=0)
        branch.close()
        return per_target

    baseline_sensing = {
        int(k): np.asarray(value).copy()
        for k, value in queued["sensing"].items()
    }
    baseline_per_target = rollout(baseline_sensing)
    target_index = int(np.argmin(baseline_per_target))
    baseline_ordered = np.sort(baseline_per_target)
    baseline_worst = float(baseline_ordered[0])
    baseline_weak3 = float(np.mean(
        baseline_ordered[:min(3, baseline_ordered.size)]))
    baseline_steady = float(np.mean(baseline_per_target))

    candidate_worst = np.zeros(len(pairs), dtype=np.float64)
    candidate_weak3 = np.zeros(len(pairs), dtype=np.float64)
    candidate_steady = np.zeros(len(pairs), dtype=np.float64)
    for pair_index, pair in enumerate(pairs):
        candidate_sensing = {
            int(k): np.asarray(value).copy()
            for k, value in baseline_sensing.items()
        }
        for agent_index in pair:
            candidate_sensing[agent_index] = blend_sensing_allocation(
                baseline_sensing[agent_index],
                target_index,
                residual_blend,
            )
        per_target = rollout(candidate_sensing)
        ordered = np.sort(per_target)
        candidate_worst[pair_index] = ordered[0]
        candidate_weak3[pair_index] = np.mean(
            ordered[:min(3, ordered.size)])
        candidate_steady[pair_index] = np.mean(per_target)

    safe_candidates = np.logical_and(
        candidate_weak3 >= baseline_weak3 - 1.0e-9,
        candidate_steady >= baseline_steady - 1.0e-9,
    )
    safe_worst = np.where(safe_candidates, candidate_worst, -np.inf)
    best_candidate = int(np.argmax(safe_worst))
    best_pair_index = (
        best_candidate
        if (safe_candidates[best_candidate]
            and candidate_worst[best_candidate] > baseline_worst + 1.0e-9)
        else -1
    )
    safe_best_worst = (
        float(candidate_worst[best_pair_index])
        if best_pair_index >= 0 else baseline_worst
    )
    safe_best_weak3 = max(
        [baseline_weak3] + [
            float(candidate_weak3[index])
            for index in range(len(pairs)) if safe_candidates[index]
        ])
    all_worst = np.concatenate([
        np.asarray([baseline_worst]), candidate_worst])
    all_weak3 = np.concatenate([
        np.asarray([baseline_weak3]), candidate_weak3])
    return {
        "agent_pairs": [list(pair) for pair in pairs],
        "target_index": target_index,
        "actual_choice": -1,
        "best_choice": best_pair_index,
        "best_pair": (
            list(pairs[best_pair_index])
            if best_pair_index >= 0 else []
        ),
        "baseline_worst": baseline_worst,
        "baseline_weak3": baseline_weak3,
        "baseline_steady": baseline_steady,
        "candidate_worst": candidate_worst.tolist(),
        "candidate_weak3": candidate_weak3.tolist(),
        "candidate_steady": candidate_steady.tolist(),
        "candidate_safe": safe_candidates.astype(float).tolist(),
        "worst_value_std": float(np.std(all_worst)),
        "worst_value_range": float(np.ptp(all_worst)),
        "worst_oracle_gap": float(safe_best_worst - baseline_worst),
        "actual_is_best": float(best_pair_index < 0),
        "weak3_value_range": float(np.ptp(all_weak3)),
        "weak3_oracle_gap": float(safe_best_weak3 - baseline_weak3),
        "residual_blend": float(np.clip(residual_blend, 0.0, 1.0)),
    }


def summarize_joint_sensing_pair_interventions(
    episodes: Sequence[Sequence[Dict[str, object]]],
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 20260723,
) -> Dict[str, object]:
    """Apply the clustered causal gate to paired bistatic interventions."""
    target_summary = summarize_target_choice_interventions(
        episodes,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    summary = {
        key.replace("eval_target_cf_", "eval_joint_sensing_cf_"): value
        for key, value in target_summary.items()
    }
    summary["eval_joint_sensing_cf_note"] = (
        "common-random-number no-op-controlled two-UAV sensing residual "
        "toward the baseline weakest target; steady and weak3 Pareto safety"
    )
    return summary
