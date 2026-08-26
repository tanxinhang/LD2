"""Same-geometry upper bounds for endpoint/power and duplex structure."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np

from uav_isac.physical.channel import compute_noise_power
from uav_isac.physical.feasibility_oracle import (
    solve_pair_only_oracle,
    solve_power_only_oracle,
    solve_joint_pair_power_oracle,
)


def per_watt_deflection_tensor(
    entries: Iterable,
    sensing_power_w: np.ndarray,
    num_uavs: int,
    num_targets: int,
) -> np.ndarray:
    """Recover linear per-watt gains from realized powered deflections."""
    power = np.asarray(sensing_power_w, dtype=np.float64)
    if power.shape != (int(num_uavs), int(num_targets)):
        raise ValueError("sensing power shape does not match K/Q")
    coefficient = np.zeros(
        (int(num_uavs), int(num_uavs), int(num_targets)),
        dtype=np.float64,
    )
    for entry in entries:
        tx = int(entry.i)
        rx = int(entry.j)
        target = int(entry.q)
        tx_power = float(power[tx, target])
        if tx != rx and tx_power > 1.0e-12 and float(entry.d_eff) > 0.0:
            coefficient[tx, rx, target] = (
                float(entry.d_eff) / tx_power
            )
    return coefficient


def per_watt_deflection_tensor_from_observables(
    alpha: np.ndarray,
    g_dd: np.ndarray,
    chi_rep: np.ndarray,
    *,
    T_sym: float,
    M: int,
    N: int,
    kT: float,
    bandwidth_hz: float,
    noise_figure_db: float,
    g_tx_dbi: float,
    g_rx_dbi: float,
    n_cpi: int,
    g_min: float,
    c_det: float = 1.0,
    use_swerling: bool = False,
) -> np.ndarray:
    """Reconstruct power-independent Deflection gain from observables.

    Dividing realized Deflection by power cannot identify an edge when its
    current target allocation is zero.  In the deterministic non-Swerling
    model, stored path magnitude, DD gate and report reliability determine the
    counterfactual coefficient analytically for every supported edge.
    """
    path = np.asarray(alpha, dtype=np.float64)
    dd = np.asarray(g_dd, dtype=np.float64)
    report = np.asarray(chi_rep, dtype=np.float64)
    if path.shape != dd.shape or path.shape != report.shape or path.ndim != 3:
        raise ValueError("alpha/g_dd/chi_rep must share shape (K,K,Q)")
    if (
        np.any(~np.isfinite(path)) or np.any(path < 0.0)
        or np.any(~np.isfinite(dd)) or np.any(dd < 0.0)
        or np.any(~np.isfinite(report)) or np.any(report < 0.0)
    ):
        raise ValueError("physical observables must be finite and non-negative")
    if bool(use_swerling):
        raise ValueError(
            "Swerling realization is not identifiable from alpha/g_dd/chi_rep")
    noise = compute_noise_power(
        float(kT), float(bandwidth_hz), float(noise_figure_db))
    antenna_gain = float(10.0 ** (
        (float(g_tx_dbi) + float(g_rx_dbi)) / 10.0))
    if float(T_sym) <= 0.0:
        raise ValueError("T_sym must be positive for energy normalization")
    detector_scale = float(c_det)
    if not np.isfinite(detector_scale) or detector_scale <= 0.0:
        raise ValueError("c_det must be finite and positive")
    scale = float(
        detector_scale * int(M) * int(N) * antenna_gain * int(n_cpi)
        / max(noise, 1.0e-15)
    )
    K = path.shape[0]
    if path.shape[1] != K:
        raise ValueError("physical observable tensor must be square in K")
    active = (
        (dd >= float(g_min))
        & (~np.eye(K, dtype=bool)[:, :, None])
    )
    return np.where(active, report * path ** 2 * scale, 0.0)


def evaluate_physical_feasibility_oracles(
    entries: Iterable,
    sensing_power_w: np.ndarray,
    deployed_pd: np.ndarray,
    deployed_selected_set: Sequence[tuple[int, int, int]],
    *,
    num_uavs: int,
    num_targets: int,
    p_fa: float,
    total_power_w: float,
    communication_reserve_w: float,
    sensing_power_cap_w: float | None = None,
    target_pair_limit: int,
    reports_per_receiver: int,
    seed: int,
    detection_fusion_mode: str = "central_oracle",
    coefficient_override: np.ndarray | None = None,
) -> Dict[str, float]:
    """Compare deployed detection to optimized single-role and duplex bounds."""
    if coefficient_override is None:
        coefficient = per_watt_deflection_tensor(
            entries,
            sensing_power_w,
            num_uavs,
            num_targets,
        )
    else:
        coefficient = np.asarray(coefficient_override, dtype=np.float64)
        expected = (int(num_uavs), int(num_uavs), int(num_targets))
        if coefficient.shape != expected:
            raise ValueError(
                "coefficient_override must have shape (K,K,Q)")
        if np.any(~np.isfinite(coefficient)) or np.any(coefficient < 0.0):
            raise ValueError(
                "coefficient_override must be finite and non-negative")
    kwargs = {
        "P_FA": float(p_fa),
        "total_power_w": float(total_power_w),
        "communication_reserve_w": float(communication_reserve_w),
        "sensing_power_cap_w": sensing_power_cap_w,
        "target_pair_limit": int(target_pair_limit),
        "reports_per_receiver": int(reports_per_receiver),
        "alternating_iterations": 4,
        "random_starts": 1,
        "seed": int(seed),
        "fusion_mode": str(detection_fusion_mode),
    }
    single = solve_joint_pair_power_oracle(
        coefficient, full_duplex=False, **kwargs)
    duplex = solve_joint_pair_power_oracle(
        coefficient, full_duplex=True, **kwargs)
    pair_only = solve_pair_only_oracle(
        coefficient,
        sensing_power_w,
        P_FA=float(p_fa),
        target_pair_limit=int(target_pair_limit),
        reports_per_receiver=int(reports_per_receiver),
        fusion_mode=str(detection_fusion_mode),
    )
    power_only = solve_power_only_oracle(
        coefficient,
        deployed_selected_set,
        P_FA=float(p_fa),
        total_power_w=float(total_power_w),
        communication_reserve_w=float(communication_reserve_w),
        sensing_power_cap_w=sensing_power_cap_w,
        fusion_mode=str(detection_fusion_mode),
    )
    deployed = np.asarray(deployed_pd, dtype=np.float64)
    deployed_ordered = np.sort(deployed)
    return {
        "fusion_mode": str(detection_fusion_mode),
        "deployed_worst": float(deployed_ordered[0]),
        "deployed_weak3": float(np.mean(
            deployed_ordered[:min(3, deployed_ordered.size)])),
        "deployed_steady": float(np.mean(deployed)),
        "pair_only_worst": pair_only.worst,
        "pair_only_weak3": pair_only.weak3,
        "pair_only_steady": pair_only.steady,
        "power_only_worst": power_only.worst,
        "power_only_weak3": power_only.weak3,
        "power_only_steady": power_only.steady,
        "single_worst": single.worst,
        "single_weak3": single.weak3,
        "single_steady": single.steady,
        "duplex_worst": duplex.worst,
        "duplex_weak3": duplex.weak3,
        "duplex_steady": duplex.steady,
        "single_worst_gap": float(single.worst - deployed_ordered[0]),
        "pair_only_worst_gap": float(
            pair_only.worst - deployed_ordered[0]),
        "power_only_worst_gap": float(
            power_only.worst - deployed_ordered[0]),
        "joint_over_best_isolated_worst_gap": float(
            single.worst - max(pair_only.worst, power_only.worst)),
        "duplex_worst_gap": float(duplex.worst - deployed_ordered[0]),
        "duplex_over_single_worst_gap": float(
            duplex.worst - single.worst),
    }


def _episode_bootstrap(
    episodes: Sequence[Sequence[Dict[str, float]]],
    key: str,
    samples: int,
    seed: int,
) -> list[float]:
    episode_values = [
        float(np.mean([float(row[key]) for row in episode]))
        for episode in episodes if episode
    ]
    if not episode_values:
        return []
    values = np.asarray(episode_values, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    return [
        float(np.mean(values[
            rng.integers(0, values.size, size=values.size)]))
        for _ in range(max(1, int(samples)))
    ]


def summarize_physical_feasibility_oracles(
    episodes: Sequence[Sequence[Dict[str, float]]],
    *,
    worst_floor: float = 0.60,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 20260727,
) -> Dict[str, object]:
    """Aggregate physical upper bounds without mixing frames across episodes."""
    rows = [row for episode in episodes for row in episode]
    summary: Dict[str, object] = {
        "eval_physical_oracle_episode_count": int(len(episodes)),
        "eval_physical_oracle_frame_count": int(len(rows)),
        "eval_physical_oracle_note": (
            "same realized geometry/channel and detector fusion boundary; "
            "exact no-ground-link RF reserve; alternating mixed-integer pair "
            "and max-min power upper benchmark"),
    }
    if not rows:
        summary["eval_physical_oracle_pair_only_gate_pass"] = False
        summary["eval_physical_oracle_single_gate_pass"] = False
        summary["eval_physical_oracle_duplex_gate_pass"] = False
        return summary
    fusion_modes = sorted({
        str(row.get("fusion_mode", "central_oracle")) for row in rows
    })
    summary["eval_physical_oracle_fusion_modes"] = fusion_modes

    keys = (
        "deployed_worst",
        "deployed_weak3",
        "deployed_steady",
        "pair_only_worst",
        "pair_only_weak3",
        "pair_only_steady",
        "power_only_worst",
        "power_only_weak3",
        "power_only_steady",
        "single_worst",
        "single_weak3",
        "single_steady",
        "duplex_worst",
        "duplex_weak3",
        "duplex_steady",
        "single_worst_gap",
        "pair_only_worst_gap",
        "power_only_worst_gap",
        "joint_over_best_isolated_worst_gap",
        "duplex_worst_gap",
        "duplex_over_single_worst_gap",
    )
    for index, key in enumerate(keys):
        name = f"eval_physical_oracle_{key}"
        summary[name] = float(np.mean([
            float(row[key]) for row in rows]))
        bootstrap = _episode_bootstrap(
            episodes,
            key,
            bootstrap_samples,
            bootstrap_seed + index,
        )
        summary[f"{name}_ci95"] = [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ]

    summary["eval_physical_oracle_single_feasible_rate"] = float(
        np.mean([
            float(row["single_worst"]) >= float(worst_floor)
            for row in rows
        ]))
    summary["eval_physical_oracle_duplex_feasible_rate"] = float(
        np.mean([
            float(row["duplex_worst"]) >= float(worst_floor)
            for row in rows
        ]))
    summary["eval_physical_oracle_pair_only_feasible_rate"] = float(
        np.mean([
            float(row["pair_only_worst"]) >= float(worst_floor)
            for row in rows
        ]))
    summary["eval_physical_oracle_power_only_feasible_rate"] = float(
        np.mean([
            float(row["power_only_worst"]) >= float(worst_floor)
            for row in rows
        ]))
    single_ci = summary[
        "eval_physical_oracle_single_worst_gap_ci95"]
    pair_only_ci = summary[
        "eval_physical_oracle_pair_only_worst_gap_ci95"]
    duplex_role_ci = summary[
        "eval_physical_oracle_duplex_over_single_worst_gap_ci95"]
    enough_episodes = sum(bool(episode) for episode in episodes) >= 10
    summary["eval_physical_oracle_pair_only_gate_pass"] = bool(
        enough_episodes
        and summary["eval_physical_oracle_pair_only_worst_gap"] >= 0.03
        and pair_only_ci[0] > 0.01
    )
    summary["eval_physical_oracle_single_gate_pass"] = bool(
        enough_episodes
        and summary["eval_physical_oracle_single_worst_gap"] >= 0.03
        and single_ci[0] > 0.01
    )
    summary["eval_physical_oracle_duplex_gate_pass"] = bool(
        enough_episodes
        and summary[
            "eval_physical_oracle_duplex_over_single_worst_gap"] >= 0.03
        and duplex_role_ci[0] > 0.01
    )
    summary["eval_physical_oracle_gate_rule"] = (
        "for pair-only, joint single-role, and duplex-over-single gates: "
        "at least 10 independent episodes, mean worst gap >= 0.03 and "
        "episode-bootstrap 95% lower bound > 0.01")
    return summary
