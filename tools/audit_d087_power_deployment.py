#!/usr/bin/env python
"""D0.87: closed-loop sensing-power override counterfactual over full episodes.

The ceiling audit showed a large same-geometry ``power_gap`` (deployed 0.31 ->
fixed-power LP 0.675 on the 8/8 test trace).  This experiment asks whether that
headroom translates into an *episode-aggregated* gain when the deployed
per-frame sensing power is replaced, keeping everything else exactly as
recorded: role, receiver owner, selected edges, communication power and
geometry.

Three strictly paired modes share the same analytic per-watt reconstruction and
the same fusion boundary, differing ONLY in the sensing power allocation:

    A  deployed   : recorded (1 - P_comm) * normalized sensing weights
    B  exact LP   : fixed-owner max-min power LP (HiGHS)
    C  Dantzig-Wolfe : finite-round quantized column generation (default 4x6-bit)

Output is episode-aggregated (steady window = last 20 frames): mean worst,
QoS-feasible rate with one-sided Wilson LCB, bottom-20% CVaR, minimum worst, and
the paired ``LP - deployed`` / ``DW - deployed`` power gap with a non-parametric
bootstrap CI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from statistics import NormalDist

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.params import load_config  # noqa: E402
from uav_isac.coordination.maxmin_power import (  # noqa: E402
    distributed_column_generation_maxmin_power,
    fixed_owner_gain_matrix,
    solve_fixed_structure_maxmin_power_lp,
)
from uav_isac.evaluation.physical_oracle_audit import (  # noqa: E402
    per_watt_deflection_tensor_from_observables,
)
from uav_isac.physical.detection import compute_detection_probabilities  # noqa: E402

STEADY_WINDOW = 20


def _recorded_power(data: dict[str, np.ndarray], row: int) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, float,
]:
    """Return (comm_power, sensing_power, budget, balance_error) for a frame."""
    weights = np.asarray(data["sensing_weights"][row], dtype=np.float64)
    normalizer = np.sum(weights, axis=1, keepdims=True)
    weights = np.divide(
        weights, normalizer,
        out=np.zeros_like(weights),
        where=normalizer > 0.0,
    )
    rate = np.asarray(data["outgoing_rate"][row], dtype=np.int64)
    token_mask = np.asarray(data["outgoing_token_mask"][row], dtype=bool)
    active = (rate > 0) & np.any(token_mask, axis=1)
    fraction = np.clip(
        np.asarray(data["comm_fraction"][row], dtype=np.float64), 0.0, 1.0)
    comm_power = np.where(active, fraction, 0.0)
    budget = 1.0 - comm_power
    sensing_power = budget[:, None] * weights
    error = float(np.max(np.abs(
        comm_power + np.sum(sensing_power, axis=1) - 1.0)))
    return comm_power, sensing_power, budget, error


def _fixed_owner_gain(data: dict[str, np.ndarray], row: int, cfg) -> tuple[
    np.ndarray, np.ndarray,
]:
    """Analytic per-watt fixed-owner gain and owner vector for one frame."""
    coefficient = per_watt_deflection_tensor_from_observables(
        np.asarray(data["privileged_alpha"][row], dtype=np.float64),
        np.asarray(data["privileged_g_dd"][row], dtype=np.float64),
        np.asarray(data["privileged_chi_rep"][row], dtype=np.float64),
        T_sym=float(cfg.otfs.T_sym),
        M=int(cfg.otfs.M),
        N=int(cfg.otfs.N),
        kT=float(cfg.channel.kT),
        bandwidth_hz=float(cfg.otfs.B),
        noise_figure_db=float(cfg.channel.NF),
        g_tx_dbi=float(cfg.otfs.g_tx_dBi),
        g_rx_dbi=float(cfg.otfs.g_rx_dBi),
        n_cpi=int(cfg.otfs.n_cpi),
        g_min=float(cfg.detection.g_min),
        use_swerling=bool(cfg.channel.use_swerling),
    )
    pair = np.asarray(data["teacher_pair"][row], dtype=bool)
    selected = tuple(tuple(int(v) for v in edge) for edge in np.argwhere(pair))
    gain, owners = fixed_owner_gain_matrix(coefficient, selected)
    return gain, owners


def _deployed_deflection(data: dict[str, np.ndarray], row: int, gain: np.ndarray) -> np.ndarray:
    _, sensing_power, _, _ = _recorded_power(data, row)
    return np.sum(gain * sensing_power, axis=0)


def _episode_qos(pd_frames: np.ndarray) -> dict[str, float]:
    """Aggregate the last-20-frame steady window to episode QoS metrics."""
    window = np.asarray(pd_frames, dtype=np.float64)[-STEADY_WINDOW:]
    per_target = np.mean(window, axis=0)
    ordered = np.sort(per_target)
    return {
        "steady": float(np.mean(ordered)),
        "weak3": float(np.mean(ordered[:min(3, ordered.size)])),
        "worst": float(ordered[0]),
    }


def _wilson_lcb(successes: int, n: int, alpha: float = 0.05) -> float:
    if n == 0:
        return 0.0
    p_hat = successes / n
    # Audit 2026-08-25: align with the project's documented 95% two-sided
    # Wilson lower endpoint (z = 1.96); the previous one-sided
    # inv_cdf(1-alpha) = 1.645 could flip a 0.70 gate vs the formal tools.
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    denom = 1.0 + z * z / n
    center = p_hat + z * z / (2.0 * n)
    radius = z * np.sqrt(p_hat * (1.0 - p_hat) / n + z * z / (4.0 * n * n))
    return float(max(0.0, (center - radius) / denom))


def audit(
    trace_path: Path,
    config_path: Path,
    *,
    dual_rounds: int = 4,
    price_bits: int = 6,
    feedback_bits: int = 16,
) -> dict[str, object]:
    with np.load(trace_path, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    cfg = load_config(str(config_path))
    seeds = np.asarray(data["seed"], dtype=np.int64)
    frames = np.asarray(data["frame"], dtype=np.int64)
    p_fa = float(np.asarray(data["p_fa"]).reshape(-1)[0])
    seed_order = list(dict.fromkeys(int(s) for s in seeds.reshape(-1)))

    episodes = {name: [] for name in ("deployed", "lp", "dw")}
    max_replay_error = 0.0
    lp_infeasible_frames = 0

    for seed in seed_order:
        rows = np.flatnonzero(seeds == seed)
        rows = rows[np.argsort(frames[rows])]
        for name in ("deployed", "lp", "dw"):
            episodes[name].append([])
        for row in rows:
            try:
                gain, _owners = _fixed_owner_gain(data, int(row), cfg)
            except ValueError:
                lp_infeasible_frames += 1
                continue
            comm_power, sensing_power, budget, _err = _recorded_power(data, int(row))

            deployed_d = np.sum(gain * sensing_power, axis=0)
            deployed_pd = compute_detection_probabilities(deployed_d, p_fa)
            if "physical_pd" in data:
                trace_pd = np.asarray(data["physical_pd"][row], dtype=np.float64)
                max_replay_error = max(
                    max_replay_error,
                    float(np.max(np.abs(deployed_pd - trace_pd))),
                )

            try:
                exact = solve_fixed_structure_maxmin_power_lp(gain, budget)
                lp_pd = compute_detection_probabilities(exact.deflection, p_fa)
            except RuntimeError:
                lp_pd = deployed_pd

            try:
                finite = distributed_column_generation_maxmin_power(
                    gain,
                    budget,
                    rounds=int(dual_rounds),
                    price_bits=int(price_bits),
                    feedback_bits=int(feedback_bits),
                )
                dw_pd = compute_detection_probabilities(finite.deflection, p_fa)
            except RuntimeError:
                dw_pd = deployed_pd

            episodes["deployed"][-1].append(deployed_pd)
            episodes["lp"][-1].append(lp_pd)
            episodes["dw"][-1].append(dw_pd)

    metrics = {}
    for name in ("deployed", "lp", "dw"):
        qos = [_episode_qos(np.stack(e)) for e in episodes[name] if len(e) >= STEADY_WINDOW]
        worst = np.array([q["worst"] for q in qos], dtype=np.float64)
        steady = np.array([q["steady"] for q in qos], dtype=np.float64)
        weak3 = np.array([q["weak3"] for q in qos], dtype=np.float64)
        feasible = (steady >= 0.80) & (weak3 >= 0.70) & (worst >= 0.60)
        n = int(worst.size)
        tail = max(1, int(np.ceil(0.20 * n))) if n else 1
        metrics[name] = {
            "episodes": n,
            "mean_worst": float(np.mean(worst)) if n else float("nan"),
            "mean_steady": float(np.mean(steady)) if n else float("nan"),
            "mean_weak3": float(np.mean(weak3)) if n else float("nan"),
            "worst_min": float(np.min(worst)) if n else float("nan"),
            "worst_cvar20": float(np.mean(np.sort(worst)[:tail])) if n else float("nan"),
            "qos_feasible_rate": float(np.mean(feasible)) if n else float("nan"),
            "qos_feasible_wilson_lcb": _wilson_lcb(int(np.sum(feasible)), n),
        }

    # Paired power gap (LP - deployed, DW - deployed) with bootstrap CI.
    rng = np.random.default_rng(20260814)
    paired = {}
    for name, tag in (("lp", "LP"), ("dw", "DW")):
        base = np.array([q["worst"] for q in
                         [_episode_qos(np.stack(e)) for e in episodes["deployed"] if len(e) >= STEADY_WINDOW]],
                        dtype=np.float64)
        other = np.array([q["worst"] for q in
                          [_episode_qos(np.stack(e)) for e in episodes[name] if len(e) >= STEADY_WINDOW]],
                         dtype=np.float64)
        if base.size and base.size == other.size:
            diff = other - base
            draws = diff[rng.integers(0, diff.size, size=(2000, diff.size))].mean(axis=1)
            paired[tag] = {
                "mean_power_gap": float(np.mean(diff)),
                "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
                "improved_episodes": int(np.sum(diff > 1e-6)),
                "degraded_episodes": int(np.sum(diff < -1e-6)),
                "unchanged_episodes": int(np.sum(np.abs(diff) <= 1e-6)),
            }

    return {
        "schema_version": 1,
        "scope": (
            "full-episode closed-loop power-override counterfactual; same "
            "role/owner/edge/comm-power/geometry, only sensing power changes"
        ),
        "trace": str(trace_path),
        "config": str(config_path),
        "dual_rounds": int(dual_rounds),
        "price_bits": int(price_bits),
        "feedback_bits": int(feedback_bits),
        "steady_window_frames": STEADY_WINDOW,
        "max_deployed_replay_error": float(max_replay_error),
        "lp_infeasible_frames": int(lp_infeasible_frames),
        "metrics": metrics,
        "paired_power_gap": paired,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dual-rounds", type=int, default=4)
    parser.add_argument("--price-bits", type=int, default=6)
    parser.add_argument("--feedback-bits", type=int, default=16)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(
        args.trace,
        args.config,
        dual_rounds=int(args.dual_rounds),
        price_bits=int(args.price_bits),
        feedback_bits=int(args.feedback_bits),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "metrics": result["metrics"],
        "paired_power_gap": result["paired_power_gap"],
        "max_deployed_replay_error": result["max_deployed_replay_error"],
        "lp_infeasible_frames": result["lp_infeasible_frames"],
    }, indent=2))


if __name__ == "__main__":
    main()
