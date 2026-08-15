#!/usr/bin/env python
"""D0.88: hold-H power cadence with gate-forced re-solve, and staleness audit.

D0.87 showed every-frame max-min power raises 8/8 mean-worst by +0.2447.  This
experiment measures how much of that gain survives when the power is held for
H frames (aligned with the structure cadence) and re-solved only on (a) the
H-frame cadence or (b) a selected-edge DD-gate support crossing.

It also measures the *actual* held-power staleness ``t*' - t_held`` and compares
it with the certified worst-case bound ``2B``, quantifying how loose the
worst-case bound is as a re-solve trigger.

Only sensing power changes; role/owner/edge, communication power and geometry
stay as recorded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.params import load_config  # noqa: E402
from uav_isac.coordination.maxmin_power import (  # noqa: E402
    fixed_owner_gain_matrix,
    solve_fixed_structure_maxmin_power_lp,
)
from uav_isac.coordination.power_staleness import (  # noqa: E402
    dd_gate_crossing,
    held_power_staleness_bound,
)
from uav_isac.evaluation.physical_oracle_audit import (  # noqa: E402
    per_watt_deflection_tensor_from_observables,
)
from uav_isac.physical.detection import compute_detection_probabilities  # noqa: E402

STEADY_WINDOW = 20


def _gain_and_budget(data, row: int, cfg):
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
    gain, _owners = fixed_owner_gain_matrix(coefficient, selected)
    rate = np.asarray(data["outgoing_rate"][row], dtype=np.int64)
    token_mask = np.asarray(data["outgoing_token_mask"][row], dtype=bool)
    active = (rate > 0) & np.any(token_mask, axis=1)
    fraction = np.clip(
        np.asarray(data["comm_fraction"][row], dtype=np.float64), 0.0, 1.0)
    budget = 1.0 - np.where(active, fraction, 0.0)
    return gain, budget


def run_hold(
    data: dict[str, np.ndarray],
    cfg,
    p_fa: float,
    g_min: float,
    *,
    hold_frames: int,
    gate_forced: bool,
) -> dict[str, float]:
    seeds = np.asarray(data["seed"], dtype=np.int64)
    frames = np.asarray(data["frame"], dtype=np.int64)
    seed_order = list(dict.fromkeys(int(s) for s in seeds.reshape(-1)))
    episode_worst = []
    re_solve_count = 0
    actual_gaps = []
    bound_gaps = []

    for seed in seed_order:
        rows = np.flatnonzero(seeds == seed)
        rows = rows[np.argsort(frames[rows])]
        last_gain = None
        last_power = None
        last_g_dd = None
        pd_frames = []
        for step, row in enumerate(rows):
            gain, budget = _gain_and_budget(data, int(row), cfg)
            g_dd = np.asarray(data["privileged_g_dd"][row], dtype=np.float64)
            pair = np.asarray(data["teacher_pair"][row], dtype=bool)

            should_resolve = (
                last_gain is None
                or (step % max(1, int(hold_frames)) == 0)
            )
            if (
                gate_forced
                and last_gain is not None
                and last_g_dd is not None
                and np.any(dd_gate_crossing(last_g_dd, g_dd, g_min) & pair)
            ):
                should_resolve = True

            if should_resolve:
                re_solve_count += 1
                last_gain = gain
                last_power = solve_fixed_structure_maxmin_power_lp(
                    gain, budget).power_w
                last_g_dd = g_dd
            else:
                t_held = float(np.min(np.sum(gain * last_power, axis=0)))
                exact = solve_fixed_structure_maxmin_power_lp(gain, budget)
                actual_gaps.append(exact.worst_deflection - t_held)
                bound_gaps.append(held_power_staleness_bound(
                    last_gain, gain, budget))

            d_q = np.sum(gain * last_power, axis=0)
            pd_frames.append(compute_detection_probabilities(d_q, p_fa))

        window = np.asarray(pd_frames, dtype=np.float64)[-STEADY_WINDOW:]
        episode_worst.append(float(np.min(np.mean(window, axis=0))))

    worst = np.asarray(episode_worst, dtype=np.float64)
    return {
        "hold_frames": int(hold_frames),
        "gate_forced": bool(gate_forced),
        "mean_worst": float(np.mean(worst)),
        "worst_min": float(np.min(worst)),
        "re_solve_count": int(re_solve_count),
        "re_solve_per_frame": float(re_solve_count / max(1, int(len(seeds)))),
        "mean_actual_staleness_deflection": float(np.mean(actual_gaps)) if actual_gaps else 0.0,
        "mean_certified_bound_deflection": float(np.mean(bound_gaps)) if bound_gaps else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--holds", type=int, nargs="+", default=[1, 2, 5, 10])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with np.load(args.trace, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    cfg = load_config(str(args.config))
    p_fa = float(np.asarray(data["p_fa"]).reshape(-1)[0])
    g_min = float(cfg.detection.g_min)

    rows = []
    for hold in sorted(set(args.holds)):
        rows.append(run_hold(data, cfg, p_fa, g_min,
                             hold_frames=hold, gate_forced=True))
    result = {"schema_version": 1, "trace": str(args.trace),
              "config": str(args.config), "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
