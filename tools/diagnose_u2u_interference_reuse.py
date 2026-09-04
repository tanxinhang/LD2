#!/usr/bin/env python
"""Diagnose certified U2U co-channel reuse without changing execution.

The diagnostic evaluates the current episode geometry under two explicit
traffic semantics:

* ``all_to_all``: every broadcast must reach every other UAV;
* ``disjoint_pairs``: sender 0 -> 1, 2 -> 3, ... (an optimistic routing bound).

The first is the semantics of the current strict control carrier.  The second
does not claim deployable performance; it only indicates whether narrowing
receiver sets could make certified spatial reuse worth implementing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.params import load_config
from uav_isac.environment.env_wrapper import UAVISACEnv
from uav_isac.environment.interference_certificate import (
    certified_spatial_reuse_schedule,
)


def _as_record(schedule) -> dict:
    return {
        "groups": [list(group) for group in schedule.groups],
        "all_links_certified": bool(schedule.all_links_certified),
        "uncertified_links": [list(link) for link in schedule.uncertified_links],
        "minimum_sinr_margin_db": float(schedule.minimum_sinr_margin_db),
        "max_concurrency": int(schedule.max_concurrency),
        "reuse_factor": float(schedule.reuse_factor),
        "pairwise_conflict_fraction": float(np.mean(
            schedule.pairwise_conflict[np.triu_indices(
                schedule.pairwise_conflict.shape[0], 1)]
        )) if schedule.pairwise_conflict.shape[0] > 1 else 0.0,
    }


def diagnose(config_path: str, seed: int, robust_margin_db: float) -> dict:
    cfg = load_config(config_path)
    env = UAVISACEnv(config=cfg)
    try:
        env.reset(seed=int(seed))
        core = env.core
        positions = np.asarray([uav.pos for uav in core.uavs], dtype=np.float64)
        active = tuple(range(core.K))
        powers = {
            sender: float(cfg.marl.comm_tx_power_w) for sender in active
        }
        physical = dict(
            carrier_hz=float(cfg.otfs.fc),
            bandwidth_hz=float(cfg.marl.comm_bandwidth_hz),
            kT=float(cfg.channel.kT),
            noise_figure_db=float(cfg.channel.NF),
            antenna_gain_dbi=float(cfg.marl.comm_antenna_gain_dbi),
            required_sinr_db=float(cfg.marl.comm_snr_threshold_db),
            desired_gain_margin_db=float(robust_margin_db),
            interference_gain_margin_db=float(robust_margin_db),
            half_duplex=True,
        )
        all_to_all = certified_spatial_reuse_schedule(
            positions,
            active,
            {
                sender: tuple(node for node in active if node != sender)
                for sender in active
            },
            powers,
            **physical,
        )

        paired_active = tuple(range(0, core.K - 1, 2))
        paired = certified_spatial_reuse_schedule(
            positions,
            paired_active,
            {sender: (sender + 1,) for sender in paired_active},
            {sender: powers[sender] for sender in paired_active},
            **physical,
        ) if paired_active else None
        return {
            "status": "shadow_diagnostic_only",
            "config": str(config_path),
            "seed": int(seed),
            "uavs": int(core.K),
            "robust_gain_margin_db_each_side": float(robust_margin_db),
            "executed_transport_unchanged": True,
            "all_to_all_current_semantics": _as_record(all_to_all),
            "disjoint_pair_optimistic_bound": (
                None if paired is None else _as_record(paired)),
            "interpretation": (
                "Enable no co-channel execution from this result alone; "
                "receiver-set routing, sensing-interference constraints and "
                "paired QoS tests remain mandatory."
            ),
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/exp_strict_distributed_k16q16.yaml",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--robust-margin-db", type=float, default=3.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = diagnose(args.config, args.seed, args.robust_margin_db)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
