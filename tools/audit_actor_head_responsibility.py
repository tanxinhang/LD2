#!/usr/bin/env python
"""Intervention audit for policy-head authority in the strict ISAC stack.

The canonical strict runner does not instantiate a learned actor.  This tool
therefore answers a second, counterfactual question: if actor-like output
fields are submitted to the same environment, which fields survive the
deterministic protocol/solver overrides and change executed physics?
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.params import load_config
from uav_isac.environment.env_wrapper import UAVISACEnv


INTERVENTIONS = (
    "movement",
    "role",
    "message_content",
    "rate",
    "comm_power_fraction",
    "sensing_weights",
)


def _run(
    config_path: str,
    *,
    seed: int,
    frames: int,
    intervention: str | None,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    env = UAVISACEnv(config=cfg)
    observations, _ = env.reset(seed=int(seed))
    trace: dict[str, list[Any]] = {
        "uav_positions": [],
        "detection": [],
        "comm_power": [],
        "sensing_power": [],
        "bits": [],
        "selected_edges": [],
    }
    try:
        for _frame in range(int(frames)):
            core = env.core
            messages = {
                sender: np.zeros(core._comm_payload_dim, dtype=np.float64)
                for sender in range(core.K)
            }
            rates = {sender: 0 for sender in range(core.K)}
            carrier_fraction = float(np.clip(
                float(cfg.marl.comm_tx_power_w)
                / max(float(cfg.uav.P_isac_total), 1.0e-12),
                0.0,
                1.0,
            ))
            fractions = {
                sender: carrier_fraction for sender in range(core.K)
            }
            weights = {
                sender: np.ones(core.Q, dtype=np.float64)
                for sender in range(core.K)
            }
            masks = {
                sender: np.ones(core.Q, dtype=np.float64)
                for sender in range(core.K)
            }
            actions = {
                str(agent): {
                    "delta_p": np.zeros(2, dtype=np.float64),
                    "role": 2,
                }
                for agent in observations
            }

            if intervention == "movement":
                actions["0"]["delta_p"] = np.asarray(
                    [float(cfg.uav.v_max) * float(cfg.scenario.dt), 0.0])
            elif intervention == "role":
                actions["0"]["role"] = 0
            elif intervention == "message_content":
                messages[0] = np.ones(core._comm_payload_dim, dtype=np.float64)
            elif intervention == "rate":
                rates[0] = min(1, len(cfg.marl.comm_rate_bits_per_dim) - 1)
            elif intervention == "comm_power_fraction":
                fractions[0] = min(0.75, float(
                    getattr(cfg.marl, "comm_power_fraction_max", 1.0)))
            elif intervention == "sensing_weights":
                weights[0] = np.zeros(core.Q, dtype=np.float64)
                weights[0][0] = 1.0
            elif intervention is not None:
                raise ValueError(f"unknown intervention: {intervention}")

            core.submit_learned_communications(
                messages=messages,
                rate_indices=rates,
                comm_power_fractions=fractions,
                sensing_target_weights=weights,
                token_masks=masks,
            )
            observations, _, terminated, _, info = env.step(actions)
            trace["uav_positions"].append(
                np.asarray(info["uav_positions"], dtype=np.float64).tolist())
            trace["detection"].append(
                np.asarray(info["P_D_q"], dtype=np.float64).tolist())
            trace["comm_power"].append(
                np.asarray(core._current_comm_power_w, dtype=np.float64).tolist())
            trace["sensing_power"].append(
                np.asarray(core._current_sensing_power_w, dtype=np.float64).tolist())
            trace["bits"].append(float(info.get(
                "total_bits_all", info.get("total_bits", 0.0))))
            trace["selected_edges"].append(sorted(
                [list(map(int, edge)) for edge in core._last_selected_set]
            ))
            if bool(terminated.get("__all__", False)):
                break
    finally:
        env.close()
    return trace


def _numeric_delta(left: list[Any], right: list[Any]) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a - b), initial=0.0))


def _compare(base: dict[str, Any], changed: dict[str, Any]) -> dict[str, Any]:
    deltas = {
        key: _numeric_delta(base[key], changed[key])
        for key in (
            "uav_positions",
            "detection",
            "comm_power",
            "sensing_power",
            "bits",
        )
    }
    structure_changed = base["selected_edges"] != changed["selected_edges"]
    physical_changed = bool(
        structure_changed
        or any(value > 1.0e-12 for value in deltas.values())
    )
    return {
        "physical_changed": physical_changed,
        "structure_changed": structure_changed,
        "max_abs_delta": deltas,
    }


def audit(config_path: str, *, seed: int, frames: int) -> dict[str, Any]:
    cfg = load_config(config_path)
    base = _run(
        config_path, seed=seed, frames=frames, intervention=None)
    results = {
        name: _compare(
            base,
            _run(
                config_path,
                seed=seed,
                frames=frames,
                intervention=name,
            ),
        )
        for name in INTERVENTIONS
    }
    return {
        "config": str(Path(config_path).resolve()),
        "seed": int(seed),
        "frames": int(frames),
        "formal_runner_has_learned_actor": False,
        "strict_overrides": {
            "movement": bool(
                cfg.marl.distributed_bistatic_bottleneck_movement_enabled),
            "role": not bool(cfg.marl.learn_roles),
            "message_content": bool(cfg.marl.hyperedge_protocol_only_enabled),
            "sensing_allocation": bool(
                cfg.marl.analytical_sensing_power_enabled),
        },
        "counterfactual_field_interventions": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/exp_strict_distributed_no_truth_pilot.yaml",
    )
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.config, seed=args.seed, frames=args.frames)
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
