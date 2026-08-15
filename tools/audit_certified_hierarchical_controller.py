#!/usr/bin/env python
"""Audit the complete certificate-routed ISAC controller on trace events.

This is a same-event full-information architecture audit. It uses the exact
analytic coefficient as a degenerate lower/upper envelope to isolate solver
and protocol headroom. It is not a causal uncertainty certificate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.params import load_config  # noqa: E402
from tools.audit_power_repair_transport_trace import (  # noqa: E402
    _communication_model,
)
from tools.audit_qos_threshold_boundary import _coefficient  # noqa: E402
from tools.audit_structure_trace_physical_bottleneck import (  # noqa: E402
    _ordered_unique,
    _recorded_power,
)
from uav_isac.coordination.certified_hierarchical_controller import (  # noqa: E402
    CertifiedHierarchicalControllerConfig,
    certified_hierarchical_isac_control,
)
from uav_isac.coordination.certified_maxmin_power_controller import (  # noqa: E402
    CertifiedMaxMinPowerConfig,
)


def audit(
    trace_path: Path,
    config_path: Path,
    *,
    seed_limit: int,
    qos_floor: float,
    owners_per_target: int,
    rounds: int,
    structure_ranking_rounds: int,
    price_bits: int,
    feedback_bits: int,
    snr_margin_db: float,
    latency_margin_s: float,
    require_pre_reserved_comm_power: bool,
) -> dict[str, object]:
    with np.load(trace_path, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    cfg = load_config(str(config_path))
    _, communication_model = _communication_model(config_path)
    seeds = np.asarray(data["seed"], dtype=np.int64)
    frames = np.asarray(data["frame"], dtype=np.int64)
    resolved = np.asarray(data["p0_resolved"], dtype=bool)
    seed_order = _ordered_unique(seeds)[:max(1, int(seed_limit))]
    K = int(np.asarray(data["num_uavs"]).reshape(-1)[0])
    Q = int(np.asarray(data["num_targets"]).reshape(-1)[0])
    p_fa = float(np.asarray(data["p_fa"]).reshape(-1)[0])
    pair_limit = int(np.asarray(data["target_pair_limit"]).reshape(-1)[0])
    receiver_limit = int(np.asarray(
        data["reports_per_receiver"]).reshape(-1)[0])
    if K != int(cfg.scenario.K) or Q != int(cfg.scenario.Q):
        raise ValueError("trace dimensions do not match supplied config")
    controller_config = CertifiedHierarchicalControllerConfig(
        qos_floor=float(qos_floor),
        target_pair_limit=pair_limit,
        reports_per_receiver=receiver_limit,
        owners_per_target=min(int(owners_per_target), K),
        structure_weak_target_count=Q,
        structure_ranking_rounds=int(structure_ranking_rounds),
        require_pre_reserved_comm_power=bool(
            require_pre_reserved_comm_power),
        snr_margin_db=float(snr_margin_db),
        latency_margin_s=float(latency_margin_s),
        power=CertifiedMaxMinPowerConfig(
            rounds=int(rounds),
            price_bits=int(price_bits),
            feedback_bits=int(feedback_bits),
            snr_margin_db=float(snr_margin_db),
            latency_margin_s=float(latency_margin_s),
            qos_floor=float(qos_floor),
        ),
    )

    rows: list[dict[str, object]] = []
    started = perf_counter()
    for seed in seed_order:
        candidates = np.flatnonzero(resolved & (seeds == seed))
        if not len(candidates):
            raise ValueError(f"seed {seed} has no resolved event")
        recorded_worst = np.min(np.asarray(
            data["physical_pd"][candidates], dtype=np.float64), axis=1)
        row_id = int(candidates[np.argmin(recorded_worst)])
        comm, sensing, power_error = _recorded_power(data, row_id)
        coefficient = _coefficient(data, row_id, cfg)
        selected = np.asarray(data["teacher_pair"][row_id], dtype=bool)
        # Trace labels use 0=Tx/1=Rx; the coordination layer uses 1=Tx/0=Rx.
        role = 1 - np.asarray(data["teacher_role"][row_id], dtype=np.int8)
        decision = certified_hierarchical_isac_control(
            coefficient,
            coefficient,
            selected,
            role,
            np.asarray(data["uav_positions"][row_id], dtype=np.float64),
            comm,
            sensing,
            communication_model=communication_model,
            control_period_s=float(cfg.scenario.dt),
            p_fa=p_fa,
            config=controller_config,
        )
        resources = decision.protocol_resources
        bid_transport = decision.owner_bid_transport
        structure = decision.structure_repair
        rows.append({
            "seed": int(seed),
            "frame": int(frames[row_id]),
            "recorded_worst": float(np.min(
                np.asarray(data["physical_pd"][row_id], dtype=np.float64))),
            "model_noop_lower_worst": float(np.min(decision.noop_lower_pd)),
            "model_noop_upper_worst": float(np.min(decision.noop_upper_pd)),
            "candidate_lower_worst": float(np.min(
                decision.candidate_lower_pd)),
            "candidate_upper_worst": float(np.min(
                decision.candidate_upper_pd)),
            "certified_worst_improvement": float(
                decision.certified_worst_pd_improvement),
            "route": decision.route.value,
            "accepted": bool(decision.accepted),
            "reason": decision.reason,
            "qos_feasible": bool(
                np.min(decision.candidate_lower_pd) >= float(qos_floor)),
            "target_no_harm": bool(np.all(
                decision.candidate_lower_pd + 1.0e-9
                >= np.minimum(decision.noop_upper_pd, float(qos_floor)))),
            "selected_edge_count": int(np.count_nonzero(decision.selected)),
            "changed_edge_count": int(np.count_nonzero(
                decision.selected != selected)),
            "owner_bid_candidate_edge_count": int(
                0 if decision.owner_bid_candidate is None
                else decision.owner_bid_candidate.candidate_edge_count),
            "owner_bid_full_edge_count": int(
                0 if decision.owner_bid_candidate is None
                else decision.owner_bid_candidate.full_edge_count),
            "owner_bid_transport_feasible": bool(
                bid_transport is None or bid_transport.feasible),
            "owner_bid_bits": int(
                0 if bid_transport is None
                else bid_transport.total_over_air_bits),
            "owner_bid_latency_s": float(
                0.0 if bid_transport is None
                else bid_transport.total_protocol_latency_s),
            "structure_candidate_count": int(
                0 if structure is None else structure.candidate_count),
            "structure_exact_verification_count": int(
                0 if structure is None else structure.exact_verification_count),
            "structure_accepted_steps": int(
                0 if structure is None else structure.accepted_steps),
            "protocol_feasible": bool(
                resources is None or resources.feasible),
            "protocol_reasons": list(
                () if resources is None else resources.reasons),
            "protocol_total_bits": int(
                0 if resources is None else resources.total_over_air_bits),
            "protocol_total_latency_s": float(
                0.0 if resources is None
                else resources.total_protocol_latency_s),
            "protocol_total_energy_j": float(
                0.0 if resources is None else resources.total_energy_j),
            "protocol_power_balance_error_w": float(
                0.0 if resources is None
                else resources.max_isac_power_balance_error_w),
            "recorded_power_balance_error_w": float(power_error),
        })

    active = [row for row in rows if row["route"] not in (
        "no_op", "slow_geometry", "hold_unverified")]
    accepted = [row for row in rows if bool(row["accepted"])]
    structure_rows = [
        row for row in rows if row["route"] == "joint_structure_power"
    ]
    summary = {
        "schema_version": 1,
        "scope": (
            "same-event exact-coefficient architecture audit of certificate-"
            "routed No-op/power/owner-auction-Top1-structure/geometry control"
        ),
        "causal_certificate": False,
        "trace": str(trace_path),
        "config": str(config_path),
        "seed_count": len(rows),
        "seed_order": [int(seed) for seed in seed_order],
        "qos_floor": float(qos_floor),
        "owners_per_target": int(owners_per_target),
        "power_rounds": int(rounds),
        "structure_ranking_rounds": int(structure_ranking_rounds),
        "require_pre_reserved_comm_power": bool(
            require_pre_reserved_comm_power),
        "route_counts": {
            route: int(sum(row["route"] == route for row in rows))
            for route in (
                "no_op", "fixed_structure_power", "joint_structure_power",
                "slow_geometry", "hold_unverified",
            )
        },
        "accepted_rate": float(np.mean([
            bool(row["accepted"]) for row in rows
        ])),
        "active_route_acceptance_rate": float(
            np.mean([bool(row["accepted"]) for row in active])
            if active else 0.0),
        "structure_route_acceptance_rate": float(
            np.mean([bool(row["accepted"]) for row in structure_rows])
            if structure_rows else 0.0),
        "model_noop_mean_worst": float(np.mean([
            float(row["model_noop_lower_worst"]) for row in rows
        ])),
        "controller_mean_worst": float(np.mean([
            float(row["candidate_lower_worst"]) for row in rows
        ])),
        "accepted_mean_worst": float(
            np.mean([float(row["candidate_lower_worst"]) for row in accepted])
            if accepted else float("nan")),
        "qos_feasible_rate": float(np.mean([
            bool(row["qos_feasible"]) for row in rows
        ])),
        "all_accepted_target_no_harm": bool(all(
            bool(row["target_no_harm"]) for row in accepted
        )),
        "all_accepted_protocol_feasible": bool(all(
            bool(row["protocol_feasible"]) for row in accepted
        )),
        "mean_active_protocol_bits": float(
            np.mean([float(row["protocol_total_bits"]) for row in active])
            if active else 0.0),
        "max_active_protocol_latency_s": float(
            np.max([float(row["protocol_total_latency_s"]) for row in active])
            if active else 0.0),
        "mean_active_protocol_energy_j": float(
            np.mean([float(row["protocol_total_energy_j"]) for row in active])
            if active else 0.0),
        "max_power_balance_error_w": float(np.max([
            float(row["protocol_power_balance_error_w"]) for row in rows
        ])),
        "elapsed_seconds": float(perf_counter() - started),
        "rows": rows,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-limit", type=int, default=20)
    parser.add_argument("--qos-floor", type=float, default=0.60)
    parser.add_argument("--owners-per-target", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--structure-ranking-rounds", type=int, default=4)
    parser.add_argument("--price-bits", type=int, default=6)
    parser.add_argument("--feedback-bits", type=int, default=16)
    parser.add_argument("--snr-margin-db", type=float, default=3.0)
    parser.add_argument("--latency-margin-s", type=float, default=5.0e-4)
    parser.add_argument(
        "--allow-unreserved-communication-power",
        action="store_true",
    )
    args = parser.parse_args()
    result = audit(
        args.trace,
        args.config,
        seed_limit=max(1, int(args.seed_limit)),
        qos_floor=float(args.qos_floor),
        owners_per_target=max(1, int(args.owners_per_target)),
        rounds=max(1, int(args.rounds)),
        structure_ranking_rounds=max(
            1, min(int(args.structure_ranking_rounds), int(args.rounds))),
        price_bits=int(args.price_bits),
        feedback_bits=int(args.feedback_bits),
        snr_margin_db=float(args.snr_margin_db),
        latency_margin_s=float(args.latency_margin_s),
        require_pre_reserved_comm_power=not bool(
            args.allow_unreserved_communication_power),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "seed_count",
        "route_counts",
        "accepted_rate",
        "active_route_acceptance_rate",
        "structure_route_acceptance_rate",
        "model_noop_mean_worst",
        "controller_mean_worst",
        "qos_feasible_rate",
        "all_accepted_target_no_harm",
        "mean_active_protocol_bits",
        "max_active_protocol_latency_s",
    )}, indent=2))


if __name__ == "__main__":
    main()
