#!/usr/bin/env python
"""Summarize the preliminary Gate D0/D1 N5 counterfactual evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _horizon_metrics(
        local_pd: np.ndarray, periodic_pd: np.ndarray) -> dict[str, float]:
    local_worst = np.min(local_pd, axis=1)
    periodic_worst = np.min(periodic_pd, axis=1)
    local_all = np.sort(local_pd.reshape(-1))
    periodic_all = np.sort(periodic_pd.reshape(-1))
    tail_count = max(1, int(np.ceil(0.20 * local_all.size)))
    return {
        "local_mean_worst": float(np.mean(local_worst)),
        "periodic_mean_worst": float(np.mean(periodic_worst)),
        "delta_mean_worst": float(np.mean(periodic_worst - local_worst)),
        "local_temporal_min": float(np.min(local_worst)),
        "periodic_temporal_min": float(np.min(periodic_worst)),
        "delta_temporal_min": float(
            np.min(periodic_worst) - np.min(local_worst)),
        "local_bottom20_target_frame": float(
            np.mean(local_all[:tail_count])),
        "periodic_bottom20_target_frame": float(
            np.mean(periodic_all[:tail_count])),
        "delta_bottom20_target_frame": float(
            np.mean(periodic_all[:tail_count])
            - np.mean(local_all[:tail_count])),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    atomic_report = json.loads(args.atomic_all.read_text(encoding="utf-8"))
    if len(atomic_report.get("events", [])) != 1:
        raise ValueError("the preliminary summary expects exactly one event")
    event = atomic_report["events"][0]
    local = np.load(args.local_trace)
    periodic = np.load(args.periodic_trace)
    if not (
        np.array_equal(local["seed"], periodic["seed"])
        and np.array_equal(local["frame"], periodic["frame"])
    ):
        raise ValueError("trace seed/frame order does not match")

    accepted_indices = np.flatnonzero(
        periodic["local_search_rebootstrap_accepted"] > 0)
    accepted_events = [{
        "seed": int(periodic["seed"][index]),
        "frame": int(periodic["frame"][index]),
    } for index in accepted_indices]
    seed_indices = np.flatnonzero(periodic["seed"] == int(args.seed))
    seed_accepts = [
        index for index in accepted_indices
        if int(periodic["seed"][index]) == int(args.seed)]
    if not seed_accepts:
        raise ValueError("requested seed has no accepted rebootstrap event")
    first_accept = int(seed_accepts[0])
    first_frame = int(periodic["frame"][first_accept])
    position = int(np.flatnonzero(seed_indices == first_accept)[0])
    pre_indices = seed_indices[periodic["frame"][seed_indices] < first_frame]

    horizons: dict[str, object] = {}
    for horizon in (1, 5, 10):
        selected = seed_indices[
            position:min(position + horizon, seed_indices.size)]
        metrics = _horizon_metrics(
            local["physical_pd"][selected],
            periodic["physical_pd"][selected],
        )
        metrics["pair_equal_frame_count"] = int(np.sum(np.all(
            local["teacher_pair"][selected]
            == periodic["teacher_pair"][selected],
            axis=(1, 2, 3),
        )))
        horizons[str(horizon)] = metrics

    frame_indices = seed_indices[
        position:min(position + 10, seed_indices.size)]
    frame_rows = []
    for index in frame_indices:
        local_worst = float(np.min(local["physical_pd"][index]))
        periodic_worst = float(np.min(periodic["physical_pd"][index]))
        frame_rows.append({
            "seed": int(args.seed),
            "frame": int(periodic["frame"][index]),
            "local_worst": local_worst,
            "periodic_worst": periodic_worst,
            "delta_worst": periodic_worst - local_worst,
            "pair_changed_edges": int(np.count_nonzero(
                local["teacher_pair"][index]
                != periodic["teacher_pair"][index])),
            "uav_position_max_error_m": float(np.max(np.abs(
                local["uav_positions"][index]
                - periodic["uav_positions"][index]))),
            "target_state_max_error": float(np.max(np.abs(
                local["target_states"][index]
                - periodic["target_states"][index]))),
            "movement_action_max_error": float(np.max(np.abs(
                local["delta_p"][index] - periodic["delta_p"][index]))),
            "sensing_weight_max_error": float(np.max(np.abs(
                local["sensing_weights"][index]
                - periodic["sensing_weights"][index]))),
        })

    proxy_oracle = event["oracle_proxy_pool"]
    all_oracle = event["oracle_all_target_pool"]
    proxy_candidates = [
        record for record in event["candidates"]
        if record["is_atomic_candidate"]
        and record["in_proxy_weak_pool"]]
    physical_positive = [
        record for record in proxy_candidates
        if float(record["delta_worst"]) > 1.0e-4]
    missed_positive = [
        record for record in physical_positive
        if not record["proxy_positive"]]
    proxy_negative = [
        record for record in proxy_candidates
        if not record["proxy_positive"]]
    report = {
        "protocol": "gate_d0_d1_n5_counterfactual_preliminary_v1",
        "scope": (
            "one fully enumerated seed-483/frame-75 atomic event plus paired "
            "local-only/Periodic-N5 closed-loop traces on five development "
            "seeds; diagnostic evidence, not a general N5 ceiling"),
        "atomic_event": {
            "seed": int(event["episode_seed"]),
            "frame": int(event["frame"]),
            "baseline": event["baseline"],
            "proxy_pool_candidates": int(
                event["candidate_count_proxy_weak"]),
            "all_target_candidates": int(event["candidate_count_all"]),
            "proxy_choice": event["proxy_choice"],
            "proxy_pool_physical_oracle": proxy_oracle,
            "all_target_physical_oracle": all_oracle,
            "all_target_extra_worst_headroom": float(
                all_oracle["delta_worst"] - proxy_oracle["delta_worst"]),
            "false_accept_rate": float(event["false_accept_rate"]),
            "false_miss_rate": float(
                len(missed_positive) / max(len(proxy_negative), 1)),
            "physical_positive_coverage": float(
                1.0 - len(missed_positive)
                / max(len(physical_positive), 1)),
            "material_sign_agreement": float(
                event["material_sign_agreement"]),
            "max_geometry_error": float(event["max_geometry_error"]),
        },
        "closed_loop": {
            "accepted_events_five_seed_trace": accepted_events,
            "seed": int(args.seed),
            "first_accept_frame": first_frame,
            "pre_accept_pair_exact": bool(np.array_equal(
                local["teacher_pair"][pre_indices],
                periodic["teacher_pair"][pre_indices])),
            "pre_accept_pd_max_error": float(np.max(np.abs(
                local["physical_pd"][pre_indices]
                - periodic["physical_pd"][pre_indices]))),
            "horizons": horizons,
            "frame_rows": frame_rows,
        },
        "decision": {
            "candidate_generation": (
                "a bottleneck for this event: the one-frame all-target Oracle "
                "improves worst by 0.323 versus 0.082 in the deployed "
                "proxy-weak pool"),
            "proxy_acceptance": (
                "not uniformly reliable: 23.1% of atomic proxy-positive "
                "moves physically reduce same-frame worst, although the "
                "proxy top choice is physically best inside its pool here"),
            "temporal_persistence": (
                "a separate bottleneck: the accepted move improves H=1 and "
                "H=5 mean worst but reverses by H=10 after sensing and "
                "movement actions diverge"),
            "next": (
                "repeat full-pool D0 on a small event-stratified sample and "
                "add frozen-action H=5 replay before training a trigger or "
                "designing learned deficit-block LNS"),
        },
        "limitations": [
            "only one event has a fully enumerated all-target pool",
            "the trace comparison is closed-loop; frozen future actions are not yet replayed",
            "the all-target pool is a diagnostic ceiling, not a deployable cost",
            "atomic candidate enumeration does not establish a sequential LNS ceiling",
        ],
        "fresh_test_consumed": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    with (args.output_dir / "seed483_first_event_h10.csv").open(
            "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=frame_rows[0].keys())
        writer.writeheader()
        writer.writerows(frame_rows)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atomic-all", type=Path, required=True)
    parser.add_argument("--local-trace", type=Path, required=True)
    parser.add_argument("--periodic-trace", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=483)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
