#!/usr/bin/env python
"""Evaluate the full-information ISAC feasibility oracle on frozen seed banks."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.params import load_config
from uav_isac.agents.trainer import (
    compute_robust_checkpoint_statistics,
    load_stratified_seed_split,
)
from uav_isac.environment.env_wrapper import UAVISACEnv
from uav_isac.physical.feasibility_oracle import (
    reachable_hungarian_geometry,
    solve_joint_pair_power_oracle,
    unit_deflection_tensor,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', default='config/exp_800_q4_u2u_joint_isac.yaml')
    parser.add_argument(
        '--seed-bank', default='config/stratified_seeds_800_q4.json')
    parser.add_argument('--split', default='selection')
    parser.add_argument('--max-seeds', type=int, default=0)
    parser.add_argument(
        '--comm-reserves', type=float, nargs='+', default=[0.0, 0.25, 0.5])
    parser.add_argument('--random-starts', type=int, default=2)
    parser.add_argument('--alternating-iterations', type=int, default=6)
    parser.add_argument(
        '--output-dir', default='results/isac_physical_feasibility_oracle')
    args = parser.parse_args()

    cfg = load_config(args.config)
    seeds = load_stratified_seed_split(args.seed_bank, args.split)
    if args.max_seeds > 0:
        seeds = seeds[:args.max_seeds]
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    env = UAVISACEnv(config=cfg, seed=0)
    rows = []
    try:
        for episode_index, seed in enumerate(seeds):
            _, info = env.reset(seed=int(seed))
            initial_uav = np.asarray(info['uav_positions'], dtype=np.float64)
            targets = np.asarray(info['target_positions'], dtype=np.float64)
            target_velocities = np.asarray([
                [target.state[2], target.state[3], 0.0]
                for target in env.core.targets
            ], dtype=np.float64)
            final_uav, direction, assignment = reachable_hungarian_geometry(
                initial_uav, targets,
                travel_distance_m=(
                    cfg.uav.v_max * cfg.scenario.T * cfg.scenario.dt))
            displacement = final_uav - initial_uav
            reached = np.linalg.norm(displacement[:, :2], axis=1) + 1e-9
            direct_velocity = direction * cfg.uav.v_max
            direct_velocity[
                reached < cfg.uav.v_max * cfg.scenario.T * cfg.scenario.dt
            ] = 0.0
            unit_power = np.ones(
                (cfg.scenario.K, cfg.scenario.Q), dtype=np.float64)
            geometries = (
                ('initial', initial_uav,
                 np.asarray([uav.vel for uav in env.core.uavs], dtype=np.float64)),
                ('reachable_final', final_uav, direct_velocity),
            )
            geometry_distances = {}
            for geometry_name, geometry_uav, geometry_velocity in geometries:
                entries = env.core.deflection_computer.compute(
                    geometry_uav,
                    geometry_velocity,
                    targets,
                    target_velocities,
                    np.zeros(cfg.scenario.K, dtype=np.int32),
                    env.core.fc_position,
                    role_agnostic=True,
                    sensing_power_w=unit_power,
                )
                coefficient = unit_deflection_tensor(
                    entries, cfg.scenario.K, cfg.scenario.Q)
                nearest = np.min(np.linalg.norm(
                    geometry_uav[:, None, :2] - targets[None, :, :2], axis=-1),
                    axis=0)
                geometry_distances[geometry_name] = float(np.max(nearest))
                for reserve in args.comm_reserves:
                    for full_duplex in (False, True):
                        solution = solve_joint_pair_power_oracle(
                            coefficient,
                            P_FA=cfg.detection.P_FA,
                            total_power_w=cfg.uav.P_isac_total,
                            communication_reserve_w=float(reserve),
                            target_pair_limit=cfg.detection.K_q_max,
                            reports_per_receiver=(
                                cfg.p0_solver.capacity_per_rx
                                // cfg.detection.B_q),
                            full_duplex=full_duplex,
                            alternating_iterations=args.alternating_iterations,
                            random_starts=args.random_starts,
                            seed=int(seed),
                        )
                        rows.append({
                            'seed': int(seed),
                            'geometry': geometry_name,
                            'mode': solution.mode,
                            'communication_reserve_w': float(reserve),
                            'steady': solution.steady,
                            'weak3': solution.weak3,
                            'worst': solution.worst,
                            'per_target_pd': json.dumps(
                                solution.P_D_q.tolist()),
                            'per_target_D': json.dumps(solution.D_q.tolist()),
                            'sensing_power_w': json.dumps(
                                solution.sensing_power_w.tolist()),
                            'tx_indices': json.dumps(solution.tx_indices),
                            'rx_indices': json.dumps(solution.rx_indices),
                            'selected_set': json.dumps(solution.selected_set),
                            'hungarian_assignment': json.dumps(
                                assignment.tolist()),
                            'mean_nearest_distance_m': float(np.mean(nearest)),
                            'worst_nearest_distance_m': float(np.max(nearest)),
                        })
            print(
                f'[{episode_index + 1}/{len(seeds)}] seed={seed} '
                f'initial/final worst_distance='
                f'{geometry_distances["initial"]:.1f}/'
                f'{geometry_distances["reachable_final"]:.1f}m',
                flush=True)
    finally:
        env.close()

    csv_path = output / 'per_seed.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summaries = []
    qos_targets = np.asarray([0.80, 0.70, 0.60], dtype=np.float64)
    for geometry in ('initial', 'reachable_final'):
        for reserve in args.comm_reserves:
            for mode in ('single_role', 'full_duplex'):
                group = [
                    row for row in rows
                    if row['geometry'] == geometry
                    and row['mode'] == mode
                    and np.isclose(row['communication_reserve_w'], reserve)
                ]
                steady = np.asarray([row['steady'] for row in group])
                weak3 = np.asarray([row['weak3'] for row in group])
                worst = np.asarray([row['worst'] for row in group])
                robust = compute_robust_checkpoint_statistics(
                    steady, weak3, worst, qos_targets)
                summaries.append({
                    'geometry': geometry,
                    'mode': mode,
                    'communication_reserve_w': float(reserve),
                    'num_seeds': len(group),
                    'steady': float(np.mean(steady)),
                    'weak3': float(np.mean(weak3)),
                    'worst': float(np.mean(worst)),
                    **robust,
                })
    payload = {
        'config': args.config,
        'seed_bank': args.seed_bank,
        'split': args.split,
        'note': (
            'Full-information terminal-geometry benchmark; optimization is '
            'feasible but alternating MILP/LP is not a global-optimality proof.'),
        'summaries': summaries,
    }
    summary_path = output / 'summary.json'
    summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n',
        encoding='utf-8')
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == '__main__':
    main()
