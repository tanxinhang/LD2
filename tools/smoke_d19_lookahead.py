#!/usr/bin/env python
"""D1.9 bottleneck-lookahead smoke eval on selected blind seeds.

Compares H=0 (baseline) vs H=lookahead on a fixed seed list by running the
same final-eval path as run_mappo.py --episodes 0 (no training), writing a
paired_eval.csv per variant, then prints per-seed steady/worst/QoS.

Usage:
    python tools/smoke_d19_lookahead.py \
        --config config/exp_800_k8q8_analytical_l0l1_movement_lex_candidates_blind.yaml \
        --warm-start results/architecture_v2_scale_k8q8_analytical_power_l0l1_movement/best_restored.pt \
        --seeds 446,598,615,696,645,98 \
        --out results/_d1_9_smoke
"""

from __future__ import annotations

import argparse
import ast
import csv
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.params import load_config  # noqa: E402
from uav_isac.agents.trainer import MAPPTrainer  # noqa: E402
from uav_isac.utils.checkpoint_loading import safe_torch_load  # noqa: E402


def _build_trainer(config_path: str, warm_start: str, lookahead: int):
    config = load_config(config_path)
    config.marl.analytical_movement_lookahead_frames = int(lookahead)
    trainer = MAPPTrainer(config)
    if warm_start:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = safe_torch_load(
            warm_start,
            map_location=device,
            description="D1.9 smoke warm-start checkpoint",
            state_dict_keys=("actor",),
            optional_state_dict_keys=("critic",),
        )
        missing, unexpected = trainer.agents[0].actor.load_state_dict(
            ckpt["actor"], strict=False)
        trainer.agents[0].actor.to(device)
        if "critic" in ckpt:
            trainer.agents[0].critic.load_state_dict(ckpt["critic"], strict=False)
    return trainer


def _evaluate_seeds(trainer, seeds: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ev = trainer._evaluate(
        n_episodes=len(seeds),
        eval_seeds=list(seeds),
        target_choice_audit_stride=0,
        sensing_choice_audit_stride=0,
        sensing_residual_blend=0.25,
        sensing_audit_horizon=0,
        sensing_oracle_control=False,
        joint_sensing_pair_audit_stride=0,
        physical_oracle_stride=0,
        evidence_trace_output="",
        structure_teacher_trace_output="",
        n5_counterfactual_output="",
        n5_counterfactual_max_events=0,
        n5_counterfactual_target_mode="both",
        n5_counterfactual_seed_filter=None,
        n5_counterfactual_max_candidates=0,
        n5_counterfactual_require_proxy_positive=False,
    )
    steady = np.asarray(ast.literal_eval(ev["eval_episode_steady_P_D"]))
    weak3 = np.asarray(ast.literal_eval(ev["eval_episode_weak3_P_D"]))
    worst = np.asarray(ast.literal_eval(ev["eval_episode_worst_P_D"]))
    return steady, weak3, worst


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--warm-start", default=None)
    ap.add_argument("--seeds", required=True, help="comma-separated seed list")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lookahead", type=int, default=0,
                    help="H frames; 0 = baseline (D1.1-B single-step scoring)")
    args = ap.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    os.makedirs(args.out, exist_ok=True)
    trainer = _build_trainer(args.config, args.warm_start, args.lookahead)
    steady, weak3, worst = _evaluate_seeds(trainer, seeds)
    qos = (steady >= 0.80) & (weak3 >= 0.70) & (worst >= 0.60)

    csv_path = os.path.join(args.out, f"paired_eval_lookahead{args.lookahead}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "steady", "weak3", "worst", "qos"])
        for s, st, wk, wo, q in zip(seeds, steady, weak3, worst, qos):
            w.writerow([s, f"{st:.6f}", f"{wk:.6f}", f"{wo:.6f}", int(q)])
    print(f"lookahead={args.lookahead}: QoS {qos.mean():.3f} "
          f"({qos.sum()}/{len(seeds)})  steady mean {steady.mean():.3f} "
          f"worst mean {worst.mean():.3f}")
    for s, st, wo in zip(seeds, steady, worst):
        print(f"  seed {s}: steady={st:.3f} worst={wo:.3f}")
    print(f"csv -> {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
