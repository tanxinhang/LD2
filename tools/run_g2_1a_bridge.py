#!/usr/bin/env python
"""Reproducible post-G2 paired bridge over the already viewed seed banks.

G2-1A is deliberately not blind and performs no tuning.  It reuses each
historical system's exact episode seeds, frozen checkpoint, and controller,
while the current physical model supplies L_eff=1 and the sensing-PA cap.
Use ``--limit`` only for smoke/falsification; formal G2-1A requires 100.
"""

from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.params import load_config
from tools.audit_detector_normalization import run_audit


@dataclass(frozen=True)
class BridgeSystem:
    config: str
    checkpoint: str
    historical_csv: str
    output: str
    student_checkpoint: str | None = None


SYSTEMS = {
    "4x4": BridgeSystem(
        "config/exp_800_q4_architecture_v2_maxmin_local_fusion_fullgraph_hold5_gate.yaml",
        "results/architecture_v2_maxmin_local_fusion_fullgraph_hold5_gate100/best_restored.pt",
        "results/architecture_v2_structure_student_u2u_resolve_bw50k_adaptive_b4b8_failclosed_gate100/paired_eval.csv",
        "results/_g2_1a_4x4_paired100",
    ),
    "8x8_v3c0": BridgeSystem(
        "config/exp_800_k8q8_analytical_l0l1_movement_lex_candidates_blind_lookahead40_indep.yaml",
        "results/architecture_v2_scale_k8q8_analytical_power_l0l1_movement/best_restored.pt",
        "results/_gate0_8x8_v3c0_indep100/paired_eval.csv",
        "results/_g2_1a_8x8_v3c0_paired100",
    ),
    "6x6_v3c0": BridgeSystem(
        "config/exp_800_k6q6_analytical_l0l1_movement_lex_candidates_v2_blind_indep.yaml",
        "results/architecture_v2_maxmin_local_fusion_fullgraph_hold5_gate100/best_restored.pt",
        "results/_gate0_6x6_v3c0_indep100/paired_eval.csv",
        "results/_g2_1a_6x6_v3c0_paired100",
    ),
    "6x6_ce": BridgeSystem(
        "config/exp_800_k6q6_analytical_l0l1_movement_lex_candidates_v2_blind_indep.yaml",
        "results/architecture_v2_maxmin_local_fusion_fullgraph_hold5_gate100/best_restored.pt",
        "results/_gate1_6x6_multiscale_ce_blind100/paired_eval.csv",
        "results/_g2_1a_6x6_multiscale_ce_paired100",
        "results/_gate1_multiscale_ce/frozen_structure_student_multiscale_ce.pt",
    ),
}


def historical_seeds(path: Path) -> list[int]:
    with path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle), None)
    if row is None or "eval_episode_seeds" not in row:
        raise ValueError(f"missing episode seeds in {path}")
    seeds = [int(seed) for seed in ast.literal_eval(row["eval_episode_seeds"])]
    if len(seeds) != 100 or len(set(seeds)) != 100:
        raise ValueError(f"G2-1A requires 100 unique historical seeds in {path}")
    return seeds


def preflight(name: str, system: BridgeSystem) -> list[int]:
    paths = [system.config, system.checkpoint, system.historical_csv]
    if system.student_checkpoint:
        paths.append(system.student_checkpoint)
    missing = [path for path in paths if not (ROOT / path).exists()]
    if missing:
        raise FileNotFoundError(f"{name}: missing {missing}")
    cfg = load_config(str(ROOT / system.config))
    if int(cfg.otfs.n_cpi) != 1:
        raise ValueError(f"{name}: G2-1A requires n_cpi=1")
    if abs(float(cfg.uav.P_sense_max) - 0.0251) > 1.0e-12:
        raise ValueError(f"{name}: unexpected sensing PA cap")
    if float(cfg.uav.P_isac_total) != 1.0:
        raise ValueError(f"{name}: unexpected joint RF cap")
    if system.student_checkpoint and "task_regret" in system.student_checkpoint:
        raise ValueError(f"{name}: task-regret checkpoint forbidden before G2-1B")
    return historical_seeds(ROOT / system.historical_csv)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--system", choices=["all", *SYSTEMS], default="all")
    parser.add_argument(
        "--limit", type=int, default=0,
        help="first N historical seeds for smoke; 0 means formal 100")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-workers", type=int, default=1)
    args = parser.parse_args()
    if args.limit < 0 or args.limit > 100:
        raise ValueError("--limit must be in [0,100]")
    if args.max_workers != 1:
        raise ValueError("G2-1A freezes max-workers=1 on Windows/MKL/GPU")

    audit = run_audit(samples=500_000, seed=20260820)
    if not audit["certification"]["ready_for_g2_1"]:
        raise RuntimeError(f"physical preflight failed: {audit['certification']}")

    selected = SYSTEMS if args.system == "all" else {
        args.system: SYSTEMS[args.system]}
    for name, system in selected.items():
        seeds = preflight(name, system)
        if args.limit:
            seeds = seeds[:args.limit]
        formal = len(seeds) == 100
        print(
            f"{name}: seeds={len(seeds)} formal_g2_1a={formal} "
            f"out={system.output}")
        if not args.execute:
            continue
        command = [
            sys.executable,
            str(ROOT / "tools/crash_isolated_seed_eval.py"),
            "--config", str(ROOT / system.config),
            "--warm-start", str(ROOT / system.checkpoint),
            "--final-eval-seeds", ",".join(map(str, seeds)),
            "--out-dir", str(ROOT / system.output),
            "--episodes", "0",
            "--max-workers", "1",
        ]
        if args.resume:
            command.append("--resume")
        if system.student_checkpoint:
            command += [
                "--structure-student-checkpoint",
                str(ROOT / system.student_checkpoint),
                "--structure-student-channel",
                "--structure-student-bits-per-dim", "8",
                "--structure-student-allow-scale-migration",
            ]
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode != 0:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
