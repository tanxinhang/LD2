"""Full system performance benchmark.

Runs multiple configs × seeds and reports:
  - QoS metrics (steady, weak3, worst, qos_success)
  - Timing (per-frame wall time, power solve time)
  - Communication overhead (bits per frame)
  - Movement coverage
  - Entropic vs baseline comparison
"""
from __future__ import annotations

import sys
import os
import time
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from config.params import load_config
from tools.run_strict_distributed_pilot import _episode

CONFIGS = [
    ("strict_pilot", "config/exp_strict_distributed_no_truth_pilot.yaml"),
    ("analytical_k6q6", "config/exp_800_k6q6_analytical_l0l1_movement_lex_candidates_v2.yaml"),
]
SEEDS = [7, 11, 23, 42, 99]
T_FRAMES = 30
TAIL_WINDOW = 10


def run_config(name: str, path: str, entropic: bool = False) -> list[dict]:
    results = []
    for seed in SEEDS:
        cfg = load_config(path)
        cfg.scenario.T = T_FRAMES
        if entropic:
            cfg.marl.entropic_dual_price_enabled = True
            cfg.marl.entropic_dual_tau = 0.01
        t0 = time.time()
        ep = _episode(cfg, seed=seed, tail_window=TAIL_WINDOW)
        wall = time.time() - t0
        r = {
            "seed": seed,
            "wall_s": wall,
            "per_frame_ms": wall / T_FRAMES * 1000,
            "steady": ep["steady"],
            "weak3": ep["weak3"],
            "worst": ep["worst"],
            "qos": float(ep["qos_success"]),
            "bits_per_frame": ep["bits_per_frame"],
            "movement_coverage": ep["movement_target_coverage"],
            "power_solve_ms": ep.get("power_solve_time_ms", 0.0),
            "controller_ms": ep.get("controller_compute_ms", 0.0),
            "closed_loop_ms": ep.get("closed_loop_critical_path_ms", 0.0),
        }
        results.append(r)
    return results


def print_summary(name: str, results: list[dict]) -> None:
    def m(k): return statistics.mean(r[k] for r in results)
    def s(k): return statistics.stdev(r[k] for r in results) if len(results) > 1 else 0
    print(f"\n  {name}")
    print(f"    QoS:  steady={m('steady'):.4f}±{s('steady'):.4f}  "
          f"weak3={m('weak3'):.4f}±{s('weak3'):.4f}  "
          f"worst={m('worst'):.4f}±{s('worst'):.4f}  "
          f"success={m('qos'):.2f}")
    print(f"    Time: {m('per_frame_ms'):.1f}±{s('per_frame_ms'):.1f} ms/frame  "
          f"total={m('wall_s'):.1f}±{s('wall_s'):.1f} s")
    print(f"    Comm: {m('bits_per_frame'):.0f} bits/frame")
    print(f"    Move: coverage={m('movement_coverage'):.4f}")
    print(f"    Pow:  solve={m('power_solve_ms'):.2f} ms  "
          f"ctrl={m('controller_ms'):.2f} ms  "
          f"closed={m('closed_loop_ms'):.2f} ms")


def main() -> None:
    print("=" * 76)
    print("Full System Performance Benchmark")
    print(f"  Configs: {len(CONFIGS)}  Seeds: {SEEDS}  T={T_FRAMES}  tail={TAIL_WINDOW}")
    print("=" * 76)

    all_results = {}
    for name, path in CONFIGS:
        print(f"\n[{name}] Baseline (entropic OFF)")
        baseline = run_config(name, path, entropic=False)
        print_summary("baseline", baseline)
        all_results[f"{name}_base"] = baseline

        print(f"\n[{name}] Entropic ON (tau=0.01)")
        entropic = run_config(name, path, entropic=True)
        print_summary("entropic", entropic)
        all_results[f"{name}_ent"] = entropic

        # Delta
        b = baseline
        e = entropic
        d_steady = np.mean([r["steady"] for r in e]) - np.mean([r["steady"] for r in b])
        d_weak3 = np.mean([r["weak3"] for r in e]) - np.mean([r["weak3"] for r in b])
        d_worst = np.mean([r["worst"] for r in e]) - np.mean([r["worst"] for r in b])
        d_time = np.mean([r["per_frame_ms"] for r in e]) - np.mean([r["per_frame_ms"] for r in b])
        print(f"\n  DELTA  steady={d_steady:+.4f}  weak3={d_weak3:+.4f}  "
              f"worst={d_worst:+.4f}  time={d_time:+.2f}ms")

    print("\n" + "=" * 76)
    print("Summary Table")
    print("=" * 76)
    print(f"  {'Config':<25} {'Mode':<10} {'steady':>7} {'weak3':>7} {'worst':>7} "
          f"{'qos':>5} {'ms/frame':>9}")
    for key, results in all_results.items():
        parts = key.rsplit("_", 1)
        cfg_name, mode = parts[0], parts[1]
        m_steady = np.mean([r["steady"] for r in results])
        m_weak3 = np.mean([r["weak3"] for r in results])
        m_worst = np.mean([r["worst"] for r in results])
        m_qos = np.mean([r["qos"] for r in results])
        m_ms = np.mean([r["per_frame_ms"] for r in results])
        print(f"  {cfg_name:<25} {mode:<10} {m_steady:>7.4f} {m_weak3:>7.4f} "
              f"{m_worst:>7.4f} {m_qos:>5.2f} {m_ms:>8.1f}")
    print("\nDone.")


if __name__ == "__main__":
    main()