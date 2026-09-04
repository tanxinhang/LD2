"""A/B benchmark: entropic dual price (ON) vs LP vertex (OFF).

Runs N seeds of the strict distributed pilot with both settings and
compares QoS metrics (steady, weak3, worst, qos_success).
"""
from __future__ import annotations

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from config.params import load_config
from tools.run_strict_distributed_pilot import _episode

PILOT = "config/exp_800_k6q6_analytical_l0l1_movement_lex_candidates_v2.yaml"
SEEDS = [7, 11, 23, 42, 99]
TAU_VALUES = [0.0, 1e-3, 1e-2, 1e-1]
T_FRAMES = 100
TAIL_WINDOW = 20


def run_seed(seed: int, entropic_on: bool, tau: float = 0.0) -> dict[str, float]:
    cfg = load_config(PILOT)
    cfg.scenario.T = T_FRAMES
    if entropic_on:
        cfg.marl.entropic_dual_price_enabled = True
        cfg.marl.entropic_dual_tau = tau
    else:
        cfg.marl.entropic_dual_price_enabled = False
    ep = _episode(cfg, seed=seed, tail_window=TAIL_WINDOW)
    return {
        "steady": ep["steady"],
        "weak3": ep["weak3"],
        "worst": ep["worst"],
        "qos_success": float(ep["qos_success"]),
        "bits_per_frame": ep["bits_per_frame"],
        "movement_target_coverage": ep["movement_target_coverage"],
    }


def summarize(results: list[dict[str, float]]) -> dict[str, float]:
    keys = results[0].keys()
    return {k: float(np.mean([r[k] for r in results])) for k in keys}


def main() -> None:
    print("=" * 72)
    print("A/B Benchmark: Entropic Dual Price vs LP Vertex")
    print(f"Config: {PILOT}  Seeds: {SEEDS}  T={T_FRAMES}  tail={TAIL_WINDOW}")
    print("=" * 72)

    # Baseline: LP vertex (current production)
    print("\n[B] Baseline: LP vertex (entropic OFF)")
    baseline_results = []
    for s in SEEDS:
        t0 = time.time()
        r = run_seed(s, entropic_on=False)
        r["wall_s"] = time.time() - t0
        baseline_results.append(r)
        print(f"  seed={s:3d}  steady={r['steady']:.4f}  weak3={r['weak3']:.4f}"
              f"  worst={r['worst']:.4f}  qos={r['qos_success']:.0f}"
              f"  {r['wall_s']:.1f}s")
    b = summarize(baseline_results)
    print(f"  MEAN   steady={b['steady']:.4f}  weak3={b['weak3']:.4f}"
          f"  worst={b['worst']:.4f}  qos={b['qos_success']:.2f}")

    # Treatment: Entropic dual price with various tau
    for tau in TAU_VALUES:
        label = f"tau={tau}" if tau > 0 else "tau=0(auto)"
        print(f"\n[E] Entropic ON  {label}")
        ent_results = []
        for s in SEEDS:
            t0 = time.time()
            r = run_seed(s, entropic_on=True, tau=tau)
            r["wall_s"] = time.time() - t0
            ent_results.append(r)
            print(f"  seed={s:3d}  steady={r['steady']:.4f}  weak3={r['weak3']:.4f}"
                  f"  worst={r['worst']:.4f}  qos={r['qos_success']:.0f}"
                  f"  {r['wall_s']:.1f}s")
        e = summarize(ent_results)
        print(f"  MEAN   steady={e['steady']:.4f}  weak3={e['weak3']:.4f}"
              f"  worst={e['worst']:.4f}  qos={e['qos_success']:.2f}")

        # Delta
        d_steady = e["steady"] - b["steady"]
        d_weak3 = e["weak3"] - b["weak3"]
        d_worst = e["worst"] - b["worst"]
        print(f"  DELTA  steady={d_steady:+.4f}  weak3={d_weak3:+.4f}"
              f"  worst={d_worst:+.4f}")

    print("\n" + "=" * 72)
    print("Done.")


if __name__ == "__main__":
    main()