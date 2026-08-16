#!/usr/bin/env python
"""Frame-level diagnostic for a single blind seed under D1.9 lookahead.

Runs one episode with the analytical L3 stack (lookahead H configurable) and
prints per-frame worst P_D, the worst target's nearest-UAV distance, and where
the steady window (last 20 frames) sits relative to convergence.

Usage:
    python tools/trace_d19_seed.py --config ..._blind_lookahead40.yaml \
        --warm-start results/.../best_restored.pt --seed 615
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.params import load_config  # noqa: E402


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--warm-start", required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args(argv)

    config = load_config(args.config)
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from uav_isac.environment.env_wrapper import UAVISACEnv
    env = UAVISACEnv(config, seed=args.seed)
    env.reset(seed=args.seed)

    core = env.core
    core.cfg.marl.analytical_movement_lookahead_frames = int(
        getattr(core.cfg.marl, 'analytical_movement_lookahead_frames', 0))
    # warm-start the actor (weights unused by the analytical L3 stack, but the
    # env may consult the structure student / learned comm on some paths; load
    # for parity with the certified runs).
    if args.warm_start:
        ckpt = torch.load(args.warm_start, map_location=device)
        if "actor" in ckpt:
            ckpt = ckpt["actor"]
        # The actor is only used through the trainer; env-only trace relies on
        # the analytical hooks, so loading is best-effort here.
        print("warm-start checkpoint loaded (analytical stack drives L0-L3)")

    # Step through one episode with zero learned actions (analytical stack
    # drives power/structure/movement; actor actions unused for movement).
    worst_pd, worst_dist = [], []
    for fr in range(core.cfg.scenario.T):
        a = {str(k): {"delta_p": np.zeros(2), "role": 0}
             for k in range(core.K)}
        _, _, _, _, info = env.step(a)
        pd = np.asarray(info.get("P_D_q", np.zeros(core.Q)), dtype=np.float64)
        worst_pd.append(float(np.min(pd)))
        # distance from each target to nearest UAV (2D)
        uav_p = np.array([u.pos[:2].copy() for u in core.uavs])
        tgt_p = np.array([t.get_position_3d()[:2] for t in core.targets])
        d = np.linalg.norm(uav_p[:, None, :] - tgt_p[None, :, :], axis=2)
        worst_dist.append(float(np.min(d, axis=0).min()))
    env.close()

    worst_pd = np.asarray(worst_pd)
    worst_dist = np.asarray(worst_dist)
    # steady = temporal mean of worst P_D over last 20 frames
    steady = float(worst_pd[-20:].mean())
    print(f"seed {args.seed}: steady(last20)={steady:.3f}")
    for fr in (0, 30, 60, 90, 110, 120, 125, 130, 135, 140, 145, 149):
        print(f"  frame {fr:3d}: worst P_D={worst_pd[fr]:.3f}  "
              f"worst nearest dist={worst_dist[fr]:.1f} m")
    # convergence frame: first frame where worst P_D >= 0.6 sustained
    above = np.where(worst_pd >= 0.6)[0]
    conv = int(above[0]) if len(above) else -1
    print(f"first frame worst P_D>=0.6: {conv}  (episode len {core.cfg.scenario.T})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
