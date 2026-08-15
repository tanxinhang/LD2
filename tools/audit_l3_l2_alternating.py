#!/usr/bin/env python
"""Alternating L2 (structure) <-> L3 (geometry) joint optimization.

Block coordinate descent on the joint problem  min_{x, S} gamma*(x, S):

    repeat
      L2: structure repair  S_{t+1} = argmin_S gamma*(x_t, S)   (priced greedy)
      L3: geometry descent  x_{t+1} = argmin_x gamma*(x, S_{t+1}) (few steps)
    until converged / max iterations.

The full (K,K,Q) coefficient tensor is rescaled exactly under the Friis path
loss  a_ijq ~ 1/(R_tx_iq^2 R_rx_jq^2)  after every geometry move, so the next
structure step sees the moved geometry (unlike the one-shot joint, which fixed
the structure at the pre-move geometry).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.params import load_config  # noqa: E402
from uav_isac.physical.detection import (  # noqa: E402
    minimum_deflection_for_detection_probability)
from uav_isac.evaluation.physical_oracle_audit import (  # noqa: E402
    per_watt_deflection_tensor_from_observables)
from uav_isac.coordination.priced_structure import priced_structure_repair  # noqa: E402

import importlib.util  # reuse long-horizon helpers
_spec = importlib.util.spec_from_file_location(
    "lh", str(ROOT / "tools" / "audit_l3_long_horizon.py"))
_lh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lh)

XI = (0.60, 0.70, 0.80, 3)


def _full_coeff(data, row, cfg):
    return per_watt_deflection_tensor_from_observables(
        np.asarray(data["privileged_alpha"][row], dtype=np.float64),
        np.asarray(data["privileged_g_dd"][row], dtype=np.float64),
        np.asarray(data["privileged_chi_rep"][row], dtype=np.float64),
        T_sym=float(cfg.otfs.T_sym), M=int(cfg.otfs.M), N=int(cfg.otfs.N),
        kT=float(cfg.channel.kT), bandwidth_hz=float(cfg.otfs.B),
        noise_figure_db=float(cfg.channel.NF),
        g_tx_dbi=float(cfg.otfs.g_tx_dBi), g_rx_dbi=float(cfg.otfs.g_rx_dBi),
        n_cpi=int(cfg.otfs.n_cpi), g_min=float(cfg.detection.g_min),
        use_swerling=bool(cfg.channel.use_swerling))


def _budget(data, row):
    rate = np.asarray(data["outgoing_rate"][row], dtype=np.int64)
    mask = np.asarray(data["outgoing_token_mask"][row], dtype=bool)
    active = (rate > 0) & np.any(mask, axis=1)
    frac = np.clip(np.asarray(data["comm_fraction"][row], dtype=np.float64), 0, 1)
    return 1.0 - np.where(active, frac, 0.0)


def friis_rescale_tensor(coeff, uav, tgt, new_uav):
    """Rescale the full (K,K,Q) tensor under 1/(R_tx^2 R_rx^2)."""
    r = np.linalg.norm(uav[:, None, :] - tgt[None, :, :], axis=2)      # (K,Q)
    rn = np.linalg.norm(new_uav[:, None, :] - tgt[None, :, :], axis=2)
    c = coeff * (r[:, None, :] ** 2) * (r[None, :, :] ** 2)
    return c / (rn[:, None, :] ** 2 * rn[None, :, :] ** 2)


def _short_l3(gain, owner, uav, tgt, budget, d_min, p_fa, horizon, step_m, area):
    """A bounded geometry descent; returns (gain, uav) at the moved geometry."""
    def cost(g):
        o = _lh._solve(g, budget, p_fa)
        return float(o[0]) if o is not None else 1e6 + float(
            np.max(_lh._ceiling_deficit(g, budget, d_min)[0]))
    for _ in range(horizon):
        out = _lh._solve(gain, budget, p_fa)
        if out is not None and out[0] <= 1.0:
            break
        delta = np.zeros_like(uav)
        if out is None:
            for k in range(gain.shape[0]):
                g_k = _lh._deficit_gradient(k, owner, uav, tgt, gain, budget, d_min)
                n = np.linalg.norm(g_k)
                if n > 1e-12:
                    delta[k] = -step_m * g_k / n
        else:
            _, power, prices = out
            support = {q: {i: (power[i, q], gain[i, q]) for i in range(gain.shape[0])}
                       for q in range(gain.shape[1])}
            for k in range(gain.shape[0]):
                g_k = _lh.local_capability_gradient_k(
                    k, owner, prices, power[k], gain[k], uav[k], tgt, support)
                n = np.linalg.norm(g_k)
                if n > 1e-12:
                    delta[k] = -step_m * g_k / n
        new_uav = np.clip(uav + delta, 0.0, area)
        new_gain = _lh._friis_rescale(gain, owner, uav, tgt, new_uav)
        if cost(new_gain) >= cost(gain) - 1e-9:
            break
        gain, uav = new_gain, new_uav
    return gain, uav


def alternating(coeff, budget, uav, tgt, p_fa, d_min, area, *, max_iters, l3_horizon, step_m):
    coeff_orig = coeff.copy()
    uav_orig = uav.copy()
    pos = uav.copy()
    coeff_cur = coeff.copy()
    best_gamma = float("inf")
    for _ in range(max_iters):
        gain, owner = priced_structure_repair(coeff_cur, budget, target_pair_limit=3)
        gain, pos = _short_l3(gain, owner, pos, tgt, budget, d_min, p_fa,
                              l3_horizon, step_m, area)
        coeff_cur = friis_rescale_tensor(coeff_orig, uav_orig, tgt, pos)
        out = _lh._solve(gain, budget, p_fa)
        gamma = out[0] if out is not None else float("inf")
        if gamma < best_gamma:
            best_gamma = gamma
        if gamma <= 1.0:
            break
    return best_gamma


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--seed-limit", type=int, default=20)
    ap.add_argument("--max-frames", type=int, default=30)
    ap.add_argument("--max-iters", type=int, default=3)
    ap.add_argument("--l3-horizon", type=int, default=25)
    ap.add_argument("--step-m", type=float, default=2.5)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    with np.load(args.trace, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    cfg = load_config(str(args.config))
    p_fa = float(np.asarray(data["p_fa"]).reshape(-1)[0])
    d_min = float(minimum_deflection_for_detection_probability(
        np.asarray([XI[0]]), p_fa)[0])
    area = tuple(float(v) for v in cfg.scenario.region_size)
    seeds = np.asarray(data["seed"], dtype=np.int64)
    frames = np.asarray(data["frame"], dtype=np.int64)
    order = list(dict.fromkeys(int(s) for s in seeds.reshape(-1)))[:args.seed_limit]

    hard = []
    for seed in order:
        rows = np.flatnonzero(seeds == seed)
        rows = rows[np.argsort(frames[rows])]
        for row in rows:
            g_t, o_t, uav, tgt, b = _lh._frame(data, int(row), cfg)
            if _lh._solve(g_t, b, p_fa) is None:
                hard.append((int(row), g_t, o_t, uav, tgt, b,
                             _full_coeff(data, int(row), cfg)))
    rng = np.random.default_rng(0)
    if len(hard) > args.max_frames:
        hard = [hard[i] for i in rng.choice(len(hard), args.max_frames, replace=False)]

    reached = 0
    gammas = []
    for row, g_t, o_t, uav, tgt, b, coeff in hard:
        # baseline L3 alone (teacher structure, full horizon)
        g_l3 = _short_l3(g_t.copy(), o_t, uav.copy(), tgt, b, d_min, p_fa,
                         args.l3_horizon * args.max_iters, args.step_m, area)[0]
        o_l3 = _lh._solve(g_l3, b, p_fa)
        g_l3 = o_l3[0] if o_l3 is not None else float("inf")
        g_alt = alternating(coeff, b, uav, tgt, p_fa, d_min, area,
                            max_iters=args.max_iters,
                            l3_horizon=args.l3_horizon, step_m=args.step_m)
        gammas.append((row, g_l3, g_alt))
        if g_alt <= 1.0:
            reached += 1

    n = len(gammas)
    l3_reach = sum(1 for _, a, _ in gammas if a <= 1.0)
    finite = [g for _, _, g in gammas if np.isfinite(g)]
    out = {
        "schema_version": 1,
        "hard_frames": n,
        "max_iters": args.max_iters, "l3_horizon": args.l3_horizon,
        "L3_alone_reach_rate": l3_reach / max(1, n),
        "alternating_joint_reach_rate": reached / max(1, n),
        "mean_alt_gamma_finite": float(np.mean(finite)) if finite else float("nan"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
