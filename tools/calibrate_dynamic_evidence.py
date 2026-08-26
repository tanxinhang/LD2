"""Dynamic-stack per-scale evidence calibration (audit priority #7).

Reproduces the deployed detector operating point on a `--evidence-trace-output`
trace produced by the current trainer (which stores the per-frame scheduled
fusion owner).  The calibration statistic replicates `estimate_quantized_
evidence_detection` exactly: owner-local evidence unquantized, delivered peer
LLRs quantized at ``llr_bits`` with ``clip_max``, and the threshold deflection
built from owner deflection plus the 2-bit confidence-quantized peer
deflection.  The H0 quantile at ``1 - p_fa`` is the standardized threshold.

  - clip_max      : minimum deflection for P_D = ``--saturation-pd`` at P_FA.
  - confidence    : 2-bit classes from log1p deflection quartiles (boundaries)
                    and per-quartile medians (representatives).
  - threshold     : 1-P_FA quantile of the replicated deployed statistic.
  - selfcheck     : deployed-statistic P_FA and episode steady/weak3/worst P_D.

The frozen K12 profile (`calibration_k12_passiverx_rawllr_seed1019466100.json`)
has threshold 3.3501; the same recipe on a fresh K12 trace gives ~3.6 (trace /
delivery differences).  Each scale must be calibrated from its own trace.

Usage:
    python tools/calibrate_dynamic_evidence.py --trace results/_cal_k6/receiver_evidence.npz \
        --out config/calibration_k6_passiverx_rawllr_seedNNN.json \
        --num-agents 6 --num-targets 6 --profile k6-passiverx-rawllr-seedNNN-engineering \
        --trace-seed NNN --samples 500000 --cal-seed 20260825 \
        --selfcheck-draws 1024 --selfcheck-seed 20260826
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uav_isac.evaluation.quantized_evidence_audit import (
    episode_detection_metrics,
)
from uav_isac.physical.detection import (
    minimum_deflection_for_detection_probability,
)
from uav_isac.physical.evidence import (
    DeflectionConfidenceQuantizer,
    quantize_llr,
)


def deployed_statistic_samples(
    receiver_deflection: np.ndarray,
    fusion_owner: np.ndarray,
    *,
    llr_bits: int,
    clip_max: float,
    confidence_quantizer: DeflectionConfidenceQuantizer | None,
    samples: int,
    seed: int,
    delivery_fraction: float = 1.0,
) -> dict:
    """Draw the deployed standardized statistic under H0.

    Mirrors `estimate_quantized_evidence_detection`: per (frame, target) the
    owner keeps its local LLR, each delivered peer contributes its quantized
    LLR, and the threshold deflection uses the confidence-quantized peer
    deflection.  Returns the statistic, its threshold deflection, and the
    owner/peer masks so the caller can take the 1-P_FA quantile.
    """
    receiver_d = np.asarray(receiver_deflection, dtype=np.float64)
    owner = np.asarray(fusion_owner, dtype=np.int64)
    frames, agents, targets = receiver_d.shape
    if owner.shape != (frames, targets):
        raise ValueError("fusion_owner must have shape (F, Q)")
    owner_mask = np.zeros((frames, agents, targets), dtype=bool)
    frame_idx, target_idx = np.nonzero(owner >= 0)
    owner_mask[frame_idx, owner[frame_idx, target_idx], target_idx] = True
    peer = ~owner_mask
    rng = np.random.default_rng(int(seed))
    count = max(1, int(samples))
    frame_ids = rng.integers(0, frames, size=count)
    target_ids = rng.integers(0, targets, size=count)
    delivered = (
        rng.random((count, agents)) <= float(delivery_fraction))
    d = receiver_d[frame_ids, :, target_ids]
    z = rng.standard_normal((count, agents))
    local_h0 = -0.5 * d + np.sqrt(d) * z
    quantized = quantize_llr(local_h0, int(llr_bits), float(clip_max))
    om = owner_mask[frame_ids, :, target_ids]
    pm = peer[frame_ids, :, target_ids] & delivered
    fused_h0 = np.sum(np.where(om, local_h0, np.where(pm, quantized, 0.0)),
                      axis=1)
    peer_d_hat = (
        receiver_d if confidence_quantizer is None
        else confidence_quantizer.quantize(receiver_d))
    threshold_d = np.sum(np.where(
        om, d, np.where(pm, peer_d_hat[frame_ids, :, target_ids], 0.0)),
        axis=1)
    return {
        "statistic": (fused_h0 + 0.5 * threshold_d) / np.sqrt(threshold_d),
        "threshold_deflection": threshold_d,
        "frame_ids": frame_ids,
        "target_ids": target_ids,
    }


def deployed_statistic_detection(
    receiver_deflection: np.ndarray,
    fusion_owner: np.ndarray,
    *,
    llr_bits: int,
    clip_max: float,
    standardized_threshold: float,
    p_fa: float,
    confidence_quantizer: DeflectionConfidenceQuantizer | None,
    draws_per_frame: int,
    seed: int,
) -> np.ndarray:
    """Per-frame P_D replicating the deployed detection statistic."""
    receiver_d = np.asarray(receiver_deflection, dtype=np.float64)
    owner = np.asarray(fusion_owner, dtype=np.int64)
    frames, agents, targets = receiver_d.shape
    owner_mask = np.zeros((frames, agents, targets), dtype=bool)
    frame_idx, target_idx = np.nonzero(owner >= 0)
    owner_mask[frame_idx, owner[frame_idx, target_idx], target_idx] = True
    peer = ~owner_mask
    peer_d_hat = (
        receiver_d if confidence_quantizer is None
        else confidence_quantizer.quantize(receiver_d))
    rng = np.random.default_rng(int(seed))
    draws = max(1, int(draws_per_frame))
    noise = rng.standard_normal((draws, frames, agents, targets))
    local_h0 = -0.5 * receiver_d[None] + np.sqrt(receiver_d[None]) * noise
    local_h1 = +0.5 * receiver_d[None] + np.sqrt(receiver_d[None]) * noise
    q0 = quantize_llr(local_h0, int(llr_bits), float(clip_max))
    q1 = quantize_llr(local_h1, int(llr_bits), float(clip_max))
    fused_h0 = np.sum(np.where(
        owner_mask[None], local_h0, np.where(peer[None], q0, 0.0)), axis=2)
    fused_h1 = np.sum(np.where(
        owner_mask[None], local_h1, np.where(peer[None], q1, 0.0)), axis=2)
    threshold_d = np.sum(np.where(
        owner_mask, receiver_d, np.where(peer, peer_d_hat, 0.0)), axis=1)
    threshold = (
        -0.5 * threshold_d[None]
        + np.sqrt(threshold_d[None]) * float(standardized_threshold))
    positive = threshold_d > 0.0
    pd = np.where(
        positive,
        np.mean(fused_h1 > threshold, axis=0),
        float(p_fa),
    )
    pfa = np.where(
        positive,
        np.mean(fused_h0 > threshold, axis=0),
        float(p_fa),
    )
    return pd, pfa


def calibrate_from_trace(
    trace_path: str | Path,
    *,
    num_agents: int,
    num_targets: int,
    trace_seed: int,
    profile: str,
    p_fa: float,
    llr_bits: int,
    confidence_bits: int,
    saturation_pd: float,
    samples: int,
    cal_seed: int,
    selfcheck_draws: int,
    selfcheck_seed: int,
) -> dict:
    trace = np.load(trace_path)
    receiver_d = np.asarray(trace["receiver_deflection"], dtype=np.float64)
    frames, agents, targets = receiver_d.shape
    if (agents, targets) != (num_agents, num_targets):
        raise ValueError(
            f"trace shape {(agents, targets)} does not match requested "
            f"{(num_agents, num_targets)}")
    if "fusion_owner" not in trace.files:
        raise ValueError(
            "trace must contain the scheduled fusion owner; regenerate it "
            "with the current trainer (which stores fusion_owner)")
    fusion_owner = np.asarray(trace["fusion_owner"], dtype=np.int64)
    if fusion_owner.shape != (frames, targets):
        raise ValueError("fusion_owner must have shape (F, Q)")

    positive = receiver_d[receiver_d > 0.0]
    if positive.size == 0:
        raise ValueError("trace has no positive deflection")
    log1p_positive = np.log1p(positive)
    boundaries = np.percentile(log1p_positive, [25.0, 50.0, 75.0])
    quartile_edges = np.quantile(log1p_positive, [0.0, 0.25, 0.5, 0.75, 1.0])
    representatives = []
    for i in range(1 << max(confidence_bits, 0)):
        member = (log1p_positive >= quartile_edges[i]) & (
            log1p_positive <= quartile_edges[i + 1])
        representatives.append(
            float(np.median(positive[member])) if np.any(member) else 0.0)
    confidence_quantizer = DeflectionConfidenceQuantizer(
        int(confidence_bits), boundaries, np.asarray(representatives))

    clip_max = float(minimum_deflection_for_detection_probability(
        np.asarray([saturation_pd]), p_fa)[0])

    drawn = deployed_statistic_samples(
        receiver_d,
        fusion_owner,
        llr_bits=llr_bits,
        clip_max=clip_max,
        confidence_quantizer=confidence_quantizer,
        samples=samples,
        seed=cal_seed,
    )
    stat = drawn["statistic"]
    threshold_d = drawn["threshold_deflection"]
    valid = threshold_d > 0.0
    if not np.any(valid):
        raise ValueError("trace has no included evidence")
    tau = float(np.quantile(stat[valid], 1.0 - float(p_fa)))
    calibration_pfa = float(np.mean(stat[valid] > tau))

    pd, pfa = deployed_statistic_detection(
        receiver_d,
        fusion_owner,
        llr_bits=llr_bits,
        clip_max=clip_max,
        standardized_threshold=tau,
        p_fa=p_fa,
        confidence_quantizer=confidence_quantizer,
        draws_per_frame=selfcheck_draws,
        seed=selfcheck_seed,
    )
    episode_metrics = episode_detection_metrics(
        pd, np.asarray(trace["episode_index"], dtype=np.int64),
        steady_window=20)
    if episode_metrics.shape[0] == 0:
        raise ValueError("selfcheck produced no episode metrics")
    steady, weak3, worst = (float(value) for value in episode_metrics[0])
    return {
        "schema_version": 1,
        "status": "single-seed engineering calibration; not reliability "
                  "certification",
        "trace_path": str(trace_path),
        "trace_seed": int(trace_seed),
        "frames": int(frames),
        "num_agents": int(agents),
        "num_targets": int(targets),
        "positive_receiver_deflection_fraction": float(
            positive.size / receiver_d.size),
        "content_mode": "normal",
        "llr_bits": int(llr_bits),
        "clip_max": clip_max,
        "clip_selection": (
            "decision-relevant saturation clip: minimum deflection for "
            f"P_D={saturation_pd} at P_FA={p_fa}"),
        "confidence_bits": int(confidence_bits),
        "confidence_log_boundaries": boundaries.tolist(),
        "confidence_representatives": representatives,
        "threshold_calibration": {
            "hypothesis": "H0 only (deployed replicated statistic)",
            "samples": int(samples),
            "seed": int(cal_seed),
            "target_pfa": float(p_fa),
            "standardized_threshold": tau,
            "calibration_pfa": calibration_pfa,
            "valid_calibration_samples": int(np.sum(valid)),
        },
        "independent_mc_selfcheck": {
            "seed": int(selfcheck_seed),
            "draws_per_frame": int(selfcheck_draws),
            "pfa": float(np.mean(pfa)),
            "steady_pd": steady,
            "weak3_pd": weak3,
            "worst_pd": worst,
        },
        "calibration_profile": profile,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-agents", type=int, required=True)
    parser.add_argument("--num-targets", type=int, required=True)
    parser.add_argument("--trace-seed", type=int, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--p-fa", type=float, default=0.001)
    parser.add_argument("--llr-bits", type=int, default=8)
    parser.add_argument("--confidence-bits", type=int, default=2)
    parser.add_argument("--saturation-pd", type=float, default=0.9999)
    parser.add_argument("--samples", type=int, default=500_000)
    parser.add_argument("--cal-seed", type=int, default=20260825)
    parser.add_argument("--selfcheck-draws", type=int, default=1024)
    parser.add_argument("--selfcheck-seed", type=int, default=20260826)
    args = parser.parse_args()

    result = calibrate_from_trace(
        args.trace,
        num_agents=args.num_agents,
        num_targets=args.num_targets,
        trace_seed=args.trace_seed,
        profile=args.profile,
        p_fa=args.p_fa,
        llr_bits=args.llr_bits,
        confidence_bits=args.confidence_bits,
        saturation_pd=args.saturation_pd,
        samples=args.samples,
        cal_seed=args.cal_seed,
        selfcheck_draws=args.selfcheck_draws,
        selfcheck_seed=args.selfcheck_seed,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
