"""Run calibrated quantized-evidence Gate 1b on saved physical traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_isac.evaluation.quantized_evidence_audit import (
    analytic_trace_bounds,
    calibrate_standardized_threshold,
    evidence_inclusion,
    mean_structured_bits_per_frame,
    simulate_quantized_detection,
    structured_broadcast_transport,
    summarize_paired_detection_delta,
    summarize_quantized_method,
)
from uav_isac.environment.communication import InterUAVCommunicationModel
from uav_isac.physical.evidence import (
    EvidencePacketLayout,
    calibrate_deflection_confidence,
    calibrate_llr_clip,
)


def _load_trace(path: Path):
    data = np.load(path)
    return {
        "receiver_d": np.asarray(
            data["receiver_deflection"], dtype=np.float64),
        "episode_index": np.asarray(
            data["episode_index"], dtype=np.int64),
        "p_fa": float(np.asarray(data["p_fa"]).item()),
        "num_agents": int(np.asarray(data["num_agents"]).item()),
        "num_targets": int(np.asarray(data["num_targets"]).item()),
        "episode_seeds": np.asarray(data["episode_seeds"], dtype=np.int64),
        "positions": (
            np.asarray(data["uav_positions"], dtype=np.float64)
            if "uav_positions" in data else None),
        "comm_power_w": (
            np.asarray(data["comm_power_w"], dtype=np.float64)
            if "comm_power_w" in data else None),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-trace", type=Path, required=True)
    parser.add_argument("--test-trace", type=Path, required=True)
    parser.add_argument("--topk", type=int, nargs="+", default=[2])
    parser.add_argument("--bits", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument(
        "--owner-aware",
        action="store_true",
        help="exclude evidence already local to the predesignated owner before "
             "applying Top-k; avoids transmitting redundant owner evidence",
    )
    parser.add_argument(
        "--physical-transport",
        action="store_true",
        help="apply the configured SNR/deadline link to every structured "
             "broadcast; requires positions and comm power in both traces",
    )
    parser.add_argument(
        "--controls",
        action="store_true",
        help="also run zero-content and value-target-roll controls with "
             "identical routing, bits, powers, and delivery events",
    )
    parser.add_argument(
        "--threshold-calibration-on-test-geometry",
        action="store_true",
        help="calibrate each detector threshold with independent H0 draws on "
             "the test geometry, then evaluate with separate H0/H1 draws; "
             "this fixes method-level PFA without using H1 outcomes",
    )
    parser.add_argument(
        "--threshold-mode",
        choices=["standardized", "global_llr"],
        default="standardized",
        help="standardized uses per-frame fused deflection; global_llr uses "
             "one H0-calibrated threshold and requires no peer-d knowledge",
    )
    parser.add_argument(
        "--confidence-bits",
        type=int,
        default=0,
        help="finite quality-class bits sent per evidence entry; 0 retains "
             "the optimistic exact-deflection threshold for diagnostics",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "config/exp_800_q4_u2u_hierarchical_multistatic_"
            "distributed_matching_hybrid50_paper_top1_eval.yaml"),
    )
    parser.add_argument("--clip-quantile", type=float, default=0.999)
    parser.add_argument("--clip-samples", type=int, default=1_000_000)
    parser.add_argument("--threshold-samples", type=int, default=1_000_000)
    parser.add_argument("--mc-draws", type=int, default=2048)
    parser.add_argument("--batch-frames", type=int, default=16)
    parser.add_argument("--steady-window", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/gate1b_quantized_evidence/summary.json"),
    )
    args = parser.parse_args()

    calibration = _load_trace(args.calibration_trace)
    test = _load_trace(args.test_trace)
    if (
        calibration["num_agents"] != test["num_agents"]
        or calibration["num_targets"] != test["num_targets"]
    ):
        raise ValueError("calibration and test traces have different shapes")
    if not np.isclose(calibration["p_fa"], test["p_fa"]):
        raise ValueError("calibration and test traces have different P_FA")

    p_fa = test["p_fa"]
    bounds = analytic_trace_bounds(test["receiver_d"], p_fa)
    layout = EvidencePacketLayout(
        test["num_agents"],
        test["num_targets"],
        confidence_bits=max(0, int(args.confidence_bits)),
    )
    transport_model = None
    if args.physical_transport:
        if (
            calibration["positions"] is None
            or calibration["comm_power_w"] is None
            or test["positions"] is None
            or test["comm_power_w"] is None
        ):
            raise ValueError(
                "physical transport requires position/power trace fields")
        from config.params import load_config
        cfg = load_config(str(args.config))
        transport_model = InterUAVCommunicationModel(
            rate_bits_per_dim=list(
                cfg.marl.comm_rate_bits_per_dim),
            header_bits=int(cfg.marl.comm_header_bits),
            bandwidth_hz=float(cfg.marl.comm_bandwidth_hz),
            deadline_s=float(cfg.marl.comm_deadline_s),
            processing_delay_s=float(
                cfg.marl.comm_processing_delay_s),
            snr_threshold_db=float(
                cfg.marl.comm_snr_threshold_db),
            antenna_gain_dbi=float(
                cfg.marl.comm_antenna_gain_dbi),
            carrier_hz=float(cfg.otfs.fc),
            tx_power_w=float(cfg.marl.comm_tx_power_w),
            kT=float(cfg.channel.kT),
            noise_figure_db=float(cfg.channel.NF),
            dt=float(cfg.scenario.dt),
            message_dim=1,
        )
    methods = {}
    method_pd = {}
    clip_max_by_topk = {}
    for topk in args.topk:
        calibration_routing = evidence_inclusion(
            calibration["receiver_d"],
            topk,
            owner_aware=args.owner_aware,
        )
        useful_calibration_d = calibration["receiver_d"][
            calibration_routing["peer_mask"]]
        confidence_quantizer = None
        calibration_peer_d_hat = None
        test_peer_d_hat = None
        if args.confidence_bits > 0:
            confidence_quantizer = calibrate_deflection_confidence(
                useful_calibration_d,
                bits=args.confidence_bits,
            )
            calibration_peer_d_hat = confidence_quantizer.quantize(
                calibration["receiver_d"])
            test_peer_d_hat = confidence_quantizer.quantize(
                test["receiver_d"])
        clip_max = calibrate_llr_clip(
            useful_calibration_d,
            quantile=args.clip_quantile,
            samples=args.clip_samples,
            seed=20260723,
        )
        clip_max_by_topk[str(int(topk))] = clip_max
        routing = evidence_inclusion(
            test["receiver_d"],
            topk,
            owner_aware=args.owner_aware,
        )
        for bits in args.bits:
            calibration_transport = None
            test_transport = None
            calibration_delivery = None
            test_delivery = None
            if transport_model is not None:
                calibration_transport = structured_broadcast_transport(
                    calibration["positions"],
                    calibration["comm_power_w"],
                    calibration_routing["selected_mask"],
                    layout,
                    bits,
                    transport_model,
                )
                test_transport = structured_broadcast_transport(
                    test["positions"],
                    test["comm_power_w"],
                    routing["selected_mask"],
                    layout,
                    bits,
                    transport_model,
                )
                calibration_delivery = calibration_transport[
                    "delivery_matrix"]
                test_delivery = test_transport["delivery_matrix"]
            content_modes = (
                ["normal", "zero", "value_roll"]
                if args.controls else ["normal"])
            for content_mode in content_modes:
                threshold_receiver_d = (
                    test["receiver_d"]
                    if args.threshold_calibration_on_test_geometry
                    else calibration["receiver_d"])
                threshold_delivery = (
                    test_delivery
                    if args.threshold_calibration_on_test_geometry
                    else calibration_delivery)
                calibration_result = calibrate_standardized_threshold(
                    threshold_receiver_d,
                    topk=topk,
                    bits=bits,
                    clip_max=clip_max,
                    p_fa=p_fa,
                    owner_aware=args.owner_aware,
                    delivery_matrix=threshold_delivery,
                    content_mode=content_mode,
                    threshold_mode=args.threshold_mode,
                    peer_deflection_estimate=(
                        test_peer_d_hat
                        if args.threshold_calibration_on_test_geometry
                        else calibration_peer_d_hat),
                    samples=args.threshold_samples,
                    seed=20260724,
                )
                simulation = simulate_quantized_detection(
                    test["receiver_d"],
                    topk=topk,
                    bits=bits,
                    clip_max=clip_max,
                    standardized_threshold=calibration_result[
                        "standardized_threshold"],
                    p_fa=p_fa,
                    owner_aware=args.owner_aware,
                    delivery_matrix=test_delivery,
                    content_mode=content_mode,
                    threshold_mode=args.threshold_mode,
                    peer_deflection_estimate=test_peer_d_hat,
                    draws_per_frame=args.mc_draws,
                    batch_frames=args.batch_frames,
                    seed=20260725,
                )
                metrics = summarize_quantized_method(
                    simulation["pd"],
                    bounds["local_pd"],
                    bounds["central_pd"],
                    test["episode_index"],
                    steady_window=args.steady_window,
                )
                bits_per_frame, active_senders = (
                    mean_structured_bits_per_frame(
                        routing["selected_mask"], layout, bits))
                base_key = f"top{topk}_{bits}bit"
                key = (
                    base_key if content_mode == "normal"
                    else f"{base_key}_{content_mode}")
                method_pd[key] = simulation["pd"]
                methods[key] = {
                    **metrics,
                    **calibration_result,
                    "content_mode": content_mode,
                    "clip_max": clip_max,
                    "test_pfa": simulation["aggregate_pfa"],
                    "clip_rate": simulation["clip_rate"],
                    "quantization_mse": simulation["quantization_mse"],
                    "sign_flip_rate": simulation["sign_flip_rate"],
                    "bits_per_frame": bits_per_frame,
                    "active_senders_per_frame": active_senders,
                    "gate_pass": bool(
                        content_mode == "normal"
                        and metrics["worst_recovery"] >= 0.50
                        and metrics[
                            "worst_delta_vs_local_ci95"][0] > 0.0
                        and abs(simulation["aggregate_pfa"] - p_fa)
                        <= max(1.0e-4, 0.20 * p_fa)
                    ),
                }
                if test_transport is not None:
                    methods[key]["transport"] = {
                        field: value
                        for field, value in test_transport.items()
                        if field not in {
                            "delivery_matrix", "payload_bits"}
                    }
                    methods[key][
                        "calibration_transport_delivery_rate"
                    ] = calibration_transport["delivery_rate"]
                if confidence_quantizer is not None:
                    methods[key]["confidence_quantizer"] = {
                        "bits": int(confidence_quantizer.bits),
                        "log_boundaries": (
                            confidence_quantizer.log_boundaries.tolist()),
                        "representatives": (
                            confidence_quantizer.representatives.tolist()),
                    }

            if args.controls:
                base_key = f"top{topk}_{bits}bit"
                for control in ("zero", "value_roll"):
                    control_key = f"{base_key}_{control}"
                    comparison = summarize_paired_detection_delta(
                        method_pd[base_key],
                        method_pd[control_key],
                        test["episode_index"],
                        steady_window=args.steady_window,
                    )
                    methods[base_key][
                        f"paired_vs_{control}"] = comparison
                methods[base_key]["content_gate_pass"] = bool(
                    methods[base_key][
                        "paired_vs_zero"]["worst_delta_ci95"][0] > 0.0
                    and methods[base_key][
                        "paired_vs_value_roll"][
                            "worst_delta_ci95"][0] > 0.0
                )
                methods[base_key]["gate_pass"] = bool(
                    methods[base_key]["gate_pass"]
                    and methods[base_key]["content_gate_pass"])

    transport_note = (
        "configured physical SNR/deadline transport"
        if args.physical_transport
        else "lossless zero-delay transport upper bound"
    )
    summary = {
        "gate": (
            "G1c-quantized-physical-transport"
            if args.physical_transport
            else "G1b-quantized-lossless-transport"),
        "calibration_trace": str(args.calibration_trace),
        "test_trace": str(args.test_trace),
        "calibration_episode_seeds": (
            calibration["episode_seeds"].tolist()),
        "test_episode_seeds": test["episode_seeds"].tolist(),
        "target_pfa": p_fa,
        "clip_quantile": float(args.clip_quantile),
        "clip_max_by_topk": clip_max_by_topk,
        "clip_samples": int(args.clip_samples),
        "threshold_samples": int(args.threshold_samples),
        "mc_draws_per_frame": int(args.mc_draws),
        "packet_layout": {
            "header_bits": layout.header_bits,
            "source_bits": layout.source_bits,
            "target_bits": layout.target_bits,
            "timestamp_bits": layout.timestamp_bits,
            "confidence_bits": layout.confidence_bits,
        },
        "note": (
            "receiver evidence is quantized and selected by pre-observation "
            f"local quality; transport={transport_note}; owner evidence "
            "remains local and unquantized; "
            f"owner-aware redundant-evidence pruning={args.owner_aware}"),
        "physical_transport": bool(args.physical_transport),
        "content_controls": bool(args.controls),
        "threshold_calibration_on_test_geometry": bool(
            args.threshold_calibration_on_test_geometry),
        "threshold_mode": args.threshold_mode,
        "methods": methods,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
