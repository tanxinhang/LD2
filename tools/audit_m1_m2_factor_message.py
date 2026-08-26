#!/usr/bin/env python
"""M1 payload and M2-0 static interval audit for factorized capability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.params import load_config  # noqa: E402
from uav_isac.coordination.certified_factor_message import (  # noqa: E402
    dense_capability_payload_bits, factor_capability_payload_bits,
    factor_coefficient_envelope, log_interval_quantize,
)
from uav_isac.coordination.coefficient_structure import (  # noqa: E402
    BistaticFactorization, balance_bistatic_factor_gauge,
    exact_bistatic_factorization,
)
from uav_isac.evaluation.physical_oracle_audit import (  # noqa: E402
    per_watt_deflection_tensor_from_observables,
)


def audit(trace_path: Path, config_path: Path, g4a_path: Path) -> dict[str, object]:
    with np.load(trace_path, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    cfg = load_config(str(config_path))
    g4a = json.loads(g4a_path.read_text(encoding="utf-8"))
    region = {(int(row["seed"]), int(row["frame"])) for row in g4a["rows"]}
    bit_depths = (3, 4, 6, 8, 10, 12, 14, 16)
    rows = []
    for index in range(np.asarray(data["seed"]).size):
        key = (int(data["seed"][index]), int(data["frame"][index]))
        if key not in region:
            continue
        alpha2 = np.asarray(data["privileged_alpha"][index], dtype=np.float64) ** 2
        report = np.asarray(data["privileged_chi_rep"][index], dtype=np.float64)
        dd = np.asarray(data["privileged_g_dd"][index], dtype=np.float64)
        active = dd >= float(cfg.detection.g_min)
        coefficient = per_watt_deflection_tensor_from_observables(
            np.sqrt(alpha2), dd, report,
            T_sym=float(cfg.otfs.T_sym), M=int(cfg.otfs.M), N=int(cfg.otfs.N),
            kT=float(cfg.channel.kT), bandwidth_hz=float(cfg.otfs.B),
            noise_figure_db=float(cfg.channel.NF),
            g_tx_dbi=float(cfg.otfs.g_tx_dBi), g_rx_dbi=float(cfg.otfs.g_rx_dBi),
            n_cpi=int(cfg.otfs.n_cpi), g_min=float(cfg.detection.g_min),
            use_swerling=bool(cfg.channel.use_swerling))
        K, _, Q = coefficient.shape
        factors = []
        exception_count = 0
        for q in range(Q):
            raw = exact_bistatic_factorization(
                alpha2[:, :, q], report[:, :, q], active[:, :, q])
            raw_dense = raw.dense()
            positive = raw_dense > 0.0
            scale = float(np.median(
                coefficient[:, :, q][positive] / raw_dense[positive]))
            factors.append(balance_bistatic_factor_gauge(BistaticFactorization(
                tx_factor=raw.tx_factor,
                rx_factor=raw.rx_factor * scale,
                inactive_offdiagonal=raw.inactive_offdiagonal)))
            exception_count += len(raw.inactive_offdiagonal)
        all_factor_values = np.concatenate([
            np.concatenate([factor.tx_factor, factor.rx_factor])
            for factor in factors])
        frame_results = {}
        for bits in bit_depths:
            common = log_interval_quantize(all_factor_values, bits)
            offset = 0
            violations = 0
            active_entries = 0
            log_widths = []
            for q, factor in enumerate(factors):
                tx_slice = slice(offset, offset + K)
                rx_slice = slice(offset + K, offset + 2 * K)
                tx_code = type(common)(
                    common.lower[tx_slice], common.upper[tx_slice],
                    common.scale_lower, common.scale_upper, bits)
                rx_code = type(common)(
                    common.lower[rx_slice], common.upper[rx_slice],
                    common.scale_lower, common.scale_upper, bits)
                offset += 2 * K
                support = active[:, :, q] & ~np.eye(K, dtype=bool)
                lower, upper = factor_coefficient_envelope(
                    tx_code, rx_code, active[:, :, q])
                truth = coefficient[:, :, q]
                violations += int(np.sum(
                    (truth < lower * (1.0 - 1.0e-12))
                    | (truth > upper * (1.0 + 1.0e-12))))
                active_entries += int(np.sum(support))
                log_widths.extend(np.log(upper[support] / lower[support]).tolist())
            dense_payload = dense_capability_payload_bits(K, Q, bits)
            factor_payload = factor_capability_payload_bits(
                K, Q, bits, exception_count)
            frame_results[str(bits)] = {
                "dense_bits": dense_payload.total_bits,
                "factor_bits": factor_payload.total_bits,
                "bit_reduction": 1.0 - factor_payload.total_bits / dense_payload.total_bits,
                "dd_bits": factor_payload.dd_bits,
                "containment_violations": violations,
                "active_entries": active_entries,
                "median_log_interval_width": float(np.median(log_widths)),
                "max_log_interval_width": float(np.max(log_widths)),
            }
        rows.append({
            "seed": key[0], "frame": key[1],
            "dd_exceptions": exception_count,
            "bits": frame_results,
        })
    summary = {}
    for bits in bit_depths:
        values = [row["bits"][str(bits)] for row in rows]
        summary[str(bits)] = {
            "dense_bits": values[0]["dense_bits"],
            "factor_bits_min": min(value["factor_bits"] for value in values),
            "factor_bits_median": float(np.median([
                value["factor_bits"] for value in values])),
            "factor_bits_max": max(value["factor_bits"] for value in values),
            "median_bit_reduction": float(np.median([
                value["bit_reduction"] for value in values])),
            "containment_violations": sum(
                value["containment_violations"] for value in values),
            "median_log_interval_width": float(np.median([
                value["median_log_interval_width"] for value in values])),
            "max_log_interval_width": max(
                value["max_log_interval_width"] for value in values),
        }
    return {
        "gate": "M1-payload-and-M2-0-static-containment",
        "scope": "semantic-equivalent shadow capability record; not current live payload",
        "frames": len(rows),
        "bit_depths": list(bit_depths),
        "payload_layout": {
            "dense": "25-bit header + 32-bit global linear scale + Q*K*(K-1)*B",
            "factor": (
                "27-bit header + two outward-float32 log range endpoints + "
                "2*K*Q*B + min(sparse DD index list, full DD mask)"),
            "dd_list": "ceil(log2(N+1)) count + s*ceil(log2(N)) indices",
        },
        "m2_scope": (
            "h=0 only; exact DD exception support; AoI and packet loss are not certified here"),
        "summary": summary,
        "m2_static_gate_pass": all(
            summary[str(bits)]["containment_violations"] == 0
            for bits in bit_depths),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--g4a", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.trace, args.config, args.g4a)
    payload = json.dumps(result, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
