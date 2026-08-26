#!/usr/bin/env python
"""G2-0.5--G2-0.7 detector, CPI, and sensing-power-cap audit.

This is deliberately cheaper than an episode evaluation.  It freezes the
declared real Gaussian shift detector, verifies its ROC by Monte Carlo, and
shows how the physical link budget accumulates through MN and n_CPI before
any structure, report-link, or DD-gate loss is applied.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.params import get_default_config
from uav_isac.physical.channel import compute_noise_power
from uav_isac.physical.detection import (
    DETECTOR_CONVENTION,
    monte_carlo_gaussian_shift_roc,
)
from uav_isac.physical.geometry import compute_path_gain
from uav_isac.physical.deflection import validate_cpi_schedule
from uav_isac.utils.math_utils import compute_PD


def link_budget_sanity(
    distances_m: list[float],
    *,
    sensing_power_w: float | None = None,
) -> dict[str, object]:
    cfg = get_default_config()
    otfs, channel, target, detection = (
        cfg.otfs, cfg.channel, cfg.target, cfg.detection)
    power = float(
        cfg.uav.P_sense if sensing_power_w is None else sensing_power_w)
    sensing_power_cap = float(getattr(
        cfg.uav, "P_sense_max", cfg.uav.P_sense))
    if power > sensing_power_cap + 1.0e-15:
        raise ValueError("audit sensing power exceeds uav.P_sense_max")
    noise = compute_noise_power(channel.kT, otfs.B, channel.NF)
    antenna_gain = 10.0 ** (
        (float(otfs.g_tx_dBi) + float(otfs.g_rx_dBi)) / 10.0)
    c_det = float(getattr(detection, "c_det", 1.0))
    rows = []
    for distance in distances_m:
        leg = float(distance)
        if not np.isfinite(leg) or leg <= 0.0:
            raise ValueError("distances must be finite and positive")
        # Symmetric bistatic geometry with both target legs equal to `leg`.
        target_pos = np.zeros(3, dtype=np.float64)
        tx_pos = np.asarray([-leg, 0.0, 0.0])
        rx_pos = np.asarray([leg, 0.0, 0.0])
        alpha = compute_path_gain(
            tx_pos, rx_pos, target_pos, otfs.fc, target.rcs)
        received_snr = power * alpha**2 * antenna_gain / noise
        after_mn = received_snr * int(otfs.M) * int(otfs.N)
        deflection = c_det * after_mn * int(otfs.n_cpi)
        pd_after_mn = float(compute_PD(
            np.asarray([c_det * after_mn]), detection.P_FA)[0])
        pd = float(compute_PD(
            np.asarray([deflection]), detection.P_FA)[0])
        rows.append({
            "leg_distance_m": leg,
            "bistatic_path_m": 2.0 * leg,
            "received_snr": float(received_snr),
            "after_MN": float(after_mn),
            "P_D_after_MN": pd_after_mn,
            "after_n_cpi_deflection": float(deflection),
            "P_D": pd,
        })
    cpi_duration, max_time_feasible_looks = validate_cpi_schedule(
        int(otfs.n_cpi), int(otfs.N), float(otfs.T_sym),
        float(cfg.scenario.dt))
    return {
        "detector_convention": DETECTOR_CONVENTION,
        "c_det": c_det,
        "P_FA": float(detection.P_FA),
        "sensing_power_w": power,
        "sensing_power_cap_w": sensing_power_cap,
        "power_cap_respected": bool(power <= sensing_power_cap + 1.0e-15),
        "noise_power_w": float(noise),
        "M": int(otfs.M),
        "N": int(otfs.N),
        "n_cpi": int(otfs.n_cpi),
        "execution_model": "one_explicit_otfs_frame_per_control_action",
        "phantom_look_free": bool(int(otfs.n_cpi) == 1),
        "max_time_feasible_looks": int(max_time_feasible_looks),
        "cpi_duration_s": float(cpi_duration),
        "control_frame_s": float(cfg.scenario.dt),
        "cpi_exceeds_control_frame": bool(
            cpi_duration > float(cfg.scenario.dt) + 1.0e-15),
        "rows": rows,
        "saturated_all": bool(all(row["P_D"] > 0.999 for row in rows)),
    }


def run_audit(*, samples: int, seed: int) -> dict[str, object]:
    budget = link_budget_sanity([100.0, 300.0, 500.0, 800.0])
    roc = monte_carlo_gaussian_shift_roc(
        np.asarray([5.0, 10.0, 15.0, 20.0]),
        float(budget["P_FA"]),
        samples=int(samples),
        seed=int(seed),
    )
    serializable_roc = {
        key: (value.tolist() if isinstance(value, np.ndarray) else value)
        for key, value in roc.items()
    }
    result = {"link_budget": budget, "roc_monte_carlo": serializable_roc}
    result["certification"] = certification_readiness(result)
    return result


def certification_readiness(result: dict[str, object]) -> dict[str, object]:
    """Return an executable pre-recertification gate, not a performance score."""
    budget = result["link_budget"]
    roc = result["roc_monte_carlo"]
    blockers: list[str] = []
    empirical_pd = np.asarray(roc["empirical_pd"], dtype=np.float64)
    analytical_pd = np.asarray(roc["analytical_pd"], dtype=np.float64)
    if np.max(np.abs(empirical_pd - analytical_pd)) > 5.0e-3:
        blockers.append("detector_mc_roc_mismatch")
    if bool(budget["saturated_all"]):
        blockers.append("link_budget_saturated_at_all_audit_ranges")
    if bool(budget["cpi_exceeds_control_frame"]):
        blockers.append("declared_cpi_exceeds_control_frame")
    if not bool(budget["phantom_look_free"]):
        blockers.append("multi_look_execution_model_not_certified")
    if not bool(budget["power_cap_respected"]):
        blockers.append("sensing_waveform_power_cap_violated")
    return {
        "ready_for_g2_1": not blockers,
        "blockers": blockers,
        "roc_absolute_tolerance": 5.0e-3,
    }


def _print_table(result: dict[str, object]) -> None:
    budget = result["link_budget"]
    print("distance | Pr/Pn | x(MN) | P_D(MN) | x(n_CPI)=D | final P_D")
    for row in budget["rows"]:
        print(
            f"{row['leg_distance_m']:7.0f}m | "
            f"{row['received_snr']:.6g} | {row['after_MN']:.6g} | "
            f"{row['P_D_after_MN']:.6f} | "
            f"{row['after_n_cpi_deflection']:.6g} | {row['P_D']:.9f}")
    print(
        f"CPI duration={budget['cpi_duration_s']:.6g}s, "
        f"control frame={budget['control_frame_s']:.6g}s, "
        f"saturated_all={budget['saturated_all']}")
    certification = result["certification"]
    print(
        f"ready_for_g2_1={certification['ready_for_g2_1']}; "
        f"blockers={certification['blockers']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--assert-ready", action="store_true",
        help="exit with status 2 when G2-1 prerequisites are not certified")
    args = parser.parse_args()
    result = run_audit(samples=args.samples, seed=args.seed)
    _print_table(result)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if args.assert_ready and not result["certification"]["ready_for_g2_1"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
