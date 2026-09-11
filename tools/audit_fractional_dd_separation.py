"""Offline two-cell identifiability audit using actual cyclic OTFS responses.

For y=a*s+b*t+n with unknown complex nuisance b and white complex noise,
project out span(t). The retained target energy is ||P_perp s||^2.
For unit signatures it equals 1-|t^H s|^2. This is an identifiability
diagnostic, not a calibrated detection probability or a resource comparison.
"""
from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_isac.physical.waveform_evidence import (
    MinimalOTFSWaveform, dd_path_response, qpsk_dd_pilot,
)


def audit() -> dict:
    cfg = MinimalOTFSWaveform(delay_bins=16, doppler_bins=8)
    impulse = np.zeros((8, 16), dtype=complex)
    impulse[0, 0] = 1.0
    rows = []
    for name, pilot in (("dd_impulse", impulse), ("qpsk", qpsk_dd_pilot(cfg))):
        for offset in (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0):
            # Both absolute path coordinates are fractional. Offset is in
            # both delay and Doppler, not a physical range in metres.
            s = dd_path_response(pilot, delay_bin=1.2, doppler_bin=0.3).ravel()
            t = dd_path_response(
                pilot, delay_bin=1.2 + offset, doppler_bin=0.3 + offset).ravel()
            s = s / np.linalg.norm(s)
            t = t / np.linalg.norm(t)
            overlap = np.vdot(t, s)
            projected = s - t * overlap
            retained = float(np.vdot(projected, projected).real)
            predicted = float(max(0.0, 1.0 - abs(overlap)**2))
            # Use only the total noiseless observation at extraction time;
            # changing nuisance amplitude must not change the projection.
            residuals = []
            for nuisance in (0j, 0.7 + 0.4j, -3.0 + 2j):
                y = (0.6 + 0.2j) * s + nuisance * t
                actual = y - t * np.vdot(t, y)
                residuals.append(float(np.linalg.norm(actual - (0.6 + 0.2j)*projected)))
            rows.append({
                "pilot": name, "dd_offset_each_axis": offset,
                "mutual_coherence": float(abs(overlap)),
                "retained_target_energy_fraction": retained,
                "projection_identity_error": abs(retained - predicted),
                "nuisance_cancellation_error": max(residuals),
            })
    passed = all(r["projection_identity_error"] < 1e-12
                 and r["nuisance_cancellation_error"] < 1e-12 for r in rows)
    return {
        "status": "PASS" if passed else "FAIL",
        "evidence_class": "CYCLIC_OTFS_TWO_CELL_IDENTIFIABILITY",
        "bandwidth_hz": cfg.delay_bins * cfg.delta_f_hz,
        "block_duration_s": cfg.doppler_bins / cfg.delta_f_hz,
        "online_behavior_changed": False,
        "assumptions": ["known hypothesis templates", "unknown complex nuisance amplitude",
                        "white complex noise", "ideal cyclic channel"],
        "pd_or_800m_certified": False,
        "rows": rows,
    }


if __name__ == "__main__":
    result = audit()
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(result["status"] != "PASS")
