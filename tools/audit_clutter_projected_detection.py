"""Known-covariance diagnostic of unknown-phase detection with nuisance nulling.

C=I+rho*c*c^H, ||c||=1. Its inverse square root is
W=I+((1+rho)^(-1/2)-1)*c*c^H. With v=P_perp(W*t) W*s,
w=W*v/||v||, w^H C w=1 and w^H t=0. Under H0 |w^H y|^2
is exponential with mean one; H1 has noncentrality 2*A^2*||v||^2.
Covariance and templates are supplied exactly: this is not runtime calibration.
"""
from pathlib import Path
import json
import sys

import numpy as np
from scipy.stats import ncx2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_isac.physical.waveform_evidence import (
    MinimalOTFSWaveform, dd_path_response, qpsk_dd_pilot,
)


def detector(s, t, c, rho):
    """Return unit-noise-variance filter, or None for unidentifiable cells."""
    def whiten(x):
        return x + ((1 + rho)**-0.5 - 1) * c * np.vdot(c, x)
    sw, tw = whiten(s), whiten(t)
    v = sw - tw * (np.vdot(tw, sw) / np.vdot(tw, tw))
    information = float(np.vdot(v, v).real)
    if information < 1e-20:
        return None, information
    return whiten(v) / np.sqrt(information), information


def _samples(w, s, t, c, rho, amplitude, trials, seed):
    rng = np.random.default_rng(seed)
    output = np.empty(trials)
    for start in range(0, trials, 1024):
        size = min(1024, trials-start)
        def cn(shape):
            return (rng.standard_normal(shape) + 1j*rng.standard_normal(shape))/np.sqrt(2)
        # Detector receives total observations, never separate target edges.
        y = cn((size, s.size)) + np.sqrt(rho)*cn((size, 1))*c
        y += cn((size, 1))*t
        y += amplitude*np.exp(2j*np.pi*rng.random((size, 1)))*s
        output[start:start+size] = np.abs(np.sum(y*w.conj(), axis=1))**2
    return output


def audit(trials=100_000):
    if trials < 10_000:
        raise ValueError("at least 10000 trials per split required")
    pilot = qpsk_dd_pilot(MinimalOTFSWaveform())
    def signature(delay, doppler):
        x = dd_path_response(pilot, delay_bin=delay, doppler_bin=doppler).ravel()
        return x/np.linalg.norm(x)
    s, c = signature(1.2, 0.3), signature(1.4, 0.5)
    pfa, amplitude = 0.001, 3.0
    rows = []
    for rho in (0.0, 4.0):
        for offset in (0.0, 0.1, 0.5, 1.0):
            t = signature(1.2+offset, 0.3+offset)
            w, info = detector(s, t, c, rho)
            row = dict(clutter_power=rho, dd_offset_each_axis=offset, information=info)
            if w is None:
                row.update(status="UNIDENTIFIABLE", pd=None, pfa=None)
            else:
                # Separate calibration H0, validation H0 and validation H1 RNGs.
                cal = _samples(w, s, t, c, rho, 0, trials, 41001)
                threshold = float(np.quantile(cal, 1-pfa, method="higher"))
                h0 = _samples(w, s, t, c, rho, 0, trials, 41002)
                h1 = _samples(w, s, t, c, rho, amplitude, trials, 41003)
                false_alarm, pd = float(np.mean(h0>threshold)), float(np.mean(h1>threshold))
                exact_pd = float(ncx2.sf(2*threshold, 2, 2*amplitude**2*info))
                variance = float(np.vdot(w,w).real + rho*abs(np.vdot(w,c))**2)
                # Both H0 calibration and validation have finite tail uncertainty.
                pfa_tolerance = 5*np.sqrt(2*pfa*(1-pfa)/trials)
                pd_tolerance = 5*np.sqrt(max(exact_pd*(1-exact_pd), 1/trials)/trials)
                passed = (abs(false_alarm-pfa)<pfa_tolerance
                          and abs(pd-exact_pd)<pd_tolerance
                          and abs(variance-1)<1e-10 and abs(np.vdot(w,t))<1e-10)
                row.update(status="PASS" if passed else "FAIL", pd=pd, pfa=false_alarm,
                           threshold=threshold, exact_pd_at_frozen_threshold=exact_pd,
                           noise_variance=variance, nuisance_residual=float(abs(np.vdot(w,t))))
            rows.append(row)
    return dict(status="PASS" if all(r['status']!='FAIL' for r in rows) else "FAIL",
                evidence_class="KNOWN_COVARIANCE_OFFLINE_NONCOHERENT_DIAGNOSTIC",
                trials_per_split=trials, target_pfa=pfa, normalized_target_amplitude=amplitude,
                range_800m_certified=False, covariance_estimated_from_data=False,
                rows=rows)


if __name__ == "__main__":
    result = audit()
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(result['status'] != 'PASS')
