"""Prediction-matched coherent receiver, centralized diagnostic only.

For unit templates u, real projections sqrt(2) Re(u^H n) have covariance
Re(Gram(u)) under proper complex Gaussian noise of unit variance. The true
signal shift projected on predicted u is amplitude * Re(u^H s_true).
Weights depend only on predicted records. No absolute value repairs a wrong
phase or negative mean shift. Cross-node pooling remains an ideal reference.
"""
import json
from pathlib import Path
import sys
import numpy as np
from scipy.stats import norm

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from config.params import get_default_config
from tools.audit_uncertain_maneuver_ensemble import _run,_trajectory
from tools.audit_short_history_horizon import _flatten
from uav_isac.physical.active_information import receiver_local_temporal_covariance
from uav_isac.physical.bistatic_waveform import real_equivalent_complex_noise_variance
from uav_isac.physical.channel import compute_noise_power
from uav_isac.physical.waveform_evidence import MinimalOTFSWaveform,qpsk_dd_pilot


def receiver_shift(predicted,truth,rho=.9):
    ids,u,mean,epochs=_flatten(predicted)
    true_ids,s,amplitude,true_epochs=_flatten(truth)
    if not np.array_equal(ids,true_ids) or not np.array_equal(epochs,true_epochs):
        raise ValueError('prediction and truth must describe identical acquisitions')
    covariance=receiver_local_temporal_covariance(ids,u,epochs,correlation=rho).real
    weights=np.linalg.solve(covariance,mean)
    variance=float(weights@covariance@weights)
    if variance<=0: raise ValueError('zero detector variance')
    projected=amplitude*np.real(np.sum(u.conj()*s,axis=1))
    return float(weights@projected/np.sqrt(variance))


def audit(seeds=8,horizon=5,samples=100000):
    if seeds<2 or samples<10000: raise ValueError('insufficient audit samples')
    cfg=get_default_config()
    wf=MinimalOTFSWaveform(cfg.otfs.M,cfg.otfs.N,cfg.otfs.delta_f,
        real_equivalent_complex_noise_variance(compute_noise_power(cfg.channel.kT,cfg.otfs.B,cfg.channel.NF)))
    pilot=qpsk_dd_pilot(wf)
    initial=np.array([[-800.,0.,20.],[700.,250.,20.],[100.,850.,20.]])
    # Calibration, validation H0, and H1 have independent random streams.
    threshold=float(np.quantile(np.random.default_rng(410000).normal(size=samples),.999,method='higher'))
    h0=np.random.default_rng(410001).normal(size=samples)
    h1=np.random.default_rng(410002).normal(size=samples)
    rows=[]
    for seed in range(seeds):
        positions,velocities,measured=_trajectory(np.random.default_rng(310000+seed),horizon)
        predicted,truth,_=_run(initial,positions,velocities,measured,wf,cfg,pilot,False,True)
        shift=receiver_shift(predicted,truth)
        oracle=receiver_shift(truth,truth)
        rows.append(dict(seed=seed,prediction_matched_shift=shift,
            prediction_matched_pd=float(np.mean(h1+shift>threshold)),
            prediction_matched_pd_theory=float(norm.sf(threshold-shift)),
            oracle_pd_theory=float(norm.sf(threshold-oracle))))
    return dict(rows=rows,threshold=threshold,held_out_pfa=float(np.mean(h0>threshold)),
        samples_per_split=samples,seeds=seeds,
        mean_receiver_pd=float(np.mean([r['prediction_matched_pd'] for r in rows])),
        worst_receiver_pd=min(r['prediction_matched_pd'] for r in rows),
        mean_oracle_pd=float(np.mean([r['oracle_pd_theory'] for r in rows])),
        deployable_detection_certified=False,
        limitations=['centralized cross-node pooling with no communication accounting',
            'proper Gaussian known noise kernel; phase-coherent signal assumption',
            'exact scalar projection sampling, not raw RF Monte Carlo',
            'one fixed threshold; empirical PFA does not certify population PFA'])


if __name__=='__main__': print(json.dumps(audit(),indent=2,allow_nan=False))
