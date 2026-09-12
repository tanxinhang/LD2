"""Frozen-schedule complex scalar receiver phase-loss decomposition.

The scalar |z|^2 GLRT eliminates one common unknown complex amplitude.
It does not eliminate independent phases across acquisitions, nor repair DD
template mismatch. The declared proper Gaussian projected-noise kernel is
known. Cross-receiver evidence is centrally pooled without transport costs.
"""
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from config.params import get_default_config
from tools.audit_uncertain_maneuver_ensemble import _run,_trajectory
from tools.audit_short_history_horizon import _flatten
from uav_isac.physical.active_information import receiver_local_temporal_covariance
from uav_isac.physical.bistatic_waveform import real_equivalent_complex_noise_variance
from uav_isac.physical.channel import compute_noise_power
from uav_isac.physical.waveform_evidence import MinimalOTFSWaveform,qpsk_dd_pilot


def complex_receiver_mean(predicted,truth,rho=.9):
    ids,u,m,epochs=_flatten(predicted)
    tid,s,amplitude,te=_flatten(truth)
    if not np.array_equal(ids,tid) or not np.array_equal(epochs,te):
        raise ValueError('acquisition identities differ')
    c=receiver_local_temporal_covariance(ids,u,epochs,correlation=rho)
    w=np.linalg.solve(c,m)
    variance=float(np.real(np.vdot(w,c@w)))
    if variance<=0: raise ValueError('nonpositive variance')
    overlap=np.sum(u.conj()*s,axis=1)
    # Existing mean is a real standardized shift; CN(0,1) uses 1/sqrt(2).
    mean=np.vdot(w,amplitude*overlap)/np.sqrt(2*variance)
    return complex(mean),overlap


def audit(samples=100000,seeds=8):
    if samples<10000 or seeds<1: raise ValueError('invalid sample count')
    cfg=get_default_config()
    wf=MinimalOTFSWaveform(cfg.otfs.M,cfg.otfs.N,cfg.otfs.delta_f,
        real_equivalent_complex_noise_variance(compute_noise_power(cfg.channel.kT,cfg.otfs.B,cfg.channel.NF)))
    pilot=qpsk_dd_pilot(wf)
    initial=np.array([[-800.,0.,20.],[700.,250.,20.],[100.,850.,20.]])
    def noise(seed):
        rng=np.random.default_rng(seed)
        return (rng.normal(size=samples)+1j*rng.normal(size=samples))/np.sqrt(2)
    calibration=noise(420000); validation=noise(420001)
    tc=float(np.quantile(np.sqrt(2)*calibration.real,.999,method='higher'))
    te=float(np.quantile(abs(calibration)**2,.999,method='higher'))
    rows=[]
    for seed in range(seeds):
        p,v,measured=_trajectory(np.random.default_rng(310000+seed),5)
        predicted,truth,_=_run(initial,p,v,measured,wf,cfg,pilot,False,True)
        mean,overlap=complex_receiver_mean(predicted,truth)
        oracle,_=complex_receiver_mean(truth,truth)
        h1=noise(421000+seed)
        rows.append(dict(seed=seed,
            predicted_coherent_pd=float(np.mean(np.sqrt(2)*(h1+mean).real>tc)),
            predicted_energy_pd=float(np.mean(abs(h1+mean)**2>te)),
            truth_coherent_pd=float(np.mean(np.sqrt(2)*(h1+oracle).real>tc)),
            truth_energy_pd=float(np.mean(abs(h1+oracle)**2>te)),
            projected_mean_abs=abs(mean),projected_mean_phase=float(np.angle(mean)),
            overlap_abs=abs(overlap).tolist(),overlap_phase=np.angle(overlap).tolist(),
            overlap_real=overlap.real.tolist()))
    keys=('predicted_coherent_pd','predicted_energy_pd','truth_coherent_pd','truth_energy_pd')
    return dict(rows=rows,mean_pd={k:float(np.mean([r[k] for r in rows])) for k in keys},
        coherent_pfa=float(np.mean(np.sqrt(2)*validation.real>tc)),
        energy_pfa=float(np.mean(abs(validation)**2>te)),
        coherent_threshold=tc,energy_threshold=te,samples_per_split=samples,
        deployable_detection_certified=False,
        limitations=['single final decision at five epochs; no early stopping',
            'scalar common-amplitude GLRT, not independent receiver-phase marginalization',
            'same schedules and powers for all receiver comparisons',
            'known proper Gaussian covariance and centralized pooling; no transport',
            'complex-coherent baseline differs from earlier real-projection receiver'])


if __name__=='__main__': print(json.dumps(audit(),indent=2,allow_nan=False))
