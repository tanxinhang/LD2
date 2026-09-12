"""Frozen predicted projections with independent acquisition phases.

Full-vector whitened energy is a GLRT for an unrestricted complex mean vector,
not a Bayesian fixed-amplitude phase marginal. Its extra degrees of freedom
are paid through an independently calibrated threshold.
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


def whitened_signal(predicted,truth):
    ids,u,m,t=_flatten(predicted); tid,s,amplitude,tt=_flatten(truth)
    if not np.array_equal(ids,tid) or not np.array_equal(t,tt):
        raise ValueError('acquisition identity mismatch')
    c=receiver_local_temporal_covariance(ids,u,t,correlation=.9)
    lower=np.linalg.cholesky(c)
    direction=np.linalg.solve(lower,m)
    direction=direction/np.linalg.norm(direction)
    signal=amplitude*np.sum(u.conj()*s,axis=1)/np.sqrt(2)
    return lower,direction,signal


def statistics(y,direction):
    return np.abs(y@direction.conj())**2,np.sum(np.abs(y)**2,axis=1)


def audit(samples=100000,seeds=8):
    cfg=get_default_config()
    wf=MinimalOTFSWaveform(cfg.otfs.M,cfg.otfs.N,cfg.otfs.delta_f,
        real_equivalent_complex_noise_variance(compute_noise_power(cfg.channel.kT,cfg.otfs.B,cfg.channel.NF)))
    pilot=qpsk_dd_pilot(wf); initial=np.array([[-800.,0.,20.],[700.,250.,20.],[100.,850.,20.]])
    rows=[]
    for seed in range(seeds):
        p,v,measured=_trajectory(np.random.default_rng(310000+seed),5)
        predicted,truth,_=_run(initial,p,v,measured,wf,cfg,pilot,False,True)
        lower,direction,signal=whitened_signal(predicted,truth)
        def noise(stream):
            rng=np.random.default_rng(430000+seed*10+stream)
            return (rng.normal(size=(samples,len(signal)))+1j*rng.normal(size=(samples,len(signal))))/np.sqrt(2)
        calibration=statistics(noise(0),direction)
        thresholds=[float(np.quantile(x,.999,method='higher')) for x in calibration]
        validation=statistics(noise(1),direction)
        base=noise(2); rng=np.random.default_rng(430003+seed*10)
        for mode in ('common','independent'):
            phases=np.exp(2j*np.pi*rng.random((samples,1 if mode=='common' else len(signal))))
            means=np.linalg.solve(lower,(phases*signal).T).T
            scores=statistics(base+means,direction)
            rows.append(dict(seed=seed,phase_mode=mode,
                scalar_pd=float(np.mean(scores[0]>thresholds[0])),
                vector_pd=float(np.mean(scores[1]>thresholds[1])),
                scalar_pfa=float(np.mean(validation[0]>thresholds[0])),
                vector_pfa=float(np.mean(validation[1]>thresholds[1]))))
    summary={}
    for mode in ('common','independent'):
        subset=[r for r in rows if r['phase_mode']==mode]
        delta=np.array([r['vector_pd']-r['scalar_pd'] for r in subset])
        draws=np.mean(delta[np.random.default_rng(440000).integers(0,seeds,(4000,seeds))],axis=1)
        summary[mode]={k:float(np.mean([r[k] for r in subset])) for k in ('scalar_pd','vector_pd','scalar_pfa','vector_pfa')}
        summary[mode]['paired_seed_delta_ci95']=np.quantile(draws,[.025,.975]).tolist()
    return dict(rows=rows,summary=summary,samples_per_split=samples,
        deployable_detection_certified=False,
        limitations=['centralized predicted projections, known noise kernel, no transport',
            'vector GLRT permits arbitrary amplitudes; not exact independent-phase marginal likelihood',
            'phases drawn before whitening; whitening does not imply independent signal phases',
            'single terminal decision; empirical false alarm is not a confidence certificate'])


if __name__=='__main__':
    result=audit(); print(json.dumps(result['summary'],indent=2,allow_nan=False))
