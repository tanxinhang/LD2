"""Correlated template-bank search and temporal-whitening receiver ablation.

No new acquisitions: all templates see the same physical projected noise.
This is a frozen-geometry, ideal-pooling diagnostic with known noise rho.
"""
import json
import numpy as np
from config.params import get_default_config
from tools.audit_active_geometry_survival import _observation
from tools.audit_otfs_multiframe_fixed_geometry import interval
from uav_isac.physical.bistatic_waveform import real_equivalent_complex_noise_variance
from uav_isac.physical.channel import compute_noise_power
from uav_isac.physical.waveform_evidence import MinimalOTFSWaveform,qpsk_dd_pilot
from uav_isac.physical.temporal_whitening import whiten_ar1
from uav_isac.physical.paired_detection import paired_difference_interval


def run(samples=100000,emit=True,geometry_seed=1031000,stress=False):
    cfg=get_default_config()
    wf=MinimalOTFSWaveform(cfg.otfs.M,cfg.otfs.N,cfg.otfs.delta_f,
        real_equivalent_complex_noise_variance(compute_noise_power(cfg.channel.kT,cfg.otfs.B,cfg.channel.NF)))
    pilot=qpsk_dd_pilot(wf); roles=np.array([0,1,1]); window=13; results=[]
    for episode in range(4):
        rng=np.random.default_rng(geometry_seed+episode)
        positions=np.array([[-800.,0.,20.],[700.,250.,20.],[100.,850.,20.]])
        positions[:,:2]+=rng.uniform(-50,50,(3,2))
        templates=[]; amplitudes=[]
        for y in (0.,30.,60.):
            for vx in (0.,6.,12.):
                _,t,a=_observation(positions,roles,wf,cfg,pilot,
                    target_position=np.array([0.,y,0.]),target_velocity=np.array([vx,0.,0.]))
                templates.append(t); amplitudes.append(a/np.sqrt(2))
        templates=np.asarray(templates).transpose(1,0,2); snr=np.asarray(amplitudes).T**2
        roots=[]
        for t in templates:
            gram=t.conj()@t.T; values,vectors=np.linalg.eigh(gram)
            if np.min(values)<-1e-10: raise RuntimeError('invalid physical template Gram')
            roots.append(vectors*np.sqrt(np.maximum(values,0))[None,:])
        truth_kwargs=(dict(target_position=np.array([0.,15.,0.]),
            target_velocity=np.array([3.,0.,0.])) if stress else {})
        _,true_t,true_a=_observation(positions,roles,wf,cfg,pilot,**truth_kwargs)
        true_projected=np.sum(templates.conj()*true_t[:,None,:],axis=2)*true_a[:,None]/np.sqrt(2)
        for rho in (0.,.6):
            # Independent target-free training; never estimate noise from H1 truth.
            rho_hat=rho
            if stress:
                train_rng=np.random.default_rng(geometry_seed+20000+episode*100+int(rho*10))
                train=(train_rng.normal(size=(32,window))+1j*train_rng.normal(size=(32,window)))/np.sqrt(2)
                for t in range(1,window): train[:,t]=rho*train[:,t-1]+np.sqrt(1-rho*rho)*train[:,t]
                rho_hat=float(np.clip(np.real(np.sum(train[:,1:]*train[:,:-1].conj()))/np.sum(abs(train[:,:-1])**2),0,.99))
            time=np.arange(window); sigma_energy=rho_hat**(2*abs(time[:,None]-time[None,:]))
            temporal_weight=np.linalg.solve(sigma_energy,np.ones(window))
            sd=np.sqrt(temporal_weight.sum()*np.sum(snr**2,axis=0))
            def scores(seed,phase):
                rng=np.random.default_rng(seed); collected={}
                for start in range(0,samples,2048):
                    count=min(2048,samples-start)
                    draws=rng.normal(size=(count,window,2,9,2))
                    white=(draws[...,0]+1j*draws[...,1])/np.sqrt(2)
                    for receiver in range(2):
                        white[:,:,receiver]=white[:,:,receiver]@roots[receiver].T
                    z=white.copy()
                    for t in range(1,window): z[:,t]=rho*z[:,t-1]+np.sqrt(1-rho*rho)*white[:,t]
                    if phase is not None:
                        phase_path={'rotating':1.7*time,'constant':np.zeros(window),
                            'slow':.2*time,'chirp':.05*time*time}[phase]
                        angles=phase_path[:,None]+np.array([0.,.3])[None,:]
                        z+=true_projected[None,None,:,:]*np.exp(1j*angles)[None,:,:,None]
                    energy=abs(z)**2
                    linear=np.einsum('ntrb,t,rb->nb',energy-1,temporal_weight,snr)/sd
                    whitened=np.sum(abs(whiten_ar1(z,rho_hat))**2,axis=(1,2))
                    methods=dict(single_energy=linear[:,4],bank_energy=linear.max(axis=1),
                        single_whitened=whitened[:,4],bank_whitened=whitened.max(axis=1),
                        hybrid_bank=np.maximum(linear.max(axis=1),(whitened.max(axis=1)-2*window)/np.sqrt(2*window)),
                        **({'reference_template_energy':linear[:,0]} if stress else {'oracle_template_energy':linear[:,0]}))
                    for name,value in methods.items(): collected.setdefault(name,[]).append(value)
                return {k:np.concatenate(v) for k,v in collected.items()}
            seed=geometry_seed+10000+episode*100+int(rho*10)
            cal=scores(seed,None); null=scores(seed+1,None)
            thresholds={name:float(np.quantile(value,.9995,method='higher')) for name,value in cal.items()}
            for phase in (('constant','slow','chirp','rotating') if stress else ('rotating','constant')):
                alternative=scores(seed+2,phase); rows={}; decisions={}
                for name,value in alternative.items():
                    decisions[name]=value>thresholds[name]
                    tp=int(decisions[name].sum()); fp=int(np.sum(null[name]>thresholds[name]))
                    rows[name]=dict(pd=tp/samples,pfa=fp/samples,threshold=thresholds[name],
                        pd_ci=interval(tp,samples,.05/(2*len(alternative))),pfa_ci=interval(fp,samples,.05/(2*len(alternative))))
                result=dict(episode=episode,rho=rho,rho_hat=rho_hat,stress=stress,phase=phase,geometry_seed=geometry_seed,methods=rows,
                    bank_minus_single=paired_difference_interval(decisions['bank_energy'],decisions['single_energy']),
                    whitened_bank_minus_single=paired_difference_interval(decisions['bank_whitened'],decisions['single_energy']),
                    hybrid_minus_single=paired_difference_interval(decisions['hybrid_bank'],decisions['single_energy']),
                    scope='same 13 blocks/0.15W; correlated template projections; stationary noise; ideal pooling; training cost excluded; no transport/CPU savings claim')
                results.append(result)
                if emit: print(json.dumps(result),flush=True)
    return results


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples',type=int,default=100000)
    parser.add_argument('--geometry-seed',type=int,default=1031000)
    parser.add_argument('--stress',action='store_true')
    args=parser.parse_args()
    run(args.samples,geometry_seed=args.geometry_seed,stress=args.stress)
