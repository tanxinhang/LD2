"""Physical geometry screening, frozen linear-energy detector, paired trials.

Outer geometry seeds define diagnostic cases; inner noise samples are not
independent geometry replicates. Cross-receiver pooling is an ideal reference.
"""
import json
import numpy as np
from config.params import get_default_config
from tools.audit_active_geometry_survival import _observation
from tools.audit_otfs_multiframe_fixed_geometry import interval
from uav_isac.physical.active_information import (
    generate_motion_candidates,check_physical_feasibility,receiver_local_temporal_covariance)
from uav_isac.physical.correlated_soft_evidence import (
    optimal_linear_soft_fusion,conditional_deflection_gradient)
from uav_isac.physical.noncoherent_information import energy_moment_model
from uav_isac.physical.bistatic_waveform import real_equivalent_complex_noise_variance
from uav_isac.physical.channel import compute_noise_power
from uav_isac.physical.waveform_evidence import MinimalOTFSWaveform,qpsk_dd_pilot
from uav_isac.physical.paired_detection import paired_difference_interval


def run(samples=50000,allow_role_switch=False,emit=True,robust_screen=False,geometry_seed=1010000):
    cfg=get_default_config()
    wf=MinimalOTFSWaveform(cfg.otfs.M,cfg.otfs.N,cfg.otfs.delta_f,
        real_equivalent_complex_noise_variance(compute_noise_power(cfg.channel.kT,cfg.otfs.B,cfg.channel.NF)))
    pilot=qpsk_dd_pilot(wf); roles=np.array([0,1,1]); history_frames=4; window=13
    results=[]
    for episode in range(4):
        rng=np.random.default_rng(geometry_seed+episode)
        initial=np.array([[-800.,0.,20.],[700.,250.,20.],[100.,850.,20.]])
        initial[:,:2]+=rng.uniform(-50,50,(3,2))
        def observation(p,role=roles):
            return _observation(p,role,wf,cfg,pilot,
                target_position=np.array([0.,30.,0.]),target_velocity=np.array([6.,0.,0.]))
        hid,ht,hm=observation(initial)
        candidates=[p for p in generate_motion_candidates(initial,[0,30,0],
            speed_limit_mps=25,frame_duration_s=.1) if check_physical_feasibility(p,initial,
            speed_limit_mps=25,frame_duration_s=.1,minimum_separation_m=20,
            lower_bound=np.array([-1000,-1000,20]),upper_bound=np.array([1000,1000,20]))]
        role_bank=[roles] if not allow_role_switch else [np.where(np.arange(3)==tx,0,1) for tx in range(3)]
        candidates=[(p,role_index) for role_index in range(len(role_bank)) for p in candidates]
        for rho in (0.,.6):
            def model(p,role_index=0):
                ids,t,m=observation(p,role_bank[role_index])
                all_t=np.concatenate([ht if frame<history_frames else t for frame in range(window)])
                all_ids=np.concatenate([hid if frame<history_frames else ids for frame in range(window)])
                epochs=np.repeat(np.arange(window),len(ids))
                mu=np.concatenate([hm if frame<history_frames else m for frame in range(window)])/np.sqrt(2)
                c=receiver_local_temporal_covariance(all_ids,all_t,epochs,correlation=rho)
                return mu,c,all_t
            gradients={}
            for role_index in range(len(role_bank)):
                mu,c,_=model(initial,role_index); dm=[]; dc=[]
                for node in range(3):
                    for axis in range(2):
                        p=initial.copy(); m=initial.copy(); p[node,axis]+=.01; m[node,axis]-=.01
                        mp,cp,_=model(p,role_index); mm,cm,_=model(m,role_index)
                        dm.append((mp-mm)/.02); dc.append((cp-cm)/.02)
                dm=np.asarray(dm); dc=np.asarray(dc)
                energy_model=energy_moment_model(mu,c,dm,dc)
                for label,moments in [('coherent', (np.sqrt(2)*mu,c.real,np.sqrt(2)*dm,dc.real)),
                                      ('energy',energy_model)]:
                    gradient=np.zeros(6)
                    for j in range(history_frames*2,window*2):
                        _,g=conditional_deflection_gradient(*moments,tuple(range(j)),j)
                        gradient+=g
                    gradients[label,role_index]=(optimal_linear_soft_fusion(*moments[:2])[0],gradient)
            models=[model(p,role_index) for p,role_index in candidates]
            energy_values=[]; coherent_values=[]; weights=[]
            for amplitude,cov,_ in models:
                value,w=optimal_linear_soft_fusion(abs(amplitude)**2,abs(cov)**2)
                energy_values.append(value); weights.append(w)
                coherent_values.append(optimal_linear_soft_fusion(np.sqrt(2)*amplitude,cov.real)[0])
            selections={'stay':0,'raw':int(np.argmax([np.sum(abs(m[0][8:])**2) for m in models]))}
            for name in ('coherent','energy'):
                predicted=[gradients[name,r][0]+gradients[name,r][1]@(p-initial)[:,:2].ravel() for p,r in candidates]
                shortlist=sorted(set([0]+list(np.argsort(-np.asarray(predicted),kind='stable')[:4])))
                values=coherent_values if name=='coherent' else energy_values
                selections[name]=int(max(shortlist,key=lambda j:values[j]))
            if robust_screen:
                # Declared support, not sampled truth or learned probabilities:
                # target y in [0,30,60] m; vx in [0,6,12] m/s. Screening uses
                # the actual energy-detector weights and template projection.
                # A signed shift avoids rewarding a wrong detector direction.
                hypotheses=[]
                for y in (0.,30.,60.):
                    for vx in (0.,6.,12.):
                        state=dict(target_position=np.array([0.,y,0.]),target_velocity=np.array([vx,0.,0.]))
                        _,htemplate,hamp=_observation(initial,roles,wf,cfg,pilot,**state)
                        hypotheses.append((state,htemplate,hamp))
                robust_values=[]
                for index,(position,role_index) in enumerate(candidates):
                    _,cov,templates=models[index]; values=[]
                    variance=float(weights[index]@(abs(cov)**2)@weights[index])
                    for state,htemplate,hamp in hypotheses:
                        _,ftemplate,famp=_observation(position,role_bank[role_index],wf,cfg,pilot,**state)
                        signal_templates=np.concatenate([htemplate if frame<4 else ftemplate for frame in range(window)])
                        amplitudes=np.concatenate([hamp if frame<4 else famp for frame in range(window)])/np.sqrt(2)
                        projected_snr=abs(amplitudes*np.sum(templates.conj()*signal_templates,axis=1))**2
                        values.append(float(weights[index]@projected_snr/np.sqrt(variance)))
                    robust_values.append(min(values))
                selections['robust_energy']=int(np.argmax(robust_values))
            # Receiver weights and actions use predictions above. Truth only
            # generates held-out alternative observations after selection.
            rng=np.random.default_rng(geometry_seed+10000+episode*100+int(rho*10))
            noise=[(rng.normal(size=(samples,26))+1j*rng.normal(size=(samples,26)))/np.sqrt(2) for _ in range(3)]
            decision={}; metrics={}
            for index in sorted(set(selections.values())):
                amplitude,cov,templates=models[index]; root=np.linalg.cholesky(cov)
                _,history_true,history_amplitude=_observation(initial,roles,wf,cfg,pilot)
                position,role_index=candidates[index]
                _,future_true,future_amplitude=_observation(position,role_bank[role_index],wf,cfg,pilot)
                true_templates=np.concatenate([history_true if frame<4 else future_true for frame in range(window)])
                true_amplitude=np.concatenate([history_amplitude if frame<4 else future_amplitude for frame in range(window)])/np.sqrt(2)
                phases=np.exp(1j*(np.repeat(np.arange(window),2)*1.7+np.tile([0.,.3],window)))
                signal=true_amplitude*np.sum(templates.conj()*true_templates,axis=1)*phases
                def score(n,signal=0):
                    return (abs(n@root.T+signal)**2-np.real(np.diag(cov)))@weights[index]
                threshold=float(np.quantile(score(noise[0]),.9995,method='higher'))
                null=score(noise[1])>threshold; alt=score(noise[2],signal)>threshold
                decision[index]=alt
                metrics[index]=dict(pd=float(alt.mean()),pfa=float(null.mean()),threshold=threshold,
                    pd_ci=interval(int(alt.sum()),samples,.05/(2*len(selections))),pfa_ci=interval(int(null.sum()),samples,.05/(2*len(selections))))
            result=dict(episode=episode,rho=rho,selections=selections,
                allow_role_switch=allow_role_switch,
                geometry_seed=geometry_seed,robust_screen=robust_screen,
                roles={name:candidates[index][1] for name,index in selections.items()},
                energy_exhaustive_index=int(np.argmax(energy_values)),
                methods={name:metrics[index] for name,index in selections.items()},
                energy_minus_raw=paired_difference_interval(decision[selections['energy']],decision[selections['raw']]),
                energy_minus_stay=paired_difference_interval(decision[selections['energy']],decision[0]),
                scope='13 acquisitions, one <=2.5m move; fixed 0.15W; ideal delivery; predicted weights; deterministic unknown phase; conditional diagnostic')
            if robust_screen:
                result['robust_minus_stay']=paired_difference_interval(decision[selections['robust_energy']],decision[0])
                result['robust_score_margin']=float(robust_values[selections['robust_energy']]-robust_values[0])
            results.append(result)
            if emit: print(json.dumps(result),flush=True)
    return results


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples',type=int,default=50000)
    parser.add_argument('--role-switch',action='store_true')
    parser.add_argument('--robust-screen',action='store_true')
    parser.add_argument('--geometry-seed',type=int,default=1010000)
    args=parser.parse_args()
    run(args.samples,args.role_switch,robust_screen=args.robust_screen,geometry_seed=args.geometry_seed)
