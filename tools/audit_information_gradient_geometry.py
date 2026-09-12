"""Model-only physical geometry screening audit; no detection certification."""
import json
import numpy as np
from config.params import get_default_config
from tools.audit_active_geometry_survival import _observation
from uav_isac.physical.active_information import generate_motion_candidates,check_physical_feasibility
from uav_isac.physical.correlated_soft_evidence import optimal_linear_soft_fusion,conditional_deflection_gradient
from uav_isac.physical.bistatic_waveform import real_equivalent_complex_noise_variance
from uav_isac.physical.channel import compute_noise_power
from uav_isac.physical.waveform_evidence import MinimalOTFSWaveform,qpsk_dd_pilot


def run():
    cfg=get_default_config()
    wf=MinimalOTFSWaveform(cfg.otfs.M,cfg.otfs.N,cfg.otfs.delta_f,
        real_equivalent_complex_noise_variance(compute_noise_power(cfg.channel.kT,cfg.otfs.B,cfg.channel.NF)))
    pilot=qpsk_dd_pilot(wf); rng=np.random.default_rng(1008001)
    rows=[]
    for episode in range(8):
        initial=np.array([[-800.,0.,20.],[700.,250.,20.],[100.,850.,20.]])
        initial[:,:2]+=rng.uniform(-40,40,size=(3,2))
        roles=np.array([0,1,1])
        _,history,history_mean=_observation(initial,roles,wf,cfg,pilot)
        for rho in (0.,.6):
            def model(positions):
                _,templates,mean=_observation(positions,roles,wf,cfg,pilot)
                # Independent receivers; declared persistent same-receiver
                # clutter. Real projection covariance under coherent model.
                cross=np.diag(rho*np.real(np.sum(templates.conj()*history,axis=1)))
                return np.r_[history_mean,mean],np.block([[np.eye(2),cross.T],[cross,np.eye(2)]])
            mean,cov=model(initial); dm=[]; ds=[]; h=.01
            for node in range(3):
                for axis in range(2):
                    plus=initial.copy(); minus=initial.copy()
                    plus[node,axis]+=h; minus[node,axis]-=h
                    mp,sp=model(plus); mm,sm=model(minus)
                    dm.append((mp-mm)/(2*h)); ds.append((sp-sm)/(2*h))
            # Chain rule for two new observations; dual price is 1 for the
            # single target. No fabricated multi-target or resource prices.
            gradient=np.zeros(6)
            for selected,candidate in (((0,1),2),((0,1,2),3)):
                _,g=conditional_deflection_gradient(mean,cov,dm,ds,selected,candidate)
                gradient+=g
            candidates=generate_motion_candidates(initial,[0,0,0],speed_limit_mps=25,frame_duration_s=.1)
            candidates=[p for p in candidates if check_physical_feasibility(p,initial,
                speed_limit_mps=25,frame_duration_s=.1,minimum_separation_m=20,
                lower_bound=np.array([-1000,-1000,20]),upper_bound=np.array([1000,1000,20]))]
            values=np.array([optimal_linear_soft_fusion(*model(p))[0] for p in candidates])
            raw=np.array([float(model(p)[0][2:]@model(p)[0][2:]) for p in candidates])
            predicted=np.array([gradient@(p-initial)[:,:2].ravel() for p in candidates])
            shortlist=sorted(set([0]+list(np.argsort(-predicted,kind='stable')[:4])))
            chosen=max(shortlist,key=lambda j:values[j])
            rows.append(dict(episode=episode,rho=rho,candidates=len(candidates),
                screened=len(shortlist),baseline=float(values[0]),selected=float(values[chosen]),
                exhaustive=float(values.max()),selected_index=int(chosen),
                raw_information_index=int(np.argmax(raw)),
                move_max_m=float(np.linalg.norm(candidates[chosen]-initial,axis=1).max())))
    print(json.dumps(dict(rows=rows,nondecreasing=all(r['selected']>=r['baseline'] for r in rows),
        exhaustive_matches=sum(abs(r['selected']-r['exhaustive'])<1e-10 for r in rows),
        raw_choice_matches=sum(r['selected_index']==r['raw_information_index'] for r in rows),
        scope='coherent Gaussian model screening only; declared rho; 0.15W research sensing; no online action, actual delivery, motion-energy or PD claim'),indent=2))


if __name__=='__main__': run()
