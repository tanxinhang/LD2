"""G0 audit for history-conditioned active geometry with frozen OTFS/power."""
import json
from pathlib import Path
import sys
import numpy as np
from scipy.stats import norm

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from config.params import get_default_config
from uav_isac.physical.active_information import (check_physical_feasibility,
    conditional_gaussian_information,generate_motion_candidates,
    top_m_exact_maxmin_selection)
from uav_isac.physical.bistatic_waveform import (bistatic_geometry_to_waveform,
    ideal_coherent_h0_deflection,real_equivalent_complex_noise_variance)
from uav_isac.physical.channel import compute_noise_power
from uav_isac.physical.waveform_evidence import MinimalOTFSWaveform,dd_path_response,qpsk_dd_pilot


def _observation(positions,roles,waveform,cfg,pilot,power=.15,
        target_position=None,target_velocity=None):
    roles=np.asarray(roles,int); tx=int(np.flatnonzero(roles==0)[0])
    receivers=np.flatnonzero(roles==1)
    allocation=np.zeros((len(positions),1)); allocation[tx,0]=power
    target=np.array([0.,0.,0.]) if target_position is None else np.asarray(target_position,float)
    velocity=np.zeros(3) if target_velocity is None else np.asarray(target_velocity,float)
    if target.shape!=(3,) or velocity.shape!=(3,) or np.any(~np.isfinite(target)) or np.any(~np.isfinite(velocity)):
        raise ValueError('target state must be finite and three-dimensional')
    physical=bistatic_geometry_to_waveform(positions,np.zeros_like(positions),
        target[None,:],velocity[None,:],roles,allocation,
        carrier_hz=cfg.otfs.fc,waveform=waveform,
        rcs_m2=cfg.target.rcs,tx_gain_dbi=cfg.otfs.g_tx_dBi,rx_gain_dbi=cfg.otfs.g_rx_dBi)
    templates=[]; means=[]
    for receiver in receivers:
        template=dd_path_response(pilot,delay_bin=physical.delay_bin[tx,receiver,0],
            doppler_bin=physical.doppler_bin[tx,receiver,0]).ravel()
        templates.append(template/np.linalg.norm(template))
        d=float(ideal_coherent_h0_deflection(physical.received_target_amplitude[tx,receiver,0],
            waveform=waveform,complex_noise_variance=waveform.noise_variance))
        means.append(np.sqrt(d))
    return receivers,np.stack(templates),np.asarray(means)


def audit(samples=200000,seed=20260911):
    cfg=get_default_config(); noise=compute_noise_power(cfg.channel.kT,cfg.otfs.B,cfg.channel.NF)
    waveform=MinimalOTFSWaveform(cfg.otfs.M,cfg.otfs.N,cfg.otfs.delta_f,
        real_equivalent_complex_noise_variance(noise)); pilot=qpsk_dd_pilot(waveform)
    # Near-800 m but deliberately non-symmetric: the old equidistant triangle
    # makes all zero-velocity bistatic DD paths reciprocal and role-degenerate.
    initial=np.array([[-800.,0.,20.],[700.,250.,20.],[100.,850.,20.]])
    candidates=generate_motion_candidates(initial,[0.,0.,0.],speed_limit_mps=25.,frame_duration_s=.8)
    feasible=np.array([check_physical_feasibility(x,initial,speed_limit_mps=25.,frame_duration_s=.8,
        minimum_separation_m=20.,lower_bound=np.array([-1000.,-1000.,20.]),
        upper_bound=np.array([1000.,1000.,20.])) for x in candidates])
    roles=(np.array([0,1,1]),np.array([1,0,1]),np.array([1,1,0]))
    history_receivers,history_templates,history_mean=_observation(initial,roles[0],waveform,cfg,pilot)
    rho=.9; raw=[]; conditional=[]; joint=[]; action_map=[]
    history_cov=np.eye(2); history_d=float(history_mean@history_mean)
    for motion_index,positions in enumerate(candidates):
      for role_index,role in enumerate(roles):
        receivers,templates,mean=_observation(positions,role,waveform,cfg,pilot)
        # Persistent receiver-local clutter correlates only statistics formed at
        # the same physical receiver. OTFS-template overlap determines its
        # projection after motion; this is not an artificial geometry weight.
        cross=np.zeros((2,2),complex)
        for a,receiver in enumerate(receivers):
            match=np.flatnonzero(history_receivers==receiver)
            if match.size:
                h=int(match[0]); cross[a,h]=rho*np.vdot(templates[a],history_templates[h])
        info=conditional_gaussian_information(mean,history_mean,np.eye(2),history_cov,cross)
        raw.append(float(mean@mean)); conditional.append(info.increment); joint.append(history_d+info.increment)
        action_map.append((motion_index,role_index))
    raw=np.asarray(raw); conditional=np.asarray(conditional); joint=np.asarray(joint)
    # One target makes max-min equal its scalar information. Top-M exact still
    # exercises the intended screening/exact boundary.
    action_feasible=np.repeat(feasible,3)
    selection=top_m_exact_maxmin_selection(conditional[:,None],joint[:,None],action_feasible,top_m=min(12,len(raw)))
    fixed=0
    range_candidates=np.array([r==0 for _,r in action_map])
    range_choice=int(np.argmax(np.where(action_feasible&range_candidates,raw,-np.inf)))
    instantaneous_choice=int(np.argmax(np.where(action_feasible,raw,-np.inf)))
    history_choice=int(selection['selected_index'])
    threshold=float(norm.ppf(.999)); rng=np.random.default_rng(seed)
    def held_out(index):
        d=joint[index]; h0=rng.normal(size=samples); h1=rng.normal(np.sqrt(d),1.,size=samples)
        return dict(information=float(d),pfa=float(np.mean(h0>threshold)),
            pd=float(np.mean(h1>threshold)),pd_theory=float(norm.sf(threshold-np.sqrt(d))))
    result=dict(evidence_class='G0_FROZEN_POWER_ACTIVE_GEOMETRY',samples=samples,
        fixed=held_out(fixed),range_only=held_out(range_choice),instantaneous_information=held_out(instantaneous_choice),
        history_conditioned=held_out(history_choice),fixed_action=action_map[fixed],range_action=action_map[range_choice],
        instantaneous_action=action_map[instantaneous_choice],history_action=action_map[history_choice],
        candidate_count=len(action_map),feasible_count=int(np.sum(action_feasible)),top_m=selection['screened_indices'].tolist(),
        pfa_design=.001,sensing_power_w=.15,communication_power_w=0.,history_clutter_correlation=rho,
        limitations=['coherent Gaussian G0; unknown phase and non-Gaussian clutter not certified',
            'single target, finite Tx roles and one slow-geometry interval',
            'cross-frame clutter correlation is declared, not estimated online',
            'survival gate, not deployed controller or 800m P_D guarantee'])
    # Do not turn independent Monte-Carlo jitter into an algorithmic claim.
    # The analytic Gaussian probability is available in this gate and is the
    # deterministic survival comparison; later non-Gaussian gates need paired
    # trials and a confidence interval instead.
    result['active_beats_fixed']=(result['history_conditioned']['pd_theory']
        > result['fixed']['pd_theory']+1e-6)
    result['history_beats_range']=(result['history_conditioned']['pd_theory']
        > result['range_only']['pd_theory']+1e-6)
    result['history_beats_instantaneous']=(result['history_conditioned']['pd_theory']
        > result['instantaneous_information']['pd_theory']+1e-6)
    result['conclusion']=('geometry_role_survives_history_novelty_does_not'
        if result['active_beats_fixed'] and not result['history_beats_instantaneous']
        else 'gate_outcome_requires_review')
    return result


if __name__=='__main__': print(json.dumps(audit(),indent=2,allow_nan=False))
