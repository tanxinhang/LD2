"""Two-epoch G0.5 audit: does delivered history change the next action?"""
import json
from pathlib import Path
import sys
import numpy as np
from scipy.stats import norm

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from config.params import get_default_config
from tools.audit_active_geometry_survival import _observation
from uav_isac.physical.active_information import (check_physical_feasibility,
    conditional_gaussian_information,generate_motion_candidates,
    generate_single_tx_roles,top_m_exact_maxmin_selection)
from uav_isac.physical.bistatic_waveform import real_equivalent_complex_noise_variance
from uav_isac.physical.channel import compute_noise_power
from uav_isac.physical.waveform_evidence import MinimalOTFSWaveform,qpsk_dd_pilot


def _actions(origin,waveform,cfg,pilot,target_position=None,target_velocity=None):
    target=np.array([0.,0.,0.]) if target_position is None else np.asarray(target_position,float)
    motions=generate_motion_candidates(origin,target,speed_limit_mps=25.,frame_duration_s=.8)
    roles=generate_single_tx_roles(len(origin)); records=[]
    for motion_index,position in enumerate(motions):
        feasible=check_physical_feasibility(position,origin,speed_limit_mps=25.,frame_duration_s=.8,
            minimum_separation_m=20.,lower_bound=np.array([-1000.,-1000.,20.]),
            upper_bound=np.array([1000.,1000.,20.]))
        for role_index,role in enumerate(roles):
            receivers,templates,mean=_observation(position,role,waveform,cfg,pilot,
                target_position=target,target_velocity=target_velocity)
            records.append(dict(motion_index=motion_index,role_index=role_index,
                position=position,receivers=receivers,templates=templates,mean=mean,
                raw=float(mean@mean),feasible=feasible))
    return records


def _conditional(record,history,rho):
    cross=np.zeros((len(record['receivers']),len(history['receivers'])),complex)
    for row,receiver in enumerate(record['receivers']):
        matches=np.flatnonzero(history['receivers']==receiver)
        if matches.size:
            column=int(matches[0])
            cross[row,column]=rho*np.vdot(record['templates'][row],history['templates'][column])
    return conditional_gaussian_information(record['mean'],history['mean'],
        np.eye(len(record['mean'])),np.eye(len(history['mean'])),cross).increment


def audit(samples=200000,seed=20260912):
    cfg=get_default_config(); noise=compute_noise_power(cfg.channel.kT,cfg.otfs.B,cfg.channel.NF)
    waveform=MinimalOTFSWaveform(cfg.otfs.M,cfg.otfs.N,cfg.otfs.delta_f,
        real_equivalent_complex_noise_variance(noise)); pilot=qpsk_dd_pilot(waveform)
    initial=np.array([[-800.,0.,20.],[700.,250.,20.],[100.,850.,20.]])
    first_bank=_actions(initial,waveform,cfg,pilot)
    first_index=max(range(len(first_bank)),key=lambda i:(first_bank[i]['feasible']*first_bank[i]['raw'],-i))
    first=first_bank[first_index]
    second_bank=_actions(first['position'],waveform,cfg,pilot); rho=.9
    increments=np.array([_conditional(record,first,rho) for record in second_bank])
    raw=np.array([record['raw'] for record in second_bank]); feasible=np.array([record['feasible'] for record in second_bank])
    history_d=first['raw']; joint=history_d+increments
    instant_index=int(np.argmax(np.where(feasible,raw,-np.inf)))
    selected=top_m_exact_maxmin_selection(increments[:,None],joint[:,None],feasible,top_m=min(12,len(joint)))
    history_index=selected['selected_index']
    threshold=float(norm.ppf(.999)); rng=np.random.default_rng(seed)
    def evaluate(index):
        d=float(joint[index]); common=rng.normal(size=samples)
        # Paired H0/H1 innovations make method differences lower variance.
        return dict(information=d,pd=float(np.mean(common+np.sqrt(d)>threshold)),
            pd_theory=float(norm.sf(threshold-np.sqrt(d))),increment=float(increments[index]),
            action=(second_bank[index]['motion_index'],second_bank[index]['role_index']))
    instant=evaluate(instant_index); history=evaluate(history_index)
    delta=history['pd_theory']-instant['pd_theory']
    return dict(evidence_class='TWO_EPOCH_HISTORY_CONDITIONED_ACTIVE_SENSING',samples=samples,
        first_action=(first['motion_index'],first['role_index']),first_information=history_d,
        instantaneous=instant,history_conditioned=history,theoretical_pd_delta=float(delta),
        action_changed=instant_index!=history_index,history_gain_survives=delta>1e-6,
        sensing_power_w=.15,communication_power_w=0.,pfa_design=.001,
        clutter_memory=rho,top_m=selected['screened_indices'].tolist(),
        limitations=['static target and known temporal covariance',
            'history is locally available; no U2U message is needed or charged',
            'two slow epochs only; no long-horizon policy claim',
            'coherent Gaussian detector; P_D>0.8 is not presumed'])


if __name__=='__main__': print(json.dumps(audit(),indent=2,allow_nan=False))
