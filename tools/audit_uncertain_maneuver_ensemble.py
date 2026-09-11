"""Paired seeded maneuver ensemble for point versus max-min active sensing."""
import json
from pathlib import Path
import sys
import numpy as np
from scipy.stats import norm

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from config.params import get_default_config
from tools.audit_active_history_sequence import _actions
from tools.audit_active_geometry_survival import _observation
from tools.audit_short_history_horizon import _increment,_total_information
from uav_isac.physical.active_information import (generate_single_tx_roles,
    robust_hypothesis_maxmin_selection)
from uav_isac.physical.bistatic_waveform import real_equivalent_complex_noise_variance
from uav_isac.physical.channel import compute_noise_power
from uav_isac.physical.waveform_evidence import MinimalOTFSWaveform,qpsk_dd_pilot


def _trajectory(rng,horizon):
    velocity=np.r_[np.array([12.,6.])+rng.normal(0.,3.,2),0.]
    acceleration=np.r_[rng.normal(0.,5.,2),0.]
    position=np.zeros(3); positions=[]; velocities=[]
    for epoch in range(horizon):
        if epoch==2: velocity=velocity+.8*acceleration
        positions.append(position.copy()); velocities.append(velocity.copy())
        position=position+.8*velocity
    measured=velocities[0]+np.r_[rng.normal(0.,6.,2),0.]
    return positions,velocities,measured


def _model_record(action,role,target,velocity,waveform,cfg,pilot):
    receivers,templates,mean=_observation(action['position'],role,waveform,cfg,pilot,
        target_position=target,target_velocity=velocity)
    return dict(action,receivers=receivers,templates=templates,mean=mean,raw=float(mean@mean))


def _run(initial,truth_positions,truth_velocities,measured_velocity,
        waveform,cfg,pilot,robust,return_histories=False):
    position=initial.copy(); truth_history=[]; actions=[]; rho=.9
    offsets=(np.array([0.,0.,0.]),np.array([6.,0.,0.]),np.array([-6.,0.,0.]),
             np.array([0.,6.,0.]),np.array([0.,-6.,0.])) if robust else (np.zeros(3),)
    model_histories=[[] for _ in offsets]
    for epoch,(true_target,true_velocity) in enumerate(zip(truth_positions,truth_velocities)):
        nominal_target=measured_velocity*(.8*epoch)
        bank=_actions(position,waveform,cfg,pilot,nominal_target,measured_velocity)
        roles=generate_single_tx_roles(len(position)); values=np.empty((len(bank),len(offsets),1))
        modeled=[[None]*len(offsets) for _ in bank]
        for action_index,action in enumerate(bank):
            role=roles[action['role_index']]
            for hypothesis,offset in enumerate(offsets):
                velocity=measured_velocity+offset; target=velocity*(.8*epoch)
                record=_model_record(action,role,target,velocity,waveform,cfg,pilot)
                modeled[action_index][hypothesis]=record
                values[action_index,hypothesis,0]=_increment(record,model_histories[hypothesis],rho)
        feasible=np.array([x['feasible'] for x in bank])
        if robust:
            chosen_index=robust_hypothesis_maxmin_selection(values,feasible)['selected_index']
        else:
            chosen_index=int(np.argmax(np.where(feasible,values[:,0,0],-np.inf)))
        proposed=bank[chosen_index]; role=roles[proposed['role_index']]
        truth=_model_record(proposed,role,true_target,true_velocity,waveform,cfg,pilot)
        truth_history.append(truth)
        for hypothesis in range(len(offsets)):
            model_histories[hypothesis].append(modeled[chosen_index][hypothesis])
        position=proposed['position']; actions.append((proposed['motion_index'],proposed['role_index']))
    if return_histories:
        return model_histories[0],truth_history,actions
    return _total_information(truth_history,rho),actions


def audit(seeds=12,horizon=5,bootstrap_draws=4000):
    if seeds<8 or horizon<2 or bootstrap_draws<1000: raise ValueError('ensemble is too small')
    cfg=get_default_config(); noise=compute_noise_power(cfg.channel.kT,cfg.otfs.B,cfg.channel.NF)
    waveform=MinimalOTFSWaveform(cfg.otfs.M,cfg.otfs.N,cfg.otfs.delta_f,
        real_equivalent_complex_noise_variance(noise)); pilot=qpsk_dd_pilot(waveform)
    initial=np.array([[-800.,0.,20.],[700.,250.,20.],[100.,850.,20.]])
    threshold=float(norm.ppf(.999)); rows=[]
    for seed in range(seeds):
        positions,velocities,measured=_trajectory(np.random.default_rng(310000+seed),horizon)
        point,point_actions=_run(initial,positions,velocities,measured,waveform,cfg,pilot,False)
        robust,robust_actions=_run(initial,positions,velocities,measured,waveform,cfg,pilot,True)
        point_pd=float(norm.sf(threshold-np.sqrt(point)))
        robust_pd=float(norm.sf(threshold-np.sqrt(robust)))
        rows.append(dict(seed=seed,point_pd=point_pd,robust_pd=robust_pd,
            paired_delta=robust_pd-point_pd,point_actions=point_actions,robust_actions=robust_actions))
    delta=np.array([x['paired_delta'] for x in rows]); rng=np.random.default_rng(320000)
    bootstrap=np.mean(delta[rng.integers(0,seeds,(bootstrap_draws,seeds))],axis=1)
    interval=np.quantile(bootstrap,[.025,.975])
    return dict(evidence_class='CENTRALIZED_ORACLE_GAUSSIAN_MANEUVER_REFERENCE',seeds=seeds,horizon=horizon,
        deployable_detection_certified=False,
        confidence_scope='paired bootstrap for mean model-PD difference; no worst-case population bound',
        rows=rows,point_worst_pd=min(x['point_pd'] for x in rows),
        robust_worst_pd=min(x['robust_pd'] for x in rows),mean_paired_delta=float(np.mean(delta)),
        paired_seed_bootstrap_ci95=interval.tolist(),robust_gain_supported=bool(interval[0]>0),
        robust_worst_passes_pd_gate=bool(min(x['robust_pd'] for x in rows)>.8),
        uncertainty_set='measured velocity plus {0,+/-6 m/s x,+/-6 m/s y}',
        pfa_design=.001,sensing_power_w=.15,communication_power_w=0.,
        limitations=['seeded random initial velocity, noisy velocity measurement, and one acceleration switch; no recursive tracker',
            'centralized pooling across receivers without transport accounting; not local history',
            'truth-matched final detector ignores receiver template and covariance estimation errors',
            'finite velocity points are not a confidence region or continuous robust guarantee',
            'velocity uncertainty set is fixed and not calibrated from a tracker',
            'known Gaussian receiver-local clutter covariance',
            'analytic detection probability; no no-ACK U2U path yet'])


if __name__=='__main__': print(json.dumps(audit(),indent=2,allow_nan=False))
