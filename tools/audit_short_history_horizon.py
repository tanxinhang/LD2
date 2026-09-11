"""Short-window active sensing audit with complete receiver-local history."""
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
from uav_isac.physical.active_information import (conditional_gaussian_information,
    generate_single_tx_roles,receiver_local_temporal_covariance)
from uav_isac.physical.bistatic_waveform import real_equivalent_complex_noise_variance
from uav_isac.physical.channel import compute_noise_power
from uav_isac.physical.waveform_evidence import MinimalOTFSWaveform,qpsk_dd_pilot


def _flatten(records):
    receivers=np.concatenate([x['receivers'] for x in records])
    templates=np.concatenate([x['templates'] for x in records])
    means=np.concatenate([x['mean'] for x in records])
    epochs=np.concatenate([np.full(len(x['mean']),epoch) for epoch,x in enumerate(records)])
    return receivers,templates,means,epochs


def _total_information(records,rho):
    receivers,templates,means,epochs=_flatten(records)
    covariance=receiver_local_temporal_covariance(receivers,templates,epochs,correlation=rho)
    inverse=np.linalg.pinv(covariance,hermitian=True,rcond=1e-10)
    return float(np.real(means@inverse@means))


def _increment(record,history,rho):
    if not history: return record['raw']
    receivers,templates,means,epochs=_flatten(history)
    history_cov=receiver_local_temporal_covariance(receivers,templates,epochs,correlation=rho)
    current_epoch=len(history); cross=np.zeros((len(record['mean']),len(means)),complex)
    offset=0
    for old_epoch,old in enumerate(history):
        for row,receiver in enumerate(record['receivers']):
            for column,old_receiver in enumerate(old['receivers']):
                if receiver==old_receiver:
                    cross[row,offset+column]=(rho**(current_epoch-old_epoch)
                        *np.vdot(record['templates'][row],old['templates'][column]))
        offset+=len(old['mean'])
    return conditional_gaussian_information(record['mean'],means,np.eye(len(record['mean'])),
        history_cov,cross).increment


def _run_policy(initial,waveform,cfg,pilot,horizon,rho,history_aware,
        target_initial=None,target_velocity=None,prediction_velocity=None):
    position=initial.copy(); model_history=[]; truth_history=[]; actions=[]; flight_energy=0.
    target0=np.zeros(3) if target_initial is None else np.asarray(target_initial,float)
    truth_velocity=np.zeros(3) if target_velocity is None else np.asarray(target_velocity,float)
    predicted_velocity=truth_velocity if prediction_velocity is None else np.asarray(prediction_velocity,float)
    for epoch in range(horizon):
        predicted_target=target0+predicted_velocity*(.8*epoch)
        true_target=target0+truth_velocity*(.8*epoch)
        # Keep predicted history for control and truth history only for offline
        # evaluation. Feeding true templates back would leak oracle state.
        bank=_actions(position,waveform,cfg,pilot,predicted_target,predicted_velocity)
        values=np.array([_increment(x,model_history,rho) if history_aware else x['raw'] for x in bank])
        index=int(np.argmax(np.where([x['feasible'] for x in bank],values,-np.inf)))
        proposed=bank[index]
        role=generate_single_tx_roles(len(position))[proposed['role_index']]
        receivers,templates,mean=_observation(proposed['position'],role,waveform,cfg,pilot,
            target_position=true_target,target_velocity=truth_velocity)
        chosen=dict(proposed,receivers=receivers,templates=templates,mean=mean,
            raw=float(mean@mean))
        speed=np.linalg.norm(chosen['position']-position,axis=1)/.8
        flight_energy+=float(np.sum(cfg.uav.P_fly_static+cfg.uav.P_fly_coeff*speed**2)*.8)
        model_history.append(proposed); truth_history.append(chosen); position=chosen['position'];
        actions.append((chosen['motion_index'],chosen['role_index']))
    return _total_information(truth_history,rho),actions,flight_energy


def audit(max_horizon=8,target_velocity=None,prediction_velocity=None):
    if int(max_horizon)!=max_horizon or not 2<=max_horizon<=8: raise ValueError('max_horizon must lie in [2,8]')
    cfg=get_default_config(); noise=compute_noise_power(cfg.channel.kT,cfg.otfs.B,cfg.channel.NF)
    waveform=MinimalOTFSWaveform(cfg.otfs.M,cfg.otfs.N,cfg.otfs.delta_f,
        real_equivalent_complex_noise_variance(noise)); pilot=qpsk_dd_pilot(waveform)
    initial=np.array([[-800.,0.,20.],[700.,250.,20.],[100.,850.,20.]])
    rho=.9; threshold=float(norm.ppf(.999)); rows=[]
    for horizon in range(1,max_horizon+1):
        instant,instant_actions,instant_flight=_run_policy(initial,waveform,cfg,pilot,horizon,rho,False,
            target_velocity=target_velocity,prediction_velocity=prediction_velocity)
        active,active_actions,active_flight=_run_policy(initial,waveform,cfg,pilot,horizon,rho,True,
            target_velocity=target_velocity,prediction_velocity=prediction_velocity)
        instant_pd=float(norm.sf(threshold-np.sqrt(instant)))
        active_pd=float(norm.sf(threshold-np.sqrt(active)))
        rows.append(dict(horizon=horizon,instantaneous_information=instant,
            history_information=active,instantaneous_pd=instant_pd,history_pd=active_pd,
            pd_delta=active_pd-instant_pd,instantaneous_actions=instant_actions,
            history_actions=active_actions,history_passes_pd_gate=active_pd>.8,
            elapsed_s=.8*horizon,instantaneous_flight_energy_j=instant_flight,
            history_flight_energy_j=active_flight,
            history_sensing_rf_energy_j=horizon*.15*waveform.doppler_bins/waveform.delta_f_hz))
    passing=[x['horizon'] for x in rows if x['history_passes_pd_gate']]
    return dict(evidence_class='COMPLETE_SHORT_HISTORY_ACTIVE_SENSING',rows=rows,
        minimum_history_horizon_for_pd_gate=min(passing) if passing else None,
        pfa_design=.001,sensing_power_w=.15,communication_power_w=0.,clutter_memory=rho,
        target_velocity_mps=np.zeros(3).tolist() if target_velocity is None else np.asarray(target_velocity,float).tolist(),
        prediction_velocity_mps=(np.zeros(3).tolist() if prediction_velocity is None and target_velocity is None
            else (np.asarray(target_velocity,float).tolist() if prediction_velocity is None
                  else np.asarray(prediction_velocity,float).tolist())),
        limitations=['known static target and known Gaussian AR(1) receiver-local clutter',
            'local complete history only; communication is not yet exercised',
            'greedy finite actions, not globally optimal dynamic programming',
            'analytic fixed-P_FA result, not field or moving-target certification'])


if __name__=='__main__': print(json.dumps(audit(),indent=2,allow_nan=False))
