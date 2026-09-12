"""Paired receiver/transport loss decomposition on one frozen OTFS scene."""
import json
from pathlib import Path
import sys
import numpy as np
from scipy.special import logsumexp

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from config.params import get_default_config
from tools.audit_active_geometry_survival import _observation
from tools.audit_otfs_multiframe_fixed_geometry import interval
from uav_isac.environment.communication import InterUAVCommunicationModel
from uav_isac.physical.bistatic_waveform import real_equivalent_complex_noise_variance
from uav_isac.physical.channel import compute_noise_power
from uav_isac.physical.energy_likelihood import energy_log_likelihood_ratio
from uav_isac.physical.receiver_report_gate import continue_reports,effective_stop
from uav_isac.physical.waveform_evidence import MinimalOTFSWaveform,qpsk_dd_pilot


def audit(samples=100000,window=13,predicted_velocity_x=6.,link_seed=650000,trial_seed=650001,report_gate=False):
    if report_gate and window<=4: raise ValueError('gate needs more than four frames')
    cfg=get_default_config()
    wf=MinimalOTFSWaveform(cfg.otfs.M,cfg.otfs.N,cfg.otfs.delta_f,
        real_equivalent_complex_noise_variance(compute_noise_power(cfg.channel.kT,cfg.otfs.B,cfg.channel.NF)))
    h=np.sqrt(800.**2-20.**2)
    positions=np.array([[-h,0,20],[h,0,20],[0,h,20]])
    roles=np.array([0,1,1]); pilot=qpsk_dd_pilot(wf)
    _,truth,amplitude=_observation(positions,roles,wf,cfg,pilot)
    _,predicted,predicted_amplitude=_observation(positions,roles,wf,cfg,pilot,
        target_position=np.array([0.,30.,0.]),target_velocity=np.array([predicted_velocity_x,0.,0.]))
    overlap=np.sum(predicted.conj()*truth,axis=1)
    # Receiver-declared velocity uncertainty: independent N(0,6^2) errors
    # in x/y, approximated with positive 3-point Gaussian quadrature per axis.
    # This is an explicit sensitivity model, not a calibrated tracker output.
    nodes,weights=np.polynomial.hermite.hermgauss(3)
    snr_bank=[]; probabilities=[]
    for a in range(3):
        for b in range(3):
            velocity=np.array([predicted_velocity_x,0.,0.])+np.array([nodes[a],nodes[b],0.])*np.sqrt(2)*6.
            _,candidate,candidate_amplitude=_observation(positions,roles,wf,cfg,pilot,
                target_position=np.array([0.,30.,0.]),target_velocity=velocity)
            match=np.sum(predicted.conj()*candidate,axis=1)
            snr_bank.append(candidate_amplitude**2/2*abs(match)**2)
            probabilities.append(weights[a]*weights[b]/np.pi)
    snr_bank=np.asarray(snr_bank); log_prior=np.log(probabilities)
    # Same declared receiver mismatch for all receiver/transport comparisons.
    radio=InterUAVCommunicationModel(rate_bits_per_dim=[0,8],header_bits=64,
        bandwidth_hz=1e6,deadline_s=.005,processing_delay_s=.0002,
        snr_threshold_db=0.,antenna_gain_dbi=16.,carrier_hz=28e9,tx_power_w=.1,
        kT=4e-21,noise_figure_db=4.,dt=.006024,finite_blocklength_enabled=True,
        snr_shadowing_std_db=4.,snr_shadowing_correlation=.7,rng=np.random.default_rng(link_seed))
    masks=[]; bits=0; energy=0.; report_energies=[]
    control_delivered=False; control_bits=0.; control_energy=0.
    for frame in range(window):
        # Fresh float32 energy + uint16 frame + uint16 schema, 64-bit header.
        packets,stats=radio.transmit({2:np.zeros(radio.message_dim)},{2:0},positions,
            tx_powers_w={2:.1},extra_payload_bits={2:64},
            base_payload_dimensions={2:0},suppress_message_payload={2:True})
        masks.append(any(p.sender==2 and p.receiver==1 for p in packets))
        bits+=stats.total_bits; energy+=stats.per_sender_energy_j.get(2,0.)
        report_energies.append(stats.per_sender_energy_j.get(2,0.))
        if report_gate and frame==3:
            # Single schedule control, not an ACK; one bit stop/continue plus
            # frame/schema metadata padded to 64 bits and a 64-bit header.
            control,cs=radio.transmit({1:np.zeros(radio.message_dim)},{1:0},positions,
                tx_powers_w={1:.1},extra_payload_bits={1:64},
                base_payload_dimensions={1:0},suppress_message_payload={1:True})
            control_delivered=any(p.sender==1 and p.receiver==2 for p in control)
            control_bits=cs.total_bits; control_energy=cs.per_sender_energy_j.get(1,0.)
    mask=np.array(masks,bool)
    gate_metrics={}
    def scores(seed,hypothesis):
        rng=np.random.default_rng(seed)
        shape=(samples,window,2)
        n=(rng.normal(size=shape)+1j*rng.normal(size=shape))/np.sqrt(2)
        e=(rng.normal(size=shape)+1j*rng.normal(size=shape))/np.sqrt(2)
        prednoise=n*overlap+e*np.sqrt(np.maximum(0,1-abs(overlap)**2))
        phase=np.exp(1j*np.arange(window)*1.7)[None,:,None]
        signal=hypothesis*phase*amplitude[None,None,:]/np.sqrt(2)
        actual=n+signal; predicted_z=prednoise+signal*overlap
        local=np.sum(abs(predicted_z[:,:,0])**2,axis=1)
        # The receiver consumes exactly the float32 scalar represented by the
        # declared payload; source/frame association is fixed by the schedule.
        wire=(abs(predicted_z[:,:,1])**2).astype(np.float32).astype(float)
        if report_gate:
            keep=continue_reports(abs(predicted_z[:,:4,0])**2,wire[:,:4][:,mask[:4]],window-4,window-4)
            # Missing control defaults to continuing, with no retransmission.
            stop=effective_stop(keep,control_delivered)
            gate_metrics[seed]=dict(stop_fraction=float(stop.mean()),
                mean_bits=float(bits+control_bits-stop.mean()*(window-4)*128),
                mean_comm_energy_j=float(energy+control_energy-stop.mean()*sum(report_energies[4:])))
        local_llr=energy_log_likelihood_ratio(abs(predicted_z[:,:,0])**2,
            predicted_amplitude[0]**2/2).sum(axis=1)
        # Same energy payload as before. Prediction-derived SNR is assumed
        # preconfigured at receiver, with no truth-overlap correction.
        remote_llr=energy_log_likelihood_ratio(wire,predicted_amplitude[1]**2/2)
        local_components=[]; delivered_components=[]; gated_components=[]
        for snr in snr_bank:
            component=energy_log_likelihood_ratio(abs(predicted_z[:,:,0])**2,snr[0]).sum(axis=1)
            remote=energy_log_likelihood_ratio(wire[:,mask],snr[1]).sum(axis=1)
            local_components.append(component); delivered_components.append(component+remote)
            if report_gate:
                prefix_llr=energy_log_likelihood_ratio(wire[:,:4][:,mask[:4]],snr[1]).sum(axis=1)
                gated_components.append(component+np.where(stop,prefix_llr,remote))
        # A single shared hypothesis spans frames AND receivers. Mixture
        # probabilities are never reweighted by information scores.
        local_mixture=logsumexp(np.stack(local_components,axis=1)+log_prior,axis=1)
        delivered_mixture=logsumexp(np.stack(delivered_components,axis=1)+log_prior,axis=1)
        result=dict(truth_local_multiframe=np.sum(abs(actual[:,:,0])**2,axis=1),
            prediction_coherent=np.sqrt(2/window)*np.sum(predicted_z[:,:,0].real,axis=1),
            prediction_multiframe=local,
            ideal_delivery=local+wire.sum(axis=1),
            actual_delivery=local+wire[:,mask].sum(axis=1),
            local_energy_likelihood=local_llr,
            delivered_energy_likelihood=local_llr+remote_llr[:,mask].sum(axis=1),
            local_uncertainty_mixture=local_mixture,
            delivered_uncertainty_mixture=delivered_mixture)
        if report_gate:
            result['prefix_gated_mixture']=logsumexp(np.stack(gated_components,axis=1)+log_prior,axis=1)
        return result
    calibration=scores(trial_seed,0); null=scores(trial_seed+1,0); alternative=scores(trial_seed+2,1)
    rows=[]; decisions={}
    for name in calibration:
        threshold=float(np.quantile(calibration[name],.9995,method='higher'))
        fp=int(np.sum(null[name]>threshold)); tp=int(np.sum(alternative[name]>threshold))
        decisions[name]=alternative[name]>threshold
        pd_ci=interval(tp,samples,.05/(2*len(calibration))); fa_ci=interval(fp,samples,.05/(2*len(calibration)))
        rows.append(dict(method=name,pd=tp/samples,pfa=fp/samples,pd_ci=pd_ci,pfa_ci=fa_ci,
            threshold=threshold,conditional_gate=pd_ci[0]>.8 and fa_ci[1]<=.001))
    paired=decisions['delivered_uncertainty_mixture'].astype(float)-decisions['local_uncertainty_mixture']
    # Paired difference in [-1,1], independent H1 trials conditional on the
    # frozen calibration thresholds/model/link mask. One prespecified contrast.
    delta=float(np.mean(paired)); lower=delta-np.sqrt(2*np.log(1/.05)/samples)
    gate_comparison=None
    if report_gate:
        gd=decisions['prefix_gated_mixture'].astype(float)-decisions['delivered_uncertainty_mixture']
        radius=np.sqrt(2*np.log(2/.05)/samples)
        gate_comparison=dict(pd_delta=float(gd.mean()),
            conditional_hoeffding_ci95=[float(gd.mean()-radius),float(gd.mean()+radius)],
            no_loss_demonstrated=bool(gd.mean()-radius>=0))
    return dict(rows=rows,window=window,samples_per_split=samples,
        gate_metrics=gate_metrics,control_delivered=control_delivered,control_bits=control_bits,
        gate_comparison=gate_comparison,
        gate_reserved_elapsed_ms=(1000*window*(wf.doppler_bins/wf.delta_f_hz+.005)+5 if report_gate else None),
        delivered_mixture_minus_local_mixture=delta,
        conditional_paired_delta_lower95=float(lower),
        conditional_cooperation_gain_supported=bool(lower>0),
        mixture_probabilities=probabilities,mixture_snr_bank=snr_bank.tolist(),
        uncertainty_model='declared velocity std 6m/s; 3x3 Gauss-Hermite; fixed position; not tracker-calibrated',
        overlap_abs=abs(overlap).tolist(),overlap_phase=np.angle(overlap).tolist(),
        delivery_mask=masks,attempted_bits=bits,communication_energy_j=energy,
        sensing_energy_j=.15*window*wf.doppler_bins/wf.delta_f_hz,
        reserved_elapsed_ms=1000*window*(wf.doppler_bins/wf.delta_f_hz+.005),
        peak_node_rf_w=[.15,0.,.1],ack_count=0,retransmission_count=0,
        deployable_detection_certified=False,
        limitations=['one frozen channel trace and static mismatched prediction; not population certification',
            'proper independent AWGN and ideal cyclic OTFS; independent deterministic frame phases',
            'real simulated FBL delivery, payload values transported as float32 sufficient statistics',
            'fixed sensing followed by reserved 5ms report slot; no receive/transmit overlap',
            'ideal delivery removes erasures, but unweighted fusion is not a performance upper bound',
            'local symmetric bistatic geometry is insensitive to the chosen velocity error; truth-template row is oracle'])


if __name__=='__main__': print(json.dumps(audit(),indent=2,allow_nan=False))
