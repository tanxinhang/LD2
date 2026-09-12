"""800 m per-leg fixed-geometry single-link multiframe GLRT reference.

Exact matched-projection sampling for the ideal cyclic OTFS AWGN model.
There is one frozen predicted path, no target search and no early stopping.
Unknown deterministic frame amplitudes are maximized, without a phase prior.
"""
import json
from pathlib import Path
import sys
import numpy as np
from scipy.stats import gamma,ncx2,beta

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from config.params import get_default_config
from uav_isac.physical.bistatic_waveform import bistatic_geometry_to_waveform,real_equivalent_complex_noise_variance
from uav_isac.physical.channel import compute_noise_power
from uav_isac.physical.waveform_evidence import MinimalOTFSWaveform,qpsk_dd_pilot,dd_path_response


def interval(k,n,error):
    return [float(beta.ppf(error/2,k,n-k+1)) if k else 0.,
            float(beta.ppf(1-error/2,k+1,n-k)) if k<n else 1.]


def audit(samples=200000):
    if samples<10000: raise ValueError('at least 10000 trials required')
    cfg=get_default_config()
    wf=MinimalOTFSWaveform(cfg.otfs.M,cfg.otfs.N,cfg.otfs.delta_f,
        real_equivalent_complex_noise_variance(compute_noise_power(cfg.channel.kT,cfg.otfs.B,cfg.channel.NF)))
    # Altitude 20m while preserving EXACT 800m slant distances.
    horizontal=np.sqrt(800.**2-20.**2)
    positions=np.array([[-horizontal,0.,20.],[horizontal,0.,20.]])
    params=bistatic_geometry_to_waveform(positions,np.zeros((2,3)),np.zeros((1,3)),
        np.zeros((1,3)),np.array([0,1]),np.array([[.15],[0.]]),carrier_hz=cfg.otfs.fc,
        waveform=wf,rcs_m2=cfg.target.rcs,tx_gain_dbi=cfg.otfs.g_tx_dBi,
        rx_gain_dbi=cfg.otfs.g_rx_dBi,require_unambiguous=True)
    template=dd_path_response(qpsk_dd_pilot(wf),delay_bin=params.delay_bin[0,1,0],
        doppler_bin=params.doppler_bin[0,1,0]).ravel()
    energy=float(np.vdot(template,template).real)
    per_frame_snr=float(params.received_target_amplitude[0,1,0]**2*energy/wf.noise_variance)
    rows=[]; frame_s=wf.doppler_bins/wf.delta_f_hz
    # Predeclared alpha/2 design margin enables an independent upper-bound
    # check against the requested alpha, without validation-driven retuning.
    alpha=.001; alpha_design=alpha/2
    for window in (1,2,4,8):
        threshold=float(gamma.isf(alpha_design,a=window))
        rng0=np.random.default_rng(610000+window)
        z0=(rng0.normal(size=(samples,window))+1j*rng0.normal(size=(samples,window)))/np.sqrt(2)
        false=int(np.count_nonzero(np.sum(abs(z0)**2,axis=1)>threshold))
        for budget in ('fixed_power','fixed_window_energy'):
            power=.15 if budget=='fixed_power' else .15/window
            snr=per_frame_snr*power/.15
            rng=np.random.default_rng(620000+window)
            z=(rng.normal(size=(samples,window))+1j*rng.normal(size=(samples,window)))/np.sqrt(2)
            # Deterministic phases suffice: circular independent noise makes
            # performance invariant to their actual values.
            phase=np.exp(1j*np.arange(window)*1.7)
            score=np.sum(abs(z+np.sqrt(snr)*phase)**2,axis=1)
            detected=int(np.count_nonzero(score>threshold))
            pd_ci=interval(detected,samples,.05/16)
            pfa_ci=interval(false,samples,.05/16)
            rows.append(dict(window=window,budget=budget,sensing_w=power,
                per_node_rf_w=[power,0.],communication_bits=0,communication_energy_j=0.,
                sensing_energy_j=power*window*frame_s,ideal_observation_ms=1000*window*frame_s,
                threshold=threshold,pd=detected/samples,pfa=false/samples,
                pd_ci95_simultaneous=pd_ci,pfa_ci95_simultaneous=pfa_ci,
                pd_theory=float(ncx2.sf(2*threshold,2*window,2*window*snr)),
                conditional_model_gate=pd_ci[0]>.8 and pfa_ci[1]<=alpha))
    return dict(rows=rows,per_frame_snr_at_015w=per_frame_snr,
        waveform_samples=wf.sample_count,template_energy=energy,samples_per_split=samples,
        false_alarm_design=alpha_design,required_false_alarm=alpha,
        physical_detection_certified=False,
        limitations=['single accurately predicted static path; no DD uncertainty or trajectory search',
            'independent proper Gaussian noise across frames, no clutter or RF impairments',
            'cyclic OTFS useful observation time excludes CP/guard/processing overhead',
            '0.15W research PA assumption exceeds legacy 0.0251W sensing cap but stays under 1W RF cap',
            'one receiver with local evidence only; no distributed cooperation claim'])


if __name__=='__main__': print(json.dumps(audit(),indent=2,allow_nan=False))
