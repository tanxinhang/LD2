"""Bistatic observations -> real repository FBL transport -> owner history.

Small three-UAV offline integration. Local and received energy values are
stored by observation/source/target ID. It uses the repository simulated
physical link, not hardware radio; no transmitter learns remote delivery state.
"""
import json
from pathlib import Path
import sys
import numpy as np
from scipy.stats import chi2, ncx2

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from config.params import get_default_config
from uav_isac.environment.communication import InterUAVCommunicationModel
from uav_isac.physical.evidence import EvidencePacketLayout,route_structured_evidence
from uav_isac.physical.delivered_energy_history import DeliveredEnergyHistory
from uav_isac.physical.bistatic_waveform import (
    bistatic_geometry_to_waveform,ideal_coherent_h0_deflection,
    real_equivalent_complex_noise_variance)
from uav_isac.physical.channel import compute_noise_power
from uav_isac.physical.waveform_evidence import MinimalOTFSWaveform


def audit(trials=100_000):
    if trials < 10000 or trials % 8:
        raise ValueError('trials must be >=10000 and divisible by eight')
    cfg=get_default_config()
    xyz=np.array([[-800.,0.,0.],[800.,0.,0.],[0.,800.,0.]])
    wf=MinimalOTFSWaveform(delay_bins=cfg.otfs.M,doppler_bins=cfg.otfs.N,
        delta_f_hz=cfg.otfs.delta_f,noise_variance=real_equivalent_complex_noise_variance(
            compute_noise_power(cfg.channel.kT,cfg.otfs.B,cfg.channel.NF)))
    sense=.15
    physical=bistatic_geometry_to_waveform(xyz,np.zeros((3,3)),np.zeros((1,3)),
        np.zeros((1,3)),np.array([0,1,1]),np.array([[sense],[0.],[0.]]),
        carrier_hz=cfg.otfs.fc,waveform=wf,rcs_m2=cfg.target.rcs,
        tx_gain_dbi=cfg.otfs.g_tx_dBi,rx_gain_dbi=cfg.otfs.g_rx_dBi)
    d=np.zeros((3,1))
    d[1:,0]=ideal_coherent_h0_deflection(physical.received_target_amplitude[0,1:,0],
        waveform=wf,complex_noise_variance=wf.noise_variance,n_cpi=1)
    rows=[]
    # Link trace bank is diagnostic, distinct from detector RNG. Confidence
    # intervals below condition on this finite bank, not unseen link histories.
    for power in (0.,.001,.01,.1):
        all_outcomes=[[],[]]; local_outcomes=[[],[]]
        delivered=[]; bits=[]; energy=[]; elapsed=[]
        for link_seed in range(8):
            model=InterUAVCommunicationModel(rate_bits_per_dim=[0,8],header_bits=64,
                bandwidth_hz=1e6,deadline_s=.005,processing_delay_s=.0002,
                snr_threshold_db=0.,antenna_gain_dbi=16.,carrier_hz=cfg.otfs.fc,
                tx_power_w=power,kT=cfg.channel.kT,noise_figure_db=cfg.channel.NF,
                dt=.1,finite_blocklength_enabled=True,snr_shadowing_std_db=4.,
                snr_shadowing_correlation=.7,rng=np.random.default_rng(81000+link_seed))
            histories=[DeliveredEnergyHistory(8),DeliveredEnergyHistory(8)]
            locals_=[DeliveredEnergyHistory(8),DeliveredEnergyHistory(8)]
            rngs=[np.random.default_rng(82000+2*link_seed+h) for h in range(2)]
            size=trials//8
            rx_count=0; charged_bits=0; charged_energy=0.; duration=0.
            for frame in range(8):
                # Only Rx2 reports; Rx1 is owner and keeps its own evidence.
                report_quality=d.copy()
                if power==0:
                    report_quality[2,0]=0.  # silence has no attempted packet
                transport=route_structured_evidence(report_quality,xyz,np.array([0.,0.,power]),
                    observation_frame=frame,topk=1,llr_bits=32,
                    layout=EvidencePacketLayout(3,1),link_model=model,
                    fusion_owner=np.array([1]))
                available=bool(transport.delivered_peer_mask[2,0])
                rx_count+=int(available)
                charged_bits+=transport.total_bits
                charged_energy+=transport.energy_j_by_sender[2]
                # Reserve full communication slot, including failed packets;
                # no failed/infinite link latency can disappear from clock.
                duration+=wf.doppler_bins/wf.delta_f_hz+.005
                for h in range(2):
                    histories[h].advance(frame); locals_[h].advance(frame)
                    for source in (1,2):
                        z=rngs[h].noncentral_chisquare(2,h*d[source,0],size=size)
                        if source==2:
                            z=z.astype(np.float32).astype(float)
                        key=(frame,source,0)
                        histories[h].accept(key,z,available=source==1 or available)
                        if source==1:
                            locals_[h].accept(key,z,available=True)
            for h in range(2):
                count,total=histories[h].total()
                all_outcomes[h].extend(total>chi2.isf(.001,2*count))
                count,total=locals_[h].total()
                local_outcomes[h].extend(total>chi2.isf(.001,2*count))
            delivered.append(rx_count);bits.append(charged_bits);energy.append(charged_energy);elapsed.append(duration)
        n=len(all_outcomes[1]);success=int(np.sum(all_outcomes[1]))
        # Conditional independent but non-identically distributed trials across
        # fixed link traces: Hoeffding is valid; pooled binomial CP is not.
        lower=max(0.,success/n-np.sqrt(np.log(4/.05)/(2*n)))
        rows.append(dict(communication_power_w=power,sensing_power_w=sense,
            per_node_power_w=[sense,0.,power],pd=float(np.mean(all_outcomes[1])),
            pd_lower_bound_conditional_on_link_bank=lower,pd_gate=lower>.8,
            pfa=float(np.mean(all_outcomes[0])),local_history_pd=float(np.mean(local_outcomes[1])),
            delivered_remote_blocks=delivered,mean_attempted_bits=float(np.mean(bits)),
            theoretical_pd_by_link_trace=[float(ncx2.sf(chi2.isf(.001,2*(8+k)),
                2*(8+k),(8+k)*d[1,0])) for k in delivered],
            mean_report_tx_energy_j=float(np.mean(energy)),
            sensing_tx_energy_j=sense*8*wf.doppler_bins/wf.delta_f_hz,
            reserved_window_ms=1000*max(elapsed)))
    return dict(evidence_class='PHYSICAL_SIMULATED_LINK_TO_OWNER_HISTORY',rows=rows,
        blocks=8,link_traces=8,trials_per_hypothesis=trials//8*8,
        confidence_method='conditional Hoeffding lower bound; family alpha .05 across four powers',
        status='PASS' if all(max(r['per_node_power_w'])<=1 and r['reserved_window_ms']<=100 for r in rows) else 'FAIL',
        assumptions=['single target, all bistatic legs 800m, ideal thermal-noise sensing',
            'FBL and correlated shadowing use existing simulated radio',
            'float32 remote energies, metadata-only router with successful payload decoding',
            'fixed powers; no history-driven power optimizer or ACK feedback yet',
            'detection confidence conditional on eight sampled link traces, not field reliability'])


if __name__=='__main__':
    result=audit();print(json.dumps(result,indent=2,allow_nan=False))
    raise SystemExit(result['status']!='PASS')
