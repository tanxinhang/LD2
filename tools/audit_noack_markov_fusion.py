"""No-ACK, once-per-block local-likelihood exchange and joint history fusion.

Static uncertain target on a physical position grid is the first Markov limit.
Independent unknown phases are marginalized exactly (Bessel likelihood).
Full covariance of matched projections is retained. No power optimizer yet.
"""
import json
from pathlib import Path
import sys
import numpy as np
from scipy.special import i0e,logsumexp

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from config.params import get_default_config
from uav_isac.physical.markov_evidence import update
from uav_isac.physical.bistatic_waveform import bistatic_geometry_to_waveform,real_equivalent_complex_noise_variance
from uav_isac.physical.channel import compute_noise_power
from uav_isac.physical.waveform_evidence import MinimalOTFSWaveform,qpsk_dd_pilot,dd_path_response
from uav_isac.environment.communication import InterUAVCommunicationModel


def model():
    cfg=get_default_config()
    wf=MinimalOTFSWaveform(delay_bins=cfg.otfs.M,doppler_bins=cfg.otfs.N,
        delta_f_hz=cfg.otfs.delta_f,noise_variance=real_equivalent_complex_noise_variance(
            compute_noise_power(cfg.channel.kT,cfg.otfs.B,cfg.channel.NF)))
    positions=np.array([[-800.,0.,0.],[800.,0.,0.],[0.,800.,0.]])
    targets=np.array([[0.,y,0.] for y in (-120.,-60.,0.,60.,120.)])
    # Each column is an alternative world, not five simultaneous allocations.
    grams=[]; signatures=[[],[]]
    for target in targets:
        physical=bistatic_geometry_to_waveform(positions,np.zeros((3,3)),target[None,:],
            np.zeros((1,3)),np.array([0,1,1]),np.array([[.15],[0.],[0.]]),
            carrier_hz=cfg.otfs.fc,waveform=wf,rcs_m2=cfg.target.rcs,
            tx_gain_dbi=cfg.otfs.g_tx_dBi,rx_gain_dbi=cfg.otfs.g_rx_dBi)
        for j in (1,2):
            s=dd_path_response(qpsk_dd_pilot(wf),delay_bin=physical.delay_bin[0,j,0],
                doppler_bin=physical.doppler_bin[0,j,0]).ravel()
            signatures[j-1].append(s*physical.received_target_amplitude[0,j,0]/np.sqrt(wf.noise_variance))
    for bank in signatures:
        b=np.stack(bank,axis=1);grams.append(b.conj().T@b)
    return positions,grams


def links(positions,seed,blocks=8):
    radio=InterUAVCommunicationModel(rate_bits_per_dim=[0,8],header_bits=64,
        bandwidth_hz=1e6,deadline_s=.005,processing_delay_s=.0002,
        snr_threshold_db=0.,antenna_gain_dbi=16.,carrier_hz=28e9,tx_power_w=.1,
        kT=4e-21,noise_figure_db=4.,dt=.1,finite_blocklength_enabled=True,
        snr_shadowing_std_db=4.,snr_shadowing_correlation=.7,rng=np.random.default_rng(seed))
    mask=[];energy=0.;bits=0
    for frame in range(blocks):
        # Five float32 emission values + frame16 + source2 + schema/grid ID16.
        packets,stats=radio.transmit({2:np.zeros(radio.message_dim)},{2:0},positions,
            tx_powers_w={2:.1},extra_payload_bits={2:194},
            base_payload_dimensions={2:0},suppress_message_payload={2:True})
        mask.append(any(x.sender==2 and x.receiver==1 for x in packets))
        energy+=stats.per_sender_energy_j.get(2,0.);bits+=stats.total_bits
    return mask,energy,bits


def trace(grams,delivery,size,hypothesis,seed):
    rng=np.random.default_rng(seed);states=rng.integers(0,5,size=size)
    mass=np.full((size,5),-np.log(5.));local_mass=mass.copy();memoryless=np.zeros(size)
    # Sampling matched-filter covariance is exact sufficient-statistic sampling,
    # including singular coincident templates, without white-noise shortcuts.
    factors=[]
    for g in grams:
        vals,vecs=np.linalg.eigh(g)
        factors.append(vecs*np.sqrt(np.maximum(vals,0))[None,:])
    snapshots={}
    for frame,received in enumerate(delivery):
        emissions=[]
        for g,factor in zip(grams,factors):
            noise=(rng.normal(size=(size,5))+1j*rng.normal(size=(size,5)))/np.sqrt(2)
            z=noise@factor.T
            if hypothesis:
                z+=np.exp(2j*np.pi*rng.random((size,1)))*g[:,states].T
            arg=2*np.abs(z)
            emissions.append(np.log(i0e(arg))+arg-np.diag(g).real)
        remote=emissions[1].astype(np.float32).astype(float) if received else None
        local_mass=update(local_mass,np.eye(5),emissions[0])
        mass=update(mass,np.eye(5),emissions[0],remote)
        fresh=emissions[0]+(remote if remote is not None else 0)
        memoryless+=logsumexp(fresh,axis=1)-np.log(5.)
        if frame+1 in (1,2,4,8):
            posterior=np.exp(mass-logsumexp(mass,axis=1)[:,None])
            local_post=np.exp(local_mass-logsumexp(local_mass,axis=1)[:,None])
            snapshots[frame+1]=dict(local=logsumexp(local_mass,axis=1),
                markov=logsumexp(mass,axis=1),memoryless=memoryless.copy(),
                state_nll=float(-np.mean(np.log(np.maximum(posterior[np.arange(size),states],1e-300)))),
                local_state_nll=float(-np.mean(np.log(np.maximum(local_post[np.arange(size),states],1e-300)))))
    return snapshots


def audit(samples=100_000):
    if samples<8000 or samples%8:
        raise ValueError('samples >=8000 and divisible by eight required')
    positions,grams=model();banks=[]
    for split in range(3):
        bank=[]
        for j in range(8):
            mask,energy,bits=links(positions,120000+1000*split+j)
            bank.append((trace(grams,mask,samples//8,int(split==2),130000+1000*split+j),mask,energy,bits))
        banks.append(bank)
    rows=[]
    for window in (1,2,4,8):
        for method in ('local','memoryless','markov'):
            scores=[np.concatenate([x[0][window][method] for x in bank]) for bank in banks]
            threshold=float(np.quantile(scores[0],.999,method='higher'))
            pd=float(np.mean(scores[2]>threshold))
            # Conditional on calibration and fixed link bank. Independent
            # non-identical trials permit Hoeffding, with 12 reported points.
            lower=float(max(0.,pd-np.sqrt(np.log(12/.05)/(2*samples))))
            rows.append(dict(window=window,method=method,
                pfa=float(np.mean(scores[1]>threshold)),pd=pd,
                conditional_pd_lower_bound=lower,pd_gate_passed=lower>.8,
                threshold=threshold,reserved_latency_ms=window*6.024,
                attempted_bits=0 if method=='local' else window*258))
    final_rows={r['method']:r for r in rows if r['window']==8}
    paired=[]
    for record in banks[2]:
        s=record[0][8]
        paired.append(float(np.mean(s['markov']>final_rows['markov']['threshold'])-
                            np.mean(s['memoryless']>final_rows['memoryless']['threshold'])))
    rng=np.random.default_rng(140000)
    draws=np.mean(np.asarray(paired)[rng.integers(0,8,(4000,8))],axis=1)
    return dict(evidence_class='NO_ACK_ONCE_ONLY_MARKOV_LOCAL_LIKELIHOODS',rows=rows,
        eight_block_markov_minus_memoryless=float(np.mean(paired)),
        paired_link_bootstrap_ci95=np.quantile(draws,[.025,.975]).tolist(),
        confidence_scope='conditional on frozen calibration threshold and eight validation links; not field guarantee',
        samples_per_split=samples,ack_packets=0,retransmissions=0,
        markov_state_nll=float(np.mean([x[0][8]['state_nll'] for x in banks[2]])),
        local_state_nll=float(np.mean([x[0][8]['local_state_nll'] for x in banks[2]])),
        delivery_masks=[x[1] for x in banks[2]],
        mean_remote_energy_j=float(np.mean([x[2] for x in banks[2]])),
        limitations=['static target uncertainty: identity Markov transition, not moving-target validation',
            'grid positions near nominal 800m geometry; no common sensing clutter',
            'exact marginalized phase emissions; sent values are fresh likelihoods, not posteriors',
            'three independent split banks, eight link traces each; no field confidence certification',
            'sensing .15W and communication .1W on different nodes, no joint adaptive power yet',
            'communication aids state prediction/detection; sensing-to-U2U coupling not established'])


if __name__=='__main__':
    print(json.dumps(audit(),indent=2,allow_nan=False))
