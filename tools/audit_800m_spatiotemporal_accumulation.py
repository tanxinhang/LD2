"""800 m equal-quality receiver accumulation; explicit transport stress model.

Each received energy is 2|z|^2/sigma_c^2 ~ chi2(2,D_edge).
Independent delivered observations sum to chi2(2*n,n*D_edge).
Condition thresholds on delivered count, never on observed energy. Communication
is serial at a declared effective rate with independent erasures, not a channel
or finite-blocklength certification. No cross-node/temporal clutter is assumed.
"""
from pathlib import Path
import json
import sys
import numpy as np
from scipy.stats import chi2, ncx2, binom

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from config.params import get_default_config
from uav_isac.physical.bistatic_waveform import (
    bistatic_geometry_to_waveform, ideal_coherent_h0_deflection,
    real_equivalent_complex_noise_variance,
)
from uav_isac.physical.channel import compute_noise_power
from uav_isac.physical.waveform_evidence import MinimalOTFSWaveform
from uav_isac.physical.evidence import EvidencePacketLayout


def expected_pd(blocks, receivers, delivery, edge_d, pfa=0.001):
    remote = blocks*(receivers-1)
    received = np.arange(remote+1)
    count = blocks+received  # owner's local observations are always available
    thresholds = chi2.isf(pfa, 2*count)
    weights = binom.pmf(received, remote, delivery)
    return float(weights @ ncx2.sf(thresholds, 2*count, count*edge_d))


def audit(trials=100_000):
    cfg = get_default_config()
    wf = MinimalOTFSWaveform(
        delay_bins=cfg.otfs.M, doppler_bins=cfg.otfs.N,
        delta_f_hz=cfg.otfs.delta_f,
        noise_variance=real_equivalent_complex_noise_variance(
            compute_noise_power(cfg.channel.kT,cfg.otfs.B,cfg.channel.NF)))
    physical = bistatic_geometry_to_waveform(
        np.array([[-800.,0,0],[800.,0,0]]), np.zeros((2,3)),
        np.zeros((1,3)), np.zeros((1,3)), np.array([0,1]),
        np.array([[cfg.uav.P_sense],[0.]]), carrier_hz=cfg.otfs.fc,
        waveform=wf, rcs_m2=cfg.target.rcs,
        tx_gain_dbi=cfg.otfs.g_tx_dBi,rx_gain_dbi=cfg.otfs.g_rx_dBi,
        require_unambiguous=True)
    edge_d = float(ideal_coherent_h0_deflection(
        physical.received_target_amplitude[0,1,0], waveform=wf,
        complex_noise_variance=wf.noise_variance,n_cpi=1))
    block_s = cfg.otfs.N/cfg.otfs.delta_f
    bits = EvidencePacketLayout(16,16).broadcast_bits(1,32)
    rate, comm_power, deadline = 1e6, 0.1, 0.1
    rows = []
    for fixed_energy in (False, True):
        for receivers, delivery in ((1,1.), (15,1.), (15,0.9)):
            for blocks in (1,2,4,8,16,32,64,128):
                d = edge_d/blocks if fixed_energy else edge_d
                packets = blocks*(receivers-1)
                latency = blocks*block_s + packets*bits/rate
                pd = expected_pd(blocks,receivers,delivery,d)
                rows.append(dict(blocks=blocks,receivers=receivers,delivery=delivery,
                    fixed_total_sensing_energy=fixed_energy,pd=pd,
                    sensing_energy_j=cfg.uav.P_sense*block_s*(1 if fixed_energy else blocks),
                    attempted_bits=packets*bits,comm_energy_j=comm_power*packets*bits/rate,
                    latency_ms=latency*1000,meets_pd_floor=pd>0.8,
                    meets_100ms_deadline=latency<=deadline))
    # Direct draws of the exact accumulated sufficient-statistic distribution.
    # Independent H0/H1 streams; no waveform Monte Carlo claim.
    validation = []
    for blocks, receivers, delivery in ((8,1,1.), (8,15,1.), (8,15,0.9)):
        outcomes=[]
        for hypothesis, seed in ((0,62001),(1,62002)):
            rng=np.random.default_rng(seed)
            count=blocks+rng.binomial(blocks*(receivers-1),delivery,size=trials)
            statistic=rng.noncentral_chisquare(2*count,hypothesis*count*edge_d)
            outcomes.append(float(np.mean(statistic>chi2.isf(0.001,2*count))))
        theory=expected_pd(blocks,receivers,delivery,edge_d)
        passed=(abs(outcomes[0]-.001)<5*np.sqrt(.001*.999/trials)
                and abs(outcomes[1]-theory)<5*np.sqrt(theory*(1-theory)/trials))
        validation.append(dict(blocks=blocks,receivers=receivers,delivery=delivery,
            pfa=outcomes[0],pd=outcomes[1],expected_pd=theory,passed=bool(passed)))
    return dict(status='PASS' if all(v['passed'] for v in validation) else 'FAIL',
        evidence_class='INDEPENDENT_ENERGY_ACCUMULATION_WITH_DECLARED_TRANSPORT',
        leg_distance_m=800,edge_deflection=edge_d,block_duration_ms=1000*block_s,
        packet_bits=bits,effective_transport_bps=rate,communication_power_w=comm_power,
        target_pfa=.001,pd_floor=.8,pd_comparison='strict_greater',trials_per_hypothesis=trials,
        limitations=['equal-quality 800m legs, not a realized K16 formation',
            'independent noise across receivers and blocks; no clutter',
            'packet precision charged at 32 bits but quantization not simulated',
            'declared transport rate/erasures, no physical link certification',
            'one target, no Q16 resource or multiple-search false-alarm certification',
            'no state-information-aided synchronization gain modeled'],
        validation=validation,rows=rows)


if __name__=='__main__':
    result=audit()
    print(json.dumps(result,indent=2,allow_nan=False))
    raise SystemExit(result['status']!='PASS')
