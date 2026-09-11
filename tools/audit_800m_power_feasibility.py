"""Separate physical-cap effects from unproven power-policy benefits.

Same independent-noise/noncoherent detector and declared transport assumptions
as the accumulation audit. This is a link-budget feasibility calculation, not
an autonomous controller evaluation. Two hardware hypotheses are explicit.
"""
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import brentq
from scipy.stats import chi2, beta

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
from tools.audit_800m_spatiotemporal_accumulation import expected_pd
from uav_isac.physical.evidence import EvidencePacketLayout


def audit(target_pd=0.8, trials=100_000):
    if not 0 < target_pd < 0.85 or trials < 10000:
        raise ValueError('require 0<target_pd<0.85 and trials>=10000')
    cfg = get_default_config()
    wf = MinimalOTFSWaveform(
        delay_bins=cfg.otfs.M, doppler_bins=cfg.otfs.N,
        delta_f_hz=cfg.otfs.delta_f,
        noise_variance=real_equivalent_complex_noise_variance(
            compute_noise_power(cfg.channel.kT,cfg.otfs.B,cfg.channel.NF)))
    # Compute per-watt coefficient from existing geometry, not prior output.
    physical = bistatic_geometry_to_waveform(
        np.array([[-800.,0.,0.],[800.,0.,0.]]),np.zeros((2,3)),
        np.zeros((1,3)),np.zeros((1,3)),np.array([0,1]),np.array([[1.],[0.]]),
        carrier_hz=cfg.otfs.fc,waveform=wf,rcs_m2=cfg.target.rcs,
        tx_gain_dbi=cfg.otfs.g_tx_dBi,rx_gain_dbi=cfg.otfs.g_rx_dBi,
        require_unambiguous=True)
    per_watt = float(ideal_coherent_h0_deflection(
        physical.received_target_amplitude[0,1,0],waveform=wf,
        complex_noise_variance=wf.noise_variance,n_cpi=1))
    rows=[]
    block_s=cfg.otfs.N/cfg.otfs.delta_f
    packet_bits=EvidencePacketLayout(16,16).broadcast_bits(1,32)
    for blocks, receivers, delivery in ((1,1,1.),(8,1,1.),(8,15,1.),(8,15,.9)):
        def pd(power):
            return expected_pd(blocks,receivers,delivery,power*per_watt)
        required=brentq(lambda power:pd(power)-target_pd,0.,1.,xtol=1e-13)
        # Predeclare 0.85 design point; validate independently rather than
        # tuning power until a Monte Carlo confidence bound passes.
        design_power=brentq(lambda power:pd(power)-.85,0.,1.,xtol=1e-13)
        validations=[]
        for hypothesis,seed in ((0,73001),(1,73002)):
            rng=np.random.default_rng(seed)
            count=blocks+rng.binomial(blocks*(receivers-1),delivery,size=trials)
            statistic=rng.noncentral_chisquare(2*count,hypothesis*count*design_power*per_watt)
            validations.append(int(np.count_nonzero(statistic>chi2.isf(.001,2*count))))
        detected=validations[1]
        # Bonferroni across the four predeclared design points: joint >=95%.
        lower=float(beta.ppf(.05/4,detected,trials-detected+1)) if detected else 0.
        rows.append(dict(blocks=blocks,receivers=receivers,delivery=delivery,
            sensing_w_at_pd_boundary=required,
            strict_boundary_is_not_a_pass=True,
            design_sensing_w=design_power,design_pd=.85,
            validation_pd=detected/trials,validation_pd_lower_bound=lower,
            validation_pfa=validations[0]/trials,
            detection_gate_passed=lower>target_pd,
            attempted_bits=blocks*(receivers-1)*packet_bits,
            sensing_energy_j=design_power*blocks*block_s,
            serial_sensing_transport_ms=1000*(blocks*block_s+blocks*(receivers-1)*packet_bits/1e6),
            network_communication_tx_energy_j=.1*blocks*(receivers-1)*packet_bits/1e6,
            pd_at_legacy_sensing_cap=pd(cfg.uav.P_sense_max),
            pd_at_half_watt=pd(.5),pd_at_one_watt=pd(1.),
            feasible_under_legacy_cap=required<=cfg.uav.P_sense_max,
            threshold_residual=abs(pd(required)-target_pd)))
    legacy_paths=[]
    for receivers,delivery in ((1,1.),(15,1.),(15,.9)):
        count=next(n for n in range(1,601) if expected_pd(
            n,receivers,delivery,cfg.uav.P_sense_max*per_watt)>target_pd)
        attempted=count*(receivers-1)*packet_bits
        elapsed=count*block_s+attempted/1e6
        legacy_paths.append(dict(receivers=receivers,delivery=delivery,blocks=count,
            pd=expected_pd(count,receivers,delivery,cfg.uav.P_sense_max*per_watt),
            attempted_bits=attempted,serial_sensing_transport_ms=1000*elapsed,
            sensing_energy_j=cfg.uav.P_sense_max*count*block_s,
            network_communication_tx_energy_j=.1*attempted/1e6,
            within_100ms=elapsed<=.1,
            detection_confidence_validated=False))
    return dict(status='PASS' if all(r['threshold_residual']<1e-9 and r['detection_gate_passed'] for r in rows) else 'FAIL',
        evidence_class='POWER_CAP_FEASIBILITY_NOT_POLICY_EVALUATION',
        shared_cap_w=1.,legacy_sensing_cap_w=cfg.uav.P_sense_max,
        target_pd_strictly_greater_than=target_pd,target_pfa=.001,
        validation_trials_per_hypothesis=trials,
        confidence_method='one-sided Clopper-Pearson; Bonferroni four points at family alpha=0.05',
        deflection_per_watt=per_watt,autonomous_policy_validated=False,
        assumptions=['800 m per bistatic leg; independent thermal noise',
            'equal-quality receivers; normalized scalar sufficient-statistic model',
            'delivery rates are declared, not predicted from communication power',
            'single transmitter illuminates; receivers have separate per-node budgets',
            '1 W sensing changes the legacy hardware cap hypothesis',
            '1 Mbps serial effective transport and 0.1 W per sending receiver are assumed',
            'no Q16 scheduling, clutter, finite-blocklength or history-policy validation'],
            rows=rows,legacy_cap_minimum_blocks=legacy_paths)


if __name__=='__main__':
    result=audit()
    print(json.dumps(result,indent=2,allow_nan=False))
    raise SystemExit(result['status']!='PASS')
