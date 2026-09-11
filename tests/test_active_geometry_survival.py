import numpy as np
from config.params import get_default_config
from tools.audit_active_geometry_survival import _observation,audit
from uav_isac.physical.bistatic_waveform import real_equivalent_complex_noise_variance
from uav_isac.physical.channel import compute_noise_power
from uav_isac.physical.waveform_evidence import MinimalOTFSWaveform,qpsk_dd_pilot


def test_g0_active_geometry_audit_is_physically_scoped_and_reproducible():
    result=audit(samples=20000,seed=7)
    assert result['candidate_count']>=result['feasible_count']>1
    assert result['communication_power_w']==0.
    assert result['fixed']['pfa']<.002
    assert abs(result['fixed']['pd']-result['fixed']['pd_theory'])<.02
    assert result['active_beats_fixed']
    assert result['history_beats_range']
    assert not result['history_beats_instantaneous']
    assert result['conclusion']=='geometry_role_survives_history_novelty_does_not'


def test_symmetric_static_geometry_is_a_role_reciprocity_counterexample():
    cfg=get_default_config(); waveform=MinimalOTFSWaveform(cfg.otfs.M,cfg.otfs.N,
        cfg.otfs.delta_f,real_equivalent_complex_noise_variance(
            compute_noise_power(cfg.channel.kT,cfg.otfs.B,cfg.channel.NF)))
    pilot=qpsk_dd_pilot(waveform)
    positions=np.array([[-800.,0.,20.],[800.,0.,20.],[0.,800.,20.]])
    observations=[_observation(positions,roles,waveform,cfg,pilot)[2] for roles in
        (np.array([0,1,1]),np.array([1,0,1]),np.array([1,1,0]))]
    assert all(np.allclose(x,observations[0]) for x in observations[1:])
