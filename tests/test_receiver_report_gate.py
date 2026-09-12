import numpy as np
from uav_isac.physical.receiver_report_gate import continue_reports,effective_stop
from tools.audit_unified_receiver_transport import audit


def test_no_remote_observations_does_not_invent_quality():
    assert continue_reports(np.ones((2,4)),np.empty((2,0)),9,9).all()


def test_lost_control_cannot_silently_stop_sender():
    decision=np.array([False,True])
    assert not effective_stop(decision,False).any()
    assert effective_stop(decision,True).tolist()==[True,False]


def test_weak_vs_strong_prefix_have_distinct_report_decisions():
    local=np.full((2,4),3.)
    remote=np.array([[1.]*4,[8.]*4])
    assert continue_reports(local,remote,9,9).tolist()==[False,True]


def test_control_cost_and_whole_policy_calibration_are_exposed():
    r=audit(samples=5000,window=6,report_gate=True)
    assert r['control_bits']==128
    assert r['gate_reserved_elapsed_ms']>r['reserved_elapsed_ms']
    assert 'prefix_gated_mixture' in {x['method'] for x in r['rows']}
    for metrics in r['gate_metrics'].values():
        expected=r['attempted_bits']+128-metrics['stop_fraction']*2*128
        assert abs(metrics['mean_bits']-expected)<1e-8
    assert r['ack_count']==r['retransmission_count']==0
