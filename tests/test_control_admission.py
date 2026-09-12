from uav_isac.physical.control_admission import delivery_lower_bound,admit_control
from tools.audit_unified_receiver_transport import audit


def test_absent_or_weak_evidence_cannot_admit():
    assert delivery_lower_bound(0,0)==0
    assert not admit_control(0,128,1152)
    assert not admit_control(.1,128,1152)
    assert admit_control(.5,128,1152)


def test_rejection_sends_no_control_and_preserves_full_evidence():
    r=audit(samples=1000,report_gate=True,stop_only_transport=True,reverse_delivery_lower=0.)
    for metrics in r['gate_metrics'].values():
        assert metrics['stop_request_fraction']==0
        assert metrics['mean_bits']==r['attempted_bits']
    assert r['gate_factorial']['evidence_delta']['delta']==0
