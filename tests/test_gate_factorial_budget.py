import pytest
from tools.audit_unified_receiver_transport import audit
from uav_isac.physical.report_budget import report_budget_check


def test_signed_factorial_deltas_telescope():
    r=audit(samples=500,window=6,report_gate=True,frozen_gate_threshold=7.)
    f=r['gate_factorial']
    assert f['evidence_delta']['delta']+f['threshold_delta']['delta']==pytest.approx(r['gate_comparison']['pd_delta'])
    assert f['full_gate']['pd']<=f['full_full']['pd']


def test_control_cost_can_reject_an_apparently_saving_gate():
    assert not report_budget_check(1664,128,1152,.06862,1664)['admissible']
    assert report_budget_check(1664,128,1152,.2,1664)['admissible']
    assert not report_budget_check(1664,128,1152,0.,1664)['admissible']
