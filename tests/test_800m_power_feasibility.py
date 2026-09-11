from tools.audit_800m_power_feasibility import audit


def test_feasibility_root_and_cap_hypotheses():
    result=audit()
    assert result['status']=='PASS'
    assert result['autonomous_policy_validated'] is False
    rows=result['rows']
    assert result['target_pd_strictly_greater_than']==.8
    assert all(r['sensing_w_at_pd_boundary']>result['legacy_sensing_cap_w'] for r in rows)
    assert rows[3]['sensing_w_at_pd_boundary']>rows[2]['sensing_w_at_pd_boundary']
    assert all(r['design_sensing_w']>r['sensing_w_at_pd_boundary'] for r in rows)
    assert all(r['detection_gate_passed'] for r in rows)
