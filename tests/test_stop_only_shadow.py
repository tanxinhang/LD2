from tools.audit_unified_receiver_transport import audit


def test_stop_request_not_success_pays_control_cost():
    r=audit(samples=2000,window=6,report_gate=True)
    for m in r['gate_metrics'].values():
        assert m['stop_request_fraction']>=m['stop_fraction']
        expected=r['attempted_bits']+128*m['stop_request_fraction']-256*m['stop_fraction']
        assert abs(m['stop_only_shadow_bits']-expected)<1e-8
        assert m['stop_only_shadow_bits']<=m['mean_bits']
        assert 'not implemented transport' in m['stop_only_scope']


def test_integrated_branch_costs_charge_requests_and_remove_only_effective_stops():
    r=audit(samples=2000,window=6,report_gate=True,stop_only_transport=True)
    assert r['stop_only_transport']
    for m in r['gate_metrics'].values():
        expected=r['attempted_bits']+128*m['stop_request_fraction']-256*m['stop_fraction']
        assert abs(m['mean_bits']-expected)<1e-8
        assert m['stop_fraction']==(m['stop_request_fraction'] if r['control_delivered'] else 0.)
        assert 'vectorized branch selection' in m['stop_only_scope']


def test_failed_control_preserves_evidence_but_costs_requests():
    r=audit(samples=2000,report_gate=True,stop_only_transport=True,link_seed=650001)
    assert not r['control_delivered']
    for m in r['gate_metrics'].values():
        assert m['stop_fraction']==0
        assert m['mean_bits']>=r['attempted_bits']
    f=r['gate_factorial']
    assert f['evidence_delta']['delta']==0
    assert f['full_full']['pd']==f['gated_full']['pd']
    assert f['full_gate']['pd']==f['gated_gate']['pd']
