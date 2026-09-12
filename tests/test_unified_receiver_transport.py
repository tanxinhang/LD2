from tools.audit_unified_receiver_transport import audit


def test_unified_audit_accounts_for_reports_and_separates_oracle():
    r=audit(samples=10000,window=4)
    assert r['attempted_bits']==4*128
    assert len(r['delivery_mask'])==4
    assert r['ack_count']==r['retransmission_count']==0
    assert max(r['peak_node_rf_w'])<=1
    assert r['communication_energy_j']>0
    assert r['reserved_elapsed_ms']>4*1.024
    assert len(r['rows'])==9
    assert abs(sum(r['mixture_probabilities'])-1)<1e-12
    assert min(r['mixture_probabilities'])>0
    assert not r['deployable_detection_certified']
