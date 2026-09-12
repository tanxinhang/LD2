from tools.audit_unified_receiver_transport import audit


def test_fixed_stop_endpoints_reproduce_local_and_full_without_extra_evidence():
    r=audit(samples=10000,window=4,stop_headroom=True)
    rows={x['method']:x for x in r['rows']}
    assert rows['fixed_stop_0']['pd']==rows['local_uncertainty_mixture']['pd']
    assert rows['fixed_stop_4']['pd']==rows['delivered_uncertainty_mixture']['pd']
    assert [x['scheduled_report_bits'] for x in r['fixed_stop_headroom']]==[0,128,256,384,512]
