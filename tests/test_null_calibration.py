from tools.audit_unified_receiver_transport import audit


def test_null_only_returns_no_detection_results():
    result=audit(samples=5000,report_gate=True,null_only=True)
    assert 'rows' not in result and 'gate_comparison' not in result
    assert 'prefix_gated_mixture' in result['thresholds']
    assert result['tail_probability_upper95']>.0005


def test_larger_sample_tightens_fixed_order_tail_bound():
    small=audit(samples=1000,null_only=True)
    large=audit(samples=10000,null_only=True)
    assert large['tail_probability_upper95']<small['tail_probability_upper95']
