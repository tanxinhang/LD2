from tools.audit_moving_target_history import audit


def test_moving_target_gate_separates_control_gain_from_alignment_gain():
    result=audit(5); cases=result['cases']
    assert cases['matched_cv']['passes_pd_gate']
    assert cases['matched_cv']['history_pd']>cases['stale_stationary']['history_pd']
    assert result['matched_control_gain']>0.
    assert result['matched_minus_stale_history_pd']>0.
    assert cases['opposite_bias']['passes_pd_gate']
    assert cases['opposite_bias']['controlled_pd_delta']>0.
