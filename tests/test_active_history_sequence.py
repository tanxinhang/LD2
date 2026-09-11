from tools.audit_active_history_sequence import audit


def test_two_epoch_history_gate_changes_action_only_for_real_information_gain():
    result=audit(samples=20000,seed=11)
    assert result['communication_power_w']==0.
    assert result['action_changed']==result['history_gain_survives']
    assert abs(result['history_conditioned']['pd']-result['history_conditioned']['pd_theory'])<.02
