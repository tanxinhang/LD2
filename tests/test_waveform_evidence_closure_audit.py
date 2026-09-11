from tools.audit_waveform_evidence_closure import run_audit


def test_waveform_evidence_closure_gates_pass_on_independent_splits():
    result = run_audit(samples=3000, p_fa=0.01, seed=62000)
    assert result["status"] == "PASS"
    assert all(result["checks"].values())
    assert result["g2"]["covariance_policy"] == "equal_covariance_pooled"
    assert (
        result["g2"]["unequal_covariance_control_policy"]
        == "h0_fixed_pfa_fallback"
    )
