from tools.audit_common_probe_energy_counterexample import audit


def test_common_probe_energy_counterexample_is_fair_and_fail_closed():
    result = audit()
    assert result["status"] == "PASS"
    assert result["resource_equality"]["passed"] is True
    assert result["communication_qos"]["same_for_both_modes"] is True
    assert result["communication_qos"]["finite_blocklength_qos_certified"] is False
    assert result["common_probe_lp_mapping_rejected"] is True
    assert result["common_to_separable_ratio"] == [2.0, 2.0]
