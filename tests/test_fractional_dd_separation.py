from tools.audit_fractional_dd_separation import audit


def test_coincident_cells_are_unidentifiable_and_projection_removes_nuisance():
    result = audit()
    assert result["status"] == "PASS"
    for pilot in ("dd_impulse", "qpsk"):
        rows = [r for r in result["rows"] if r["pilot"] == pilot]
        assert rows[0]["retained_target_energy_fraction"] < 1e-24
        assert rows[1]["retained_target_energy_fraction"] < 0.01
        assert rows[-1]["retained_target_energy_fraction"] > 0.9
    assert result["block_duration_s"] == 8 / 15625
