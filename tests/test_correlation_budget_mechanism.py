from tools.audit_correlation_budget_mechanism import audit


def test_controlled_grid_is_falsifiable_and_has_expected_limits():
    report = audit(
        cases=4,
        correlations=(0.0, 0.8, 0.95),
        budget_evidence_counts=(1, 2, 3, 4, 5, 6, 7, 8),
        bits_per_evidence=64,
        p_fa=1.0e-3,
        target_pd=0.9,
        seed_offset=44000,
    )
    assert report["status"] == "PASS"
    assert report["checks"] == {
        "rho_zero_reduces_to_unaware": True,
        "high_rho_has_positive_paired_ci": True,
        "high_rho_saves_bits_at_target_pd": True,
        "proposed_never_exceeds_exhaustive_reference": True,
    }
    assert {
        row["budget_bits"] for row in report["rows"]
    } == {64, 128, 192, 256, 320, 384, 448, 512}
    assert all(
        row["budget_bits"] == 64 * row["budget_evidence_count"]
        for row in report["rows"]
    )
