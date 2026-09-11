from tools.audit_uncertain_maneuver_ensemble import audit


def test_seeded_maneuver_ensemble_reports_paired_uncertainty_without_overclaim():
    result=audit(seeds=8,horizon=3,bootstrap_draws=1000)
    assert len(result['rows'])==8
    assert not result['deployable_detection_certified']
    assert result['evidence_class']=='CENTRALIZED_ORACLE_GAUSSIAN_MANEUVER_REFERENCE'
    assert len(result['paired_seed_bootstrap_ci95'])==2
    assert result['robust_gain_supported']==(result['paired_seed_bootstrap_ci95'][0]>0)
    assert result['robust_worst_passes_pd_gate']==(result['robust_worst_pd']>.8)
