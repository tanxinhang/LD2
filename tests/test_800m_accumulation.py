import pytest
from scipy.stats import chi2, ncx2
from tools.audit_800m_spatiotemporal_accumulation import expected_pd


def test_erased_remote_evidence_reduces_to_local_only():
    assert expected_pd(8,15,0.,.573)==pytest.approx(expected_pd(8,1,1.,.573))


def test_delivered_count_threshold_and_independent_sum():
    assert expected_pd(8,15,1.,.573)==pytest.approx(ncx2.sf(chi2.isf(.001,240),240,120*.573))
    assert expected_pd(8,15,.9,.573)<expected_pd(8,15,1.,.573)
