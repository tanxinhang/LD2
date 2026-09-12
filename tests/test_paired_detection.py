import numpy as np
import pytest
from uav_isac.physical.paired_detection import paired_difference_interval


def test_paired_count_and_swap_symmetry():
    a=np.array([True,True,False,False]); b=np.array([False,True,True,True])
    r=paired_difference_interval(a,b); reverse=paired_difference_interval(b,a)
    assert r['candidate_only']==1 and r['reference_only']==2
    assert r['delta']==-.25
    np.testing.assert_allclose(r['interval'],[-reverse['interval'][1],-reverse['interval'][0]])


def test_identical_observed_decisions_do_not_prove_population_equality():
    a=np.ones(100,dtype=bool); r=paired_difference_interval(a,a)
    assert r['interval'][0]<0<r['interval'][1]
    with pytest.raises(ValueError): paired_difference_interval([1],[1])
