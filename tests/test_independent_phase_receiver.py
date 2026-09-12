import numpy as np
from tools.audit_independent_phase_receiver import statistics


def test_vector_energy_keeps_opposing_observations_lost_in_scalar_sum():
    direction=np.ones(2)/np.sqrt(2)
    scalar,vector=statistics(np.array([[1.,-1.]],complex),direction)
    assert scalar[0]==0
    assert vector[0]==2


def test_both_statistics_preserve_global_phase():
    rng=np.random.default_rng(3)
    y=rng.normal(size=(20,3))+1j*rng.normal(size=(20,3))
    direction=np.ones(3)/np.sqrt(3)
    for a,b in zip(statistics(y,direction),statistics(y*np.exp(.7j),direction)):
        np.testing.assert_allclose(a,b)
