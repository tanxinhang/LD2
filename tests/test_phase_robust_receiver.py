import numpy as np
import pytest
from tools.audit_phase_robust_receiver import complex_receiver_mean


def record(s):
    return [dict(receivers=np.array([0]),templates=np.array([s],complex),mean=np.array([2.]))]


def test_quadrature_phase_preserves_complex_energy_and_loses_real_shift():
    prediction=record([1,0])
    aligned,_=complex_receiver_mean(prediction,prediction)
    rotated,_=complex_receiver_mean(prediction,record([1j,0]))
    assert abs(rotated)**2==pytest.approx(abs(aligned)**2)
    assert rotated.real==pytest.approx(0.)
    assert abs(aligned)**2==pytest.approx(2.)
    orthogonal,_=complex_receiver_mean(prediction,record([0,1]))
    assert abs(orthogonal)==pytest.approx(0.)
