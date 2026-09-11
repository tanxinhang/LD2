import numpy as np
from tools.audit_clutter_projected_detection import detector


def test_colored_noise_normalization_nuisance_null_and_information_loss():
    s = np.array([1, 0, 0], dtype=complex)
    t = np.array([0, 1, 0], dtype=complex)
    c = np.array([1, 1j, 1], dtype=complex)/np.sqrt(3)
    w, info = detector(s, t, c, 4)
    covariance = np.eye(3)+4*np.outer(c,c.conj())
    assert abs(np.vdot(w, covariance@w)-1)<1e-12
    assert abs(np.vdot(w,t))<1e-12
    assert 0<info<1
    assert detector(s,s,c,4)[0] is None
