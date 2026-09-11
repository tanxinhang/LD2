import numpy as np
import pytest
from tools.audit_prediction_matched_receiver import receiver_shift


def record(template):
    return [dict(receivers=np.array([0]),templates=np.array([template],complex),mean=np.array([2.]))]


def test_receiver_loses_orthogonal_signal_and_preserves_phase_sign():
    predicted=record([1.,0.])
    assert receiver_shift(predicted,predicted)==pytest.approx(2.)
    assert receiver_shift(predicted,record([0.,1.]))==pytest.approx(0.)
    assert receiver_shift(predicted,record([-1.,0.]))==pytest.approx(-2.)
