import numpy as np
import pytest
from uav_isac.physical.receiver_report_gate import history_predictive_continue


def test_history_order_and_state_permutation_do_not_change_decision():
    rng=np.random.default_rng(12)
    local=rng.exponential(size=(100,4)); remote=rng.exponential(size=(100,3))
    bank=np.array([[.2,.1],[2.,1.]])
    prior=np.array([.3,.7])
    a=history_predictive_continue(local,remote,bank,prior,9,9)
    b=history_predictive_continue(local[:,::-1],remote[:,::-1],bank[::-1],prior[::-1],9,9)
    assert np.array_equal(a,b)


def test_missing_remote_fails_closed_and_invalid_prior_rejected():
    local=np.ones((3,4)); remote=np.empty((3,0)); bank=np.array([[1.,.5]])
    assert history_predictive_continue(local,remote,bank,[1.],9,9).all()
    with pytest.raises(ValueError):
        history_predictive_continue(local,remote,bank,[2.],9,9)
