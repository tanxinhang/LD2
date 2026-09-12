import numpy as np
import pytest
from uav_isac.physical.mixture_rollout_gate import mixture_rollout_continue


def test_no_remote_fails_closed():
    assert mixture_rollout_continue(np.ones((2,4)),np.empty((2,0)),
        [[1.,1.]],[1.],9,4,4.).all()


def test_zero_information_reports_do_not_create_gain_or_stop_on_tie():
    assert mixture_rollout_continue(np.ones((2,4)),np.ones((2,4)),
        [[1.,0.]],[1.],9,4,4.).all()


def test_fixed_rollouts_are_reproducible_and_history_order_invariant():
    rng=np.random.default_rng(99)
    local=rng.exponential(size=(20,4)); remote=rng.exponential(size=(20,3))
    args=([[.5,.2],[1.,2.]],[.4,.6],9,4,4.)
    a=mixture_rollout_continue(local,remote,*args)
    b=mixture_rollout_continue(local[:,::-1],remote[:,::-1],*args)
    assert np.array_equal(a,b)
    with pytest.raises(ValueError):
        mixture_rollout_continue(local,remote,[[1.,1.]],[1.],9,2,4.)
