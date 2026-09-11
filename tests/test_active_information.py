import numpy as np
import pytest
from uav_isac.physical.active_information import (check_physical_feasibility,
    conditional_gaussian_information,generate_motion_candidates,
    generate_single_tx_roles,top_m_exact_maxmin_selection)
from uav_isac.physical.active_information import receiver_local_temporal_covariance
from uav_isac.physical.active_information import robust_hypothesis_maxmin_selection


def test_conditional_information_suppresses_redundant_history():
    novel=conditional_gaussian_information([1.],[1.],[[1.]],[[1.]],[[0.]])
    redundant=conditional_gaussian_information([1.],[1.],[[1.]],[[1.]],[[.999]])
    assert novel.increment==pytest.approx(1.)
    assert redundant.increment<1e-3


def test_conditional_information_rejects_non_psd_joint_model():
    with pytest.raises(ValueError,match='joint_covariance'):
        conditional_gaussian_information([1.],[1.],[[1.]],[[1.]],[[1.1]])


def test_motion_and_physical_feasibility_contracts():
    old=np.array([[-10.,0.,20.],[10.,0.,20.],[0.,20.,20.]])
    candidates=generate_motion_candidates(old,[0.,0.,0.],speed_limit_mps=10.,frame_duration_s=.1)
    assert len(candidates)>1 and any(abs(item[0,1])>0 for item in candidates)
    args=dict(speed_limit_mps=10.,frame_duration_s=.1,minimum_separation_m=5.,
        lower_bound=np.array([-20.,-20.,10.]),upper_bound=np.array([20.,20.,30.]))
    assert check_physical_feasibility(old,old,**args)
    too_fast=old.copy(); too_fast[0,0]+=2.
    assert not check_physical_feasibility(too_fast,old,**args)


def test_roles_are_half_duplex_and_exactly_one_transmitter():
    roles=generate_single_tx_roles(3)
    assert len(roles)==3
    assert all(np.count_nonzero(x==0)==1 and np.count_nonzero(x==1)==2 for x in roles)


def test_top_m_proxy_only_screens_and_exact_score_decides():
    result=top_m_exact_maxmin_selection([[4.,4.],[3.,3.],[2.,2.]],
        [[1.,1.],[5.,5.],[9.,9.]],np.ones(3,dtype=bool),top_m=2)
    assert result['screened_indices'].tolist()==[0,1]
    assert result['selected_index']==1


def test_receiver_local_temporal_covariance_is_psd_and_cross_receiver_zero():
    covariance=receiver_local_temporal_covariance([0,0,1],[[1,0],[0,1],[1,0]],
        [0,1,1],correlation=.8)
    assert np.min(np.linalg.eigvalsh(covariance))>=-1e-12
    assert covariance[0,2]==0 and covariance[1,2]==0


def test_robust_selection_maximizes_worst_hypothesis_without_weights():
    result=robust_hypothesis_maxmin_selection(
        np.array([[[9.],[1.]],[[4.],[4.]],[[3.],[5.]]]),np.ones(3,dtype=bool))
    assert result['selected_index']==1
    assert result['worst_information']==4.
