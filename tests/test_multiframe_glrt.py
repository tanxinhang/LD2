import numpy as np
import pytest
from uav_isac.physical.multiframe_glrt import trajectory_bank_scores


def test_duplicate_columns_do_not_double_information():
    y=np.array([[2.,1.]],complex)
    one=np.array([[1.],[0.]])
    a,_,rank=trajectory_bank_scores(y,[np.column_stack([one,one])],np.eye(2))
    b,_,_=trajectory_bank_scores(y,[one],np.eye(2))
    np.testing.assert_allclose(a,b)
    assert rank.tolist()==[1]


def test_frame_amplitude_phase_changes_preserve_subspace():
    rng=np.random.default_rng(5)
    y=rng.normal(size=(30,4))+1j*rng.normal(size=(30,4))
    design=np.array([[1,0],[1j,0],[0,1],[0,1j]])
    a,_,_=trajectory_bank_scores(y,[design],np.eye(4))
    b,_,_=trajectory_bank_scores(y,[design@np.diag([2j,-3])],np.eye(4))
    np.testing.assert_allclose(a,b)


def test_bank_chooses_whole_trajectory_not_per_frame_cherry_picking():
    designs=[np.eye(4)[:,[0,2]],np.eye(4)[:,[1,3]]]
    score,_,_=trajectory_bank_scores(np.array([[1,0,0,1]]),designs,np.eye(4))
    assert score[0]==pytest.approx(1.)


def test_colored_noise_single_rank_has_unit_mean_null_score():
    rng=np.random.default_rng(7); c=np.array([[2.,.6j],[-.6j,1.]])
    z=(rng.normal(size=(30000,2))+1j*rng.normal(size=(30000,2)))/np.sqrt(2)
    y=z@np.linalg.cholesky(c).T
    score,_,_=trajectory_bank_scores(y,[np.ones((2,1))],c)
    assert np.mean(score)==pytest.approx(1.,abs=.025)
