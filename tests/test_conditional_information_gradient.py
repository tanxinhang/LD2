import numpy as np
import pytest
from uav_isac.physical.correlated_soft_evidence import (
    conditional_deflection_gain,conditional_deflection_gradient)


def test_gradient_matches_finite_difference_with_changing_covariance():
    rng=np.random.default_rng(1008)
    for _ in range(20):
        a=rng.normal(size=(4,4)); sigma=a@a.T+np.eye(4)
        mean=rng.normal(size=4); dm=rng.normal(size=(3,4))
        ds=rng.normal(size=(3,4,4)); ds=(ds+ds.transpose(0,2,1))/2
        gain,gradient=conditional_deflection_gradient(mean,sigma,dm,ds,(0,1),2)
        assert gain==pytest.approx(conditional_deflection_gain(mean,sigma,(0,1),2))
        for coordinate in range(3):
            h=1e-5
            plus=conditional_deflection_gain(mean+h*dm[coordinate],sigma+h*ds[coordinate],(0,1),2)
            minus=conditional_deflection_gain(mean-h*dm[coordinate],sigma-h*ds[coordinate],(0,1),2)
            assert gradient[coordinate]==pytest.approx((plus-minus)/(2*h),rel=2e-7,abs=2e-7)


def test_high_correlation_is_not_sufficient_for_redundancy():
    covariance=np.array([[1.,.99],[.99,1.]])
    assert conditional_deflection_gain([1.,1.],covariance,(0,),1)<.01
    assert conditional_deflection_gain([1.,-1.],covariance,(0,),1)>100


def test_duplicate_and_invalid_covariance_derivatives():
    gain,gradient=conditional_deflection_gradient([1.,2.],np.eye(2),[[1.,1.]],
        np.zeros((1,2,2)),(0,),0)
    assert gain==0 and np.array_equal(gradient,[0.])
    with pytest.raises(ValueError,match='symmetric'):
        conditional_deflection_gradient([1.,2.],np.eye(2),[[1.,1.]],
            [[[0.,1.],[0.,0.]]],(0,),1)
