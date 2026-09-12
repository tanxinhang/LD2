import numpy as np
import pytest
from uav_isac.physical.noncoherent_information import energy_moment_model
from uav_isac.physical.correlated_soft_evidence import conditional_deflection_gain,conditional_deflection_gradient


def test_complex_energy_covariance_matches_sampled_h0():
    c=np.array([[1.,.3+.4j],[.3-.4j,1.5]])
    shift,cov,_,_=energy_moment_model([1,2j],c,np.zeros((1,2)),np.zeros((1,2,2)))
    rng=np.random.default_rng(1009)
    n=(rng.normal(size=(150000,2))+1j*rng.normal(size=(150000,2)))/np.sqrt(2)
    e=abs(n@np.linalg.cholesky(c).T)**2
    np.testing.assert_allclose(np.cov(e,rowvar=False),cov,atol=.035)
    np.testing.assert_allclose(shift,[1,4])
    assert cov[0,1]==pytest.approx(.25)


def test_phase_invariance_and_energy_gradient_chain_rule():
    mu=np.array([1+.5j,.8-.3j]); c=np.array([[1,.2j],[-.2j,1.]])
    dm=np.array([[.2+.1j,-.1j]]); dc=np.array([[[.1,.03j],[-.03j,-.05]]])
    model=energy_moment_model(mu,c,dm,dc)
    _,gradient=conditional_deflection_gradient(*model,(0,),1)
    h=1e-5
    plus=energy_moment_model(mu+h*dm[0],c+h*dc[0],dm,dc)
    minus=energy_moment_model(mu-h*dm[0],c-h*dc[0],dm,dc)
    numeric=(conditional_deflection_gain(*plus[:2],(0,),1)-conditional_deflection_gain(*minus[:2],(0,),1))/(2*h)
    assert gradient[0]==pytest.approx(numeric,rel=1e-8,abs=1e-8)
    rotated=energy_moment_model(mu*np.exp(1j*1.7),c,dm*np.exp(1j*1.7),dc)
    for a,b in zip(model,rotated): np.testing.assert_allclose(a,b,atol=1e-14)
