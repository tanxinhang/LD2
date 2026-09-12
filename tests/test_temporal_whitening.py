import numpy as np
import pytest
from uav_isac.physical.temporal_whitening import whiten_ar1


def test_whitening_inverts_stationary_ar_noise_and_preserves_templates():
    rng=np.random.default_rng(11001)
    innovations=(rng.normal(size=(100,13,2,3))+1j*rng.normal(size=(100,13,2,3)))/np.sqrt(2)
    noise=innovations.copy(); rho=.6
    for t in range(1,13): noise[:,t]=rho*noise[:,t-1]+np.sqrt(1-rho*rho)*innovations[:,t]
    recovered=whiten_ar1(noise,rho)
    np.testing.assert_allclose(recovered,innovations,atol=1e-14)
    np.testing.assert_array_equal(whiten_ar1(noise,0),noise)
    with pytest.raises(ValueError): whiten_ar1(noise,1.)


@pytest.mark.parametrize('omega',[0.,.2,1.7])
def test_whitening_signal_gain_depends_on_temporal_frequency(omega):
    rho=.6
    signal=np.exp(1j*omega*np.arange(13))[None,:]
    output=whiten_ar1(signal,rho)
    expected=abs(np.exp(1j*omega)-rho)**2/(1-rho**2)
    np.testing.assert_allclose(abs(output[:,1:])**2,expected,atol=1e-14)
    np.testing.assert_allclose(output[:,0],signal[:,0])
