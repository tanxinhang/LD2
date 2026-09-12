import numpy as np
import pytest
from scipy.stats import ncx2
from uav_isac.physical.energy_likelihood import energy_log_likelihood_ratio


def test_energy_density_matches_noncentral_chisquare_ratio():
    energy=np.array([.01,.5,2.,10.]); snr=1.7
    expected=ncx2.logpdf(2*energy,2,2*snr)-ncx2.logpdf(2*energy,2,0)
    np.testing.assert_allclose(energy_log_likelihood_ratio(energy,snr),expected)
    np.testing.assert_allclose(energy_log_likelihood_ratio(energy,0),0)


def test_negative_energy_fails_closed():
    with pytest.raises(ValueError): energy_log_likelihood_ratio([-1],1)
