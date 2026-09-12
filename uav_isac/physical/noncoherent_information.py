"""Exact first two H0 energy moments for proper complex Gaussian evidence.

For z=mu+n, n~CN(0,C), the phase-invariant mean-energy shift is |mu|^2
and Cov_H0(|z_i|^2,|z_j|^2)=|C_ij|^2. These are not Gaussian energies;
their H0-Deflection is a screening score, not an analytical Gaussian ROC.
"""
import numpy as np


def energy_moment_model(amplitude,noise_covariance,amplitude_derivatives,noise_derivatives):
    """Return (mean shift, H0 covariance, and their coordinate derivatives).

    Amplitude phase is irrelevant for these moments. H1 energy covariance
    generally depends on phase and differs from H0. Calibrate and evaluate
    the final detector independently, including the chosen phase scenario.
    """
    mu=np.asarray(amplitude,complex)
    c=np.asarray(noise_covariance,complex)
    dm=np.asarray(amplitude_derivatives,complex)
    dc=np.asarray(noise_derivatives,complex)
    if mu.ndim!=1 or c.shape!=(len(mu),len(mu)) or dm.ndim!=2 or dm.shape[1:]!=mu.shape or dc.shape!=(len(dm),len(mu),len(mu)):
        raise ValueError('inconsistent energy moment dimensions')
    if any(np.any(~np.isfinite(v)) for v in (mu,c,dm,dc)):
        raise ValueError('finite energy model required')
    if not np.allclose(c,c.conj().T,atol=1e-12,rtol=1e-12) or not np.allclose(dc,dc.conj().transpose(0,2,1),atol=1e-12,rtol=1e-12):
        raise ValueError('Hermitian covariance and derivatives required')
    try:
        np.linalg.cholesky(c)
    except np.linalg.LinAlgError as exc:
        raise ValueError('positive definite noise covariance required') from exc
    return (abs(mu)**2,abs(c)**2,
        2*np.real(mu.conj()[None,:]*dm),2*np.real(c.conj()[None,:,:]*dc))
