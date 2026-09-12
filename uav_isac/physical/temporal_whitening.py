"""Known-AR(1) receiver-local noise whitening, with no target phase input."""
import numpy as np


def whiten_ar1(observations,rho):
    """Whiten a stationary unit-variance AR(1) time axis (axis 1).

    Rows are independent trials; any trailing template/receiver axes are
    preserved. Cross-template covariance is NOT removed by this operation.
    rho must be a known/calibrated noise parameter, not fitted to H1 truth.
    """
    z=np.asarray(observations)
    if z.ndim<2 or z.shape[1]<1 or np.any(~np.isfinite(z)):
        raise ValueError('finite trial-by-time observations required')
    if not np.isfinite(rho) or not 0<=rho<1:
        raise ValueError('rho must lie in [0,1)')
    out=z.astype(np.result_type(z.dtype,np.float64),copy=True)
    out[:,1:]=(z[:,1:]-rho*z[:,:-1])/np.sqrt(1-rho*rho)
    return out
