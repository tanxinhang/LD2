"""Likelihood ratio of normalized complex-projection energy.

For deterministic unknown phase, E=|sqrt(snr) exp(j phi)+CN(0,1)|^2
has density exp(-E-snr) I0(2 sqrt(snr E)). This energy distribution is
phase invariant: no probabilistic phase prior is needed. SNR is supplied
by a receiver-visible model, not estimated from simulator truth.
"""
import numpy as np
from scipy.special import i0e


def energy_log_likelihood_ratio(energy,snr):
    energy=np.asarray(energy,float); snr=np.asarray(snr,float)
    if np.any(~np.isfinite(energy)) or np.any(energy<0) or np.any(~np.isfinite(snr)) or np.any(snr<0):
        raise ValueError('energy and SNR must be finite and nonnegative')
    argument=2*np.sqrt(energy)*np.sqrt(snr)
    return np.log(i0e(argument))+argument-snr
