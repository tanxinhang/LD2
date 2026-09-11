"""Log-domain forward evidence update with fresh source-local emissions.

Local and remote log-likelihood ratios must be conditionally independent given
the hidden state. Sending a posterior containing old evidence violates this
contract. Correlated clutter needs a joint emission model instead of addition.
"""
import numpy as np
from scipy.special import logsumexp


def update(log_state_mass, transition, local_loglr, remote_loglr=None):
    """Unnormalized H1 state masses relative to H0, preserving total log LR."""
    old=np.asarray(log_state_mass,float)
    p=np.asarray(transition,float)
    local=np.asarray(local_loglr,float)
    if old.ndim<1 or old.shape!=local.shape or p.shape!=(old.shape[-1],)*2:
        raise ValueError('inconsistent state dimensions')
    if np.any(~np.isfinite(p)) or np.any(p<0) or not np.allclose(p.sum(axis=1),1,atol=1e-12,rtol=0):
        raise ValueError('transition must be stochastic')
    if np.any(np.isnan(old)) or np.any(np.isposinf(old)) or np.any(~np.isfinite(local)):
        raise ValueError('invalid log evidence')
    logp=np.full(p.shape,-np.inf)
    np.log(p,out=logp,where=p>0)
    predicted=logsumexp(old[..., :, None]+logp,axis=-2)
    result=predicted+local
    if remote_loglr is not None:
        remote=np.asarray(remote_loglr,float)
        if remote.shape!=old.shape or np.any(~np.isfinite(remote)):
            raise ValueError('invalid remote evidence')
        result=result+remote
    return result
