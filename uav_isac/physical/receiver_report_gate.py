"""Prefix-only report continuation heuristic, no truth or future evidence.

Moment SNR estimates predict terminal noncoherent energy-test power. This
surrogate does not certify mixture-detector benefit; the complete adaptive
detector must be calibrated and validated independently.
"""
import numpy as np
from scipy.stats import gamma,ncx2


def effective_stop(continue_decision,control_delivered):
    """Sender only stops after the schedule-control packet is delivered."""
    decision=np.asarray(continue_decision)
    if decision.dtype!=bool: raise ValueError('boolean decisions required')
    return (~decision)&bool(control_delivered)


def continue_reports(local_prefix,received_remote_prefix,remaining_local,remaining_reports):
    local=np.asarray(local_prefix,float); remote=np.asarray(received_remote_prefix,float)
    if local.ndim!=2 or remote.ndim!=2 or len(local)!=len(remote) or local.shape[1]<1:
        raise ValueError('trial-by-observation prefixes required')
    if np.any(~np.isfinite(local)) or np.any(local<0) or np.any(~np.isfinite(remote)) or np.any(remote<0):
        raise ValueError('finite nonnegative energies required')
    if remaining_local<0 or remaining_reports<0: raise ValueError('negative remaining count')
    if remote.shape[1]==0: return np.ones(len(local),bool)
    ls=np.maximum(local.mean(axis=1)-1.,0.)
    rs=np.maximum(remote.mean(axis=1)-1.,0.)
    count=local.shape[1]+remaining_local+remote.shape[1]
    noncentrality=2*((local.shape[1]+remaining_local)*ls+remote.shape[1]*rs)
    stop_pd=ncx2.sf(2*gamma.isf(.0005,a=count),2*count,noncentrality)
    more_pd=ncx2.sf(2*gamma.isf(.0005,a=count+remaining_reports),
        2*(count+remaining_reports),noncentrality+2*remaining_reports*rs)
    return more_pd>stop_pd
