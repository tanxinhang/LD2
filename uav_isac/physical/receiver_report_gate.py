"""Prefix-only report continuation heuristic, no truth or future evidence.

Moment SNR estimates predict terminal noncoherent energy-test power. This
surrogate does not certify mixture-detector benefit; the complete adaptive
detector must be calibrated and validated independently.
"""
import numpy as np
from scipy.stats import gamma,ncx2
from scipy.special import logsumexp
from .energy_likelihood import energy_log_likelihood_ratio


def history_predictive_continue(local,remote,snr_bank,probabilities,remaining_local,remaining_reports):
    """H1-conditional posterior predictive energy-test power, not mixture power.

    A static shared state is the identity-transition special case of a Markov
    belief update. All delivered observations enter once; no H0/H1 prior is
    introduced. Future delivery is assumed perfect for this forecasting proxy.
    """
    local=np.asarray(local,float); remote=np.asarray(remote,float)
    bank=np.asarray(snr_bank,float); prior=np.asarray(probabilities,float)
    if local.ndim!=2 or remote.ndim!=2 or len(local)!=len(remote):
        raise ValueError('paired observation matrices required')
    if bank.ndim!=2 or bank.shape[1]!=2 or prior.shape!=(len(bank),):
        raise ValueError('state bank and probabilities disagree')
    if any(np.any(~np.isfinite(x)) or np.any(x<0) for x in (local,remote,bank,prior)):
        raise ValueError('finite nonnegative inputs required')
    if np.any(prior<=0) or not np.isclose(prior.sum(),1):
        raise ValueError('positive normalized state probabilities required')
    if remaining_local<1 or remaining_reports<0:
        raise ValueError('positive remaining local count required')
    logs=np.log(prior)[None,:]+sum(
        energy_log_likelihood_ratio(obs[:,:,None],bank[None,None,:,j]).sum(axis=1)
        for j,obs in enumerate((local,remote)))
    posterior=np.exp(logs-logsumexp(logs,axis=1,keepdims=True))
    observed=local.sum(axis=1)+remote.sum(axis=1)
    prefix_count=local.shape[1]+remote.shape[1]
    powers=[]
    for extra in (0,remaining_reports):
        count=remaining_local+extra
        residual=gamma.isf(.0005,a=prefix_count+count)-observed
        power=ncx2.sf(2*np.maximum(residual[:,None],0),2*count,
            2*(remaining_local*bank[None,:,0]+extra*bank[None,:,1]))
        powers.append(np.sum(posterior*power,axis=1))
    keep=powers[1]>powers[0]
    if remote.shape[1]==0: keep[:]=True
    return keep


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
