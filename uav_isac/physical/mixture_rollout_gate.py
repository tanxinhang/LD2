"""Experimental shared-state terminal-score rollout; no future inputs."""
import numpy as np
from scipy.special import logsumexp
from .energy_likelihood import energy_log_likelihood_ratio as llr
from .observable_link_predictor import predict_delivery_paths


def mixture_rollout_continue(local,remote,bank,prior,remaining,prefix_attempts,
                             threshold,rollouts=64,seed=810001,delivery_prefix=None,transition=None):
    local=np.asarray(local,float); remote=np.asarray(remote,float)
    bank=np.asarray(bank,float); prior=np.asarray(prior,float)
    if local.ndim!=2 or remote.ndim!=2 or len(local)!=len(remote):
        raise ValueError('paired prefix matrices required')
    if bank.shape!=(len(prior),2) or remaining<1 or rollouts<2:
        raise ValueError('invalid state bank or rollout dimensions')
    if any(np.any(~np.isfinite(x)) or np.any(x<0) for x in (local,remote,bank,prior)):
        raise ValueError('finite nonnegative inputs required')
    if np.any(prior<=0) or not np.isclose(prior.sum(),1) or not np.isfinite(threshold):
        raise ValueError('normalized positive prior and finite threshold required')
    if prefix_attempts<remote.shape[1]: raise ValueError('more deliveries than attempts')
    if remote.shape[1]==0: return np.ones(len(local),bool)
    logs=np.log(prior)[None,:]+sum(llr(obs[:,:,None],bank[None,None,:,j]).sum(axis=1)
        for j,obs in enumerate((local,remote)))
    posterior=np.exp(logs-logsumexp(logs,axis=1,keepdims=True))
    rng=np.random.default_rng(seed)
    # Explicit exchangeable Bernoulli approximation, NOT the true correlated
    # radio state. Receiver knows attempted slots and received packet count.
    p=rng.beta(1+remote.shape[1],1+prefix_attempts-remote.shape[1],size=rollouts)
    delivered=rng.random((rollouts,remaining))<p[:,None]
    if transition is not None:
        if len(delivery_prefix)!=prefix_attempts or np.count_nonzero(delivery_prefix)!=remote.shape[1]:
            raise ValueError('delivery prefix disagrees with received evidence')
        delivered=predict_delivery_paths(delivery_prefix,transition,remaining,rollouts,seed+1)
    # Separate stream and interleaved real/imag draws keep rollouts nested.
    draws=np.random.default_rng(seed+2).normal(size=(rollouts,remaining,2,2))
    noise=(draws[:,:,:,0]+1j*draws[:,:,:,1])/np.sqrt(2)
    gain=np.zeros(len(local))
    for state in range(len(prior)):
        energy=abs(noise+np.sqrt(bank[state])[None,None,:])**2
        future_local=llr(energy[:,:,0,None],bank[None,None,:,0]).sum(axis=1)
        remote_wire=energy[:,:,1].astype(np.float32).astype(float)
        future_remote=(llr(remote_wire[:,:,None],bank[None,None,:,1])*delivered[:,:,None]).sum(axis=1)
        for start in range(0,len(local),512):
            sl=slice(start,start+512)
            stopped=logsumexp(logs[sl,None,:]+future_local[None,:,:],axis=2)>threshold
            continued=logsumexp(logs[sl,None,:]+future_local[None,:,:]+future_remote[None,:,:],axis=2)>threshold
            gain[sl]+=posterior[sl,state]*(continued.astype(float)-stopped).mean(axis=1)
    # A Monte Carlo tie is not evidence that communication is worthless.
    return gain>=0
