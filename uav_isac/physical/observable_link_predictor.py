"""First-order predictor of observed scheduled packet delivery, not hidden CSI."""
import numpy as np


def fit_delivery_markov(traces):
    traces=np.asarray(traces)
    if traces.ndim!=2 or traces.shape[1]<2 or traces.dtype!=bool:
        raise ValueError('boolean trace matrix required')
    counts=np.ones((2,2))  # explicit unit pseudocount, not calibrated CSI
    np.add.at(counts,(traces[:,:-1].ravel().astype(int),traces[:,1:].ravel().astype(int)),1)
    return counts/counts.sum(axis=1,keepdims=True)


def predict_delivery_paths(prefix,transition,remaining,rollouts,seed):
    prefix=np.asarray(prefix); transition=np.asarray(transition,float)
    if prefix.ndim!=1 or not len(prefix) or prefix.dtype!=bool:
        raise ValueError('ordered boolean received-slot history required')
    if transition.shape!=(2,2) or np.any(~np.isfinite(transition)) or np.any(transition<0) or not np.allclose(transition.sum(axis=1),1):
        raise ValueError('stochastic transition matrix required')
    rng=np.random.default_rng(seed)
    # Row-major uniforms make each path stable when rollout count increases.
    uniforms=rng.random((rollouts,remaining))
    state=np.full(rollouts,int(prefix[-1])); result=np.empty((rollouts,remaining),bool)
    for t in range(remaining):
        state=(uniforms[:,t]<transition[state,1]).astype(int)
        result[:,t]=state
    return result
