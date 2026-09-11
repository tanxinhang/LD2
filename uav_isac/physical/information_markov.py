"""Finite-state reference solver for terminal detection-value maximization.

This is an offline reference, not a distributed policy implementation. Caller
states must be sufficient information states and terminal values must use a
fixed false-alarm rule. No surrogate rewards or resource penalty weights.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class InformationAction:
    name: str
    transition: np.ndarray
    sensing_w: np.ndarray
    communication_w: np.ndarray
    feasible: np.ndarray


def solve_terminal_detection(actions, terminal_pd, stages, power_cap_w=1.):
    """Exact backward recursion for an explicit finite information-state MDP.

    Resource balances, clock, queues, and local knowledge must be included in
    state/transitions. Power arrays have shape (states,nodes). The solver only
    certifies the supplied finite model; it cannot certify its physical origin.
    Every state must have a feasible continuation (e.g. absorbing stop/idle).
    """
    value=np.asarray(terminal_pd,dtype=float)
    if value.ndim!=1 or value.size==0 or np.any(~np.isfinite(value)) or np.any((value<0)|(value>1)):
        raise ValueError('terminal detection values must be probabilities')
    if not isinstance(stages,int) or stages<1 or not np.isfinite(power_cap_w) or power_cap_w<=0:
        raise ValueError('invalid horizon or power cap')
    if not actions or len({a.name for a in actions})!=len(actions):
        raise ValueError('unique nonempty action set required')
    size=value.size; checked=[]; node_count=None
    for a in actions:
        p=np.asarray(a.transition,dtype=float)
        ps=np.asarray(a.sensing_w,dtype=float);pc=np.asarray(a.communication_w,dtype=float)
        feasible=np.asarray(a.feasible)
        if p.shape!=(size,size) or np.any(~np.isfinite(p)) or np.any(p<0) or not np.allclose(p.sum(axis=1),1,rtol=0,atol=1e-12):
            raise ValueError('transition must be row stochastic')
        if ps.ndim!=2 or ps.shape[0]!=size or ps.shape!=pc.shape or ps.shape[1]<1:
            raise ValueError('power arrays must have shape (states,nodes)')
        if node_count is not None and ps.shape[1]!=node_count:
            raise ValueError('inconsistent node count')
        node_count=ps.shape[1]
        if np.any(~np.isfinite(ps)) or np.any(~np.isfinite(pc)) or np.any(ps<0) or np.any(pc<0):
            raise ValueError('invalid powers')
        if feasible.shape!=(size,) or feasible.dtype!=bool:
            raise ValueError('explicit boolean feasibility per state required')
        allowed=feasible & np.all(ps+pc<=power_cap_w+1e-12,axis=1)
        checked.append((p,allowed))
    if not np.all(np.any([a[1] for a in checked],axis=0)):
        raise ValueError('state has no feasible continuation')
    values=[value.copy()]; policies=[]
    for _ in range(stages):
        q=np.stack([np.where(ok,p@value,-np.inf) for p,ok in checked])
        chosen=np.argmax(q,axis=0)
        value=q[chosen,np.arange(size)]
        policies.append(chosen);values.append(value.copy())
    return dict(values_by_steps_remaining=np.asarray(values),
                policy_by_steps_remaining=np.asarray(policies),
                action_names=tuple(a.name for a in actions))
