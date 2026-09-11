"""Auditable primitives for information-state active sensing.

The information score screens finite geometry/role candidates. It never
replaces held-out detection evaluation at a fixed false-alarm probability.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ConditionalInformation:
    increment: float
    innovation_mean: np.ndarray
    innovation_covariance: np.ndarray
    effective_rank: int


def _psd(matrix, name, tolerance):
    value=np.asarray(matrix,dtype=np.complex128)
    if value.ndim!=2 or value.shape[0]!=value.shape[1] or np.any(~np.isfinite(value)):
        raise ValueError(f'{name} must be a finite square matrix')
    value=.5*(value+value.conj().T)
    scale=max(1.,float(np.linalg.norm(value,ord=2)))
    if float(np.min(np.linalg.eigvalsh(value))) < -tolerance*scale:
        raise ValueError(f'{name} must be positive semidefinite')
    return value


def _pinv_psd(matrix, cutoff):
    values,vectors=np.linalg.eigh(matrix)
    keep=values>cutoff*max(1.,float(np.max(values)))
    return (vectors[:,keep]/values[keep])@vectors[:,keep].conj().T,int(np.count_nonzero(keep))


def conditional_gaussian_information(candidate_mean_shift,history_mean_shift,
        candidate_covariance,history_covariance,candidate_history_covariance,
        *,relative_cutoff=1e-10):
    """Return the Schur-complement increment in Gaussian Deflection.

    This is twice the equal-covariance Gaussian KL increment, not P_D. Models
    with non-PSD joint covariance or unsupported deterministic mean shifts are
    rejected instead of being regularised into artificial information.
    """
    if not np.isfinite(relative_cutoff) or not 0.<relative_cutoff<1.:
        raise ValueError('relative_cutoff must lie in (0,1)')
    da=np.asarray(candidate_mean_shift,np.complex128).reshape(-1)
    dh=np.asarray(history_mean_shift,np.complex128).reshape(-1)
    ca=_psd(candidate_covariance,'candidate_covariance',relative_cutoff)
    ch=_psd(history_covariance,'history_covariance',relative_cutoff)
    cross=np.asarray(candidate_history_covariance,np.complex128)
    if ca.shape!=(da.size,da.size) or ch.shape!=(dh.size,dh.size) or cross.shape!=(da.size,dh.size):
        raise ValueError('mean and covariance dimensions do not agree')
    _psd(np.block([[ca,cross],[cross.conj().T,ch]]),'joint_covariance',relative_cutoff)
    ch_inv,_=_pinv_psd(ch,relative_cutoff)
    if np.linalg.norm(dh-ch@ch_inv@dh)>np.sqrt(relative_cutoff)*max(1.,np.linalg.norm(dh)):
        raise ValueError('history mean shift lies outside covariance support')
    mean=da-cross@ch_inv@dh
    cov=ca-cross@ch_inv@cross.conj().T
    cov=_psd(cov,'innovation_covariance',10.*relative_cutoff)
    inverse,rank=_pinv_psd(cov,relative_cutoff)
    if np.linalg.norm(mean-cov@inverse@mean)>np.sqrt(relative_cutoff)*max(1.,np.linalg.norm(mean)):
        raise ValueError('conditional mean shift lies outside covariance support')
    value=float(np.real(mean.conj()@inverse@mean))
    return ConditionalInformation(max(0.,value),mean,cov,rank)


def generate_motion_candidates(positions,target_belief_position,*,speed_limit_mps,frame_duration_s):
    """Generate stay and one-UAV-at-a-time planar motion primitives."""
    xyz=np.asarray(positions,float); target=np.asarray(target_belief_position,float).reshape(-1)
    step=float(speed_limit_mps)*float(frame_duration_s)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or target.shape!=(3,) or np.any(~np.isfinite(xyz)) or np.any(~np.isfinite(target)):
        raise ValueError('positions and target belief must be finite 3-D coordinates')
    if not np.isfinite(step) or step<=0:
        raise ValueError('speed and frame duration must define a positive step')
    result=[xyz.copy()]
    cardinal=(np.array([1.,0.,0.]),np.array([-1.,0.,0.]),np.array([0.,1.,0.]),np.array([0.,-1.,0.]))
    for node in range(len(xyz)):
        radial=target[:2]-xyz[node,:2]; directions=list(cardinal)
        if np.linalg.norm(radial)>1e-12:
            toward=np.r_[radial/np.linalg.norm(radial),0.]
            lateral=np.array([-toward[1],toward[0],0.])
            directions.extend((toward,lateral,-lateral))
        for direction in directions:
            moved=xyz.copy(); moved[node]+=step*direction
            if not any(np.allclose(moved,item,atol=1e-12,rtol=0) for item in result):
                result.append(moved)
    return tuple(result)


def check_physical_feasibility(positions,previous_positions,*,speed_limit_mps,
        frame_duration_s,minimum_separation_m,lower_bound,upper_bound):
    """Check one-step speed, region bounds, and pairwise separation."""
    xyz=np.asarray(positions,float); old=np.asarray(previous_positions,float)
    lower=np.asarray(lower_bound,float); upper=np.asarray(upper_bound,float)
    if xyz.shape!=old.shape or xyz.ndim!=2 or xyz.shape[1]!=3 or lower.shape!=(3,) or upper.shape!=(3,):
        raise ValueError('inconsistent physical-feasibility dimensions')
    scalars=np.asarray([speed_limit_mps,frame_duration_s,minimum_separation_m],float)
    if np.any(~np.isfinite(np.r_[xyz.ravel(),old.ravel(),lower,upper,scalars])) or speed_limit_mps<0 or frame_duration_s<=0 or minimum_separation_m<0 or np.any(lower>upper):
        raise ValueError('invalid physical-feasibility inputs')
    if np.any(xyz<lower) or np.any(xyz>upper) or np.any(np.linalg.norm(xyz-old,axis=1)>speed_limit_mps*frame_duration_s+1e-10):
        return False
    distance=np.linalg.norm(xyz[:,None]-xyz[None,:],axis=-1); np.fill_diagonal(distance,np.inf)
    return bool(np.min(distance)>=minimum_separation_m-1e-10)


def generate_single_tx_roles(node_count):
    """Return half-duplex roles with exactly one Tx (0), all others Rx (1)."""
    if int(node_count)!=node_count or node_count<2:
        raise ValueError('node_count must be an integer of at least two')
    return tuple(np.where(np.arange(node_count)==tx,0,1) for tx in range(node_count))


def top_m_exact_maxmin_selection(surrogate_information,exact_information,feasible,*,top_m):
    """Use the proxy only to screen; decide among Top-M using exact values."""
    proxy=np.asarray(surrogate_information,float); exact=np.asarray(exact_information,float)
    allowed=np.asarray(feasible)
    if proxy.ndim!=2 or exact.shape!=proxy.shape or allowed.shape!=(len(proxy),) or allowed.dtype!=bool:
        raise ValueError('candidate information arrays have incompatible shapes')
    if np.any(~np.isfinite(proxy)) or np.any(~np.isfinite(exact)) or np.any(proxy<0) or np.any(exact<0):
        raise ValueError('information values must be finite and non-negative')
    if int(top_m)!=top_m or not 1<=top_m<=len(proxy) or not np.any(allowed):
        raise ValueError('invalid Top-M request or no feasible candidate')
    order=sorted(np.flatnonzero(allowed),key=lambda i:(-float(np.min(proxy[i])),int(i)))
    screened=np.asarray(order[:min(top_m,len(order))],int)
    selected=min(screened,key=lambda i:(-float(np.min(exact[i])),int(i)))
    return dict(selected_index=int(selected),screened_indices=screened,
        selected_worst_exact_information=float(np.min(exact[selected])))


def receiver_local_temporal_covariance(receiver_ids,templates,epoch_ids,*,correlation):
    """Build a PSD covariance for persistent receiver-local Gaussian clutter.

    The kernel is zero across different physical receivers. For a common
    receiver it is the product of an AR(1) temporal kernel and the complex OTFS
    template Gram matrix, hence PSD by the Schur product theorem.
    """
    receivers=np.asarray(receiver_ids,int).reshape(-1)
    waveform=np.asarray(templates,np.complex128)
    epochs=np.asarray(epoch_ids,int).reshape(-1)
    rho=float(correlation)
    if waveform.ndim!=2 or len(receivers)!=len(waveform) or epochs.shape!=receivers.shape:
        raise ValueError('receiver, template, and epoch dimensions must agree')
    if len(receivers)<1 or np.any(~np.isfinite(waveform)) or np.any(epochs<0) or not np.isfinite(rho) or not 0<=rho<1:
        raise ValueError('invalid temporal covariance inputs')
    norms=np.linalg.norm(waveform,axis=1)
    if np.any(norms<=0): raise ValueError('templates must have positive norm')
    normalized=waveform/norms[:,None]
    gram=normalized.conj()@normalized.T
    temporal=rho**np.abs(epochs[:,None]-epochs[None,:])
    same=receivers[:,None]==receivers[None,:]
    covariance=np.where(same,temporal*gram,0.)
    return _psd(covariance,'temporal_covariance',1e-10)


def robust_hypothesis_maxmin_selection(information,feasible):
    """Choose the action maximizing worst information over hypotheses/targets."""
    values=np.asarray(information,float); allowed=np.asarray(feasible)
    if values.ndim!=3 or values.shape[0]<1 or values.shape[1]<1 or values.shape[2]<1:
        raise ValueError('information must have shape (actions,hypotheses,targets)')
    if allowed.shape!=(len(values),) or allowed.dtype!=bool or not np.any(allowed):
        raise ValueError('a nonempty boolean feasible action mask is required')
    if np.any(~np.isfinite(values)) or np.any(values<0):
        raise ValueError('information must be finite and non-negative')
    worst=np.min(values,axis=(1,2)); index=int(np.argmax(np.where(allowed,worst,-np.inf)))
    return dict(selected_index=index,worst_information=float(worst[index]),
        hypothesis_target_information=values[index].copy())
