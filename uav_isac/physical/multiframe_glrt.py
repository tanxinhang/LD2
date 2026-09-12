"""Known-covariance multiframe GLRT with deterministic unknown amplitudes.

Each supplied design is one complete, externally constrained trajectory.
Columns correspond to independently unknown frame/source complex amplitudes.
No per-frame maximization across different trajectory identities is performed.
"""
import numpy as np


def whitened_trajectory_basis(design,noise_covariance):
    design=np.asarray(design,complex)
    covariance=np.asarray(noise_covariance,complex)
    if design.ndim!=2 or min(design.shape)<1 or covariance.shape!=(len(design),len(design)):
        raise ValueError('incompatible nonempty design and covariance')
    if np.any(~np.isfinite(design)) or np.any(~np.isfinite(covariance)):
        raise ValueError('finite inputs required')
    if not np.allclose(covariance,covariance.conj().T,rtol=1e-10,atol=1e-12):
        raise ValueError('noise covariance must be Hermitian')
    lower=np.linalg.cholesky(covariance)
    whitened=np.linalg.solve(lower,design)
    u,s,_=np.linalg.svd(whitened,full_matrices=False)
    cutoff=max(whitened.shape)*np.finfo(float).eps*(s[0] if len(s) else 0.)
    rank=int(np.count_nonzero(s>cutoff))
    if rank==0: raise ValueError('zero signal subspace')
    return lower,u[:,:rank]


def trajectory_bank_scores(observations,designs,noise_covariance):
    """Return per-trial maximum over complete trajectory GLRT scores.

    For a single rank-r trajectory, H0 score is Gamma(r,1) for proper complex
    Gaussian noise. A bank maximum needs a separate global H0 calibration.
    """
    y=np.asarray(observations,complex)
    if y.ndim!=2 or np.any(~np.isfinite(y)) or not designs:
        raise ValueError('finite trial-by-observation array and designs required')
    scores=[]; ranks=[]
    for design in designs:
        lower,basis=whitened_trajectory_basis(design,noise_covariance)
        if y.shape[1]!=len(lower): raise ValueError('observation dimension mismatch')
        white=np.linalg.solve(lower,y.T).T
        scores.append(np.sum(abs(white@basis.conj())**2,axis=1))
        ranks.append(basis.shape[1])
    bank=np.stack(scores,axis=1)
    return bank.max(axis=1),bank,np.asarray(ranks)
