"""Conservative simultaneous confidence interval for paired success rates."""
import numpy as np
from scipy.stats import beta


def paired_difference_interval(candidate,reference,error=.05):
    """Bound p(candidate only)-p(reference only) using binomial intervals.

    Two Clopper-Pearson intervals, each with error/2, imply coverage >=1-error
    by the union bound. Pair dependence is preserved; trials must be independent.
    Callers allocate error across prespecified comparisons when appropriate.
    """
    a=np.asarray(candidate); b=np.asarray(reference)
    if a.dtype!=bool or b.dtype!=bool or a.ndim!=1 or a.shape!=b.shape or not len(a):
        raise ValueError('nonempty paired boolean arrays required')
    if not 0<error<1: raise ValueError('error must be in (0,1)')
    positive=int(np.sum(a&~b)); negative=int(np.sum(~a&b)); n=len(a)
    def bounds(k):
        lo=float(beta.ppf(error/4,k,n-k+1)) if k else 0.
        hi=float(beta.ppf(1-error/4,k+1,n-k)) if k<n else 1.
        return lo,hi
    pl,pu=bounds(positive); nl,nu=bounds(negative)
    return dict(delta=(positive-negative)/n,interval=[pl-nu,pu-nl],
        candidate_only=positive,reference_only=negative,trials=n)
