import numpy as np
import pytest
from scipy.special import logsumexp
from uav_isac.physical.markov_evidence import update


def test_forward_matches_explicit_trajectory_likelihood():
    p=np.array([[.8,.2],[.3,.7]])
    prior=np.array([.4,.6]);a=np.array([.2,-.3]);b=np.array([-.4,.7]);c=np.array([.8,-.1])
    first=update(np.log(prior),p,a,b)
    second=update(first,p,c)
    exact=sum(prior[i]*p[i,j]*np.exp(a[j]+b[j])*p[j,k]*np.exp(c[k])
              for i in range(2) for j in range(2) for k in range(2))
    assert np.exp(logsumexp(second))==pytest.approx(exact)


def test_missing_remote_message_is_neutral_not_a_negative_observation():
    old=np.log([.5,.5]);a=np.array([.1,.2])
    assert np.allclose(update(old,np.eye(2),a),old+a)
