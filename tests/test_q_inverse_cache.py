"""Property tests for the memoized scalar Q_inverse path (roadmap O7).

O7 guarantees:
  (a) scalar (size==1) inputs are memoized but BIT-FOR-BIT identical to the
      vectorized formula on the same platform;
  (b) compute_PD behaviour is unchanged;
  (c) output shape mirrors the input shape (0-d for Python float, (n,) for
      arrays), as before the change.
"""

import numpy as np

from uav_isac.utils.math_utils import (
    Q_inverse,
    compute_PD,
)


def test_scalar_path_matches_vectorized_formula():
    """For p in a spread of (0,1), scalar memoized result == vector formula."""
    from scipy.special import erfinv

    def vec(p):
        p = np.clip(np.asarray(p, dtype=np.float64), 1e-15, 1.0 - 1e-15)
        return np.sqrt(2.0) * erfinv(1.0 - 2.0 * p)

    for p in (1e-10, 0.001, 0.05, 0.5, 0.9, 1.0 - 1e-12):
        cached = float(Q_inverse(np.array([p]))[0])
        direct = float(vec(np.array([p]))[0])
        np.testing.assert_allclose(cached, direct, rtol=0.0, atol=0.0)
        # exact equality must hold: same formula, same library, deterministic
        assert cached == direct


def test_vector_path_unaffected_by_cache():
    p_vals = np.array([0.001, 0.37, 0.9, 1e-8], dtype=np.float64)
    out = Q_inverse(p_vals)
    assert out.shape == p_vals.shape
    # element 0 equals the single-element (cached) call bit-for-bit
    single = Q_inverse(np.array([0.001]))
    np.testing.assert_array_equal(out[:1], single)


def test_output_shape_mirrors_input():
    assert Q_inverse(0.001).shape == ()          # python float -> 0-d
    assert Q_inverse(np.array(0.001)).shape == ()  # numpy scalar -> 0-d
    assert Q_inverse(np.array([0.001])).shape == (1,)
    assert Q_inverse(np.array([0.001, 0.5])).shape == (2,)


def test_compute_pd_unchanged():
    """compute_PD still matches P_D = Q(Q^{-1}(P_FA) - sqrt(D)) exactly."""
    from scipy.special import erfc, erfinv

    p_fa = 0.001
    d = np.array([0.0, 0.5, 1.0, 5.0, 20.0], dtype=np.float64)
    q_inv = np.sqrt(2.0) * erfinv(1.0 - 2.0 * p_fa)
    expected = 0.5 * erfc((q_inv - np.sqrt(np.maximum(d, 1e-10))) / np.sqrt(2.0))
    np.testing.assert_allclose(compute_PD(d, p_fa), expected, rtol=0.0, atol=0.0)