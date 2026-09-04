"""O2 property tests: canonical vectorized deflection path is bit-for-bit
identical to the scalar slow-path formulas (roadmap 2026-08-29).

Reference implementation below re-executes the ORIGINAL scalar slow path
(compute_raw_deflection + compute_dd_effectiveness + compute_dd_phys_gain,
same call order) so the vectorized branch can be diffed field-by-field.

Conventions kept for the reference:
- chi_rep is pinned to a constant via patch (the reliability draw is RNG-state
  and must not be consumed twice) -- the reference uses the same constant;
- Swerling tests use two independent same-seed DeflectionComputer instances so
  the Swerling RNG stream (drawn in i -> j -> q order) stays aligned;
- binary mode reference implements the legacy 1[g_dd >= g_min] gate exactly.
"""

import numpy as np
from unittest.mock import patch

from uav_isac.physical.deflection import DeflectionComputer
from uav_isac.physical.otfs import (
    compute_dd_effectiveness,
    compute_dd_phys_gain,
)
from uav_isac.utils.types import DeflectionEntry

# Patched compute_report_link_reliability return value; the reference uses the
# same pinned constant so both sides see identical chi_rep.
CHI_REP_PIN = 0.8


def _geometry(rng: np.random.Generator, K: int, Q: int):
    uav_pos = rng.uniform(50.0, 950.0, size=(K, 3))
    uav_pos[:, 2] = 100.0
    uav_vel = np.zeros((K, 3), dtype=np.float64)
    tgt_pos = rng.uniform(0.0, 1000.0, size=(Q, 3))
    tgt_pos[:, 2] = 0.0
    tgt_vel = np.zeros((Q, 3), dtype=np.float64)
    return uav_pos, uav_vel, tgt_pos, tgt_vel


def _make(
    seed: int,
    *,
    dd_gain_mode: str,
    use_swerling: bool,
    use_report_link: bool = True,
) -> DeflectionComputer:
    rng = np.random.default_rng(seed)
    return DeflectionComputer(
        fc=2.8e10, delta_f=1.5625e4, T_sym=6.4e-5, M=64, N=16,
        kT=4.0e-21, B=1.0e6, NF_dB=4.0, P_sense=0.0251, P_report=0.25,
        ric_K=6.0, rcs=1.0, g_min=0.5, rng=rng,
        g_tx_dBi=16.0, g_rx_dBi=16.0, n_cpi=1, c_det=1.0,
        use_los_prob=False, use_swerling=use_swerling,
        use_report_link=use_report_link, dd_gain_mode=dd_gain_mode,
    )


def _scalar_entries(
    dc: DeflectionComputer,
    uav_pos, uav_vel, tgt_pos, tgt_vel, roles, fc_position,
    sensing_power_w,
):
    """Original scalar slow path, same call order as pre-O2 code."""
    from uav_isac.physical.geometry import compute_all_bistatic_params
    from uav_isac.physical.deflection import compute_raw_deflection

    tau, nu, alpha = compute_all_bistatic_params(
        uav_pos, uav_vel, tgt_pos, tgt_vel, roles, dc.fc, dc.rcs,
        role_agnostic=False)
    tx = np.where(roles == 0)[0]
    rx = np.where(roles == 1)[0]
    chi_rep_by_rx = {
        int(j): CHI_REP_PIN if dc.use_report_link else 1.0
        for j in rx
    }
    entries = []
    for i in tx:
        for j in rx:
            if i == j:
                continue
            chi_rep = chi_rep_by_rx[int(j)]
            for q in range(tgt_pos.shape[0]):
                if np.isinf(tau[i, j, q]):
                    continue
                target_power_w = (
                    dc.P_sense if sensing_power_w is None
                    else max(float(sensing_power_w[i, q]), 0.0))
                d_raw = compute_raw_deflection(
                    alpha[i, j, q], target_power_w, dc.T_sym, dc.M, dc.N,
                    dc.noise_power, antenna_gain=dc.antenna_gain,
                    n_cpi=dc.n_cpi, c_det=dc.c_det)
                g_dd = compute_dd_effectiveness(
                    tau[i, j, q], nu[i, j, q], dc.delta_f, dc.T_sym,
                    dc.M, dc.N)
                if dc.use_swerling:
                    d_raw = d_raw * float(dc.rng.exponential(1.0))
                if dc.dd_gain_mode == "continuous":
                    phys_gain = compute_dd_phys_gain(
                        tau[i, j, q], nu[i, j, q], dc.delta_f, dc.T_sym,
                        dc.M, dc.N)
                    d_eff = chi_rep * d_raw * phys_gain
                else:
                    d_eff = chi_rep * d_raw if g_dd >= dc.g_min else 0.0
                entries.append(DeflectionEntry(
                    i=int(i), j=int(j), q=int(q),
                    tau=float(tau[i, j, q]), nu=float(nu[i, j, q]),
                    alpha=float(alpha[i, j, q]),
                    d_raw=float(d_raw), g_dd=float(g_dd),
                    chi_rep=float(chi_rep), d_eff=float(d_eff)))
    return entries


def _run(dc: DeflectionComputer, *, seed: int, K: int, Q: int, roles,
         sensing_power_w=None):
    rng = np.random.default_rng(seed + 1)
    uav_pos, uav_vel, tgt_pos, tgt_vel = _geometry(rng, K, Q)
    fc_position = np.array([500.0, 500.0, 100.0], dtype=np.float64)
    entries = dc.compute(
        uav_pos, uav_vel, tgt_pos, tgt_vel, roles, fc_position,
        sensing_power_w=sensing_power_w)
    return entries, (uav_pos, uav_vel, tgt_pos, tgt_vel, fc_position)


def _assert_entries_equal(entries, ref):
    """Numerical equivalence: 1-2 ULP observed from expression-order
    differences between the batch (array) and scalar paths.  Bit-for-bit
    equality is intentionally NOT claimed (floating point, C1); the bound is
    1e-13 relative, which is ~1e3x looser than one ULP and far below any
    physical significance of these coefficients."""
    assert len(entries) == len(ref)
    for got, exp in zip(entries, ref):
        for field in ("i", "j", "q"):
            assert getattr(got, field) == getattr(exp, field)
        for field in ("tau", "nu", "alpha", "d_raw", "g_dd", "chi_rep", "d_eff"):
            assert np.isclose(
                float(getattr(got, field)), float(getattr(exp, field)),
                rtol=1e-13, atol=0.0), (
                f"field {field} mismatch: {getattr(got, field)} vs "
                f"{getattr(exp, field)}")


def _gen(K: int, Q: int, seed: int):
    rng = np.random.default_rng(seed)
    return _geometry(rng, K, Q)


def test_canonical_continuous_batch_matches_scalar_reference():
    K, Q = 4, 2
    roles = np.array([0, 0, 1, 1], dtype=np.int64)
    with patch(
        "uav_isac.physical.deflection.compute_report_link_reliability",
        return_value=CHI_REP_PIN,
    ):
        dc = _make(7, dd_gain_mode="continuous", use_swerling=False)
        entries, (uav_pos, uav_vel, tgt_pos, tgt_vel, fc_position) = _run(
            dc, seed=99, K=K, Q=Q, roles=roles)
        ref = _scalar_entries(
            dc, uav_pos, uav_vel, tgt_pos, tgt_vel, roles, fc_position, None)
        _assert_entries_equal(entries, ref)


def test_canonical_with_sensing_power_matrix_matches_reference():
    K, Q = 4, 2
    roles = np.array([0, 0, 1, 1], dtype=np.int64)
    sensing = np.array([
        [0.0251, 0.01], [0.005, 0.0251], [0.0, 0.02], [0.015, 0.0],
    ], dtype=np.float64)
    with patch(
        "uav_isac.physical.deflection.compute_report_link_reliability",
        return_value=CHI_REP_PIN,
    ):
        dc = _make(11, dd_gain_mode="continuous", use_swerling=False)
        entries, (uav_pos, uav_vel, tgt_pos, tgt_vel, fc_position) = _run(
            dc, seed=3, K=K, Q=Q, roles=roles, sensing_power_w=sensing)
        ref = _scalar_entries(
            dc, uav_pos, uav_vel, tgt_pos, tgt_vel, roles, fc_position,
            sensing)
        _assert_entries_equal(entries, ref)


def test_dense_u2u_unit_geometry_scales_linearly_and_matches_reference():
    """The tensor shortcut preserves physics and canonical entry ordering."""
    K, Q = 4, 3
    roles = np.array([0, 0, 1, 1], dtype=np.int64)
    rng = np.random.default_rng(20260829)
    uav_pos, uav_vel, tgt_pos, tgt_vel = _geometry(rng, K, Q)
    fc_position = np.array([500.0, 500.0, 100.0], dtype=np.float64)
    power = rng.uniform(0.0, 0.04, size=(K, Q))
    dc = _make(
        71, dd_gain_mode="continuous", use_swerling=False,
        use_report_link=False)

    unit = dc.compute_dense(
        uav_pos, uav_vel, tgt_pos, tgt_vel, roles, fc_position,
        sensing_power_w=np.ones((K, Q), dtype=np.float64))
    scaled_entries = unit.to_entries(power)
    direct_entries = dc.compute(
        uav_pos, uav_vel, tgt_pos, tgt_vel, roles, fc_position,
        sensing_power_w=power)
    scalar_reference = _scalar_entries(
        dc, uav_pos, uav_vel, tgt_pos, tgt_vel, roles, fc_position,
        power)

    _assert_entries_equal(scaled_entries, direct_entries)
    _assert_entries_equal(scaled_entries, scalar_reference)


def test_out_of_support_target_yields_zero_deflection():
    K, Q = 4, 1
    roles = np.array([0, 0, 1, 1], dtype=np.int64)
    with patch(
        "uav_isac.physical.deflection.compute_report_link_reliability",
        return_value=CHI_REP_PIN,
    ):
        dc = _make(2, dd_gain_mode="continuous", use_swerling=False)
        uav_pos = np.array([
            [100, 100, 100], [900, 100, 100], [100, 900, 100], [900, 900, 100],
        ], dtype=np.float64)
        uav_vel = np.zeros((K, 3), dtype=np.float64)
        # target placed far beyond the DD unambiguous delay range:
        # tau = R/c >= 100 us > 1/delta_f = 64 us -> out of support, so the
        # continuous gain is exactly 0 and d_eff must be 0 too.
        tgt_pos = np.array([[30000.0, 0.0, 0.0]], dtype=np.float64)
        tgt_vel = np.zeros((1, 3), dtype=np.float64)
        fc_position = np.array([500.0, 500.0, 100.0], dtype=np.float64)
        entries = dc.compute(
            uav_pos, uav_vel, tgt_pos, tgt_vel, roles, fc_position)
        assert entries
        for e in entries:
            assert e.d_eff == 0.0


def test_binary_scalar_path_preserved():
    K, Q = 4, 2
    roles = np.array([0, 0, 1, 1], dtype=np.int64)
    with patch(
        "uav_isac.physical.deflection.compute_report_link_reliability",
        return_value=CHI_REP_PIN,
    ):
        dc = _make(5, dd_gain_mode="binary", use_swerling=False)
        entries, (uav_pos, uav_vel, tgt_pos, tgt_vel, fc_position) = _run(
            dc, seed=17, K=K, Q=Q, roles=roles)
        ref = _scalar_entries(
            dc, uav_pos, uav_vel, tgt_pos, tgt_vel, roles, fc_position, None)
        _assert_entries_equal(entries, ref)


def test_swerling_scalar_path_preserved():
    """Swerling branch keeps the original RNG-order scalar loop."""
    K, Q = 4, 2
    roles = np.array([0, 0, 1, 1], dtype=np.int64)
    with patch(
        "uav_isac.physical.deflection.compute_report_link_reliability",
        return_value=CHI_REP_PIN,
    ):
        # two independent same-seed instances: chi_rep is patched (no RNG use),
        # the only RNG consumption is the per-edge Swerling draw in i->j->q
        # order, so both streams stay aligned.
        dc = _make(13, dd_gain_mode="continuous", use_swerling=True)
        dc_ref = _make(13, dd_gain_mode="continuous", use_swerling=True)
        entries, (uav_pos, uav_vel, tgt_pos, tgt_vel, fc_position) = _run(
            dc, seed=23, K=K, Q=Q, roles=roles)
        ref = _scalar_entries(
            dc_ref, uav_pos, uav_vel, tgt_pos, tgt_vel, roles,
            fc_position, None)
        _assert_entries_equal(entries, ref)
