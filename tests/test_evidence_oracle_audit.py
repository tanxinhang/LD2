import numpy as np

from uav_isac.evaluation.evidence_oracle_audit import (
    lossless_quality_topk_pd,
    receiver_local_and_global_pd,
    summarize_evidence_oracle,
    summarize_lossless_topk_capacity,
)
from uav_isac.utils.types import DeflectionEntry


def _entry(i, j, q, d_eff):
    return DeflectionEntry(
        i=i, j=j, q=q, tau=0.0, nu=0.0, alpha=1.0,
        d_raw=d_eff, g_dd=1.0, chi_rep=1.0, d_eff=d_eff)


def test_global_evidence_fuses_multiple_receiver_deflections():
    entries = [_entry(0, 1, 0, 4.0), _entry(2, 3, 0, 4.0)]
    result = receiver_local_and_global_pd(
        [(0, 1, 0), (2, 3, 0)],
        entries,
        num_agents=4,
        num_targets=1,
        p_fa=0.001,
    )
    assert result["global_pd"][0] > result["local_best_pd"][0]
    np.testing.assert_allclose(
        result["receiver_deflection"][:, 0],
        np.array([0.0, 4.0, 0.0, 4.0]),
    )


def test_evidence_oracle_gate_uses_paired_episode_gap():
    local = [np.full((20, 4), 0.50) for _ in range(12)]
    central = [np.full((20, 4), 0.56) for _ in range(12)]
    summary = summarize_evidence_oracle(
        central, local, bootstrap_samples=20)
    np.testing.assert_allclose(
        summary["eval_evidence_oracle_worst_delta"], 0.06)
    assert summary["eval_evidence_oracle_gate_pass"] is True


def test_lossless_quality_topk_uses_local_pre_observation_ranking():
    receiver_d = np.asarray([
        [9.0, 1.0, 0.0],
        [2.0, 8.0, 0.0],
        [3.0, 4.0, 7.0],
    ])
    top1 = lossless_quality_topk_pd(receiver_d, p_fa=0.001, topk=1)
    np.testing.assert_array_equal(
        top1["selected_mask"],
        np.asarray([
            [True, False, False],
            [False, True, False],
            [False, False, True],
        ]),
    )
    # Owners keep their local evidence; no peer chose a second target.
    np.testing.assert_allclose(
        top1["fused_deflection"], np.asarray([9.0, 8.0, 7.0]))

    top2 = lossless_quality_topk_pd(receiver_d, p_fa=0.001, topk=2)
    np.testing.assert_allclose(
        top2["fused_deflection"], np.asarray([11.0, 13.0, 7.0]))


def test_lossless_topk_capacity_gate_recovers_oracle_gap():
    local = [np.full((20, 4), 0.50) for _ in range(12)]
    central = [np.full((20, 4), 0.60) for _ in range(12)]
    top1 = [np.full((20, 4), 0.56) for _ in range(12)]
    top2 = [np.full((20, 4), 0.59) for _ in range(12)]
    summary = summarize_lossless_topk_capacity(
        central,
        local,
        {1: top1, 2: top2},
        bootstrap_samples=20,
    )
    np.testing.assert_allclose(
        summary["eval_evidence_quality_top1_worst_recovery"], 0.60)
    assert summary["eval_evidence_topk_capacity_gate_pass"] is True
