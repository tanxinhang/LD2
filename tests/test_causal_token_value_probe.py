import numpy as np
import pytest

from tools.probe_causal_token_value import (
    PolicyDecision,
    PolicyRuntime,
    choose_candidate,
    qos_components,
    summarize_branch,
)


def test_qos_components_use_medium_thresholds():
    assert qos_components(np.array([0.90, 0.80, 0.70, 0.60])) == pytest.approx(
        (0.75, 0.70, 0.60, 0.0))
    steady, weak3, worst, feasible = qos_components(
        np.array([0.95, 0.90, 0.85, 0.80]))
    assert steady >= 0.80
    assert weak3 >= 0.70
    assert worst >= 0.60
    assert feasible == 1.0


def test_summarize_branch_keeps_bits_out_of_qos_value():
    base = {
        "P_D_q": np.array([0.90, 0.85, 0.80, 0.75]),
        "learned_comm_bits": 100.0,
    }
    expensive = dict(base, learned_comm_bits=1000.0)
    a = summarize_branch([base], gamma=0.95)
    b = summarize_branch([expensive], gamma=0.95)
    assert a["qos"] == b["qos"]
    assert a["bits"] != b["bits"]
    np.testing.assert_allclose(a["pd"], base["P_D_q"])
    assert a["worst_id"] == 3


def test_choose_candidate_cycles_active_edges_and_records_rank():
    decision = PolicyDecision(
        actions={},
        messages=np.zeros((2, 4)),
        rates=np.array([2, 2]),
        token_masks=np.array([[1, 0], [1, 1]], dtype=float),
        claim_scores=np.array([[0.9, 0.1], [0.2, 0.8]], dtype=float),
        policy_latent=np.zeros((2, 8), dtype=float),
        comm_fractions=None,
        sensing_weights=None,
        next_runtime=PolicyRuntime(None, 0, None),
    )
    assert choose_candidate(decision, 0) == (0, 0, 1)
    assert choose_candidate(decision, 1) == (1, 0, 2)
    assert choose_candidate(decision, 2) == (1, 1, 1)
    assert choose_candidate(decision, 3) == (0, 0, 1)
