import numpy as np

from tools.probe_crisis_gate import choose_threshold, evaluate, temporal_hysteresis_gate


def test_threshold_is_fit_from_scores_and_separates_simple_crises():
    score = np.asarray([0.05, 0.10, 0.20, 0.75, 0.85, 0.95])
    label = np.asarray([0, 0, 0, 1, 1, 1], dtype=bool)
    threshold = choose_threshold(score, label)
    prediction = score >= threshold
    np.testing.assert_array_equal(prediction, label)


def test_binary_metrics_give_half_credit_to_auc_ties():
    result = evaluate(
        np.asarray([0.1, 0.5, 0.5, 0.9]),
        np.asarray([False, False, True, True]),
        threshold=0.5,
    )
    assert result["roc_auc"] == 0.875
    assert result["balanced_accuracy"] == 0.75
    assert result["precision"] == 2 / 3
    assert result["recall"] == 1.0


def test_temporal_gate_repairs_underload_immediately_and_waits_on_quality():
    rows = {
        "seed": np.asarray([1, 1, 1, 1, 1]),
        "step": np.arange(5),
        "target": np.zeros(5, dtype=np.int64),
        "claim_count": np.asarray([1, 2, 2, 2, 2]),
        "semantic_crisis": np.asarray([0.1, 0.8, 0.8, 0.8, 0.8]),
    }
    gate = temporal_hysteresis_gate(
        rows,
        semantic_threshold=0.7,
        stagnation_frames=3,
        improvement_epsilon=0.02,
        ema_alpha=0.3,
    )
    np.testing.assert_array_equal(gate, [True, False, False, True, True])
