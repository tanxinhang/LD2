import numpy as np

from tools.probe_targetwise_causal_value import (
    aligned_probabilities,
    select_conservative_gate,
)


class _ProbabilityModel:
    classes_ = np.array([0, 2])

    def predict_proba(self, x):
        return np.tile(np.array([[0.25, 0.75]]), (x.shape[0], 1))


def test_aligned_probabilities_preserve_missing_target_slots():
    result = aligned_probabilities(_ProbabilityModel(), np.zeros((2, 3)), 4)
    np.testing.assert_allclose(result[0], [0.25, 0.0, 0.75, 0.0])
    np.testing.assert_allclose(result.sum(axis=1), 1.0)


def test_conservative_gate_obeys_critical_recall_constraint():
    effect = np.array([[1.0], [-1.0], [0.8], [-0.5]])
    uncertainty = np.zeros_like(effect)
    worst_probability = np.ones_like(effect)
    delta_qos = np.array([1.0, -1.0, 0.5, -0.5])
    repair = np.array([True, False, True, False])
    result = select_conservative_gate(
        effect, uncertainty, worst_probability, delta_qos, repair, 1.0)
    assert result["recall_constraint_satisfied"]
    assert result["critical_recall"] == 1.0
    assert result["gain"] > 0.0
